"""Source-assisted visible color-support envelopes. No anatomy, segmentation model or verified identity."""
import json
import math
import re
import time

PARAMS = dict(hue_distance=12, saturation_delta=80, value_delta=70, lab_distance=35,
              minimum_saturation=35, closing_kernel=3, minimum_area=8,
              area_ratio=[.25, 2.2], displacement_per_frame=60, nearest_margin=6,
              shared_fraction=.25, axis_minimum_area=12, axis_eigen_ratio=1.5)


def validate(job):
    if 'regions' not in job:
        return []
    regions = job['regions']
    if (type(regions) is not list or not 1 <= len(regions) <= 8
            or job.get('mode') != 'single-anchor'):
        raise ValueError('regions require 1..8 explicit single-anchor definitions')
    w, h = job['width'], job['height']
    if type(w) is not int or type(h) is not int or w*h > 2073600 or min(w,h) < 2:
        raise ValueError('bounded region image required')
    point_ids = {p['point_id'] for p in job['points']}
    ids = []
    for r in regions:
        if (type(r) is not dict or set(r) != {'point_id','label','seed_box','seed_xy'}
                or not re.fullmatch('[A-Za-z0-9_-]{1,80}', str(r['point_id']))
                or r['point_id'] not in point_ids or type(r['label']) is not str
                or not 1 <= len(r['label']) <= 100 or any(ord(c)<32 for c in r['label'])):
            raise ValueError('explicit bounded region and existing point ID required')
        box, xy = r['seed_box'], r['seed_xy']
        if (type(box) is not list or len(box)!=4 or any(type(x) is not int for x in box)
                or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h
                or min(box[2]-box[0],box[3]-box[1]) < 2
                or type(xy) is not list or len(xy)!=2 or any(type(x) is not int for x in xy)
                or not box[0] <= xy[0] < box[2] or not box[1] <= xy[1] < box[3]):
            raise ValueError('region uses native integer exclusive-edge box and interior color seed')
        ids.append(r['point_id'])
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate region ID')
    return regions


def _converted(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _components(hsv, lab, profile):
    import cv2
    import numpy as np
    hue = np.abs(hsv[:,:,0].astype(np.int16)-profile['hsv'][0])
    hue = np.minimum(hue, 180-hue)
    delta = lab.astype(np.int16)-np.asarray(profile['lab'], dtype=np.int16)
    lab2 = np.sum(delta.astype(np.int32)**2, axis=2)
    mask = ((hue<=PARAMS['hue_distance'])
            & (np.abs(hsv[:,:,1].astype(np.int16)-profile['hsv'][1])<=PARAMS['saturation_delta'])
            & (hsv[:,:,1]>=PARAMS['minimum_saturation'])
            & (np.abs(hsv[:,:,2].astype(np.int16)-profile['hsv'][2])<=PARAMS['value_delta'])
            & (lab2<=PARAMS['lab_distance']**2)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return labels, stats, centers


def _support(labels, stats, centers, index):
    import numpy as np
    x,y,w,h,area = map(int, stats[index])
    mask = labels==index
    axis = None
    if area >= PARAMS['axis_minimum_area']:
        ys,xs = np.nonzero(mask)
        cov = np.cov(np.stack([xs,ys]), bias=True)
        values,vectors = np.linalg.eigh(cov)
        if values[1] > 0 and values[1]/max(values[0],1e-12) >= PARAMS['axis_eigen_ratio']:
            v = vectors[:,1]
            axis = math.degrees(math.atan2(float(v[1]),float(v[0])))%180.
    return dict(bbox=[x,y,x+w,y+h], centroid=[float(v) for v in centers[index]],
                area_px=area, axis_degrees_mod180=axis, mask=mask)


def calibrate(rgb, regions):
    """Only initial source pixels. Area means support pixel count after closing, not bbox area."""
    import numpy as np
    hsv,lab = _converted(rgb)
    result = {}
    for r in regions:
        x,y = r['seed_xy']; x0,y0,x1,y1 = r['seed_box']
        p = dict(hsv=np.median(hsv[max(0,y-1):y+2,max(0,x-1):x+2].reshape(-1,3),axis=0).astype(int).tolist(),
                 lab=np.median(lab[max(0,y-1):y+2,max(0,x-1):x+2].reshape(-1,3),axis=0).astype(int).tolist())
        anchor_h=int(hsv[y,x,0]); patch_h=hsv[max(0,y-1):y+2,max(0,x-1):x+2,0].astype(int)
        p['hsv'][0]=int((anchor_h+np.median((patch_h-anchor_h+90)%180-90))%180)
        if p['hsv'][1] < PARAMS['minimum_saturation']:
            raise ValueError('selected region lacks a supported chromatic seed')
        labels,stats,centers = _components(hsv,lab,p); index = int(labels[y,x])
        if index == 0 or int(stats[index,4]) < PARAMS['minimum_area']:
            raise ValueError('source color seed has no sufficiently sized component')
        support = _support(labels,stats,centers,index); box = support['bbox']
        if not (x0-8 <= box[0] and y0-8 <= box[1] and box[2] <= x1+8 and box[3] <= y1+8):
            raise ValueError('initial support extends beyond the declared source part')
        support.pop('mask'); p.update(source=support)
        result[r['point_id']] = p
    return result


def track(rgb_frames, job):
    """Current-frame pixels + frozen source only; never consumes NN coordinates or future references."""
    import numpy as np
    started = time.monotonic(); definitions = validate(job)
    profiles = calibrate(rgb_frames[0], definitions)
    points = {p['point_id']:p for p in job['points']}
    for pid,p in profiles.items():
        if math.dist(points[pid]['anchors'][0]['xy'],p['source']['centroid']) > 1e-5:
            raise ValueError('control query must equal the same initial support centroid')
    last = {pid:dict(frame=job['start_frame'],centroid=p['source']['centroid']) for pid,p in profiles.items()}
    previous = {}; had_gap = set(); frames = []
    for offset,rgb in enumerate(rgb_frames):
        n = job['start_frame']+offset; hsv,lab = _converted(rgb); choices = {}; reasons = {}
        for r in definitions:
            pid = r['point_id']; p = profiles[pid]; labels,stats,centers = _components(hsv,lab,p)
            distance_limit = PARAMS['displacement_per_frame']*max(1,n-last[pid]['frame'])
            choices_with_distance = [(math.dist(last[pid]['centroid'],centers[i]),i) for i in range(1,len(stats))
                                     if stats[i,4]>=PARAMS['minimum_area']
                                     and math.dist(last[pid]['centroid'],centers[i])<=distance_limit]
            choices_with_distance.sort()
            if not choices_with_distance:
                choices[pid]=None; reasons[pid]='no_local_color_support'; continue
            choice = _support(labels,stats,centers,choices_with_distance[0][1]); choices[pid]=choice
            if len(choices_with_distance)>1 and choices_with_distance[1][0]-choices_with_distance[0][0] < PARAMS['nearest_margin']:
                reasons[pid]='ambiguous_nearest_support'
            ratio = choice['area_px']/p['source']['area_px']
            if not PARAMS['area_ratio'][0] <= ratio <= PARAMS['area_ratio'][1]:
                reasons[pid]='support_area_changed'
        ids = list(choices)
        for i,pid in enumerate(ids):
            a = choices[pid]
            if a is None:continue
            for other in ids[i+1:]:
                b = choices[other]
                if b is not None and np.count_nonzero(a['mask']&b['mask'])/min(a['area_px'],b['area_px']) > PARAMS['shared_fraction']:
                    reasons[pid]=reasons[other]='ambiguous_shared_appearance_support'
        observations = []
        for r in definitions:
            pid = r['point_id']; choice = choices[pid]; reason = reasons.get(pid)
            common = dict(region_id=pid,label=r['label'],identity_verified=False,
                          claim_scope='source_selected_color_support',physical_occlusion_verified=False)
            if reason:
                had_gap.add(pid)
                observations.append(dict(common,status='unknown',reason=reason,bbox=None,centroid=None,
                    axis_degrees_mod180=None,continuous_from_previous=False,displacement_px=None,
                    diagnostic_candidate_bbox=None if choice is None else choice['bbox']))
            else:
                usable = offset>0; continuous = usable and previous.get(pid,{}).get('status')=='region_hypothesis'
                delta = [choice['centroid'][i]-last[pid]['centroid'][i] for i in range(2)] if continuous else None
                observations.append(dict(common,status='region_hypothesis' if usable else 'source_anchor',reason=None,
                    bbox=choice['bbox'],centroid=choice['centroid'],area_px=choice['area_px'],
                    axis_degrees_mod180=choice['axis_degrees_mod180'],continuous_from_previous=continuous,
                    displacement_px=delta,reacquired_after_gap=pid in had_gap))
                last[pid]=dict(frame=n,centroid=choice['centroid'])
        previous={r['region_id']:r for r in observations}
        frames.append(dict(frame=n,regions=observations))
    return dict(schema='team-eye.part-regions/1',parameters=PARAMS,profiles=profiles,frames=frames,
                elapsed_seconds=time.monotonic()-started,
                coordinate_convention='native pixels, [x0,y0,x1,y1) exclusive edges',
                color_convention='OpenCV8bit HSV H0..179 circular distance; Lab8bit Euclidean',
                limits='Selected visible color support only, not full part segmentation/anatomy. Shared support is an ambiguity diagnostic, not proof of occlusion. Nearest matches may swap. Reacquisition is unverified and never bridges a gap.')


def _ranges(values):
    out=[]
    for n in values:
        if out and n==out[-1][1]+1:out[-1][1]=n
        else:out.append([n,n])
    return out


def _axis_class(angle):
    if angle is None:return 'unknown'
    if angle<22.5 or angle>=157.5:return 'horizontal'
    if 67.5<=angle<=112.5:return 'vertical'
    return 'oblique'


def _sampled_trajectory(run, times):
    """Observed color-centroid polyline only; no interpolation, motion model or noise correction."""
    xy = [r['centroid'] for _, r in run]
    distances = [math.dist(xy[0], p) for p in xy]
    farthest = max(range(len(run)), key=lambda i: distances[i])
    path = math.fsum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    roles = {}
    for role, i in [('start', 0), ('end', len(run)-1),
                    ('min_x', min(range(len(run)), key=lambda i: xy[i][0])),
                    ('max_x', max(range(len(run)), key=lambda i: xy[i][0])),
                    ('min_y', min(range(len(run)), key=lambda i: xy[i][1])),
                    ('max_y', max(range(len(run)), key=lambda i: xy[i][1])),
                    ('max_excursion', farthest)]:
        roles.setdefault(i, []).append(role)
    return dict(sample_count=len(run), sampled_path_length_px=path,
                net_distance_px=distances[-1], max_excursion_from_start_px=distances[farthest],
                max_excursion_frame=run[farthest][0],
                max_excursion_pts_seconds=times[run[farthest][0]],
                net_to_path_ratio=distances[-1]/path if path else None,
                key_observations=[dict(frame=run[i][0], pts_seconds=times[run[i][0]],
                                       centroid_px=list(xy[i]), roles=roles[i]) for i in sorted(roles)])


def build_report(packet, bindings, start, end, validate_bindings, evidence_frames=None):
    """Compact current source-indexed facts. Cached read path uses stdlib only."""
    if 'shot_check'in packet:
        import runpy
        from pathlib import Path
        shot_context=runpy.run_path(str(Path(__file__).with_name('shot_boundaries.py')))['context'](packet)
        if shot_context['status']=='requires_shot_local_retracking':
            raise ValueError('cross-shot region calibration/model context requires fresh source observations per shot')
    if type(start) is not int or type(end) is not int or not 0<=start<end or end-start+1>160:
        raise ValueError('bounded native region report interval required')
    if evidence_frames is not None and (type(evidence_frames) is not list or not 1<=len(evidence_frames)<=3
            or any(type(n) is not int or not start<=n<=end for n in evidence_frames)
            or len(set(evidence_frames))!=len(evidence_frames)):
        raise ValueError('choose 1..3 unique native source frames inside the report interval')
    region_data=packet.get('region_observations');job=packet['job'];definitions=validate(job)
    if (not definitions or type(region_data) is not dict or region_data.get('schema')!='team-eye.part-regions/1'
            or region_data.get('parameters')!=PARAMS):
        raise ValueError('supported explicit region observations required')
    bound=validate_bindings(packet,bindings);wanted={x['point_id'] for x in bound};available={r['point_id']for r in definitions}
    if not wanted<=available:raise ValueError('report binding is not an observed region')
    raw=region_data.get('frames');base=packet.get('frames');w,h=job['width'],job['height']
    if (type(raw) is not list or type(base) is not list or not 1<=len(raw)==len(base)<=160
            or [f.get('frame') for f in raw]!=list(range(job['start_frame'],job['end_frame']+1))
            or [f.get('frame') for f in base]!=[f['frame']for f in raw]):
        raise ValueError('complete unique native region frames required')
    times={};previous={};table={}
    for f,b in zip(raw,base):
        t=b.get('pts_seconds')
        if type(t) not in(int,float) or not math.isfinite(t) or t<0 or times and t<=list(times.values())[-1]:
            raise ValueError('finite increasing source PTS required')
        times[f['frame']]=t;rows=f.get('regions')
        if type(rows) is not list or len(rows)!=len(available) or {r.get('region_id')for r in rows}!=available:
            raise ValueError('one observation per defined region required')
        for r in rows:
            pid=r['region_id'];status=r.get('status')
            if (status not in('unknown','source_anchor','region_hypothesis')
                    or r.get('identity_verified')is not False or r.get('physical_occlusion_verified')is not False
                    or r.get('claim_scope')!='source_selected_color_support'
                    or type(r.get('continuous_from_previous'))is not bool):
                raise ValueError('region claim or state contract violated')
            if status=='unknown':
                diagnostic=r.get('diagnostic_candidate_bbox')
                if diagnostic is not None and(type(diagnostic)is not list or len(diagnostic)!=4 or any(type(v)not in(int,float)or not math.isfinite(v)for v in diagnostic)or not 0<=diagnostic[0]<diagnostic[2]<=w or not 0<=diagnostic[1]<diagnostic[3]<=h):
                    raise ValueError('diagnostic region crop must be finite and in-frame')
                if type(r.get('reason'))is not str or not 1<=len(r['reason'])<=100:raise ValueError('bounded region uncertainty reason required')
                if any(r.get(k)is not None for k in ['bbox','centroid','axis_degrees_mod180','displacement_px'])or r['continuous_from_previous']:
                    raise ValueError('unknown region cannot carry usable geometry')
            else:
                box,xy=r.get('bbox'),r.get('centroid');axis=r.get('axis_degrees_mod180')
                if (type(box)is not list or len(box)!=4 or any(type(v)not in(int,float)or not math.isfinite(v)for v in box)
                        or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h
                        or type(xy)is not list or len(xy)!=2 or any(type(v)not in(int,float)or not math.isfinite(v)for v in xy)
                        or not box[0]<=xy[0]<box[2] or not box[1]<=xy[1]<box[3]
                        or axis is not None and(type(axis)not in(int,float)or not math.isfinite(axis)or not 0<=axis<180)):
                    raise ValueError('finite in-frame support geometry required')
                if (status=='source_anchor') != (f['frame']==job['start_frame']):
                    raise ValueError('source anchor must stay excluded at initial frame only')
                old=previous.get(pid)
                if r['continuous_from_previous']:
                    delta=r.get('displacement_px')
                    if (not old or old['status']!='region_hypothesis' or status!='region_hypothesis'
                            or type(delta)is not list or len(delta)!=2
                            or any(type(v)not in(int,float)or not math.isfinite(v)for v in delta)
                            or any(abs(delta[i]-(xy[i]-old['centroid'][i]))>1e-5 for i in range(2))):
                        raise ValueError('region displacement bridges a gap or corrupts coordinates')
                elif r.get('displacement_px')is not None:raise ValueError('disconnected hypothesis cannot carry motion')
            previous[pid]=r
        table[f['frame']]={r['region_id']:r for r in rows}
    if start not in table or end not in table:raise ValueError('requested region interval is missing')
    summaries=[];evidence=set();ambiguity_ranges=[]
    for feature in bound:
        pid=feature['point_id'];runs=[];current=[];unknown=[];reasons={};axis_counts={k:0 for k in ['horizontal','vertical','oblique','unknown']}
        for n in range(start,end+1):
            r=table[n][pid]
            if r['status']!='region_hypothesis':
                if current:runs.append(current);current=[]
                if r['status']=='unknown':unknown.append(n);reasons[r['reason']]=reasons.get(r['reason'],0)+1
                continue
            if current and not r['continuous_from_previous']:runs.append(current);current=[]
            current.append((n,r));axis_counts[_axis_class(r['axis_degrees_mod180'])]+=1
        if current:runs.append(current)
        segments=[]
        for run in runs:
            first,last=run[0],run[-1];dx=last[1]['centroid'][0]-first[1]['centroid'][0];dy=last[1]['centroid'][1]-first[1]['centroid'][1]
            direction='single_frame_no_motion'if len(run)==1 else'within_2px'if math.hypot(dx,dy)<2 else['right','down_right','down','down_left','left','up_left','up','up_right'][int(math.floor((math.degrees(math.atan2(dy,dx))+22.5)/45))%8]
            segments.append(dict(frames=[first[0],last[0]],net_displacement_px=[dx,dy],screen_direction=direction,
                                 trajectory=_sampled_trajectory(run,times)))
            evidence.update([first[0],last[0]])
        ambiguity_ranges.extend(_ranges(unknown))
        good=[(n,table[n][pid])for n in range(start,end+1)if table[n][pid]['status']=='region_hypothesis']
        widths=[r['bbox'][2]-r['bbox'][0]for n,r in good];heights=[r['bbox'][3]-r['bbox'][1]for n,r in good]
        summaries.append(dict(**feature,usable_region_frames=len(good),unknown_ranges=_ranges(unknown),unknown_reasons=reasons,
            width_range_px=[min(widths),max(widths)]if widths else None,height_range_px=[min(heights),max(heights)]if heights else None,
            undirected_support_axis_counts=axis_counts,continuous_segments=segments[:8],omitted_segments=max(0,len(segments)-8)))
    all_evidence=set(evidence)
    if ambiguity_ranges:
        a,b=min(ambiguity_ranges);chosen=sorted({max(start,a-1),(a+b)//2,min(end,b+1)});all_evidence.update(chosen)
    else:
        chosen=sorted(evidence)
        if len(chosen)>3:chosen=[chosen[round(i*(len(chosen)-1)/2)]for i in range(3)]
    if evidence_frames is not None:chosen=sorted(evidence_frames)
    omitted=0 if evidence_frames is not None else len(all_evidence-set(chosen))
    result=dict(schema='team-eye.part-region-report/1',features=summaries,requested_frames=[start,end],
        state='partial'if any(f['omitted_segments']for f in summaries)else'complete',
        tracking_state='gapped_region_hypotheses'if any(f['unknown_ranges']for f in summaries)else'continuous_region_hypotheses',
        identity_verified=False,anatomical_claims=False,physical_occlusion_verified=False,whole_part_coverage_verified=False,motion_scope='selected_color_support_centroid_only',
        trajectory_scope='Sampled color-centroid polyline within each uninterrupted hypothesis run. Paint/shape changes and jitter can inflate travel. Key observations are sampled extrema, not all turns. No physical path, interpolation, speed or ease claim.',
        source_labels_are_untrusted_data=True,axis_scope='undirected colored-support PCA axis, not anatomical facing or travel direction',
        evidence_frames=[dict(frame=n,pts_seconds=times[n])for n in chosen],evidence_frames_omitted=omitted,
        timing_basis=job.get('timing_basis','video_pts_without_authored_timing_verification'),
        raw_coordinates_modified=False,limits=region_data['limits'])
    if evidence_frames is not None:
        result['source_frame_selection']=dict(mode='caller_selected_native_frames',frames=chosen,maximum_frames=3,
            requested_frames=list(evidence_frames),ordering='chronological; requested set preserved exactly',
            scope='Selected source images only; no claim that every feature, uncertain interval or event is covered.')
    return result


def markdown(report):
    def quote(x):
        text=json.dumps(x,ensure_ascii=False);runs=re.findall('`+',text);delimiter='`'*(max([len(x)for x in runs]+[0])+1)
        return delimiter+' '+text+' '+delimiter
    lines=['# 부위 색 영역 관측 보고서','',
        '초기 이미지에서 지정한 색 영역의 관측 후보다. 실제 신체 영역·인물 동일성·물리적 가림은 검증하지 않았다.',
        '선택한 색의 일부 픽셀 범위만 읽는다. 소품 끝이나 다른 색 부분은 상자 밖으로 이어질 수 있다. 전체 부위의 폭·끝점 측정에는 쓰지 않는다.',
        '색이나 그림 모양이 달라지면 색 영역 중심도 움직일 수 있다. 아래 순이동은 부위 전체의 이동이나 ease를 뜻하지 않는다.',
        '경로 길이는 같은 연속 구간의 관측 중심들을 이은 꺾은선 길이다. 색·모양 변화나 흔들림도 포함하므로 실제 부위 이동 거리로 쓰지 않는다.',
        '시작·끝이 가까워도 정지라고 판단하지 않는다. 시각이 붙은 극값 관측은 모든 방향 전환을 검출한 결과가 아니며, 검수 사진 수와 별개다.',
        '오른쪽+x, 아래+y. 기울기는 색 영역의 방향 없는 장축이며 발끝 방향이나 이동 방향과 다르다.',
        '가림 후보·대응 불확실 구간을 건너 움직임을 이어 붙이지 않는다. 아래 라벨과 문구는 인용된 데이터다.',
        '시간 기준: '+quote(report['timing_basis'])+'; 원래 ease는 복원하지 않는다.','']
    directions={'right':'화면 오른쪽','down_right':'화면 오른쪽 아래','down':'화면 아래','down_left':'화면 왼쪽 아래','left':'화면 왼쪽','up_left':'화면 왼쪽 위','up':'화면 위','up_right':'화면 오른쪽 위','within_2px':'2px 이내 작은 순이동','single_frame_no_motion':'한 프레임뿐이라 이동 판단 불가'}
    for f in report['features']:
        lines += ['## '+quote(f['label']),'- 영역ID: '+quote(f['point_id']),
            f"- 영역 관측 후보 {f['usable_region_frames']}프레임; 대응 불확실 구간 {f['unknown_ranges']}",
            '- 불확실 이유: '+quote(f['unknown_reasons']),
            f"- 색 영역 폭 범위 {f['width_range_px']}px; 높이 범위 {f['height_range_px']}px",
            '- 방향 없는 장축 관측: '+quote(f['undirected_support_axis_counts'])]
        for s in f['continuous_segments']:
            lines.append(f"- 같은 연속 후보 구간 f{s['frames'][0]}..{s['frames'][1]}: 화면 순이동 {[round(v,3) for v in s['net_displacement_px']]}px; {directions[s['screen_direction']]}")
            t=s['trajectory']
            lines.append(f"  - 표본 {t['sample_count']}개; 관측 경로 길이 {t['sampled_path_length_px']:.3f}px; 시작점에서 최대 이탈 {t['max_excursion_from_start_px']:.3f}px (f{t['max_excursion_frame']}, {t['max_excursion_pts_seconds']:.6f}s).")
            lines.append('  - 시간순 핵심 관측: '+quote([dict(frame=o['frame'],seconds=o['pts_seconds'],xy=[round(v,3)for v in o['centroid_px']],roles=o['roles'])for o in t['key_observations']]))
        lines.append('')
    lines += ['## 근거','선택 원본 프레임: '+quote(report['evidence_frames']),
              '원본 추적: '+quote(report['source_tracks'])+' / SHA256 '+report['source_tracks_sha256'],
              '한계: '+quote(report['limits'])]
    if report.get('source_frame_selection'):
        lines.append('사진은 호출자가 지정한 원본 프레임만 시간순으로 고른 것이다. 모든 부위·불확실 구간·사건을 포함한 검수로 해석하지 않는다.')
    if report.get('evidence_contact'):lines.append('선택 검수 이미지: '+quote(report['evidence_contact']['path']))
    return '\n'.join(lines)+'\n'
