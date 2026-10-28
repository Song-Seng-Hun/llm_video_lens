"""Observed 2D surface timing in an explicit interval; no model or interpolation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import runpy

from image_lens.paths import workspace_root
SHOT = runpy.run_path(str(Path(__file__).with_name('shot_boundaries.py')))
PARAMS = dict(min_samples=12, static_extent_px=1.5, min_travel_px=12.,
              max_perpendicular_ratio=.15, max_backward_step_ratio=.08,
              max_fit_rmse=.08, min_fit_margin=.02, background_agreement_px=2.)
PROFILES = {'approximately_constant_speed': lambda u: u,
            'accelerating': lambda u: u*u, 'decelerating': lambda u: 2*u-u*u}

def finite(value):
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        return False

def norm(a):
    return math.hypot(*a)

def sub(a,b):
    return [a[0]-b[0],a[1]-b[1]]

def valid_xy(xy):
    return type(xy) is list and len(xy)==2 and all(finite(v) for v in xy)

def infer(times, coordinates):
    """Numerical helper; caller supplies independently validated contiguous observations."""
    if len(times)!=len(coordinates) or len(times)<PARAMS['min_samples']:
        return dict(label='unknown',reasons=['insufficient_samples'])
    if (not all(finite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:]))
            or not all(valid_xy(p) for p in coordinates)):
        return dict(label='unknown',reasons=['invalid_time_or_coordinate'])
    start,end=coordinates[0],coordinates[-1]
    extent=max(norm(sub(a,b)) for a in coordinates for b in coordinates)
    common=dict(extent_px=extent,duration_seconds=times[-1]-times[0],samples=len(times))
    if extent<=PARAMS['static_extent_px']:
        return dict(common,label='near_static_at_resolution',reasons=[],
                    interpretation='Observed extent only, not proof of held drawings or zero physical motion.')
    direction=sub(end,start);travel=norm(direction)
    if travel<PARAMS['min_travel_px']:
        return dict(common,label='unknown',reasons=['small_or_returning_net_travel'],net_travel_px=travel)
    unit=[v/travel for v in direction]
    progress=[sum(a*b for a,b in zip(sub(p,start),unit))/travel for p in coordinates]
    perpendicular=max(abs((p[0]-start[0])*unit[1]-(p[1]-start[1])*unit[0])/travel for p in coordinates)
    backward=max([0.]+[a-b for a,b in zip(progress,progress[1:])])
    common.update(net_travel_px=travel,perpendicular_ratio=perpendicular,max_backward_step_ratio=backward)
    if perpendicular>PARAMS['max_perpendicular_ratio'] or backward>PARAMS['max_backward_step_ratio']:
        return dict(common,label='unknown',reasons=['not_near_straight_monotone_motion'])
    u=[(t-times[0])/(times[-1]-times[0]) for t in times]
    scores={name:math.sqrt(sum((p-f(t))**2 for p,t in zip(progress,u))/len(u)) for name,f in PROFILES.items()}
    order=sorted(scores,key=scores.get);margin=scores[order[1]]-scores[order[0]]
    good=scores[order[0]]<=PARAMS['max_fit_rmse'] and margin>=PARAMS['min_fit_margin']
    return dict(common,label=order[0] if good else 'unknown',
                reasons=[] if good else ['weak_or_ambiguous_profile_fit'],
                normalized_rmse=scores,fit_margin=margin,
                interpretation='Observed 2D projected timing family, not original Bezier, 3D ease or anatomy.')

def collect_observations(packet,point_id,start,end,camera_ids=(),*,min_samples=12):
    if (packet.get('schema')!='team-eye.observed-tracks/1'
            or packet.get('observation_scope')!='visible_surface_2d'
            or packet.get('anatomical_claims') is not False):
        raise ValueError('explicit observed surface tracks required')
    if (type(start) is not int or type(end) is not int or not 0<=start<end
            or end-start+1>160 or not isinstance(point_id,str) or not point_id):
        raise ValueError('explicit bounded native-frame interval and point ID required')
    if camera_ids and (len(camera_ids)!=2 or len(set(camera_ids))!=2 or point_id in camera_ids):
        raise ValueError('camera correction requires exactly two distinct background IDs separate from target')
    ids=[point_id,*camera_ids]
    base=dict(point_id=point_id,requested_frames=[start,end],
              coordinate_scope='background_relative_2d' if camera_ids else 'screen_2d',
              camera_feature_ids=list(camera_ids),parameters=PARAMS,
              anatomical_interpretation=None,semantic_identity_verified_by_model=False,
              camera_assumption='Selected features are world-static; common co-motion is not detectable.' if camera_ids else None)
    def unknown(reason,**details):
        return dict(base,label='unknown',reasons=[reason],**details)
    shot_context=SHOT['context'](packet)
    if shot_context and shot_context['status']=='requires_shot_local_retracking':
        return unknown('offline_tracking_context_crosses_detected_shots',shot_context=shot_context)
    frames=packet.get('frames')
    if type(frames) is not list or len(frames)>160:
        raise ValueError('bounded source frames required')
    selected=[f for f in frames if type(f) is dict and type(f.get('frame')) is int and start<=f['frame']<=end]
    if [f['frame'] for f in selected]!=list(range(start,end+1)):
        return unknown('missing_duplicate_or_out_of_order_native_frame')
    times=[f.get('pts_seconds') for f in selected]
    if not all(finite(t) and t>=0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        return unknown('nonfinite_or_nonincreasing_native_pts')
    rows=[];indices={};skipped=[]
    width,height=packet.get('job',{}).get('width'),packet.get('job',{}).get('height')
    if type(width) is not int or type(height) is not int or min(width,height)<=1 or width*height>2073600:
        raise ValueError('source image dimensions required')
    for index,frame in enumerate(selected):
        points=frame.get('points')
        if type(points) is not list or any(type(p) is not dict for p in points):
            return unknown('malformed_point_rows')
        matches={p:[v for v in points if v.get('point_id')==p] for p in ids}
        if any(len(v)!=1 for v in matches.values()):
            return unknown('missing_or_duplicate_point_id')
        row={p:v[0] for p,v in matches.items()}
        if any(v.get('status')=='anchor_input_not_prediction' for v in row.values()):
            if index==0:
                skipped.append(frame['frame']);continue
            return unknown('interior_anchor_reset',at_frame=frame['frame'])
        if frame.get('tracking_metric_eligible') is not True:
            return unknown('frame_not_prediction_eligible',at_frame=frame['frame'])
        for pid,v in row.items():
            if (v.get('visible') is not True or v.get('model_visible') is not True
                    or v.get('in_frame') is not True or v.get('status')!='tracked_unverified'
                    or v.get('claim_scope')!='visible_surface_2d'
                    or v.get('anatomical_interpretation') is not None
                    or not valid_xy(v.get('xy')) or not 0<=v['xy'][0]<width or not 0<=v['xy'][1]<height
                    or type(v.get('query_index')) is not int
                    or v['query_index']<0 or type(v.get('anchor_index')) is not int or v['anchor_index']<0
                    or type(v.get('anchor_frame')) is not int or not 0<=v['anchor_frame']<frame['frame']
                    or v.get('tracking_metric_eligible') is not True
                    or v.get('metric_excluded_common_anchor_frame') is not False
                    or v.get('motion_pair_usable') is not True):
                return unknown('unusable_or_unknown_surface_observation',at_frame=frame['frame'],failed_point_id=pid)
            identity=(v['query_index'],v.get('anchor_index'),v.get('anchor_frame'))
            if pid in indices and identity!=indices[pid]:
                return unknown('query_or_anchor_identity_transition',at_frame=frame['frame'])
            indices[pid]=identity
        if len({v['query_index'] for v in row.values()})!=len(ids):
            return unknown('selected_surface_ids_share_query')
        rows.append((frame['frame'],frame['pts_seconds'],row))
    if len(rows)<min_samples:
        return unknown('insufficient_samples_after_anchor_exclusion')
    xy=[];agreement=0.
    origin={p:rows[0][2][p]['xy'] for p in camera_ids}
    for _,_,row in rows:
        target=row[point_id]['xy']
        if camera_ids:
            deltas=[sub(row[p]['xy'],origin[p]) for p in camera_ids]
            disagreement=norm(sub(*deltas));agreement=max(agreement,disagreement)
            if disagreement>PARAMS['background_agreement_px']:
                return unknown('background_displacement_disagreement',max_background_disagreement_px=agreement)
            shift=[statistics.median(d[i] for d in deltas) for i in range(2)]
            target=sub(target,shift)
        xy.append(target)
    return base,rows,xy,skipped,agreement

def analyze(packet,point_id,start,end,camera_ids=()):
    collected=collect_observations(packet,point_id,start,end,camera_ids)
    if type(collected) is dict:
        return collected
    base,rows,xy,skipped,agreement=collected
    result=infer([r[1] for r in rows],xy)
    return dict(base,**result,used_frames=[rows[0][0],rows[-1][0]],
                used_pts_seconds=[rows[0][1],rows[-1][1]],excluded_source_anchor_frames=skipped,
                max_background_disagreement_px=agreement if camera_ids else None,
                observations_are_model_estimates=True,smoothing=False,interpolation=False)

def read(path):
    p=Path(path);resolved=p.resolve()
    if p.is_symlink() or p!=resolved or not resolved.is_relative_to(workspace_root()) or p.stat().st_size>16*1024*1024:
        raise ValueError('input must be a bounded regular local task JSON')
    data=p.read_bytes()
    def reject(value):raise ValueError('nonfinite JSON '+value)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(data,parse_constant=reject,object_pairs_hook=pairs),hashlib.sha256(data).hexdigest()

def chord_errors(times, paths, selected):
    """Distance to time-linear chords, for review compression only; never output imputed coordinates."""
    errors=[]
    for left,right in zip(selected,selected[1:]):
        for index in range(left+1,right):
            u=(times[index]-times[left])/(times[right]-times[left])
            for point,path in enumerate(paths):
                implied=[a+(b-a)*u for a,b in zip(path[left],path[right])]
                errors.append((norm(sub(path[index],implied)),index,point))
    return errors

def break_context_frames(packet,point_ids,start,end,max_frames):
    """Only source-image context around failed observation pairs; no inferred motion or identities."""
    table={f['frame']:f for f in packet['frames']};events=[]
    for pid in point_ids:
        bad=[]
        for frame in range(start,end):
            left=table[frame]
            if frame==start and any(p.get('point_id')==pid and p.get('status')=='anchor_input_not_prediction' for p in left['points']):continue
            pair=collect_observations(packet,pid,frame,frame+1,min_samples=2)
            if type(pair) is dict:bad.append((frame,pair['reasons'][0]))
        groups=[]
        for frame,reason in bad:
            if groups and frame==groups[-1][-1][0]+1:groups[-1].append((frame,reason))
            else:groups.append([(frame,reason)])
        for group in groups:
            first,last=group[0][0],group[-1][0]
            events.append(dict(point_id=pid,failed_pair_left_frames=[first,last],reasons=sorted({r for _,r in group}),
                context=[dict(frame=first,role='before_or_start_context'),dict(frame=(first+last+1)//2,role='during_break_context'),dict(frame=last+1,role='after_or_end_context')]))
    # Round-robin events, so one actor's long list cannot silently consume the entire image budget.
    wanted={}
    if not events:return None
    for slot in range(3):
        for event in events:
            request=event['context'][slot];frame=request['frame']
            wanted.setdefault(frame,[]).append(dict(point_id=event['point_id'],role=request['role']))
    chosen=list(wanted)[:max_frames];selected=set(chosen)
    return dict(status='diagnostic_only',reasons=['tracking_discontinuity_requires_visual_review'],
        requested_frames=[start,end],point_ids=point_ids,image_cap=max_frames,selected_count=len(selected),
        selected_frames=[dict(frame=f,pts_seconds=table[f]['pts_seconds'],context_requests=wanted[f],observed_points={})for f in sorted(selected)],
        discontinuities=[dict(**event,all_context_selected=all(r['frame']in selected for r in event['context']))for event in events],
        omitted_context_frame_count=len(wanted)-len(selected),budget_exhausted=len(wanted)>len(selected),
        motion_inference='unknown_across_breaks',semantic_identity_verified_by_model=False,reappearance_identity_verified=False,
        generated_coordinates=False,generated_frames=False,raw_coordinates_modified=False,
        interpretation='Bad-pair context, not physical occlusion timing. Reappearing estimates are not verified identity; no motion interpolation.')

def select_review_frames(packet, point_ids, start, end, max_frames=8,*,inspect_breaks=False):
    """One shared image budget across explicit surface IDs, in screen coordinates."""
    if (type(point_ids) is not list or not 1<=len(point_ids)<=8
            or any(type(p) is not str or not p for p in point_ids)
            or len(set(point_ids))!=len(point_ids)
            or type(max_frames) is not int or not 2<=max_frames<=12):
        raise ValueError('1..8 distinct point IDs and global image cap2..12 required')
    base=dict(point_ids=point_ids,requested_frames=[start,end],coordinate_scope='screen_2d',
              image_cap=max_frames,tolerance_px=4.,policy='worst-PTS-chord-deviation/1',
              anatomical_interpretation=None,semantic_identity_verified_by_model=False,
              interpretation='Observed screen-motion review candidates; camera/jitter included. Not pose truth, events or original keyframes.',
              smoothing=False,generated_coordinates=False,generated_frames=False)
    collected=[]
    for pid in point_ids:
        data=collect_observations(packet,pid,start,end)
        if type(data) is dict:
            structural={'missing_duplicate_or_out_of_order_native_frame','nonfinite_or_nonincreasing_native_pts','malformed_point_rows'}
            if inspect_breaks and not structural.intersection(data['reasons']):
                selected=[f for f in packet['frames'] if type(f)is dict and type(f.get('frame'))is int and start<=f['frame']<=end]
                if any(type(f.get('points'))is not list or any(type(p)is not dict for p in f['points']) for f in selected):
                    return dict(base,status='unavailable',reasons=['malformed_point_rows'],selected_frames=[],selected_count=0)
                # Validate every requested feature before deriving source contexts.
                probes=[collect_observations(packet,p,start,end) for p in point_ids]
                if not any(type(x)is dict and structural.intersection(x['reasons']) for x in probes):
                    context=break_context_frames(packet,point_ids,start,end,max_frames)
                    if context is not None:return context
            return dict(base,status='unavailable',reasons=data['reasons'],failed_point_id=pid,
                        at_frame=data.get('at_frame'),selected_frames=[],selected_count=0,
                        note='No continuity or coordinates synthesized. Inspect the failed interval manually.')
        collected.append(data)
    rows=collected[0][1];times=[r[1] for r in rows]
    if any([r[0] for r in data[1]]!=[r[0] for r in rows] for data in collected):
        return dict(base,status='unavailable',reasons=['point_prediction_ranges_differ'],selected_frames=[],selected_count=0)
    if len({data[1][0][2][pid]['query_index'] for data,pid in zip(collected,point_ids)})!=len(point_ids):
        return dict(base,status='unavailable',reasons=['selected_surface_ids_share_query'],selected_frames=[],selected_count=0)
    paths=[data[2] for data in collected];selected=[0,len(rows)-1]
    reasons={0:dict(reason='prediction_range_start'),len(rows)-1:dict(reason='prediction_range_end')}
    while len(selected)<min(max_frames,len(rows)):
        errors=chord_errors(times,paths,selected)
        if not errors:break
        distance,index,point=max(errors,key=lambda row:(row[0],-row[1],-row[2]))
        if distance<=base['tolerance_px']:break
        selected.append(index);selected.sort()
        reasons[index]=dict(reason='largest_observed_deviation',trigger_point_id=point_ids[point],
                            deviation_at_selection_px=distance)
    residuals=chord_errors(times,paths,selected)
    per_point={pid:max([e[0] for e in residuals if e[2]==i],default=0.) for i,pid in enumerate(point_ids)}
    worst=max(per_point.values())
    return dict(base,status='selected',reasons=[],used_frames=[rows[0][0],rows[-1][0]],
                used_pts_seconds=[times[0],times[-1]],excluded_source_anchor_frames=collected[0][3],
                selected_count=len(selected),budget_exhausted=worst>base['tolerance_px'],
                max_omitted_deviation_px=worst,per_point_max_omitted_deviation_px=per_point,
                selected_frames=[dict(frame=rows[i][0],pts_seconds=times[i],**reasons[i],
                    observed_points={pid:paths[j][i] for j,pid in enumerate(point_ids)}) for i in selected],
                residual_measures='Distance to PTS-linear chords through selected observations; not ground-truth tracking error.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracks',required=True)
    ap.add_argument('--shots',help='complete source-bound shot scan; legacy cross-shot model context requires retracking')
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--point');mode.add_argument('--review-points',help='comma-separated physical feature IDs; explicit review mode')
    mode.add_argument('--report-bindings',help='pinned source-frame actor/feature label packet')
    ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--camera-points',default='');ap.add_argument('--max-frames',type=int,default=8);ap.add_argument('--out',required=True)
    ap.add_argument('--format',choices=['json','markdown'],default='json')
    ap.add_argument('--evidence-dir',help='new task-local source contact; report or review with --inspect-breaks')
    ap.add_argument('--evidence-frames',help='region report only: 1..3 unique native frame numbers, requires --evidence-dir')
    ap.add_argument('--inspect-breaks',action='store_true',help='review mode: bounded source context for broken tracks; no motion inference')
    ap.add_argument('--region-report',action='store_true',help='explicit source-color region report with --report-bindings')
    ap.add_argument('--trajectory-dir',help='new task-local whole-frame path sheet plus appearance-change candidates; requires --report-bindings')
    ap.add_argument('--pose',help='image-lens pose packet for the same source; marks pose overlap/jump frames on the trajectory view, requires --trajectory-dir')
    ap.add_argument('--foreground-extent',action='store_true',help='optional flat-background inspection mask, requires --region-report and --evidence-dir')
    args=ap.parse_args();out=Path(args.out).resolve()
    if not out.is_relative_to(workspace_root()) or out.exists() or out.is_symlink():raise ValueError('new task-local output required')
    if args.foreground_extent and not(args.region_report and args.report_bindings and args.evidence_dir):
        raise ValueError('--foreground-extent requires --region-report, --report-bindings and --evidence-dir')
    if args.trajectory_dir and not args.report_bindings or args.trajectory_dir and args.region_report:
        raise ValueError('--trajectory-dir requires --report-bindings without --region-report')
    if args.pose and not args.trajectory_dir:raise ValueError('--pose requires --trajectory-dir')
    evidence_frames=None
    if args.evidence_frames is not None:
        if not(args.region_report and args.report_bindings and args.evidence_dir):
            raise ValueError('--evidence-frames requires --region-report, --report-bindings and --evidence-dir')
        tokens=[s.strip() for s in args.evidence_frames.split(',')]
        if not 1<=len(tokens)<=3 or any(not s.isascii() or not s.isdecimal() for s in tokens):
            raise ValueError('choose 1..3 distinct nonnegative native frame numbers')
        evidence_frames=[int(s) for s in tokens]
        if len(set(evidence_frames))!=len(evidence_frames) or any(not args.start<=n<=args.end for n in evidence_frames):
            raise ValueError('evidence frames must be unique and inside the requested interval')
    packet,pin=read(args.tracks)
    if args.region_report and not args.report_bindings:raise ValueError('--region-report requires --report-bindings')
    if args.inspect_breaks and not args.review_points:raise ValueError('--inspect-breaks requires --review-points')
    if args.shots:
        scan,scan_pin=read(args.shots)
        SHOT['validate'](scan,packet['job'],packet['frames'])
        if packet.get('shot_check') is not None and packet['shot_check']!=scan:
            raise ValueError('conflicting producer shot evidence; do not overwrite it with another scan; fresh tracking required')
        packet=dict(packet,shot_check=scan)
    shot_context=SHOT['context'](packet)
    if shot_context and shot_context['status']=='requires_shot_local_retracking':
        if not packet['job']['start_frame']<=args.start<args.end<=packet['job']['end_frame'] or args.end-args.start+1>160:
            raise ValueError('requested native interval is outside tracked generation range')
        if args.report_bindings:
            if args.camera_points or args.max_frames!=8:raise ValueError('report is screen2D with shared8frame cap')
            blocked_binding,_=read(args.report_bindings)
            if blocked_binding.get('source_tracks_sha256')!=pin:raise ValueError('binding source hash mismatch')
            runpy.run_path(str(Path(__file__).with_name('motion_report.py')))['validate_bindings'](packet,blocked_binding)
        elif args.format!='json' or args.evidence_dir and not(args.review_points and args.inspect_breaks):
            raise ValueError('report-only output options; review evidence requires --inspect-breaks')
        if args.foreground_extent or evidence_frames is not None:
            raise ValueError('cross-shot model context: no foreground/motion evidence; use boundary source inspection')
        scan=packet['shot_check'];job=packet['job'];segments=[]
        for s in scan['analysis_segments']:
            a=max(s['start_frame'],job['start_frame']);b=min(s['end_frame'],job['end_frame'])
            if a>b:continue
            segments.append(dict(s,start_frame=a,end_frame=b,start_seconds=scan['native_pts_seconds'][a-scan['native_range'][0]],
                                 last_frame_seconds=scan['native_pts_seconds'][b-scan['native_range'][0]],
                                 analysis_eligibility='inspection_only_single_frame'if a==b else'fresh_tracking_job_required'))
        result=dict(schema='team-eye.shot-blocked-report/1',state='requires_shot_local_retracking',label='unknown',
                    requested_frames=[args.start,args.end],shot_context=shot_context,
                    reasons=['Old whole-clip inference saw multiple detected shots; cropping its output cannot remove that context.'],
                    analysis_segments=segments,segment_scope='intersection with original bounded generation job; new explicit seeds required',
                    source_tracks=str(Path(args.tracks).resolve()),source_tracks_sha256=pin,
                    source_video=packet['job']['source_video'],source_video_sha256=packet['job']['source_sha256'],
                    model_loaded=False,raw_coordinates_modified=False,generated_coordinates=False,
                    motion_inference=False,interpolation=False,identity_transfer=False)
        if args.evidence_dir:
            index=packet['shot_check']['events'].index(shot_context['detected_events'][0])
            result['evidence_contact']=SHOT['render'](packet['shot_check'],args.evidence_dir,index)
        text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        if args.format=='markdown':
            text='# 컷 경계를 포함한 추적 — 구간별 재추적 필요\n\n기존 전체 영상 추적에 여러 컷이 섞여 있어 움직임·속도·ease를 계산하지 않았습니다. 각 구간에서 새 기준점을 지정하세요.\n\n'+''.join(
                '- %s: native f%d..%d / %.6f..%.6fs (마지막 프레임 시각)\n'%(s['segment_id'],s['start_frame'],s['end_frame'],s['start_seconds'],s['last_frame_seconds'])for s in result['analysis_segments'])
            if result.get('evidence_contact'):text+='\n경계 검수 이미지: '+result['evidence_contact']['path']+'\n'
        with out.open('x')as f:f.write(text)
        print(json.dumps(dict(status='written',report_state=result['state'],output=str(out),model_loaded=False)))
        return
    if args.report_bindings:
        if args.camera_points or args.max_frames!=8:raise ValueError('report is screen2D with a fixed shared8frame evidence cap')
        import importlib.util
        spec=importlib.util.spec_from_file_location('motion_report',Path(__file__).with_name('motion_report.py'))
        report_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(report_module)
        bindings,binding_pin=read(args.report_bindings)
        region_module=None
        if args.region_report:
            spec=importlib.util.spec_from_file_location('part_regions',Path(__file__).with_name('part_regions.py'))
            region_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(region_module)
            result=region_module.build_report(packet,bindings,args.start,args.end,report_module.validate_bindings,
                                              evidence_frames=evidence_frames)
        else:result=report_module.build_report(packet,bindings,args.start,args.end,collect_observations,infer)
        result.update(source_tracks=str(Path(args.tracks).resolve()),source_tracks_sha256=pin,
                      bindings_sha256=binding_pin,bindings_path=str(Path(args.report_bindings).resolve()),
                      analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reporter_sha256=hashlib.sha256(Path(report_module.__file__).read_bytes()).hexdigest(),model_loaded=False,
                      source_video=packet.get('job',{}).get('source_video'),source_video_sha256=packet.get('job',{}).get('source_sha256'))
        if bindings.get('source_tracks_sha256')!=pin:raise ValueError('binding source hash mismatch')
        foreground_module=None;foreground_masks=None
        if args.foreground_extent:
            spec=importlib.util.spec_from_file_location('foreground_regions',Path(__file__).with_name('foreground_regions.py'))
            foreground_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(foreground_module)
            inspection,foreground_masks=foreground_module.inspect(packet,result,region_module,report_module.read_source_frames(result,packet,workspace_root()))
            result['foreground_inspection']=inspection
            result['foreground_inspector_sha256']=hashlib.sha256(Path(foreground_module.__file__).read_bytes()).hexdigest()
        # Validate numerical serialization before optional media side effects or opening output.
        json.dumps(result,ensure_ascii=False,allow_nan=False)
        if args.evidence_dir:result['evidence_contact']=report_module.render_contact(result,packet,args.evidence_dir,workspace_root(),foreground_masks=foreground_masks)
        if args.trajectory_dir:
            spec=importlib.util.spec_from_file_location('motion_view',Path(__file__).with_name('motion_view.py'))
            view_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(view_module)
            pose_flags=None
            if args.pose:
                from image_lens import paths as lens_paths
                pose_packet,pose_path=lens_paths.load(args.pose,max_bytes=64*1024*1024)
                if pose_packet.get('schema')!='image-lens.pose/1' or pose_packet.get('source_sha256')!=result['source_video_sha256']:
                    raise ValueError('pose packet for the same pinned source required')
                pose_flags=dict(path=str(pose_path),sha256=lens_paths.sha(pose_path),native_range=pose_packet['native_range'],
                                params=pose_packet['flags']['params'],frames=pose_packet['flags']['frames'])
            result['trajectory_view']=view_module.render(result,packet,args.trajectory_dir,workspace_root(),report_module._source_video(result,workspace_root()),pose_flags)
            result['trajectory_view']['renderer_sha256']=hashlib.sha256(Path(view_module.__file__).read_bytes()).hexdigest()
        if region_module:result['region_reporter_sha256']=hashlib.sha256(Path(region_module.__file__).read_bytes()).hexdigest()
        if shot_context:result['shot_context']=shot_context
        text=(region_module.markdown(result)if region_module else report_module.markdown(result)) if args.format=='markdown' else json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        if args.format=='markdown' and shot_context:text+='\n컷 검사: 생성 구간 안에서 검출된 경계 없음. 컷 미검출이 연속성·동일 인물의 증거는 아닙니다.\n'
        if args.format=='markdown'and foreground_module:text+=foreground_module.markdown(result['foreground_inspection'])
        with out.open('x') as f:
            f.write(text)
        receipt=dict(status='written',report_state=result['state'],output=str(out),model_loaded=False)
        if foreground_module:receipt['foreground_state']=result['foreground_inspection']['state']
        print(json.dumps(receipt))
        return
    if args.format!='json' or (args.evidence_dir and not(args.review_points and args.inspect_breaks)):raise ValueError('report-only output options; review evidence requires --inspect-breaks')
    if args.evidence_dir and args.max_frames>8:raise ValueError('rendered source contacts are capped at8frames')
    if args.review_points:
        if args.camera_points:raise ValueError('review mode is screen2D only; no implicit camera correction')
        result=select_review_frames(packet,args.review_points.split(','),args.start,args.end,args.max_frames,inspect_breaks=args.inspect_breaks)
    else:
        if args.max_frames!=8:raise ValueError('--max-frames applies only to --review-points')
        result=analyze(packet,args.point,args.start,args.end,tuple(args.camera_points.split(',')) if args.camera_points else ())
    result.update(schema='team-eye.review-frames/1' if args.review_points else 'team-eye.motion-timing/1',source_tracks=str(Path(args.tracks).resolve()),
                  source_tracks_sha256=pin,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_loaded=False,source_video=packet.get('job',{}).get('source_video'),
                  source_video_sha256=packet.get('job',{}).get('source_sha256'))
    if shot_context:result['shot_context']=shot_context
    if args.evidence_dir:
        if not result['selected_frames']:raise ValueError('no valid source contexts selected; no image written')
        import importlib.util
        spec=importlib.util.spec_from_file_location('motion_report',Path(__file__).with_name('motion_report.py'))
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        contact=dict(result,evidence_frames=result['selected_frames'],features=[dict(point_id=pid,actor_id='F')for pid in args.review_points.split(',')])
        result['evidence_contact']=renderer.render_contact(contact,packet,args.evidence_dir,workspace_root())
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    receipt=dict(status='written',output=str(out),model_loaded=False)
    if args.review_points:receipt['selection_status']=result['status']
    else:receipt['label']=result['label']
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
