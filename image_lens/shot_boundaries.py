"""Source-bound hard-cut candidates. No pose, identity transfer or video re-encoding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
from image_lens.paths import workspace_root
COMPARISON_SHA256 = '430a74560012f3a335e812c02547a59b90e2b00ea7f7b6a6a4d99d55beabbe07'
PARAMS = dict(detector='PySceneDetect.AdaptiveDetector', version='0.6.7.1',
              adaptive_threshold=3.0, min_content_val=15.0, min_scene_len=1,
              window_width=2, max_side=192, edge_padding=2,
              brief_boundary_gap=2, frame_skip=0)
LIMITS = dict(max_frames=18000, max_seconds=600, max_pixels=8294400,
              max_file_bytes=512*1024*1024, scan_timeout_seconds=120, evidence_frames=3)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def finite(x):
    try:return type(x) in (int,float) and math.isfinite(x)
    except OverflowError:return False

def detector():
    pin_path=HERE/'shot-detector-pins.json'; pin=json.loads(pin_path.read_text())
    if pin.get('version')!='0.6.7.1' or pin.get('wheel_sha256')!='3808ef4436ab0fc6fee8a155e95759b42e2446c925c45746a82ff119be4eb3e1':
        raise ValueError('unsupported shot detector pin')
    base=HERE/'vendor/shot-detector'
    for n,h in pin['files'].items():
        p=base/n
        if p.is_symlink() or not p.resolve().is_relative_to(base) or sha(p)!=h:raise ValueError('shot detector source changed: '+n)
    if 'scenedetect' in sys.modules and Path(sys.modules['scenedetect'].__file__).resolve()!=base/'scenedetect/__init__.py':
        raise ValueError('another scenedetect module is already loaded')
    sys.path.insert(0,str(base))
    try:
        from scenedetect.detectors import AdaptiveDetector
        return AdaptiveDetector(adaptive_threshold=PARAMS['adaptive_threshold'],min_content_val=PARAMS['min_content_val'],
                                min_scene_len=PARAMS['min_scene_len'],window_width=PARAMS['window_width'])
    finally:sys.path.remove(str(base))

def thumbnail(bgr):
    import cv2
    h,w=bgr.shape[:2];scale=min(1.,PARAMS['max_side']/max(h,w))
    return cv2.resize(bgr,(max(2,round(w*scale)),max(2,round(h*scale))),interpolation=cv2.INTER_AREA)

class Scan:
    def __init__(self):
        self.impl=detector();self.first=None;self.last=None;self.n=0;self.cuts=[]
    def add(self,bgr):
        im=thumbnail(bgr)
        if self.n==0:
            self.first=im
            for i in range(PARAMS['edge_padding']):self.impl.process_frame(i,im)
        self.cuts.extend(self.impl.process_frame(self.n+PARAMS['edge_padding'],im));self.n+=1;self.last=im
    def finish(self):
        if self.n<2:raise ValueError('at least two decoded native frames required')
        for i in range(PARAMS['edge_padding']):self.cuts.extend(self.impl.process_frame(self.n+PARAMS['edge_padding']+i,self.last))
        return sorted({i-PARAMS['edge_padding'] for i in self.cuts if 0<i-PARAMS['edge_padding']<self.n})

def assemble(job,times,local_cuts):
    start=job['start_frame'];end=job['end_frame'];cuts=[start+i for i in local_cuts]
    events=[]
    for i,c in enumerate(cuts):
        nearby=any(0<abs(c-other)<=PARAMS['brief_boundary_gap'] for other in cuts)
        events.append(dict(before_frame=c-1,after_frame=c,before_seconds=times[c-start-1],after_seconds=times[c-start],
                           kind='brief_transition_ambiguous' if nearby else 'hard_cut_candidate',
                           score_is_probability=False,verified=False))
    bounds=[start,*cuts,end+1]
    segments=[dict(segment_id='shot-%04d'%i,start_frame=a,end_frame=b-1,
                   start_seconds=times[a-start],last_frame_seconds=times[b-1-start],
                   analysis_eligibility='inspection_only_single_frame' if b-a==1 else 'fresh_tracking_job_required',
                   needs_fresh_source_observations=True,identity_continues_from_previous=False)
              for i,(a,b) in enumerate(zip(bounds,bounds[1:]))]
    return dict(schema='team-eye.shots/1',source_video=job['source_video'],source_sha256=job['source_sha256'],
                width=job['width'],height=job['height'],fps=job['fps'],native_range=[start,end],
                frame_count=len(times),native_pts_seconds=times,decode_complete=True,
                parameters=PARAMS,detector_source_manifest_sha256=sha(HERE/'shot-detector-pins.json'),
                detector_helper_sha256=sha(__file__),events=events,analysis_segments=segments,
                limitations=['Hard-cut candidates only; no calibrated probability or semantic scene/identity proof.',
                             'Gradual dissolves/fades and missed edits are not protected by this detector.',
                             'Brief flashes and short inserted shots may be indistinguishable; retain both boundaries.',
                             'No short segment merging, frame skipping, synthesized frames or cross-boundary interpolation.'])

def detect_images(images,times,job):
    """Compatibility name; supplied interval never creates a second cut context."""
    if len(images)!=len(times) or len(images)!=job['end_frame']-job['start_frame']+1:raise ValueError('incomplete decoded tracking interval')
    return canonical_scan(job,times)

def validate(scan,job=None,frames=None):
    if type(scan)is not dict or scan.get('schema')!='team-eye.shots/1' or scan.get('parameters')!=PARAMS or scan.get('decode_complete')is not True:
        raise ValueError('supported complete shot scan required')
    bounds=scan.get('native_range');times=scan.get('native_pts_seconds')
    if (type(bounds)is not list or len(bounds)!=2 or any(type(n)is not int for n in bounds)
            or not 0<=bounds[0]<bounds[1] or bounds[1]-bounds[0]+1>LIMITS['max_frames']
            or scan.get('frame_count')!=bounds[1]-bounds[0]+1 or type(times)is not list or len(times)!=scan['frame_count']
            or any(not finite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:]))):
        raise ValueError('complete finite increasing native shot frames/PTS required')
    if (not finite(scan.get('fps')) or not 0<scan['fps']<=240 or times[-1]-times[0]>LIMITS['max_seconds']
            or any(type(scan.get(k))is not int or scan[k]<2 for k in ('width','height'))
            or scan['width']*scan['height']>LIMITS['max_pixels']):raise ValueError('shot source limits')
    import re
    for k in ['source_sha256','detector_source_manifest_sha256','detector_helper_sha256']:
        if type(scan.get(k))is not str or not re.fullmatch('[0-9a-f]{64}',scan[k]):raise ValueError('shot pin missing')
    if scan['detector_source_manifest_sha256']!=sha(HERE/'shot-detector-pins.json') or scan['detector_helper_sha256']!=sha(__file__):
        raise ValueError('shot detector provenance mismatch; rescan with current helper')
    events=scan.get('events')
    if type(events)is not list or len(events)>=scan['frame_count']:raise ValueError('bounded shot event list required')
    cuts=[]
    for e in events:
        if (type(e)is not dict or type(e.get('after_frame'))is not int or not bounds[0]<e['after_frame']<=bounds[1]
                or e.get('before_frame')!=e['after_frame']-1 or e.get('verified')is not False or e.get('score_is_probability')is not False):raise ValueError('invalid native shot boundary')
        c=e['after_frame'];cuts.append(c)
        if e.get('before_seconds')!=times[c-bounds[0]-1] or e.get('after_seconds')!=times[c-bounds[0]]:raise ValueError('shot event PTS mismatch')
    if cuts!=sorted(set(cuts)):raise ValueError('duplicate/unordered shot boundary')
    expected=assemble(dict(source_video=scan['source_video'],source_sha256=scan['source_sha256'],width=scan['width'],height=scan['height'],fps=scan['fps'],start_frame=bounds[0],end_frame=bounds[1]),times,[c-bounds[0]for c in cuts])
    if events!=expected['events'] or scan.get('analysis_segments')!=expected['analysis_segments']:raise ValueError('shot partition or event classification mismatch')
    if job is not None:
        if (scan['source_video']!=job['source_video'] or scan['source_sha256']!=job['source_sha256']
                or any(scan[k]!=job[k] for k in ['width','height','fps'])
                or not bounds[0]<=job['start_frame']<=job['end_frame']<=bounds[1]):raise ValueError('shot scan belongs to another source/range')
    if frames is not None:
        if [f['frame']for f in frames]!=list(range(job['start_frame'],job['end_frame']+1)):raise ValueError('incomplete tracked generation interval')
        for f in frames:
            if f.get('pts_seconds')!=times[f['frame']-bounds[0]]:raise ValueError('shot/track native PTS mismatch')
    return scan

def context(packet):
    scan=packet.get('shot_check')
    if scan is None:return None
    validate(scan,packet['job'],packet['frames'])
    job=packet['job'];events=[e for e in scan['events']if job['start_frame']<e['after_frame']<=job['end_frame']]
    return dict(status='requires_shot_local_retracking' if events else 'no_cut_detected_in_generation_interval',
                detected_events=events,checked_generation_interval=[job['start_frame'],job['end_frame']],
                scan_range=scan['native_range'],source_sha256=scan['source_sha256'],
                detector=PARAMS,identity_scope='shot_local_unverified',
                no_cut_is_not_proof_of_continuity=True)

def scan_video(path,start=0,end=None):
    import cv2
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(workspace_root()) or not path.is_file() or path.stat().st_size>LIMITS['max_file_bytes']:
        raise ValueError('bounded task-local regular video required')
    path=path.resolve();pin=sha(path);cap=cv2.VideoCapture(str(path));started=time.monotonic()
    try:
        width,height,count=(int(cap.get(v))for v in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_COUNT]);fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not finite(fps) or not 0<fps<=240 or count<2 or width*height>LIMITS['max_pixels']:raise ValueError('unsupported video metadata')
        if end is None:end=count-1
        if type(start)is not int or type(end)is not int or not 0<=start<end<count or end-start+1>LIMITS['max_frames'] or (end-start)/fps>LIMITS['max_seconds']:raise ValueError('bounded native scan interval required')
        scan=Scan();times=[]
        for n in range(end+1):
            if time.monotonic()-started>LIMITS['scan_timeout_seconds']:raise TimeoutError('shot scan budget')
            ok,im=cap.read()
            if not ok:raise ValueError('incomplete video decode; no scan published')
            if n<start:continue
            if im.shape!=(height,width,3):raise ValueError('source dimension changed')
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if not finite(t) or t<0 or times and t<=times[-1]:raise ValueError('nonincreasing/nonfinite decoder PTS')
            times.append(t);scan.add(im)
        if end==count-1:
            ok,_=cap.read()
            if ok:raise ValueError('decoder frame count disagrees; no scan published')
        if sha(path)!=pin:raise ValueError('source changed during shot scan')
        result=assemble(dict(source_video=str(path),source_sha256=pin,width=width,height=height,fps=fps,start_frame=start,end_frame=end),times,scan.finish())
        result.update(scan_scope='full_source' if start==0 and end==count-1 else 'partial_source',source_frame_count=count)
        validate(result);return result
    finally:cap.release()

def canonical_scan(job,times,cached=None):
    """One full-source detector context, including for a short tracking job."""
    import cv2
    path=Path(job['source_video'])
    if (path.is_symlink() or not path.resolve().is_relative_to(workspace_root()) or not path.is_file()
            or path.stat().st_size>LIMITS['max_file_bytes'] or sha(path)!=job['source_sha256']):
        raise ValueError('canonical shot source mismatch')
    scan=scan_video(path) if cached is None else cached
    validate(scan,job,[dict(frame=job['start_frame']+i,pts_seconds=t)for i,t in enumerate(times)])
    cap=cv2.VideoCapture(str(path))
    try:
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (not cap.isOpened() or scan.get('scan_scope')!='full_source'
                or scan.get('source_frame_count')!=count or scan['native_range']!=[0,count-1]
                or scan['width']!=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                or scan['height']!=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                or abs(scan['fps']-cap.get(cv2.CAP_PROP_FPS))>1e-6):
            raise ValueError('canonical full-source/EOF shot scan required; partial/local scans cannot admit tracking')
    finally:cap.release()
    if sha(path)!=job['source_sha256']:raise ValueError('canonical source changed')
    return scan

def selected_frames(scan,chosen):
    import cv2
    validate(scan)
    if (type(chosen)is not list or not 1<=len(chosen)<=LIMITS['evidence_frames']
            or chosen!=sorted(set(chosen))or any(type(n)is not int or not scan['native_range'][0]<=n<=scan['native_range'][1]for n in chosen)):
        raise ValueError('choose 1..3 distinct native source frames')
    path=Path(scan['source_video'])
    if path.is_symlink() or not path.resolve().is_relative_to(workspace_root()) or sha(path)!=scan['source_sha256']:raise ValueError('boundary image source mismatch')
    cap=cv2.VideoCapture(str(path));panels=[];started=time.monotonic()
    try:
        for n in range(max(chosen)+1):
            if time.monotonic()-started>LIMITS['scan_timeout_seconds']:raise TimeoutError('boundary evidence decode budget')
            ok,im=cap.read()
            if not ok:raise ValueError('boundary image decode incomplete')
            if n not in chosen:continue
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if t!=scan['native_pts_seconds'][n-scan['native_range'][0]]:raise ValueError('boundary image PTS mismatch')
            panels.append((n,t,im))
    finally:cap.release()
    if sha(path)!=scan['source_sha256']:raise ValueError('source changed during boundary rendering')
    return panels

def render(scan,destination,event_index=0):
    import cv2
    from PIL import Image,ImageDraw
    dest=Path(destination).resolve()
    if not dest.is_relative_to(workspace_root()) or dest.exists():raise ValueError('new task-local boundary image directory required')
    events=scan['events']
    if events:
        if type(event_index)is not int or not 0<=event_index<len(events):raise ValueError('boundary event index not present')
        e=events[event_index];chosen={e['before_frame'],e['after_frame']};wanted=set(chosen)
        for other in events:
            if abs(other['after_frame']-e['after_frame'])<=PARAMS['brief_boundary_gap']:wanted.update([other['before_frame'],other['after_frame']])
        for n in sorted(wanted-chosen):
            if len(chosen)==LIMITS['evidence_frames']:break
            chosen.add(n)
        chosen=sorted(chosen);omitted=len(wanted)-len(chosen)
    else:
        if event_index!=0:raise ValueError('no boundary event')
        chosen=scan['native_range'][:];omitted=0
    panels=[]
    for n,t,im in selected_frames(scan,chosen):
        rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));rgb.thumbnail((380,420));panels.append((n,t,rgb))
    canvas=Image.new('RGB',(400*len(panels),470),'white');d=ImageDraw.Draw(canvas)
    for i,(n,t,im)in enumerate(panels):canvas.paste(im,(400*i,35));d.text((400*i+5,5),'native f%d / %.6fs'%(n,t),fill='black')
    d.text((5,450),'Source frames only. Cut/flash candidates, not verified scene or identity.',fill='black')
    dest.mkdir();out=dest/'boundary.png';canvas.save(out)
    return dict(path=str(out),frames=chosen,image_cap=3,selected_event_index=event_index if events else None,
                omitted_context_frame_count=omitted,other_event_count=max(0,len(events)-1),overlays=False)

def render_comparison(scan,destination,frames):
    import runpy
    import cv2
    from PIL import Image,ImageDraw
    dest=Path(destination).resolve()
    if not dest.is_relative_to(workspace_root())or dest.exists():raise ValueError('new task-local comparison directory required')
    if type(frames)is not list or len(frames)!=2 or frames[0]>=frames[1]:raise ValueError('two increasing native frames required')
    helper=HERE/'appearance_inspection.py'
    if helper.is_symlink()or sha(helper)!=COMPARISON_SHA256:raise ValueError('appearance inspection helper pin mismatch')
    api=runpy.run_path(str(helper));panels=selected_frames(scan,frames)
    before,after=[api['thumbnail'](x[2])for x in panels];report,overlay=api['inspect'](before,after)
    images=[Image.fromarray(cv2.cvtColor(before,cv2.COLOR_BGR2RGB)),Image.fromarray(cv2.cvtColor(after,cv2.COLOR_BGR2RGB)),Image.fromarray(overlay)]
    height=images[0].height;canvas=Image.new('RGB',(3*500,height+140),'white');d=ImageDraw.Draw(canvas)
    labels=['SOURCE f%d / %.6fs'%(panels[0][0],panels[0][1]),'SOURCE f%d / %.6fs'%(panels[1][0],panels[1][1]),'TARGET + UNVERIFIED RESIDUAL']
    for i,(im,label)in enumerate(zip(images,labels)):canvas.paste(im,(i*500,32));d.text((i*500+4,8),label,fill='black')
    d.text((5,height+45),'Red: motion/copy/redraw OR failed alignment. Gray: fitted coverage exclusion, NOT actual occlusion.',fill='black')
    d.text((5,height+65),'Target-conditioned fit, NOT predicted/interpolated motion. Border/copy alignment may be wrong; inspect original frames.',fill='black')
    d.text((5,height+85),'No automatic redraw/action decision; does not change the shot partition or tracking admission.',fill='black')
    dest.mkdir();path=dest/'comparison.png';canvas.save(path)
    report.update(source_sha256=scan['source_sha256'],frames=frames,native_pts_seconds=[x[1]for x in panels],helper_sha256=COMPARISON_SHA256,
                  image=dict(path=str(path),sha256=sha(path),source_panels=2,derived_panels=1,image_cap=3),
                  native_size=[scan['width'],scan['height']],thumbnail_size=[images[0].width,images[0].height])
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--video',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int);ap.add_argument('--evidence-dir');ap.add_argument('--event-index',type=int,default=0)
    ap.add_argument('--compare-frames',help='two native frames a,b; diagnostic originals + residual only, no new cut')
    a=ap.parse_args();out=Path(a.out).resolve()
    if not out.is_relative_to(workspace_root()) or out.exists():raise ValueError('new task-local shot JSON required')
    if a.event_index and not a.evidence_dir:raise ValueError('--event-index requires --evidence-dir')
    comparison=None
    if a.compare_frames is not None:
        parts=a.compare_frames.split(',')
        if not a.evidence_dir or a.event_index or len(parts)!=2 or any(not x.isascii()or not x.isdecimal()for x in parts):raise ValueError('--compare-frames needs a,b plus --evidence-dir, without --event-index')
        comparison=[int(x)for x in parts]
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('shot scan wall budget')));signal.alarm(60 if comparison is not None else 120)
    result=scan_video(a.video,a.start,a.end)
    json.dumps(result,allow_nan=False)
    if comparison is not None:result['appearance_inspection']=render_comparison(result,a.evidence_dir,comparison)
    elif a.evidence_dir:result['evidence_contact']=render(result,a.evidence_dir,a.event_index)
    with out.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if hasattr(signal, 'SIGALRM'):signal.alarm(0)
    print(json.dumps(dict(status='written',output=str(out),frames=result['frame_count'],events=len(result['events']),segments=len(result['analysis_segments']),model_loaded=False)))

if __name__=='__main__':
    try:main()
    except Exception as e:print(json.dumps({'status':'failed','type':type(e).__name__,'error':str(e)[:1000]}),file=sys.stderr);raise SystemExit(1)
