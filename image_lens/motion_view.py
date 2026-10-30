"""Whole-frame trajectory sheet and appearance-change candidates for one motion report.
Reads the pinned source only; no tracking, interpolation, gap filling or automatic redraw/action decision."""
import copy,hashlib,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHANGE=dict(thumb=(160,120),min_mean_abs_diff=12.,median_factor=4.,max_candidates=6)
POSE_NEAR_FRAMES=2
SIZE=dict(width=1600,main=(1200,900),thumbs=4)
_WINDOW=[]

def usable(point,width,height):
    xy=point.get('xy')
    return (point.get('visible') is True and point.get('model_visible') is True and point.get('in_frame') is True
            and point.get('status')=='tracked_unverified' and type(xy) is list and len(xy)==2
            and all(type(v) in (int,float) and math.isfinite(v) for v in xy) and 0<=xy[0]<width and 0<=xy[1]<height)

def change_candidates(frames,start):
    """Consecutive-frame jumps in downscaled gray difference. Diagnostic hint for manual compare, never a decision."""
    import numpy as np
    scores=[(start+i,float(np.mean(np.abs(b.astype(np.int16)-a.astype(np.int16))))) for i,(a,b) in enumerate(zip(frames,frames[1:]),1)]
    if not scores:return dict(threshold=None,candidates=[],omitted_candidate_count=0)
    median=sorted(s for _,s in scores)[len(scores)//2]
    threshold=max(CHANGE['min_mean_abs_diff'],CHANGE['median_factor']*median)
    hits=sorted(((n,s) for n,s in scores if s>=threshold),key=lambda x:(-x[1],x[0]))
    kept=sorted(hits[:CHANGE['max_candidates']])
    return dict(threshold=threshold,median_mean_abs_diff=median,
                candidates=[dict(after_frame=n-1,frame=n,mean_abs_diff=s,compare_frames=[n-1,n]) for n,s in kept],
                omitted_candidate_count=max(0,len(hits)-CHANGE['max_candidates']))

def pose_overlay(flags,start,end,candidates):
    """Pose-layer trouble (overlapping people, skeleton jumps) inside this range, and which change candidates sit near it."""
    if flags is None:return None
    lo,hi=flags['native_range']
    if start<lo or end>hi:raise ValueError('pose packet does not cover the report range')
    rows=[r for r in flags['frames'] if start<=r['frame']<=end];ranges=[]
    for r in rows:
        if ranges and r['frame']==ranges[-1]['frames'][1]+1:ranges[-1]['frames'][1]=r['frame']
        else:ranges.append(dict(frames=[r['frame'],r['frame']],overlapping_people=False,skeleton_jump=False,max_people=0))
        ranges[-1]['overlapping_people']|=r['overlapping_people'];ranges[-1]['skeleton_jump']|=bool(r['skeleton_jump'])
        ranges[-1]['max_people']=max(ranges[-1]['max_people'],r['people'])
    flagged={r['frame'] for r in rows}
    near=[c['frame'] for c in candidates if any(abs(c['frame']-n)<=POSE_NEAR_FRAMES for n in flagged)]
    return dict(path=flags['path'],sha256=flags['sha256'],params=flags['params'],near_frames=POSE_NEAR_FRAMES,
                ranges=ranges,candidates_near_pose_flags=near,
                basis='pose model trouble frames; overlap or echo copies make every tracker less reliable there, diagnostic only')

def render(report,packet,directory,root,source_video,pose_flags=None):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    destination=Path(directory);resolved=destination.resolve()
    if destination!=resolved or not resolved.is_relative_to(root) or destination.exists():raise ValueError('new canonical task-local trajectory directory required')
    start,end=report['requested_frames'];w,h=packet['job']['width'],packet['job']['height']
    table={f['frame']:f for f in packet['frames']}
    if any(n not in table for n in range(start,end+1)):raise ValueError('trajectory view needs every requested source frame row')
    paths={}
    for feature in report['features']:
        paths[feature['point_id']]=[(n,table[n]['pts_seconds'],p['xy']) for n in range(start,end+1)
                                    for p in table[n]['points'] if p.get('point_id')==feature['point_id'] and usable(p,w,h)]
    runs={f['point_id']:[run['frames'] for run in f['continuous_runs']] for f in report['features']}
    def same_run(pid,a,b):return b==a+1 and any(lo<=a and b<=hi for lo,hi in runs[pid])
    units=[(f['point_id'],u) for f in report['features'] for run in f['continuous_runs'] for u in run['strokes']['units']]
    picks=[start]+[u[k] for _,u in units if u['major'] for k in ('start_frame','end_frame')]+[end]
    picks=sorted(set(picks));picks=picks if len(picks)<=SIZE['thumbs'] else [picks[round(i*(len(picks)-1)/(SIZE['thumbs']-1))] for i in range(SIZE['thumbs'])]
    # Every group of one window asks for the same frames: the last window's checked frames and change candidates are kept.
    stat=Path(source_video).stat();key=(str(source_video),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,start,end,w,h)
    if _WINDOW[:1]!=[key]:
        from image_lens import frames as shared
        frames=shared.checked(source_video,start,end,w,h,'source decode geometry/index mismatch')
        # Each thumb is independent; threads give the same pixels (clip B: 7.4 s of serial resizing per look).
        with ThreadPoolExecutor(8) as pool:small=list(pool.map(lambda bgr:cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),CHANGE['thumb'],interpolation=cv2.INTER_AREA),frames))
        _WINDOW[:]=[key,frames,change_candidates(small,start)]
    frames=_WINDOW[1];full={n:Image.fromarray(frames[n-start][:,:,::-1].copy()) for n in set(picks)|{start}}
    changes=copy.deepcopy(_WINDOW[2]);pose=pose_overlay(pose_flags,start,end,changes['candidates'])
    many=len(report['features'])>2
    font=ImageFont.load_default(size=18);tiny=ImageFont.load_default(size=14)
    def color(t):
        u=0. if end==start else (t-start)/(end-start)
        return (round(40+215*u),round(90*(1-abs(2*u-1))),round(255*(1-u)))
    def draw_path(mark,scale,until=None,radius=5):
        for pid,rows in paths.items():
            rows=[r for r in rows if until is None or r[0]<=until]
            for (a,_,p),(b,_,q) in zip(rows,rows[1:]):
                # Only validated continuous runs are joined; breaks and identity transitions stay unconnected.
                if same_run(pid,a,b):mark.line((p[0]*scale,p[1]*scale,q[0]*scale,q[1]*scale),fill=color(b),width=3)
            if rows:
                x,y=rows[-1][2];mark.ellipse((x*scale-radius,y*scale-radius,x*scale+radius,y*scale+radius),outline='white',fill=color(rows[-1][0]),width=2)
    main_w,main_h=SIZE['main'];scale=min(main_w/w,main_h/h);thumb_w=SIZE['width']//SIZE['thumbs']-10;tscale=thumb_w/w;thumb_h=round(h*tscale)
    sheet=Image.new('RGB',(SIZE['width'],round(h*scale)+40+150+thumb_h+70),'white');draw=ImageDraw.Draw(sheet)
    base=full[start].resize((round(w*scale),round(h*scale)));base=Image.blend(base,Image.new('RGB',base.size,'white'),.35);mark=ImageDraw.Draw(base)
    draw_path(mark,scale,radius=7)
    for pid,u in units:
        rows={r[0]:r[2] for r in paths[pid]}
        for k in ('start_frame','end_frame'):
            # With many points only major corners get labels, otherwise they pile up.
            if u[k] in rows and (u['major'] or not many):
                x,y=rows[u[k]];mark.ellipse((x*scale-4,y*scale-4,x*scale+4,y*scale+4),fill='black')
                mark.text((x*scale+6,y*scale-18),f"f{u[k]}",fill='black',font=tiny,stroke_width=2,stroke_fill='white')
    sheet.paste(base,(0,40));draw.text((8,10),f"f{start}..{end}: full-frame path over f{start} (dimmed). Blue=early, red=late. Black dots = stroke corners.",fill='black',font=font)
    x0=base.width+16;y=50
    shown=[(pid,u) for pid,u in units if u['major'] or not many]
    for line in ['Major strokes (median-filtered, screen px):' if many else 'Strokes (median-filtered, screen px):']+[f"{pid} {u['start_seconds']:.2f}-{u['end_seconds']:.2f}s {'hold' if u['direction']=='hold' else u['direction']} {u['distance_px']:.0f}px"+(' MAJOR' if u['major'] else '') for pid,u in shown][:14]:
        draw.text((x0,y),line,fill='black',font=tiny);y+=20
    y+=10;draw.text((x0,y),'Possible drawing/appearance change',fill='firebrick',font=tiny);y+=20
    draw.text((x0,y),'(candidates only; compare frames):',fill='firebrick',font=tiny);y+=20
    for c in changes['candidates']:
        tag=' + pose overlap/jump' if pose and c['frame'] in pose['candidates_near_pose_flags'] else ''
        draw.text((x0,y),f"f{c['after_frame']}->{c['frame']} diff {c['mean_abs_diff']:.0f}{tag}",fill='firebrick',font=tiny);y+=20
    if not changes['candidates']:draw.text((x0,y),'none above threshold',fill='gray',font=tiny)
    top=round(h*scale)+50;left,right=40,SIZE['width']-40
    def tx(n):return left+(right-left)*(0 if end==start else (n-start)/(end-start))
    draw.line((left,top+60,right,top+60),fill='black',width=1)
    for r in (pose['ranges'] if pose else []):
        a,b=r['frames'];draw.rectangle((tx(a)-2,top+22,tx(b)+2,top+28),fill='darkorange')
    for pid,u in units:
        fill='gray' if u['direction']=='hold' else ('black' if u['major'] else 'steelblue')
        draw.rectangle((tx(u['start_frame']),top+30 if u['major'] else top+40,tx(u['end_frame']),top+56),fill=fill)
        if u['major']:draw.text((tx(u['start_frame']),top+8),f"{u['direction']} {u['distance_px']:.0f}px",fill='black',font=tiny)
    for c in changes['candidates']:
        draw.line((tx(c['frame']),top+62,tx(c['frame']),top+90),fill='firebrick',width=3);draw.text((tx(c['frame'])-12,top+92),f"f{c['frame']}",fill='firebrick',font=tiny)
    for n in range(start,end+1,max(1,(end-start)//10)):
        draw.line((tx(n),top+60,tx(n),top+66),fill='black');draw.text((tx(n)-14,top+118),f"{table[n]['pts_seconds']:.2f}s",fill='black',font=tiny)
    draw.text((left,top+136),'Timeline: bars = strokes (black major, blue minor, gray hold); red ticks = appearance-change candidates'+('; orange = pose overlap/jump frames.' if pose else '.'),fill='black',font=tiny)
    ty=top+165
    for i,n in enumerate(picks):
        image=full[n].resize((thumb_w,thumb_h));draw_path(ImageDraw.Draw(image),tscale,until=n,radius=6)
        sheet.paste(image,(i*(thumb_w+10)+5,ty+22));draw.text((i*(thumb_w+10)+8,ty),f"f{n} {table[n]['pts_seconds']:.2f}s (full frame, path so far)",fill='black',font=tiny)
    draw.text((8,sheet.height-26),'Model surface estimates on screen 2D. No skeleton/person-ID/ease verification. Missing observations are never drawn as motion.',fill='black',font=tiny)
    destination.mkdir();path=destination/'trajectory.png';sheet.save(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),thumbnail_frames=picks,
                appearance_change=dict(changes,params=CHANGE,diagnostic_only=True,automatic_redraw_decision=None,
                                       basis='mean absolute gray difference between consecutive downscaled frames; motion, effects or camera can also trigger it'),
                pose_flags=pose,not_recomputed_by_brief=True)
