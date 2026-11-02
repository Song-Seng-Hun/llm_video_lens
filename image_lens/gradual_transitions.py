"""One opt-in unverified transition policy. Static readers are stdlib-only."""
import math
import struct
from pathlib import Path
from image_lens import paths

PILOT_SHA256='0a3ac57bdc42c1452c347eb73913f93e636457fbb0f61a34240b301d29fdf46e'
CHEAP_PARAMS = dict(anchor_frames=3,stable_mae=1.0,min_mixed_frames=3,max_seconds=2.0,
              min_endpoint_mae=12.0,alpha_tolerance=.03,max_alpha_step=.5,min_alpha_span=.4,
              max_input_bytes=128*1024*1024,max_frames=18000,max_pixels=16384,delta_chunk=128,
              max_frame_residual=.15)


def recipe():
    return dict(name='many_and_endpoint_mixture_union',many_threshold=.3,many_comparison='strict_greater_float32',
                min_many_run=3,cheap_parameters=dict(CHEAP_PARAMS),cheap_config='primary',padding=0,
                helper_sha256=paths.sha(__file__),pilot_sha256=PILOT_SHA256,verified=False)


def many_runs(values):
    threshold=struct.unpack('<f',struct.pack('<f',.3))[0];found=[];start=None
    for n,v in enumerate(values):
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('invalid transition probability')
        positive=struct.unpack('<f',struct.pack('<f',v))[0]>threshold
        if positive and start is None:start=n
        if start is not None and (not positive or n==len(values)-1):
            end=n if positive else n-1
            if end-start+1>=3:found.append([start,end])
            start=None
    return found


def union(ranges):
    out=[]
    for a,b in sorted(ranges):
        if out and a<=out[-1][1]+1:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


def evidence(frames,fps,many,source_sha):
    return dict(recipe=recipe(),source_sha256=source_sha,frame_count=len(frames),
                many_runs=many_runs(many),cheap_intervals=detect(frames,fps))


def validate(data,many,n,fps,source_sha):
    """Derive the union from frozen heads/scalars, without claiming pixel recomputation."""
    if (type(data)is not dict or data.get('recipe')!=recipe() or data.get('source_sha256')!=source_sha
            or data.get('frame_count')!=n or data.get('many_runs')!=many_runs(many)):
        raise ValueError('transition policy/source/run provenance mismatch')
    records=data.get('cheap_intervals')
    if type(records)is not list or len(records)>n//3:raise ValueError('bounded cheap interval evidence required')
    ranges=[]
    for row in records:
        if type(row)is not dict or row.get('kind')!='endpoint_mixture_candidate' or row.get('verified')is not False or row.get('config')!='primary':raise ValueError('invalid cheap candidate')
        span=row.get('frames');e=row.get('evidence')
        if (type(span)is not list or len(span)!=2 or any(type(v)is not int for v in span)
                or not 0<=span[0]<=span[1]<n or not 3<=span[1]-span[0]+1<=math.floor(fps*2)
                or type(e)is not dict):raise ValueError('invalid cheap native span')
        a=e.get('anchor_before');b=e.get('anchor_after')
        if any(type(x)is not list or len(x)!=2 or any(type(v)is not int for v in x) for x in (a,b)) or not 0<=a[0]<=a[1]==span[0]-1 or not span[1]+1==b[0]<=b[1]<n or min(a[1]-a[0]+1,b[1]-b[0]+1)<3:raise ValueError('invalid cheap anchors')
        keys=('endpoint_mae','max_normalized_residual','mean_normalized_residual','alpha_first','alpha_last','max_alpha_step')
        if set(e)!=set(keys)|{'anchor_before','anchor_after'} or any(type(e[k])not in(int,float) or not math.isfinite(e[k]) for k in keys):raise ValueError('finite bounded cheap scalars required')
        if (not 12<=e['endpoint_mae']<=255 or not 0<=e['mean_normalized_residual']<=e['max_normalized_residual']<=.15
                or not -.03<=e['alpha_first']<=1.03 or not -.03<=e['alpha_last']<=1.03
                or e['alpha_last']-e['alpha_first']<.4 or not 0<=e['max_alpha_step']<=.5):raise ValueError('cheap scalar policy mismatch')
        ranges.append(span)
    if ranges!=sorted(ranges) or any(b>=c for (a,b),(c,d)in zip(ranges,ranges[1:])):raise ValueError('ordered disjoint cheap evidence required')
    return union(data['many_runs']+ranges)


def _stable_spans(change,n,minimum,threshold):
    spans=[];start=0
    for at,value in enumerate(change,1):
        if value>threshold:
            if at-start>=minimum:spans.append((start,at-1))
            start=at
    if n-start>=minimum:spans.append((start,n-1))
    return spans


def detect(frames,fps):
    """Return unverified endpoint-mixture intervals; no mutation, global state or per-clip tuning."""
    import numpy as np
    if isinstance(fps,bool) or not math.isfinite(float(fps)) or not 0<float(fps)<=240:
        raise ValueError('finite source fps in(0,240] required')
    frames=np.asarray(frames)
    if frames.dtype!=np.uint8 or frames.ndim!=4 or frames.shape[-1]!=3 or min(frames.shape[1:3])<1:
        raise ValueError('dense RGB uint8 frames[N,H,W,3] required')
    n=len(frames);p=CHEAP_PARAMS
    if n>p['max_frames'] or frames.nbytes>p['max_input_bytes'] or frames.shape[1]*frames.shape[2]>p['max_pixels']:
        raise ValueError('bounded small-frame transition input required')
    if n<2*p['anchor_frames']+p['min_mixed_frames']:return []
    flat=frames.reshape(n,-1);change=np.empty(n-1,np.float64)
    for at in range(1,n,p['delta_chunk']):
        end=min(n,at+p['delta_chunk'])
        delta=flat[at:end].astype(np.float32)-flat[at-1:end-1].astype(np.float32)
        np.abs(delta,out=delta);change[at-1:end-1]=delta.mean(axis=1,dtype=np.float64)
    spans=_stable_spans(change,n,p['anchor_frames'],p['stable_mae']);intervals=[]
    maximum=int(math.floor(float(fps)*p['max_seconds']))
    for left,right in zip(spans,spans[1:]):
        start,end=left[1]+1,right[0]-1
        if not p['min_mixed_frames']<=end-start+1<=maximum:continue
        a=np.median(flat[left[1]-p['anchor_frames']+1:left[1]+1],axis=0).astype(np.float32)
        b=np.median(flat[right[0]:right[0]+p['anchor_frames']],axis=0).astype(np.float32)
        direction=b-a;scale=float(np.mean(np.abs(direction),dtype=np.float64))
        if scale<p['min_endpoint_mae']:continue
        energy=float(np.einsum('i,i->',direction,direction,dtype=np.float64))
        if energy<=0:continue
        centered=flat[start:end+1].astype(np.float32)-a
        alpha=np.einsum('ij,j->i',centered,direction,dtype=np.float64)/energy
        steps=np.diff(np.concatenate(([0.],alpha,[1.])))
        if (alpha.min()<-p['alpha_tolerance'] or alpha.max()>1+p['alpha_tolerance']
                or steps.min()<-p['alpha_tolerance'] or steps.max()>p['max_alpha_step']
                or alpha[-1]-alpha[0]<p['min_alpha_span']):continue
        centered-=alpha.astype(np.float32)[:,None]*direction
        np.abs(centered,out=centered)
        residual=centered.mean(axis=1,dtype=np.float64)/scale
        if float(residual.max())>p['max_frame_residual']:continue
        intervals.append(dict(frames=[start,end],kind='endpoint_mixture_candidate',verified=False,config='primary',
                              evidence=dict(anchor_before=list(left),anchor_after=list(right),endpoint_mae=scale,
                                            max_normalized_residual=float(residual.max()),mean_normalized_residual=float(residual.mean()),
                                            alpha_first=float(alpha[0]),alpha_last=float(alpha[-1]),max_alpha_step=float(steps.max()))))
    return intervals
