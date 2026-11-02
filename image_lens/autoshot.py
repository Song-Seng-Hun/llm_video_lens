"""Optional fixed AutoShot recipe. Static provenance/readers never import Torch or fetch weights."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from image_lens import paths

HERE=Path(__file__).resolve().parent
WEIGHT_SHA='3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6'
WEIGHT_BYTES=57243097
TORCH_THREADS=8  # 4 -> 8 on the tested Mac: AutoShot 9.6 -> 9.0 s on the AI clip, heads bit-identical; 12 adds ~0.2 s


def pins():
    data=json.loads((HERE/'autoshot-pins.json').read_text())
    if (data.get('schema')!='image-lens.autoshot-pins/1' or data.get('commit')!='77c82ff826a9301bb173d9be786297a49d73d081'
            or data['checkpoint']['sha256']!=WEIGHT_SHA or data['checkpoint']['bytes']!=WEIGHT_BYTES):
        raise ValueError('unsupported AutoShot pins')
    for name,expected in data['files'].items():
        file=HERE/name
        if file.is_symlink() or not file.resolve().is_relative_to(HERE/'vendor/source/autoshot') or paths.sha(file)!=expected:
            raise ValueError('AutoShot source pin mismatch: '+name)
    return data


def recipe():
    data=pins()
    return dict(name='AutoShot',commit=data['commit'],source_manifest_sha256=paths.sha(HERE/'autoshot-pins.json'),helper_sha256=paths.sha(__file__),
                checkpoint_sha256=WEIGHT_SHA,checkpoint_bytes=WEIGHT_BYTES,threshold=.296,comparison='strict_greater_float32',
                input='native_dense_RGB_uint8_INTER_LINEAR48x27',window=100,step=50,left_repeat=25,keep=[25,75],trim='native_N',
                mapping='positive_run_end_plus_one; prefix allowed; suffix raw/unmapped',device='cpu',torch_threads=TORCH_THREADS,interop_threads=1,decoder_threads=4)


def bounded(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('thorough cut scan wall budget exhausted')


def checkpoint(deadline):
    """Only the explicit inference entrypoint calls this. Publish only a verified newly created file."""
    from urllib.request import urlopen
    data=pins()['checkpoint'];root=paths.confined(paths.workspace_root()/'models/autoshot');root.mkdir(parents=True,exist_ok=True)
    final=paths.confined(root/data['filename']);bounded(deadline)
    if final.exists():
        if final.stat().st_size!=WEIGHT_BYTES or paths.sha(final)!=WEIGHT_SHA:raise ValueError('corrupt AutoShot cache; refusing overwrite')
        return final,False
    partial=paths.confined(root/(data['filename']+'.partial'),new=True);created=False
    try:
        with partial.open('xb') as stream:
            created=True;size=0;digest=hashlib.sha256()
            with urlopen(data['url'],timeout=max(.001,min(30.,deadline-time.monotonic()))) as response:
                while True:
                    bounded(deadline);chunk=response.read(65536)
                    if not chunk:break
                    size+=len(chunk)
                    if size>WEIGHT_BYTES:raise ValueError('AutoShot download exceeds pinned byte count')
                    digest.update(chunk);stream.write(chunk)
            if size!=WEIGHT_BYTES or digest.hexdigest()!=WEIGHT_SHA:raise ValueError('AutoShot download hash/size mismatch')
        bounded(deadline);os.link(partial,final)
        return final,True
    finally:
        if created and partial.exists():partial.unlink()


def runs(probabilities):
    """Same float32 threshold comparison after JSON as on the original NumPy model head."""
    def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
    threshold=f32(.296);found=[];start=None
    for n,value in enumerate(probabilities):
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('invalid AutoShot probability')
        positive=f32(value)>threshold
        if positive and start is None:start=n
        if start is not None and (not positive or n==len(probabilities)-1):
            end=n if positive else n-1
            found.append(dict(start=start,end=end,length=end-start+1,first_new=end+1 if end+1<len(probabilities) else None));start=None
    return found


def validate(evidence,n,source_sha):
    if (not isinstance(evidence,dict) or evidence.get('recipe')!=recipe() or evidence.get('source_sha256')!=source_sha
            or evidence.get('frame_count')!=n or evidence.get('model_loaded') is not True or evidence.get('forward_calls')!=(n+49)//50):
        raise ValueError('AutoShot evidence source/model/recipe mismatch')
    single=evidence.get('single');many=evidence.get('many')
    if not isinstance(single,list) or not isinstance(many,list) or len(single)!=n or len(many)!=n:raise ValueError('AutoShot native head length mismatch')
    expected=runs(single);runs(many)
    if evidence.get('positive_runs')!=expected:raise ValueError('AutoShot threshold/run mapping mismatch')
    return [r['first_new'] for r in expected if r['first_new'] is not None]


def window(frames,k):
    """The k-th 100-frame input: native frames 50k-25..50k+74, the first and last frame repeated past the ends."""
    import numpy as np
    return np.stack([frames[min(max(n,0),len(frames)-1)] for n in range(50*k-25,50*k+75)])


def dependencies():
    try:
        import torch
        import einops
    except ImportError as error:
        raise ImportError('thorough cuts require the optional cuts dependencies; install them explicitly') from error
    if str(torch.__version__).split('+')[0]!='2.9.0' or einops.__version__!='0.8.1':raise ValueError('thorough cuts require pinned torch2.9.0/einops0.8.1')
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_interop_threads()!=1:
        try:torch.set_num_interop_threads(1)
        except RuntimeError as error:raise ValueError(f'host cannot admit fixed AutoShot CPU{TORCH_THREADS}/interop1 settings') from error
    return torch


def _module(name,file):
    directory=HERE/'vendor/source/autoshot'
    spec=importlib.util.spec_from_file_location(name,directory/file);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out


def load_model(torch,weights):
    previous=sys.modules.get('linear');sys.modules['linear']=_module('autoshot_linear','linear.py')
    try:source=_module('autoshot_model','supernet_flattransf_3_8_8_8_13_12_0_16_60.py')
    finally:
        if previous is None:sys.modules.pop('linear',None)
        else:sys.modules['linear']=previous
    model=source.TransNetV2Supernet().cpu().eval()
    if model.color_hist_layer is not None:model.color_hist_layer.device='cpu'
    if model.frame_sim_layer is not None:model.frame_sim_layer.device='cpu'
    try:
        payload=torch.load(weights,map_location='cpu',weights_only=True)
        model.load_state_dict(payload['net'],strict=True)
    except (KeyError,RuntimeError) as error:raise ValueError('AutoShot strict checkpoint load failed: '+str(error)) from error
    return model


class Stream:
    """predict, fed a frame at a time while the scan decodes: a thread readies the model and runs each window once its
    frames are in, so AutoShot overlaps the decode instead of following it (AI clip, 1807 f: scan 13.9 -> 9.9 s).
    Same windows, same inputs, same CPU/interop thread settings as one call on the stacked frames."""
    def __init__(self,source_sha,deadline):
        import queue,threading
        self.source_sha=source_sha;self.deadline=deadline;self.frames=[];self.sent=0;self.todo=queue.Queue()
        self.error=None;self.single=[];self.many=[];self.calls=0
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def feed(self,rgb):
        import numpy as np
        if self.error is not None:raise self.error
        if rgb.dtype!=np.uint8 or rgb.shape!=(27,48,3):raise ValueError('native RGB27x48 uint8 input required')
        self.frames.append(rgb)
        if len(self.frames)==50*self.sent+75:self.todo.put(self.sent);self.sent+=1  # its last native frame is in

    def finish(self):
        import numpy as np
        n=len(self.frames)
        if n<2:raise ValueError('native RGB27x48 uint8 input required')
        for k in range(self.sent,(n+49)//50):self.todo.put(k)
        self.close();self.thread.join()
        if self.error is not None:raise self.error
        bounded(self.deadline);single=np.concatenate(self.single)[:n].tolist();many=np.concatenate(self.many)[:n].tolist()
        evidence=dict(recipe=self.configuration,source_sha256=self.source_sha,frame_count=n,single=single,many=many,positive_runs=runs(single),
                      model_loaded=True,forward_calls=self.calls,weights=str(self.weights),downloaded=self.downloaded,torch_version=str(self.torch.__version__))
        validate(evidence,n,self.source_sha)
        return evidence

    def close(self):
        self.todo.put(None)

    def _run(self):
        try:
            import numpy as np
            bounded(self.deadline);self.configuration=recipe();torch=self.torch=dependencies();bounded(self.deadline)
            self.weights,self.downloaded=checkpoint(self.deadline);model=load_model(torch,self.weights);bounded(self.deadline)
            with torch.inference_mode():
                while (k:=self.todo.get()) is not None:
                    bounded(self.deadline);a,b=model(torch.from_numpy(window(self.frames,k)[None]).permute(0,4,1,2,3).float());self.calls+=1
                    if tuple(a.shape)!=(1,100,1) or tuple(b.shape)!=(1,100,1):raise ValueError('AutoShot official head shape mismatch')
                    a=torch.sigmoid(a)[0,:,0].cpu().numpy();b=torch.sigmoid(b)[0,:,0].cpu().numpy()
                    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite AutoShot head')
                    self.single.append(a[25:75]);self.many.append(b[25:75])
        except BaseException as error:
            self.error=error


def predict(frames,source_sha,deadline):
    import numpy as np
    if frames.dtype!=np.uint8 or frames.ndim!=4 or frames.shape[1:]!=(27,48,3) or len(frames)<2:raise ValueError('native RGB27x48 uint8 input required')
    stream=Stream(source_sha,deadline)
    try:
        for rgb in frames:stream.feed(rgb)
        return stream.finish()
    finally:stream.close()
