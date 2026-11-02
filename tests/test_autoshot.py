"""Optional cut admission/routing guards; authored frames and fake model/fetch only."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from image_lens import autoshot,cli,jobs,look,paths,shot_boundaries

REPO=Path(__file__).resolve().parents[1]


class Cuts(unittest.TestCase):
    def setUp(self):
        self.previous=paths.workspace_root();self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve();paths.configure(self.root)
        self.video=self.root/'source.avi';writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'FFV1'),25,(64,48))
        self.assertTrue(writer.isOpened())
        for n in range(16):writer.write(np.full((48,64,3),(10,20,100) if n<8 else (200,30,40),np.uint8))
        writer.release()

    def tearDown(self):
        paths.configure(self.previous);self.temp.cleanup()

    def evidence(self,n,source_sha):
        p=[0.]*n;p[5]=1.
        return dict(recipe=autoshot.recipe(),source_sha256=source_sha,frame_count=n,single=p,many=[0.]*n,
                    positive_runs=autoshot.runs(p),model_loaded=True,forward_calls=(n+49)//50)

    def test_fresh_default_and_explicit_fast_never_import_or_fetch(self):
        """The actual CLI fast path remains runnable with optional imports/network forbidden."""
        script='''import sys,urllib.request,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class NoOptional:
 def find_spec(self,name,*a,**k):
  if name.split('.')[0] in ('torch','einops'):raise AssertionError('optional import: '+name)
sys.meta_path.insert(0,NoOptional())
def blocked(*a,**k):raise AssertionError('network')
urllib.request.urlopen=blocked
from image_lens.cli import main
root=Path(sys.argv[2])
for label,extra in [('default',[]),('fast',['--cuts','fast'])]:
 assert main(['--workspace',str(root),'scan','--video','source.avi','--out',label+'.json',*extra])==0
a=json.loads((root/'default.json').read_text());b=json.loads((root/'fast.json').read_text())
assert a['events']==b['events'] and a['analysis_segments']==b['analysis_segments']
assert [e['after_frame'] for e in a['events']]==[8]
assert 'torch' not in sys.modules and 'einops' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-I','-B','-c',script,str(REPO),str(self.root)],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('"model_loaded": false',result.stdout)

    def test_native_union_provenance_barrier_and_no_partial_downgrade(self):
        """Union must retain adaptive cuts, add native onsets, and survive frozen admission without NN."""
        def predict(frames,source_sha,deadline):
            self.assertEqual(frames.shape,(16,27,48,3));np.testing.assert_array_equal(frames[0,0,0],[100,20,10])
            return self.evidence(len(frames),source_sha)
        class Stream:
            def __init__(self,source_sha,deadline):self.source_sha=source_sha;self.deadline=deadline;self.frames=[]
            def feed(self,rgb):self.frames.append(rgb)
            def finish(self):return predict(np.stack(self.frames),self.source_sha,self.deadline)
            def close(self):pass
        with patch.object(autoshot,'Stream',side_effect=Stream) as inference:
            scan=shot_boundaries.scan_video(self.video,cuts='thorough')
            self.assertEqual([e['after_frame'] for e in scan['events']],[6,8])
            with self.assertRaisesRegex(ValueError,'full-source'):shot_boundaries.scan_video(self.video,start=1,cuts='thorough')
            self.assertEqual(inference.call_count,1)
        broken=copy.deepcopy(scan);broken['temporal']['positive_runs'][0]['first_new']=7
        with self.assertRaises(ValueError):shot_boundaries.validate(broken)
        job=dict(source_video=str(self.video),source_sha256=paths.sha(self.video),width=64,height=48,fps=25.,start_frame=0,end_frame=15)
        missing=shot_boundaries.assemble(job,scan['native_pts_seconds'],[6],mode='thorough')
        missing.update(scan_scope='full_source',source_frame_count=16,temporal=scan['temporal'])
        with self.assertRaisesRegex(ValueError,'preserve every adaptive'):shot_boundaries.validate(missing)
        paths.write_json('shots.json',scan)
        query=dict(schema='team-eye.queries/1',jobs=[dict(job,name='a',end_frame=4,seed_frame=0,points=[dict(point_id='p',xy=[10.,20.])])])
        paths.write_json('queries.json',query);jobs.prepare('queries.json','shots.json','freeze.json','a')
        frozen,_=paths.load('freeze.json');self.assertEqual(frozen['cuts_mode'],'thorough')
        self.assertIn('autoshot.py',frozen['helper_pins']);self.assertIn('autoshot-pins.json',frozen['helper_pins'])
        stale=copy.deepcopy(frozen);stale['helper_pins']['autoshot.py']='0'*64;paths.write_json('stale-helper.json',stale)
        with self.assertRaisesRegex(ValueError,'helper pins'):jobs.verify_frozen(self.root/'stale-helper.json',self.root/'queries.json')
        broken=copy.deepcopy(frozen);broken.pop('canonical_shots');paths.write_json('missing.json',broken)
        with self.assertRaisesRegex(ValueError,'canonical shot reference'):jobs.verify_frozen(self.root/'missing.json',self.root/'queries.json')
        with self.assertRaisesRegex(ValueError,'policy mismatch'):jobs.check_cut_policy(dict(frozen,cuts_mode='fast'),scan)

    def test_float32_runs_and_source_model_recipe_pins(self):
        """JSON roundtrips cannot change strict thresholds or admit stale model/helper/source evidence."""
        p=[.5,0.,float(np.float32(.296)),.5,.5,0.,1.]
        self.assertEqual([r['first_new'] for r in autoshot.runs(json.loads(json.dumps(p)))],[1,5,None])
        evidence=self.evidence(16,'a'*64);evidence['recipe']['checkpoint_sha256']='0'*64
        with self.assertRaises(ValueError):autoshot.validate(evidence,16,'a'*64)
        evidence=self.evidence(16,'a'*64)
        with self.assertRaises(ValueError):autoshot.validate(evidence,16,'b'*64)
        original=paths.sha
        with patch.object(paths,'sha',side_effect=lambda path:'0'*64 if str(path).endswith('/linear.py') else original(path)):
            with self.assertRaisesRegex(ValueError,'source pin'):autoshot.recipe()
        frames=np.broadcast_to(np.arange(51,dtype=np.uint8)[:,None,None,None],(51,27,48,3)).copy()
        windows=[autoshot.window(frames,k) for k in range(2)]
        np.testing.assert_array_equal(np.concatenate([w[25:75,0,0,0] for w in windows])[:51],np.arange(51))

    def test_hash_pinned_fetch_cache_and_own_partial_cleanup(self):
        """Cold fetch is atomic, verified and never replaces corrupt/pre-existing user artifacts."""
        data=b'owned tiny checkpoint fixture';digest=hashlib.sha256(data).hexdigest()
        pin=dict(checkpoint=dict(filename='tiny.pth',url='https://example.invalid/fixed'))
        with patch.object(autoshot,'pins',return_value=pin),patch.object(autoshot,'WEIGHT_BYTES',len(data)),patch.object(autoshot,'WEIGHT_SHA',digest):
            with patch('urllib.request.urlopen',return_value=io.BytesIO(data)) as fetch:
                file,downloaded=autoshot.checkpoint(time.monotonic()+5);self.assertTrue(downloaded);self.assertEqual(file.read_bytes(),data)
                self.assertFalse(autoshot.checkpoint(time.monotonic()+5)[1]);self.assertEqual(fetch.call_count,1)
            file.write_bytes(b'corrupt')
            with patch('urllib.request.urlopen') as fetch:
                with self.assertRaisesRegex(ValueError,'corrupt'):autoshot.checkpoint(time.monotonic()+5)
                fetch.assert_not_called();self.assertEqual(file.read_bytes(),b'corrupt')
            file.unlink();partial=file.with_name(file.name+'.partial');partial.write_bytes(b'keep')
            with self.assertRaises(ValueError):autoshot.checkpoint(time.monotonic()+5)
            self.assertEqual(partial.read_bytes(),b'keep');partial.unlink()
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaisesRegex(ValueError,'hash/size'):autoshot.checkpoint(time.monotonic()+5)
            self.assertFalse(file.exists());self.assertFalse(partial.exists())

    def test_optional_dependency_strict_checkpoint_and_cli_selection(self):
        """Missing deps and incompatible state fail explicitly; the public selector reaches look unchanged."""
        with patch.object(autoshot,'dependencies',side_effect=ImportError('missing optional')),patch.object(autoshot,'checkpoint') as cache:
            with self.assertRaises(ImportError):autoshot.predict(np.zeros((2,27,48,3),np.uint8),'0'*64,time.monotonic()+5)
            cache.assert_not_called()
        class Model:
            color_hist_layer=types.SimpleNamespace(device='cuda')
            frame_sim_layer=types.SimpleNamespace(device='cuda')
            def cpu(self):return self
            def eval(self):return self
            def load_state_dict(self,state,strict):
                if strict and 'unexpected' in state:raise RuntimeError('unexpected checkpoint key')
        torch=types.SimpleNamespace(load=Mock(return_value={'net':{'unexpected':1}}))
        with patch.object(autoshot,'_module',return_value=types.SimpleNamespace(TransNetV2Supernet=Model)):
            with self.assertRaisesRegex(ValueError,'strict checkpoint'):autoshot.load_model(torch,'unused')
        self.assertEqual(Model.color_hist_layer.device,'cpu');self.assertEqual(Model.frame_sim_layer.device,'cpu')
        torch.load.assert_called_once_with('unused',map_location='cpu',weights_only=True)
        with patch.object(cli,'models',side_effect=lambda a,*k:a),patch.object(look,'run',return_value={}) as run:
            cli.dispatch('look',['--video','source.avi','--out','look','--cuts','thorough'])
            self.assertEqual(run.call_args.kwargs['cuts'],'thorough')
            cli.dispatch('look',['--video','source.avi','--out','look'])
            self.assertNotIn('cuts',run.call_args.kwargs)
