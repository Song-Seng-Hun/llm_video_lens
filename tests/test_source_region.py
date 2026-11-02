"""Source-region pixel/timing/provenance boundaries; no model or camera evaluation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from image_lens import cli,digest,look,paths,redo,source_region


class Region(unittest.TestCase):
    def setUp(self):
        self.previous = paths.workspace_root();self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve();paths.configure(self.root)
        self.video = self.root/'parent.avi';self.frames = []
        rng = np.random.default_rng(19)
        writer = cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'FFV1'),25,(64,48))
        self.assertTrue(writer.isOpened())
        for n in range(4):
            image = rng.integers(0,256,(48,64,3),np.uint8);self.frames.append(image);writer.write(image)
        writer.release()

    def tearDown(self):
        paths.configure(self.previous);self.temp.cleanup()

    def prepare(self,name='crop',**kwargs):
        root = self.root/name;root.mkdir()
        return look.fit(self.video,root,scene_box=[4,6,52,38],**kwargs)

    def test_exact_pixels_frames_timing_mapping_and_no_overwrite(self):
        """Crop/resize must preserve decoded pixels, native indices and invertible raw mapping."""
        before = self.video.read_bytes()
        for name,cap in [('native',1920*1080),('small',384)]:
            source,info = self.prepare(name,max_pixels=cap);r = info['source_region']
            size = (r['analysis']['width'],r['analysis']['height']);c = cv2.VideoCapture(str(source))
            for n,frame in enumerate(self.frames):
                ok,image = c.read();self.assertTrue(ok)
                expected = frame[6:38,4:52]
                if size != (48,32):expected = cv2.resize(expected,size,interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(image,expected);self.assertEqual(c.get(cv2.CAP_PROP_PTS),n)
            self.assertFalse(c.read()[0]);c.release();self.assertEqual(self.video.read_bytes(),before)
            matrix = r['raw_analysis_to_parent']['matrix3x3']
            sx,sy = 48/size[0],32/size[1]
            np.testing.assert_allclose(np.array(matrix)@np.array([0.,0.,1.]),[4+(sx-1)/2,6+(sy-1)/2,1.])
            self.assertEqual(source_region.validate(source,r),r)
            with self.assertRaises(ValueError):look.fit(self.video,source.parent,scene_box=[4,6,52,38])
            broken = copy.deepcopy(r);broken['box'][0] += 2
            with self.assertRaises(ValueError):source_region.validate(source,broken)
        self.video.write_bytes(before+b'stale')
        with self.assertRaises(ValueError):source_region.validate(source,r)

    def test_invalid_bounds_and_non_cfr_fail_without_final_copy(self):
        """No clipping or fabricated frame timing before downstream model startup."""
        for box in ('0,0,2.0,4','nan,0,2,4',[-1,0,3,4],[0,0,3,4],[0,0,0,4],[0,0,66,48]):
            root = self.root/'bad';root.mkdir(exist_ok=True)
            with self.assertRaises(ValueError):look.fit(self.video,root,scene_box=box)
            self.assertFalse((root/'source-region.avi').exists())
        factory = cv2.VideoCapture
        class VFR:
            def __init__(self,path):self.cap = factory(path)
            def __getattr__(self,key):return getattr(self.cap,key)
            def get(self,key):
                value = self.cap.get(key)
                return value+.5 if key==cv2.CAP_PROP_PTS and value==2 else value
        with patch.object(cv2,'VideoCapture',VFR):
            with self.assertRaisesRegex(ValueError,'VFR'):self.prepare('vfr')
        self.assertEqual(list((self.root/'vfr').iterdir()),[])

    def test_actual_look_scan_pose_and_reuse_boundaries(self):
        """Real look routing must scan the crop, and a regional redo must skip fit entirely."""
        def step(command,args):
            self.assertEqual(command,'scan');source = Path(args[args.index('--video')+1])
            self.assertNotEqual(source,self.video)
            paths.write_json(args[args.index('--out')+1],dict(source_sha256=paths.sha(source)))
        def child(args):
            scan,_ = paths.load(args[args.index('--shots')+1]);self.assertTrue(scan['source_sha256'])
            raise RuntimeError('verified pose boundary')
        with patch.object(look,'_step',side_effect=step),patch.object(look,'_pose_child',side_effect=child):
            with self.assertRaisesRegex(RuntimeError,'verified pose boundary'):
                look._run(self.video,'d','p','c',self.root/'run',scene_box=[4,6,52,38])
        crop,info = self.prepare('reuse')
        with patch.object(look,'fit',side_effect=AssertionError('double crop')),patch.object(look,'_step',side_effect=step),patch.object(look,'_pose_child',side_effect=child):
            with self.assertRaisesRegex(RuntimeError,'verified pose boundary'):
                look._run(crop,'d','p','c',self.root/'rerun',source_region=info['source_region'])

    def test_redo_preserves_region_and_headers(self):
        """Existing redo source/identity route must retain the parent map and displayed scope."""
        crop,info = self.prepare();r = info['source_region'];root = crop.parent
        record = dict(source_video=str(crop),source_sha256=paths.sha(crop),source_region=r,shots=[dict(segment_id='shot-0000',windows=[dict(window=dict(seed=0,end=3))])])
        packet = dict(source_sha256=paths.sha(crop),models=dict(detector=dict(kind='humanart')),frames=[])
        paths.write_json(root/'look.json',record);paths.write_json(root/'pose.json',packet)
        def rerun(*args,**kwargs):
            self.assertEqual(Path(args[0]),crop);self.assertEqual(kwargs['source_region'],r);self.assertNotIn('scene_box',kwargs)
            out = Path(args[4]);out.mkdir();paths.write_json(out/'pose.json',packet);return dict(status='written')
        with patch.object(look,'run',side_effect=rerun),patch('image_lens.person_models.resolve',return_value=dict(kind='humanart')),patch.object(redo,'boxes'):
            redo.run(root,'shot-0000-w0','d','p','c',self.root/'redo')
        text = digest.markdown(dict(record,shots=[],check=[]),dict(fps=25,source_frame_count=4),'look-full.md')
        self.assertIn(source_region.header(r),text)
        paths.write(root/'look-full.md',source_region.header(r)+'\n\n## shot-0000 details\nbody\n')
        self.assertTrue(digest.focus(root,'shot-0000').startswith(source_region.header(r)))
        from image_lens import check
        at_look = self.root/'at-look';at_look.mkdir()
        paths.write_json(at_look/'look.json',dict(record,shots=[],check=[]))
        paths.write_json(at_look/'shots.json',dict(fps=25,source_frame_count=4))
        at = check.at(at_look,'0-0.12',self.root/'at')
        self.assertIn(source_region.header(r),Path(at['output']).read_text())

    def test_cli_optional_dispatch_keeps_default_absent(self):
        """The real public look flag reaches preprocessing without altering the default call."""
        common=['look','--video',str(self.video),'--out',str(self.root/'cli')]
        with patch.object(cli,'models',side_effect=lambda a,*k:a),patch.object(look,'run',return_value=dict(status='fake')) as run:
            cli.dispatch('look',common[1:]+['--scene-box','4,6,52,38'])
            self.assertEqual(run.call_args.kwargs['scene_box'],[4,6,52,38])
            cli.dispatch('look',common[1:]);self.assertNotIn('scene_box',run.call_args.kwargs)
