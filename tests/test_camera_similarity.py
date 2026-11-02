"""Analytic similarity/consumer regressions; source packets and model execution stay untouched."""
import copy
import json
import math
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from image_lens import camera, onion, body_motion, motion_report


def similarity(scale=1., angle=0., shift=(0.,0.), centre=(320.,240.)):
    a,b=scale*math.cos(angle),scale*math.sin(angle)
    x,y=centre
    return np.array([[a,-b,x-a*x+b*y+shift[0]],[b,a,y-b*x-a*y+shift[1]],[0.,0.,1.]])


class Similarity(unittest.TestCase):
    def test_retained_fallback_reference_survives_gaps_and_resets_on_global_recovery(self):
        """Avoid adjacent-fit accumulation; existing rekey test covers only one fallback step."""
        import cv2
        source=[np.full((480,640,3),n,np.uint8) for n in range(10)];calls=[]
        # Small-image translations. Adjacent fits can drift despite a usable retained reference.
        shifts={(0,1):15.,(1,2):15.,(1,3):30.,(2,3):16.,(3,4):15.,
                (3,6):45.,(4,6):31.,(0,7):105.,(7,8):15.,(7,9):30.,(8,9):16.}
        class ORB:
            def detectAndCompute(self,gray,mask):return [],np.array([[int(gray[0,0])]],np.uint8)
        def step(cv2,np,matcher,base,keys,desc,shape,*spread):
            a,b=int(base[1][0,0]),int(desc[0,0]);calls.append((a,b,bool(spread)))
            return similarity(shift=(shifts[a,b],0.))[:2] if (a,b) in shifts else None
        with patch('image_lens.frames.window',return_value=source),patch.object(cv2,'ORB_create',return_value=ORB()),patch.object(camera,'_step',side_effect=step):
            result=camera._path('unused',0,9)
        self.assertEqual(calls,[(0,1,False),(0,2,False),(1,2,True),(0,3,False),(1,3,True),
                              (0,4,False),(1,4,True),(3,4,True),(0,5,False),(3,5,True),(4,5,True),
                              (0,6,False),(3,6,True),(0,7,False),(0,8,False),(7,8,True),(0,9,False),(7,9,True)])
        self.assertEqual(sorted(result),[0,1,2,3,4,6,7,8,9])
        for n,matrix in result.items():
            np.testing.assert_allclose(matrix,similarity(shift=(30.*n,0.)),atol=1e-9)
        # A retained fit cannot bypass the old admission after returning near the origin.
        calls.clear();shifts={(0,1):15.,(1,2):-10.,(1,3):15.}
        with patch('image_lens.frames.window',return_value=source[:4]),patch.object(cv2,'ORB_create',return_value=ORB()),patch.object(camera,'_step',side_effect=step):
            result=camera._path('unused',0,3)
        self.assertEqual(calls,[(0,1,False),(0,2,False),(1,2,True),(0,3,False)])
        self.assertEqual(sorted(result),[0,1,2])
        np.testing.assert_allclose(result[2],similarity(shift=(10.,0.)),atol=1e-9)

    def test_subwindow_body_basis_has_one_explicit_reference(self):
        from tests.test_body_motion import packet
        import cv2
        path=camera.CameraPath({n:similarity(4. if n>=5 else 1.,centre=(0.,0.)) for n in range(26)},size=(3000,3000))
        own={'shoulder-l':{n:(300.+8*n,300.) for n in range(26)},
             'shoulder-r':{n:(100.+8*n,300.) for n in range(26)},'wrist-l':{n:(320.+8*n,400.) for n in range(26)}}
        source=packet({pid:{n:camera.apply(path[n],xy) for n,xy in points.items()} for pid,points in own.items()})
        body=onion.body_path([source],5,25,camera=path)
        class Capture:
            def get(self,key):return 3000
            def release(self):pass
        kwargs=dict(camera=path)
        if 'reference' in inspect.signature(onion.body_basis).parameters:kwargs['reference']=5
        with patch.object(cv2,'VideoCapture',return_value=Capture()):
            found=onion.body_basis('unused',[400,800,2000,2000],body,[source],**kwargs)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found,(body[25][0]-body[6][0],body[25][1]-body[6][1]))

    def test_subwindow_diagnostics_match_corrected_report_reference(self):
        from tests.test_look import Camera
        from image_lens import analyze_motion
        legacy={n:(100.+10*n,50.-2*n,1.) for n in range(26)}
        packet,_,_=Camera().view([(400.+n,200.) for n in range(26)],None)
        binding=dict(schema='team-eye.motion-bindings/1',source_tracks_sha256='0'*64,basis='source_frame_grouping_only',
                     actors=[dict(actor_id='marked',label='marked',features=[dict(point_id='hand',label='hand')])])
        report=motion_report.build_report(packet,binding,5,25,analyze_motion.collect_observations,analyze_motion.infer)
        view=motion_report.camera_relative(report,packet,binding,legacy,analyze_motion.collect_observations,analyze_motion.infer)
        self.assertEqual(view['coordinate_transform']['camera_path'][0],[5,0.,0.])
        path=camera.CameraPath({n:similarity(1+.01*n,.01*n) for n in range(26)},size=(640,480))
        kwargs=dict(reference=5) if 'reference' in inspect.signature(camera.centre_path).parameters else {}
        np.testing.assert_allclose(camera.centre_path(path,**kwargs)[5],(0.,0.,1.),atol=1e-10)
        self.assertAlmostEqual(camera.describe(path,5,25,15)['rotation_clockwise_degrees'],math.degrees(.20))
        np.testing.assert_allclose(onion._views(path,640,480,reference=5)[5],(0.,0.,640.,480.),atol=1e-10)

    def test_canonical_size_required(self):
        for size in (None,(0,480),(640,-1),(float('nan'),480)):
            with self.assertRaises(ValueError):camera.CameraPath({0:np.eye(3)},size=size)

    def test_metadata_ownership(self):
        source={'video':'original','tags':['a']};bounds=[0,2];scales={0:1.}
        path=camera.CameraPath({0:np.eye(3),1:np.eye(3),2:np.eye(3)},size=(640,480),source=source,native_range=bounds,diagnostic_scales=scales)
        source['tags'].append('mutated');bounds[1]=99;scales[0]=9.
        self.assertEqual(path.source['tags'],['a']);self.assertEqual(path.native_range,[0,2]);self.assertEqual(path.diagnostic_scales[0],1.)
        prefix=path.prefix(1);prefix.source['tags'].append('prefix');prefix.diagnostic_scales[0]=7.
        saved=camera.record(path);saved['source']['tags'].append('record');saved['native_range'][1]=88
        self.assertEqual(path.source['tags'],['a']);self.assertEqual(path.native_range,[0,2]);self.assertEqual(path.diagnostic_scales[0],1.)
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'v.bin';video.write_bytes(b'cache boundary, no decoder')
            camera._LAST.clear()
            with patch.object(camera,'_path',return_value=path):
                first=camera.path(video,0,2);first.source['video']='caller changed';first.diagnostic_scales[0]=5.
                second=camera.path(video,0,1)
            self.assertEqual(second.source['video'],str(video));self.assertEqual(second.diagnostic_scales[0],1.)
            camera._LAST.clear()

    def test_derived_unknown_visibility_retains_exact_source_provenance(self):
        from tests.test_look import Camera
        source,_,_=Camera().view([(400.+n,200.) for n in range(26)],None);before=copy.deepcopy(source)
        path=camera.CameraPath({n:np.eye(3) for n in range(26) if n!=13},size=(1920,1080))
        derived=camera.correct_packet(source,path,0);point=derived['frames'][13]['points'][0]
        self.assertEqual(derived.get('coordinate_scope'),'camera_similarity_relative_2d')
        self.assertFalse(point['visible']);self.assertTrue(point['source_visible']);self.assertTrue(point['model_visible'])
        self.assertEqual(point['source_xy'],before['frames'][13]['points'][0]['xy'])
        point['source_xy'][0]=999.
        self.assertEqual(source,before)
        self.assertEqual([f['pts_seconds'] for f in derived['frames']],[f['pts_seconds'] for f in source['frames']])

    def test_native_inverse_composition_persistence_and_explicit_legacy(self):
        a=similarity(1.1,.2,(10.,-5.));b=similarity(.8,-.1,(-8.,6.))@a
        path=camera.CameraPath({0:np.eye(3),1:a,2:b},size=(640,480),native_range=[0,2])
        for ref in (0,1):
            for n,m in path.items():
                native=camera.apply(m,(210.,170.));expected=camera.apply(path[ref],(210.,170.))
                np.testing.assert_allclose(camera.apply(camera.transforms(path,ref)[n],native),expected,atol=1e-9)
        saved=camera.record(path,'source-pin');restored=camera.normalize(json.loads(json.dumps(saved)))
        self.assertEqual(path,restored);self.assertEqual(saved['schema'],'image-lens.camera/1')
        self.assertEqual(saved['source_sha256'],'source-pin')
        legacy=camera.normalize({0:(0.,0.,1.),1:(0.,0.,2.)})
        self.assertEqual(legacy.kind,'legacy_translation')
        self.assertEqual(camera.apply(camera.transforms(legacy)[1],(200.,100.)),(200.,100.))
        for bad in (np.zeros((3,3)),np.diag([-1.,1.,1.]),np.diag([2.,1.,1.])):
            with self.assertRaises(ValueError):camera.CameraPath({0:bad},size=(640,480))
        with self.assertRaises(ValueError):camera.normalize(dict(saved,schema='image-lens.camera/99'))

    def test_numeric_report_trails_native_bounds_and_source_invariance(self):
        from tests.test_look import Camera
        path=camera.CameraPath({n:similarity(1+n*.01,n*.008,(n*2.,-n)) for n in range(26)},size=(1920,1080))
        screen=[camera.apply(path[n],(300.+2*n,200.)) for n in range(26)]
        packet,raw,view=Camera().view(screen,path);before=copy.deepcopy(packet)
        self.assertEqual(view['coordinate_scope'],'camera_similarity_relative_2d')
        self.assertTrue(view['coordinate_transform']['scale_rotation_corrected'])
        self.assertEqual(view['features'][0]['usable_prediction_frames'],25)
        trail=onion.trails([packet],['hand'],0,25,path)[0]['xy']
        np.testing.assert_allclose(trail[-1][1:],(350.,200.),atol=1e-8)
        self.assertEqual(packet,before)
        # Valid source points can inverse-map beyond the seed viewport; source in_frame remains true.
        outside=camera.CameraPath({n:similarity(shift=(100*n,0.)) for n in range(26)},size=(1920,1080))
        source,_,corrected=Camera().view([(50.,100.)]*26,outside)
        self.assertEqual(corrected['features'][0]['usable_prediction_frames'],25)
        self.assertTrue(all(f['points'][0]['in_frame'] for f in source['frames']))
        self.assertLess(onion.trails([source],['hand'],0,25,outside)[0]['xy'][-1][1],0)

    def test_camera_gaps_break_runs_and_body_smoothing_never_crosses_gap(self):
        from tests.test_look import Camera
        path=camera.CameraPath({n:np.eye(3) for n in range(26) if n!=13},size=(1920,1080))
        packet,_,view=Camera().view([(300.+n,200.) for n in range(26)],path)
        self.assertEqual(view['coordinate_transform']['unsupported_frames'],[[13,13]])
        self.assertEqual([run['frames'] for run in view['features'][0]['continuous_runs']],[[1,12],[14,25]])
        self.assertNotIn(13,[n for n,_,_ in onion.trails([packet],['hand'],0,25,path)[0]['xy']])
        self.assertEqual(onion._smooth({1:(0.,0.),2:(0.,0.),10:(100.,0.),11:(100.,0.)}),
                         {1:(0.,0.),2:(0.,0.),10:(100.,0.),11:(100.,0.)})

    def test_point_image_band_and_body_composition_share_inverse(self):
        import cv2
        m=similarity(1.15,.12,(3.,-2.),centre=(50.,50.));path=camera.CameraPath({0:np.eye(3),1:m},size=(100,100))
        seed=np.zeros((100,100,3),np.uint8);cv2.circle(seed,(40,40),3,(255,255,255),-1)
        own=seed.copy();own[:]=0;cv2.circle(own,(48,40),3,(255,255,255),-1)
        current=cv2.warpAffine(own,m[:2],(100,100))
        source=dict(frames=[dict(frame=0,points=[dict(point_id='p',xy=[40.,40.],visible=True)]),
                            dict(frame=1,points=[dict(point_id='p',xy=list(camera.apply(m,(48.,40.))),visible=True)])])
        points=np.asarray([[[40.,40.]]]*12,np.float32)
        def flow(a,b,cur,*args,**kwargs):
            return np.asarray([camera.apply(m,xy) for xy in cur.reshape(-1,2)],np.float32).reshape(cur.shape),np.ones((len(cur),1),np.uint8),None
        for shift in (None,(5.,0.)):
            body={0:(30.,30.),1:(35.,30.)} if shift else None
            trail=onion.trails([source],['p'],0,1,path,body=body)
            with patch('image_lens.frames.window',return_value=[seed,current]),patch.object(cv2,'goodFeaturesToTrack',return_value=points),\
                 patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=flow),patch.object(onion,'_image') as image:
                onion.bands('unused',0,1,[10,10,90,90],path,'unused.png',trail,view_shift=shift)
            warped=image.call_args.args[3];y,x=np.nonzero(warped>100)
            expected=43. if shift else 48.
            self.assertAlmostEqual(float(x.mean()),expected,delta=1.)
            self.assertAlmostEqual(trail[0]['xy'][-1][1],expected,delta=1e-6)
        crop=camera.warp(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),camera.transforms(path)[1],(50,50),offset=(20,20))
        y,x=np.nonzero(crop>100);self.assertAlmostEqual(float(x.mean()),28.,delta=1.)

    def test_body_raw_timing_and_actor_context_use_each_native_transform(self):
        from tests.test_body_motion import packet
        from tests.test_pose import track
        own={'shoulder-l':{n:(300.+n,300.) for n in range(21)},'shoulder-r':{n:(100.+n,300.) for n in range(21)},
             'wrist-l':{n:(320.+n+.4*n*n,400.) for n in range(21)}}
        path=camera.CameraPath({n:similarity(1+.01*n,.01*n) for n in range(21)},size=(1440,1200))
        transformed={pid:{n:camera.apply(path[n],xy) for n,xy in points.items()} for pid,points in own.items()}
        raw=packet(transformed);before=copy.deepcopy(raw)
        expected=body_motion.describe([packet(own)],dict(seed=0,end=20))
        got=body_motion.describe([raw],dict(seed=0,end=20),timing_packets=[raw],camera_path=path)
        self.assertEqual(got['limbs'][0]['strokes'][0]['ease'],expected['limbs'][0]['strokes'][0]['ease'])
        self.assertEqual(got['limbs'][0]['strokes'][0]['speeds'],expected['limbs'][0]['strokes'][0]['speeds'])
        self.assertEqual(raw,before)
        poses={}
        for n in range(21):
            if n<3:continue  # Entrant's own interval, not the window-wide camera net.
            person=track('a');person['keypoints']=[list(camera.apply(path[n],xy)) for xy in person['keypoints']]
            poses[n]=[person]
        native=copy.deepcopy(poses);people=body_motion.people(poses,dict(seed=0,end=20),25,path)
        self.assertEqual(people['coordinate_scope'],'camera_similarity_relative_2d')
        self.assertEqual(people['people'][0]['move']['widths'],0.)
        self.assertEqual(people['people'][0]['bbox'],poses[3][0]['bbox']);self.assertEqual(poses,native)

    def test_centre_zoom_rotation_rekeys_and_composes_without_translation(self):
        import cv2
        for first in (similarity(1.2,0.),similarity(1.,.2)):
            small=np.diag([.5,.5,1.])@first@np.diag([2.,2.,1.]);local=similarity(1.,.04,centre=(160.,120.))
            source=[np.full((480,640,3),n,np.uint8) for n in range(3)];calls=[]
            class ORB:
                def detectAndCompute(self,gray,mask):return [],np.array([[int(gray[0,0])]],np.uint8)
            def step(cv2,np,matcher,base,keys,desc,shape,*spread):
                a,b=int(base[1][0,0]),int(desc[0,0]);calls.append((a,b,bool(spread)))
                if b==1:return small[:2]
                return local[:2] if a==1 else None
            with patch('image_lens.frames.window',return_value=source),patch.object(cv2,'ORB_create',return_value=ORB()),patch.object(camera,'_step',side_effect=step):
                result=camera._path('unused',0,2)
            self.assertIn((1,2,True),calls)
            np.testing.assert_allclose(result[2],np.diag([2.,2.,1.])@local@small@np.diag([.5,.5,1.]),atol=1e-9)
            self.assertAlmostEqual(math.hypot(*camera.centre_path(result)[1][:2]),0.,delta=1e-9)
            corners=camera.view_corners(result,640,480)[2]
            self.assertNotAlmostEqual(corners[0][1],corners[1][1])
            self.assertIsNotNone(onion.moved(result))
