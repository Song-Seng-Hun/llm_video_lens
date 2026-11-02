"""Native transition safety regressions; fake heads and authored tiny frames, never NN."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from image_lens import gradual_transitions as gradual,shot_boundaries as shots,autoshot,paths


class Intervals(unittest.TestCase):
    def setUp(self):
        self.previous=paths.workspace_root();self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name).resolve();paths.configure(self.root)
        self.video=self.root/'source.bin';self.video.write_bytes(b'authored structural source')
        self.job=dict(source_video=str(self.video),source_sha256=paths.sha(self.video),width=64,height=48,fps=24.,start_frame=0,end_frame=19)

    def tearDown(self):
        paths.configure(self.previous);self.tmp.cleanup()

    def scan(self,ranges=((4,6),),cuts=(10,),n=20):
        many=[0.]*n
        for a,b in ranges:many[a:b+1]=[1.]*(b-a+1)
        single=[0.]*n
        for c in cuts:single[c-1]=1.
        temporal=dict(recipe=autoshot.recipe(),source_sha256=self.job['source_sha256'],frame_count=n,single=single,many=many,
                      positive_runs=autoshot.runs(single),model_loaded=True,forward_calls=(n+49)//50,adaptive_cuts=list(cuts))
        detection=dict(recipe=gradual.recipe(),source_sha256=self.job['source_sha256'],frame_count=n,many_runs=gradual.many_runs(many),cheap_intervals=[])
        union=gradual.validate(detection,many,n,24.,self.job['source_sha256'])
        scan=shots.assemble(dict(self.job,end_frame=n-1),[n/24 for n in range(n)],list(cuts),mode='thorough',transitions=True,unsafe=union)
        scan.update(scan_scope='full_source',source_frame_count=n,temporal=temporal,transition_detection=detection)
        return shots.validate(scan)

    def test_inclusive_union_safe_partition_and_actual_hardcuts(self):
        """Touching intervals exclude each endpoint without erasing or inventing hardcuts."""
        self.assertEqual(gradual.union([[5,8],[0,2],[3,4],[11,11]]),[[0,8],[11,11]])
        scan=self.scan(((0,2),(6,12),(17,19)))
        self.assertEqual([e['after_frame'] for e in scan['events']],[10])
        self.assertEqual([(s['start_frame'],s['end_frame']) for s in scan['analysis_segments']],[(3,5),(13,16)])
        self.assertEqual(scan['frame_accounting'],dict(safe=7,excluded_transition=13,total=20))
        self.assertEqual(scan['transition_intervals'][1]['last_frame_seconds'],12/24)
        self.assertEqual(self.scan(((0,19),))['analysis_segments'],[])
        broken=copy.deepcopy(scan);broken['analysis_segments'][0]['end_frame']=6
        with self.assertRaisesRegex(ValueError,'partition'):shots.validate(broken)
        old=copy.deepcopy(scan);old['schema']='team-eye.shots/1'
        with self.assertRaises(ValueError):shots.validate(old)

    def test_transition_summary_is_bounded_without_hiding_exclusion_totals(self):
        """Many candidate intervals must not turn a bounded LLM brief into an unbounded dump."""
        from image_lens import brief
        scan=self.scan(tuple((n,n+2) for n in range(2,60,5)),cuts=(),n=60)
        paths.write_json('many-shots.json',scan);result=brief.build('many-shots.json');coverage=result['coverage']
        self.assertEqual(len(coverage['transition_intervals']),8)
        self.assertEqual(coverage['total_transition_intervals'],12)
        self.assertEqual(coverage['omitted_transition_intervals'],4)
        self.assertEqual(coverage['frame_accounting'],dict(safe=24,excluded_transition=36,total=60))
        self.assertEqual(coverage['transition_intervals_locator'],str(self.root/'many-shots.json')+'#/transition_intervals')
        text=brief.markdown(result)
        self.assertIn('excluded frames: 36/60',text)
        self.assertEqual(text.count('- Candidate f'),8)
        self.assertIn('Other transition candidates: 4',text)

    def test_frozen_primary_math_and_float32_native_runs(self):
        """JSON threshold equality, suffix runs and stable-anchor exclusion cannot drift."""
        p=[float(np.float32(.3))]*3+[1.]*3+[0.]+[1.]*3
        self.assertEqual(gradual.many_runs(json.loads(json.dumps(p))),[[3,5],[7,9]])
        levels=[0]*3+[30,60,90,120,150,180]+[210]*3
        frames=np.broadcast_to(np.array(levels,np.uint8)[:,None,None,None],(12,2,2,3)).copy()
        before=frames.copy();found=gradual.detect(frames,24.)
        self.assertEqual([r['frames']for r in found],[[3,8]])
        np.testing.assert_array_equal(frames,before)
        data=gradual.evidence(frames,24.,[0.]*12,self.job['source_sha256'])
        self.assertEqual(gradual.validate(data,[0.]*12,12,24.,self.job['source_sha256']),[[3,8]])
        data['cheap_intervals'][0]['evidence']['max_normalized_residual']=float('nan')
        with self.assertRaises(ValueError):gradual.validate(data,[0.]*12,12,24.,self.job['source_sha256'])

    def test_policy_requires_thorough_and_static_validation_has_no_optional_imports(self):
        """Invalid opt-in must fail before model work; static scans need no array/model runtime."""
        with patch.object(autoshot,'Stream') as model:
            with self.assertRaisesRegex(ValueError,'requires --cuts thorough'):shots.scan_video(self.video,transitions=True)
            model.assert_not_called()
        scan=self.scan();original=__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','numpy','einops'):raise AssertionError('static import '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):shots.validate(scan)
        off=shots.assemble(self.job,[n/24 for n in range(20)],[10]);shots.validate(off)
        self.assertFalse(off['parameters']['transitions']);self.assertEqual(off['transition_intervals'],[])
        self.assertEqual([(s['start_frame'],s['end_frame'])for s in off['analysis_segments']],[(0,9),(10,19)])

    def test_original_generation_not_safe_crop_controls_all_admission(self):
        """An unsafe seed/end or cached generation cannot regain motion claims through a safe crop."""
        from image_lens import jobs,analyze_motion,brief,part_regions,cli
        scan=self.scan(cuts=());paths.write_json('shots.json',scan)
        job=dict(self.job,name='a',seed_frame=0,points=[dict(point_id='p',xy=[10.,20.])])
        query=dict(schema='team-eye.queries/1',jobs=[job]);paths.write_json('queries.json',query)
        with self.assertRaisesRegex(ValueError,'transition candidate'):jobs.prepare('queries.json','shots.json','bad.freeze.json','a')
        self.assertFalse((self.root/'bad.freeze.json').exists())
        for start,end in ((0,4),(6,9),(4,6)):
            with self.assertRaisesRegex(ValueError,'transition candidate'):jobs.check_job(dict(query,jobs=[dict(job,start_frame=start,end_frame=end,seed_frame=start)]),'a',scan)
        packet=dict(schema='team-eye.observed-tracks/1',observation_scope='visible_surface_2d',anatomical_claims=False,
                    job=job,shot_check=scan,frames=[dict(frame=n,pts_seconds=n/24,points=[])for n in range(20)])
        paths.write_json('tracks.json',packet)
        self.assertEqual(analyze_motion.collect_observations(packet,'p',0,3)['label'],'unknown')
        with self.assertRaisesRegex(ValueError,'requires fresh'):part_regions.build_report(packet,{},0,3,lambda *_:None)
        cli.dispatch('analyze',['--tracks','tracks.json','--point','p','--start','0','--end','3','--out','blocked.json'])
        result=json.loads((self.root/'blocked.json').read_text())
        self.assertEqual(result['state'],'requires_shot_local_retracking');self.assertFalse(result['motion_inference'])
        # A forged apparently safe cached timing report is checked against original tracks first.
        report=dict(source_video=str(self.video),source_video_sha256=job['source_sha256'],analyzer_sha256=paths.sha(Path(analyze_motion.__file__)),
                    source_tracks=str(self.root/'tracks.json'),source_tracks_sha256=paths.sha(self.root/'tracks.json'),requested_frames=[0,3])
        paths.write_json('cached.json',report)
        with self.assertRaisesRegex(ValueError,'transition candidate'):brief.admitted_report('cached.json',scan)
        safe=dict(query,jobs=[dict(job,end_frame=3)]);paths.write_json('safe.json',safe)
        jobs.prepare('safe.json','shots.json','safe.freeze.json','a')
        frozen,_=paths.load('safe.freeze.json');self.assertTrue(frozen['transitions'])
        with self.assertRaisesRegex(ValueError,'policy mismatch'):jobs.check_cut_policy(dict(frozen,transitions=False),scan)
        catalog=brief.build('shots.json');self.assertFalse(catalog['shots'][1]['cut_candidate_before'])
        self.assertTrue(catalog['shots'][1]['transition_candidate_before'])
        # Transition-only evidence selects actual interval pixels without indexing a missing hardcut.
        with patch.object(shots,'selected_frames',return_value=[(4,4/24,np.zeros((4,4,3),np.uint8)),(6,6/24,np.zeros((4,4,3),np.uint8))]):
            contact=shots.render(scan,self.root/'evidence',transition_index=0)
        self.assertEqual(contact['selected_transition_index'],0);self.assertIsNone(contact['selected_event_index'])

    def test_pose_prefetch_exclusions_dense_parts_and_fresh_tracker_identity(self):
        """Read-ahead must decode unsafe frames without detection/pose, retaining dense parts and fresh IDs."""
        import cv2,sys,types
        from unittest.mock import Mock
        from image_lens import pose,person_models
        for ranges in (((4,6),(17,19)),((0,19),)):
            with self.subTest(ranges=ranges):
                scan=self.scan(ranges,cuts=());scan_path=paths.write_json('pose-shots-'+str(len(ranges))+'.json',scan)
                detected=[];estimated=[];trackers=[];parts=[]
                class Detector:
                    def __call__(self,image):
                        detected.append(int(image[0,0,0]));return np.array([[1,1,2,6]],np.float32),np.array([.9],np.float32)
                class Estimator:
                    session=None
                    def preprocess(self,image,bbox):
                        estimated.append(int(image[0,0,0]));return np.zeros((2,2,3)),None,None
                    def postprocess(self,outputs,center,scale):return np.zeros((1,133,2),np.float32),np.zeros((1,133),np.float32)
                class Detections:
                    def __init__(self,**kwargs):self.__dict__.update(kwargs)
                    def __len__(self):return len(self.xyxy)
                class Tracker:
                    def __init__(self,**kwargs):trackers.append(self)
                    def update_with_detections(self,d):d.tracker_id=[1]*len(d);return d
                class Capture:
                    index=0
                    def isOpened(self):return True
                    def set(self,prop,value):self.index=int(value)
                    def read(self):
                        image=np.full((48,64,3),self.index,np.uint8);self.index+=1;return True,image
                    def get(self,prop):return self.index if prop==cv2.CAP_PROP_POS_FRAMES else (self.index-1)*1000/24
                    def release(self):pass
                session=types.SimpleNamespace(get_providers=lambda:['CoreMLExecutionProvider','CPUExecutionProvider'],get_inputs=lambda:[types.SimpleNamespace(name='x')],
                                         run=lambda names,feed:[np.zeros((len(feed['x']),1)),np.zeros((len(feed['x']),1))])
                modules={'onnxruntime':types.SimpleNamespace(__version__='fake',InferenceSession=lambda *a,**k:session),
                         'rtmlib':types.SimpleNamespace(RTMPose=lambda *a,**k:Estimator()),
                         'supervision':types.SimpleNamespace(__version__='fake',ByteTrack=Tracker,Detections=Detections)}
                with (patch.dict(sys.modules,modules),patch.object(person_models,'resolve',return_value=dict(kind='humanart',path='fake')),
                     patch.object(pose,'_model',return_value=Path('fake')),patch.object(pose,'person_detector',return_value=Detector()),
                     patch.object(pose,'_coreml_cache',return_value='fake'),patch.object(cv2,'VideoCapture',return_value=Capture()),patch.object(pose,'PART_FRAMES',2)):
                    packet=pose.run(scan_path,'fake','fake',part=parts.append)
                safe=[n for s in scan['analysis_segments'] for n in range(s['start_frame'],s['end_frame']+1)]
                self.assertEqual(detected,safe);self.assertEqual(estimated,safe)
                self.assertEqual(len(trackers),len(scan['analysis_segments']))
                self.assertEqual([r['frame']for r in packet['frames']],list(range(20)))
                self.assertEqual([r['pts_seconds']for r in packet['frames']],scan['native_pts_seconds'])
                for row in packet['frames']:
                    if row['frame'] not in safe:self.assertEqual(row['status'],'excluded_transition');self.assertEqual(row['tracks'],[])
                self.assertEqual([r['frame']for part in parts for r in part['frames']],list(range(20)))
                if safe:
                    self.assertEqual(packet['frames'][0]['tracks'][0]['track_id'],'shot-0000/1')
                    self.assertEqual(packet['frames'][7]['tracks'][0]['track_id'],'shot-0001/1')

    def test_public_policy_coverage_and_mark_box_excluded_seed_barrier(self):
        """Opt-in must reach the real CLI, cap counts safe segments, and corner fallback cannot admit exclusions."""
        from image_lens import cli,look,mark,box,digest
        scan=self.scan();paths.write_json('shots.json',scan)
        with patch.object(cli,'models') as models,patch.object(look,'run') as run:
            with self.assertRaisesRegex(ValueError,'requires --cuts thorough'):cli.dispatch('look',['--video','source.bin','--out','x','--transitions'])
            models.assert_not_called();run.assert_not_called()
        with patch.object(cli,'models',side_effect=lambda a,*_:a),patch.object(look,'run',return_value={}) as run:
            cli.dispatch('look',['--video','source.bin','--out','x','--cuts','thorough','--transitions','--scene-box','0,0,32,24'])
            self.assertTrue(run.call_args.kwargs['transitions']);self.assertEqual(run.call_args.kwargs['scene_box'],[0,0,32,24])
        coverage=look.transition_coverage(scan,max_shots=1)
        self.assertEqual(coverage['excluded_frames']+coverage['selected_safe_frames']+coverage['cap_omitted_safe_frames'],20)
        self.assertEqual(coverage['selected_safe_segments'],1);self.assertEqual(coverage['cap_omitted_safe_segments'],2)
        self.assertIn('이 구간의 움직임은 미분석',digest.transition_line(coverage))
        self.assertIn('3/20프레임 제외 / f4..6 (0.17..0.25s)',digest.transition_line(coverage))
        for action in (lambda:mark.run(self.video,self.root/'shots.json',4,['p=1,1'],'fake',self.root/'m'),
                       lambda:box.run(self.video,self.root,4,'0,0,32,32','fake',self.root/'b')):
            with patch.object(mark,'_frame') as frame,patch.object(box,'corners') as corners:
                with self.assertRaisesRegex(ValueError,'excluded transition'):action()
                frame.assert_not_called();corners.assert_not_called()

    def test_all_excluded_actual_look_finishes_without_tracking_windows(self):
        """A complete exclusion must publish honest zero-motion coverage instead of waiting on nonexistent safe parts."""
        import io,types
        from image_lens import look,onion,zoom,check
        scan=self.scan(((0,19),),cuts=())
        def step(command,args):
            self.assertEqual(command,'scan');self.assertIn('--transitions',args)
            paths.write_json(args[args.index('--out')+1],scan)
        def child(args):
            out=Path(args[args.index('--out')+1])
            paths.write_json(out,dict(native_range=[0,19],flags=dict(frames=[]),frames=[dict(frame=n,tracks=[])for n in range(20)]))
            return types.SimpleNamespace(stdout=io.StringIO(''),wait=lambda:0)
        with patch.object(look,'fit',return_value=(self.video,None)),patch.object(look,'_step',side_effect=step),patch.object(look,'_pose_child',side_effect=child),\
             patch.object(onion,'sheets',return_value=[]),patch.object(zoom,'overview',return_value=[]),patch.object(check,'sheets'):
            result=look._run(self.video,'fake','fake','fake',self.root/'look',cuts='thorough',transitions=True)
        record=json.loads((self.root/'look/look.json').read_text());self.assertEqual(result['windows'],0)
        self.assertEqual(record['transition_coverage']['excluded_frames'],20)
        self.assertEqual(record['transition_coverage']['selected_safe_frames'],0)
        self.assertIn('이 구간의 움직임은 미분석',(self.root/'look/look.md').read_text())
