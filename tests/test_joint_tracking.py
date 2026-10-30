"""Model-fusion contracts; synthetic geometry, not a quality benchmark."""
from pathlib import Path
import copy
import tempfile
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_lens import joint_tracking as jt, paths


def person(identity='shot-0/1'):
    k = [[0., 0.] for _ in range(133)]; s = [0.] * 133
    for i, xy in {5:[100., 100.],6:[300., 100.],7:[120., 150.]}.items():
        k[i] = xy; s[i] = .8
    return dict(track_id=identity, keypoints=k, scores=s)


def packet(visible=True, xy=None, anchor=False):
    row = dict(point_id='elbow-l', xy=xy or [130., 150.], visible=visible, model_visible=visible,
               status='anchor_input_not_prediction' if anchor else 'tracked_unverified', in_frame=True,
               claim_scope='visible_surface_2d', motion_pair_usable=visible)
    return dict(schema='team-eye.observed-tracks/1', observation_scope='visible_surface_2d', limits='raw',
                job=dict(width=640, height=480), frames=[dict(frame=0 if anchor else 1, points=[row])])


class JointTracking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.before = paths.workspace_root(); paths.configure(self.root)
        self.raw = paths.write_json('raw.json', {}); self.pose = paths.write_json('pose.json', {})
    def tearDown(self):
        paths.configure(self.before); self.tmp.cleanup()
    def fused(self, source=None, poses=None):
        return jt.reconcile(source or packet(), {1:poses or [person()]}, 'shot-0/1', dict(seed=0,end=2), self.raw, self.pose)
    def test_correction_is_explicit_and_does_not_change_raw(self):
        raw = packet(); before = copy.deepcopy(raw)
        out = self.fused(raw); row = out['frames'][0]['points'][0]
        self.assertEqual(raw, before)
        self.assertEqual(row['xy'], [120.,150.]); self.assertEqual(row['coordinate_source'], 'pose')
        self.assertEqual(row['status'], 'pose_assisted_joint_unverified'); self.assertFalse(row['motion_pair_usable'])
        self.assertEqual(out['schema'], 'image-lens.joint-tracks/1')
        self.assertEqual(out['joint_tracking']['raw_tracks']['sha256'], paths.sha(self.raw))
    def test_agreement_recovery_does_not_invent_raw_model_visibility(self):
        row = self.fused(packet(False))['frames'][0]['points'][0]
        self.assertTrue(row['visible']); self.assertFalse(row['model_visible'])
        self.assertEqual(row['visibility_source'], 'pose_tracker_agreement')
        for raw in (packet(False,[500.,150.]), packet(False,[640.,150.])):
            self.assertFalse(self.fused(raw)['frames'][0]['points'][0]['visible'])
    def test_no_other_person_fallback_or_weak_joint_recovery(self):
        for t in (person('shot-0/2'), person()):
            if t['track_id']=='shot-0/1':t['scores'][7]=.49
            row = self.fused(packet(False),[t])['frames'][0]['points'][0]
            self.assertFalse(row['visible']); self.assertEqual(row['xy'],[130.,150.])
    def test_seed_other_surfaces_and_beyond_cut_stay_raw(self):
        for raw in (packet(anchor=True), packet()):
            if raw['frames'][0]['frame']!=0: raw['frames'][0]['points'][0]['point_id']='hair-front'
            self.assertEqual(self.fused(raw)['frames'],raw['frames'])
        raw=packet();raw['frames'][0]['frame']=3
        self.assertEqual(self.fused(raw)['frames'],raw['frames'])
    def test_raw_path_does_not_match_selected_scorer_glob(self):
        stem=Path('shot-0000-w0-arms')
        self.assertEqual(jt.track_path(stem,'arms').name,'shot-0000-w0-arms.raw-tracks.json')
        self.assertFalse(jt.track_path(stem,'arms').match('*.tracks.json'))
        self.assertEqual(jt.track_path(stem,'head'),stem.with_suffix('.tracks.json'))


class Timing(unittest.TestCase):
    def test_fused_positions_do_not_manufacture_raw_ease(self):
        from image_lens import body_motion
        frames=[]
        for n in range(1,31):
            points=[dict(point_id=p,xy=[x+float(n),y],visible=True) for p,x,y in
                    [('shoulder-l',100.,100.),('shoulder-r',300.,100.),('wrist-l',120.+n*n,150.)]]
            frames.append(dict(frame=n,points=points))
        fused=dict(frames=frames); raw=copy.deepcopy(fused)
        for f in raw['frames']:f['points'][-1]['visible']=False
        result=body_motion.describe([fused],dict(seed=0,end=30),timing_packets=[raw])
        self.assertTrue(result['limbs'])
        self.assertTrue(all(s['ease'] is None and s['speeds'] is None for l in result['limbs'] for s in l['strokes']))
        # The unchanged default still reports its continuous raw path timing.
        result=body_motion.describe([fused],dict(seed=0,end=30))
        self.assertTrue(any(s['speeds'] is not None for l in result['limbs'] for s in l['strokes']))


    def test_raw_face_timing_is_not_labelled_as_shoulder_relative_speed(self):
        from image_lens import body_motion
        frames=[]
        for n in range(1,31):
            pts=[dict(point_id=p,xy=[x+n,y],visible=True) for p,x,y in
                 [('shoulder-l',100.,100.),('shoulder-r',300.,100.),('wrist-l',120.+n*n,150.)]]
            pts += [dict(point_id='eye-l',xy=[100.+n+n*n,80.],visible=True),
                    dict(point_id='eye-r',xy=[140.+n+n*n,80.],visible=True)]
            frames.append(dict(frame=n,points=pts))
        fused=dict(frames=frames);raw=copy.deepcopy(fused)
        for f in raw['frames']:f['points'][0]['visible']=False
        result=body_motion.describe([fused],dict(seed=0,end=30),timing_packets=[raw])
        self.assertEqual(result['reference'],'shoulders');self.assertTrue(result['limbs'])
        self.assertTrue(all(s['ease'] is None and s['speeds'] is None for l in result['limbs'] for s in l['strokes']))


    def test_hidden_limb_does_not_rename_reference_or_suppress_other_timing(self):
        from image_lens import body_motion
        frames=[]
        for n in range(31):
            points=[dict(point_id=p,xy=[x+n,y],visible=vis or n==0) for p,x,y,vis in
                    [('shoulder-l',100.,100.,True),('shoulder-r',300.,100.,True),
                     ('wrist-l',120.,150.,False),('knee-r',130.+n*n,300.,True)]]
            frames.append(dict(frame=n,points=points))
        packet=dict(frames=frames)
        result=body_motion.describe([packet],dict(seed=0,end=30),timing_packets=[packet])
        self.assertTrue(result['unknown'])
        self.assertEqual(result['reference'],'shoulders')
        self.assertTrue(any(s['speeds'] is not None for l in result['limbs'] for s in l['strokes']))


class GeometrySources(unittest.TestCase):
    def packets(self, switch='wrist-l'):
        from tests.test_body_motion import packet
        paths={p:{n:xy for n in range(31)} for p,xy in
               [('shoulder-l',(300.,300.)),('shoulder-r',(100.,300.)),('elbow-l',(330.,350.)),('wrist-l',(360.,400.))]}
        raw=packet(paths); selected=copy.deepcopy(raw)
        for f in selected['frames']:
            for q in f['points']:
                q['coordinate_source']='pose' if q['point_id']==switch and 10<=f['frame']<=20 else 'point_tracker'
                if q['coordinate_source']=='pose':q['xy'][0]+=160
        return selected,raw
    def test_switch_spike_is_not_a_limb_or_centre_move(self):
        from image_lens import body_motion
        for pid in ('wrist-l','shoulder-l'):
            selected,raw=self.packets(pid);before=copy.deepcopy(selected)
            r=body_motion.describe([selected],dict(seed=0,end=30),timing_packets=[raw])
            self.assertEqual(r['limbs'],[])
            self.assertEqual(r['centre_shift']['distance_px'],0)
            self.assertEqual(selected,before)
            self.assertEqual(r['geometry']['source'],'raw_tracker_source_switch')
            self.assertIn(pid,r['geometry']['source_switches'])
            from image_lens import digest
            done=dict(window=dict(seed=0,end=30),body=r,body_geometry=r['geometry'],groups=[],motif='A')
            for first in (None,done):
                line=digest.window_line(done,30,first)
                self.assertIn('실제 동작은 모름',line)
                self.assertIn(pid,line)
    def test_source_switch_scope_and_missing_frame(self):
        from image_lens import body_motion
        selected,raw=self.packets()
        # One pose source after the seed, plus a raw seed and a post-cut row, is stable.
        for f in selected['frames']:
            for q in f['points']:
                q['coordinate_source']='pose' if 1<=f['frame']<=20 else 'point_tracker'
        p=body_motion.geometry_policy([selected],dict(seed=0,end=20),[raw])
        self.assertEqual(p['source'],'selected_estimates')
        # Keep source history across missing visibility; the first later tracker row changes source.
        for f in selected['frames']:
            for q in f['points']:
                if q['point_id']=='wrist-l' and f['frame']==10:q['visible']=False
                if q['point_id']=='wrist-l' and f['frame']>=11:q['coordinate_source']='point_tracker'
        p=body_motion.geometry_policy([selected],dict(seed=0,end=20),[raw])
        self.assertEqual(p['source_switches']['wrist-l'],[11])
    def test_constant_pose_source_keeps_selected_geometry(self):
        from image_lens import body_motion
        selected,raw=self.packets()
        for f in selected['frames']:
            for q in f['points']:
                q['coordinate_source']='pose'
                if q['point_id']=='wrist-l':q['xy'][0]=360+4*f['frame']
        r=body_motion.describe([selected],dict(seed=0,end=30),timing_packets=[raw])
        self.assertEqual(r['geometry']['source'],'selected_estimates')
        self.assertTrue(r['limbs']);self.assertEqual(r['limbs'][0]['strokes'][0]['direction'],'right')
    def test_missing_raw_reference_keeps_window_provenance_in_first_and_repeat(self):
        from image_lens import body_motion,digest
        selected,raw=self.packets()
        for f in raw['frames']:
            for q in f['points']:
                if q['point_id'].startswith('shoulder'):q['visible']=False
        w=dict(seed=0,end=30);policy=body_motion.geometry_policy([selected],w,[raw])
        self.assertIsNone(body_motion.describe([selected],w,timing_packets=[raw]))
        done=dict(window=w,body=None,body_geometry=policy,motif='A',groups=[dict(moves=[])])
        first=dict(window=w,body=None,groups=[],motif='A')
        for original in (None,first):
            line=digest.window_line(done,30,original)
            self.assertIn('출처 전환',line);self.assertIn('모름',line)
            self.assertNotIn('24px 이상 이동 없음',line)
    def test_lost_raw_wrist_is_not_whole_limb_stillness(self):
        from image_lens import body_motion,digest
        selected,raw=self.packets()
        for f in raw['frames']:
            for q in f['points']:
                if q['point_id']=='wrist-l' and f['frame']>0:q['visible']=False
        w=dict(seed=0,end=30);policy=body_motion.geometry_policy([selected],w,[raw])
        r=body_motion.describe([selected],w,timing_packets=[raw])
        self.assertEqual(r['limbs'],[]);self.assertTrue(policy['lost_observations'])
        self.assertIn('모름','\n'.join(body_motion.lines(r)))
        line=digest.window_line(dict(window=w,body=r,body_geometry=policy,groups=[]),30)
        self.assertIn('모름',line);self.assertNotIn('팔다리 12px 미만',line)
    def test_mixed_source_without_raw_geometry_is_refused(self):
        from image_lens import body_motion
        selected,_=self.packets()
        with self.assertRaisesRegex(ValueError,'raw geometry'):
            body_motion.describe([selected],dict(seed=0,end=30))


if __name__ == '__main__': unittest.main()
