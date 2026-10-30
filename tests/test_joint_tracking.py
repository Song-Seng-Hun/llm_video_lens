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


if __name__ == '__main__': unittest.main()
