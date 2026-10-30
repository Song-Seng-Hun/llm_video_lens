"""Pose seed geometry, pose trouble flags and their report wording; synthetic, not pose-quality evidence."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import motion_report, motion_view, pose


def track(track_id='shot-0000/1', box=(100., 100., 500., 900.), dx=0., score=.9):
    k = [[0., 0.]]*133; s = [0.]*133
    joints = dict(nose=(300, 260), eye_l=(330, 220), eye_r=(270, 220), ear_l=(360, 230), ear_r=(240, 230),
                  shoulder_l=(380, 380), shoulder_r=(220, 380), elbow_l=(420, 500), elbow_r=(180, 500),
                  wrist_l=(440, 600), wrist_r=(160, 600), hip_l=(340, 650), hip_r=(260, 650),
                  knee_l=(340, 790), knee_r=(260, 790), ankle_l=(340, 1075), ankle_r=(260, 1075))
    for name, (x, y) in joints.items():
        k[pose.BODY[name]] = [x+dx, float(y)]; s[pose.BODY[name]] = score
    return dict(track_id=track_id, bbox=[box[0]+dx, box[1], box[2]+dx, box[3]], keypoints=k, scores=s)



class Split(unittest.TestCase):
    """The pose call is faked: each crop gets a head at the given nose x and eye distance."""
    def run_split(self, heads, box=(0., 0., 1000., 600.)):
        import numpy as np
        def fake(image, bboxes):
            k = np.zeros((len(bboxes), 133, 2), np.float32); s = np.full((len(bboxes), 133), .9, np.float32)
            for i, (x, eyes) in enumerate(heads):
                k[i, pose.BODY['nose']] = (x, 200); k[i, pose.BODY['eye_l']] = (x+eyes/2, 180); k[i, pose.BODY['eye_r']] = (x-eyes/2, 180)
            return k, s
        boxes = np.array([box], np.float32); conf = np.array([.85], np.float32)
        return pose._split(None, boxes, conf, np.zeros((1, 133, 2), np.float32), np.zeros((1, 133), np.float32), fake)

    def test_two_far_heads_split_the_box(self):
        boxes, conf, _, _, split = self.run_split([(200, 100), (800, 100)])
        self.assertEqual(split, 1); self.assertEqual(boxes.tolist(), [[0, 0, 600, 600], [400, 0, 1000, 600]])

    def test_same_head_tall_box_and_eye_sized_heads_stay(self):
        self.assertEqual(self.run_split([(500, 100), (510, 100)])[4], 0)
        self.assertEqual(self.run_split([(200, 100), (800, 100)], box=(0., 0., 500., 900.))[4], 0)
        self.assertEqual(self.run_split([(200, 40), (800, 40)])[4], 0)  # a close-up's two eyes, not two faces

class Seeds(unittest.TestCase):
    def test_hair_sits_above_eyes_and_beside_them(self):
        seeds = pose.seed_points(track(), 1440, 1080)
        front, top = seeds['hair_front']['xy'], seeds['head_top']['xy']
        self.assertAlmostEqual(front[0], 300., delta=.01); self.assertLess(front[1], 220.)
        self.assertLess(top[1], front[1])
        self.assertGreater(seeds['hair_side_l']['xy'][0], 330.); self.assertLess(seeds['hair_side_r']['xy'][0], 270.)

    def test_head_frame_follows_a_tilted_head(self):
        t = track()
        t['keypoints'][pose.BODY['eye_l']] = [300., 190.]; t['keypoints'][pose.BODY['eye_r']] = [300., 250.]
        t['keypoints'][pose.BODY['nose']] = [340., 220.]
        front = pose.seed_points(t, 1440, 1080)['hair_front']['xy']
        self.assertLess(front[0], 300.)  # up points away from the nose, here to the left

    def test_edge_clamped_and_low_score_joints_are_dropped(self):
        t = track(); t['scores'][pose.BODY['elbow_r']] = .1
        seeds = pose.seed_points(t, 1440, 1080)
        self.assertNotIn('ankle_l', seeds); self.assertNotIn('elbow_r', seeds); self.assertIn('knee_l', seeds)

    def test_no_hair_without_both_eyes_and_nose(self):
        t = track(); t['scores'][pose.BODY['eye_l']] = 0.
        seeds = pose.seed_points(t, 1440, 1080)
        self.assertFalse(any(n.startswith(('hair', 'head')) for n in seeds))
        self.assertIn('nose', seeds)

    def test_ears_only_frame_writes_nothing(self):
        import tempfile, json
        from image_lens import paths
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as root:
            paths.configure(root); root = Path(root)
            video = root/'v.bin'; video.write_bytes(b'x')
            scan = dict(analysis_segments=[dict(start_frame=0, end_frame=9)])
            paths.write_json('shots.json', scan)
            t = track(); t['scores'] = [0.]*133; t['scores'][pose.BODY['ear_l']] = .9
            packet = dict(schema='image-lens.pose/1', source_video=str(video), source_sha256=paths.sha(video), width=1440, height=1080,
                          fps=50., canonical_shots=dict(path=str(root/'shots.json'), sha256=paths.sha(root/'shots.json')),
                          flags=dict(frames=[]), frames=[dict(frame=0, tracks=[t])])
            paths.write_json('pose.json', packet)
            with self.assertRaisesRegex(ValueError, 'no usable pose seeds'):
                pose.seed(str(root/'pose.json'), 0, 'j', 'crops')
            self.assertFalse((root/'crops').exists())

    def test_groups_fit_report_and_observation_caps(self):
        from image_lens import onion
        # move tracks up to 12 candidates (observation cap) but reports at most onion.GAPS['points'] (report cap 8)
        self.assertTrue(all(len(names) <= 8 for group, names in pose.GROUPS.items() if group != 'move'))
        self.assertLessEqual(len(pose.GROUPS['move']), 12); self.assertLessEqual(onion.GAPS['points'], 8)


class Flags(unittest.TestCase):
    def frames(self, rows):
        return [dict(frame=n, tracks=tracks) for n, tracks in rows]

    def test_overlap_and_jump(self):
        rows = [(0, [track()]), (1, [track(), track('shot-0000/2', dx=150.)]), (2, [track(dx=300.)]), (3, [track(dx=302.)])]
        result = pose.flags(self.frames(rows))['frames']
        self.assertEqual([r['frame'] for r in result], [1, 2])
        self.assertTrue(result[0]['overlapping_people']); self.assertIsNone(result[0]['skeleton_jump'])
        self.assertFalse(result[1]['overlapping_people']); self.assertEqual(result[1]['skeleton_jump']['track_id'], 'shot-0000/1')

    def test_jump_needs_consecutive_frames_of_one_track(self):
        rows = [(0, [track()]), (5, [track(dx=300.)]), (6, [track('shot-0000/9', dx=600.)])]
        self.assertEqual(pose.flags(self.frames(rows))['frames'], [])

    def test_parts_cut_at_a_shot_flag_as_the_whole(self):
        # look's pose parts end at cuts: track IDs carry the shot, so no jump is read across the cut.
        rows = [(0, [track()]), (1, [track(dx=300.)]), (2, [track('shot-0001/1', dx=900.)]), (3, [track('shot-0001/1', dx=1200.), track('shot-0001/2', dx=1250.)])]
        whole = pose.flags(self.frames(rows))['frames']
        self.assertEqual(pose.flags(self.frames(rows[:2]))['frames']+pose.flags(self.frames(rows[2:]))['frames'], whole)
        self.assertEqual([r['frame'] for r in whole], [1, 3])

    def test_overlay_ranges_and_nearby_candidates(self):
        flags = dict(path='p', sha256='0'*64, native_range=[0, 99], params=pose.FLAGS,
                     frames=[dict(frame=n, people=3, overlapping_people=True, skeleton_jump=None) for n in (40, 41, 42)])
        candidates = [dict(frame=44), dict(frame=60)]
        overlay = motion_view.pose_overlay(flags, 30, 70, candidates)
        self.assertEqual(overlay['ranges'], [dict(frames=[40, 42], overlapping_people=True, skeleton_jump=False, max_people=3)])
        self.assertEqual(overlay['candidates_near_pose_flags'], [44])
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            motion_view.pose_overlay(flags, 50, 120, candidates)


class Wording(unittest.TestCase):
    def report(self, units, ranges):
        feature = dict(point_id='hair-front', continuous_runs=[dict(strokes=dict(units=units))])
        view = dict(path='t.png', appearance_change=dict(candidates=[dict(after_frame=63, frame=64, mean_abs_diff=50.)],
                    threshold=16., omitted_candidate_count=0),
                    pose_flags=dict(ranges=ranges, candidates_near_pose_flags=[64]))
        return dict(features=[feature, dict(feature, point_id='nose')]), view

    def test_one_grouped_warning_per_cause(self):
        unit = dict(major=True, start_frame=62, end_frame=74, direction='left', distance_px=244.)
        report, view = self.report([unit], [dict(frames=[32, 63], overlapping_people=True, skeleton_jump=False, max_people=3)])
        text = motion_report.trajectory_lines(report, view)
        warnings = [l for l in text if l.startswith('- 주의')]
        self.assertEqual(len(warnings), 1)
        self.assertIn('hair-front f62..74', warnings[0]); self.assertIn('nose f62..74', warnings[0])
        self.assertIn('f64', warnings[0]); self.assertIn('포즈', warnings[0])
        self.assertTrue(any('포즈 겹침/튐 구간 근처' in l for l in text))

    def test_small_strokes_are_not_major(self):
        fs = list(range(30)); ts = [n/50 for n in fs]; xy = [[100.+n*.6, 100.] for n in fs]
        units = motion_report.strokes(fs, ts, xy)['units']
        self.assertFalse(any(u['major'] for u in units))


class Together(unittest.TestCase):
    def feature(self, pid, start, end, direction, distance):
        unit = dict(major=True, start_frame=start, end_frame=end, start_seconds=start/50, end_seconds=end/50,
                    direction=direction, distance_px=distance)
        return dict(point_id=pid, continuous_runs=[dict(strokes=dict(units=[unit]))])

    def test_same_move_groups_and_different_moves_do_not(self):
        report = dict(features=[self.feature('nose', 262, 274, 'left', 244.), self.feature('eye-l', 263, 274, 'left', 240.),
                                self.feature('wrist-l', 262, 274, 'right', 240.), self.feature('hair', 262, 274, 'left', 90.)])
        groups = motion_report.together(report)
        self.assertEqual([g['points'] for g in groups], [['nose', 'eye-l']])


    def test_group_members_all_overlap(self):
        report = dict(features=[self.feature('a', 0, 100, 'right', 100.), self.feature('b', 0, 50, 'right', 100.),
                                self.feature('c', 50, 100, 'right', 100.)])
        groups = motion_report.together(report)
        self.assertTrue(all(not ({'b', 'c'} <= set(g['points'])) for g in groups))


class CoremlCache(unittest.TestCase):
    def test_caches_of_unpinned_models_are_removed(self):
        import tempfile
        from unittest import mock
        from image_lens import paths
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, '_ROOT', Path(tmp)):
            cache = Path(tmp)/'.cache'/'coreml'
            (cache/'old-model'/'compiled').mkdir(parents=True); (cache/pose.MODELS['detector'][1]).mkdir()
            self.assertEqual(pose._coreml_cache('pose'), str(cache/pose.MODELS['pose'][1]))
            self.assertEqual(sorted(p.name for p in cache.iterdir()), sorted(d for _, d in pose.MODELS.values()))

if __name__ == '__main__': unittest.main()
