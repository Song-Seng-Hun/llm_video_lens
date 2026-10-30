"""Shot-window selection for the one-command look; synthetic, no models."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import look


class Windows(unittest.TestCase):
    def frames(self, n0, n1):
        return {n: [dict(track_id='t')] for n in range(n0, n1+1)}

    def test_long_shot_splits_into_reseeded_windows(self):
        found, skipped = look.windows(dict(start_frame=0, end_frame=399), self.frames(0, 399), set())
        self.assertEqual(found, [dict(seed=0, end=159), dict(seed=160, end=319), dict(seed=320, end=399)])
        self.assertEqual(skipped, [])

    def test_seed_skips_flagged_and_empty_frames(self):
        poses = self.frames(10, 99)
        found, _ = look.windows(dict(start_frame=0, end_frame=99), poses, {10, 11, 12})
        self.assertEqual(found, [dict(seed=13, end=99)])

    def test_short_tail_and_no_pose_are_reported(self):
        found, skipped = look.windows(dict(start_frame=0, end_frame=165), self.frames(0, 165), set())
        self.assertEqual(found, [dict(seed=0, end=159)])
        self.assertIn('fewer than', skipped[0]['reason'])
        found, skipped = look.windows(dict(start_frame=0, end_frame=50), {}, set())
        self.assertEqual(found, []); self.assertEqual(skipped[0]['reason'], 'no unflagged pose track')



class Summary(unittest.TestCase):
    def test_no_observation_is_unknown_not_still(self):
        report = dict(features=[dict(point_id='hip-l', continuous_runs=[])])
        lines, _ = look._summary(report, None, 'legs', 'x', 'y')
        self.assertEqual(len(lines), 1); self.assertIn('모름', lines[0])

    def span(self, a, b):
        return dict(frames=[a, b], strokes=dict(units=[]))

    def test_group_cut_off_early_is_unknown_not_still(self):
        report = dict(features=[dict(point_id=p, continuous_runs=[self.span(340, 353)]) for p in ('elbow-l', 'elbow-r', 'wrist-l')])
        lines, summary = look._summary(report, dict(seed=339, end=383), 'arms', 'x', 'y')
        self.assertIn('큰 이동 없음', lines[0]); self.assertIn('f353 뒤로 점 3/3개 끊김', lines[1])
        self.assertEqual(summary['lost'], dict(count=3, total=3, after=353))

    def test_one_point_dropping_is_not_a_cut(self):
        report = dict(features=[dict(point_id='a', continuous_runs=[self.span(0, 99)]), dict(point_id='b', continuous_runs=[self.span(0, 99)]),
                                dict(point_id='c', continuous_runs=[self.span(0, 40)])])
        lines, summary = look._summary(report, dict(seed=0, end=99), 'arms', 'x', 'y')
        self.assertEqual(lines, []); self.assertIsNone(summary['lost'])


class Cut(unittest.TestCase):
    def packet(self, last_visible, seed=0, end=99):
        return dict(frames=[dict(frame=n, points=[dict(point_id=p, visible=n <= e) for p, e in last_visible.items()]) for n in range(seed, end+1)])

    def test_group_dropping_together_cuts_the_window(self):
        arms = self.packet(dict(a=40, b=41, c=99)); head = self.packet(dict(x=99, y=99))
        self.assertEqual(look.cut_frame([head, arms], dict(seed=0, end=99)), 41)

    def test_no_cut_when_points_last_to_the_end(self):
        self.assertIsNone(look.cut_frame([self.packet(dict(a=99, b=95, c=30))], dict(seed=0, end=99)))

    def test_reseed_starts_after_the_cut(self):
        poses = {n: [dict(track_id='t')] for n in range(0, 200)}
        window, _ = look.next_window(dict(start_frame=0, end_frame=199), 42, poses, {42, 43})
        self.assertEqual(window, dict(seed=44, end=199))


class Camera(unittest.TestCase):
    def test_strokes_that_only_follow_the_frame_are_set_aside(self):
        unit = lambda dx: dict(major=True, start_frame=1, end_frame=20, start_seconds=.04, end_seconds=.8, dx_px=dx, dy_px=0., direction='left', distance_px=abs(dx))
        report = dict(features=[dict(point_id='a', continuous_runs=[dict(frames=[1, 20], strokes=dict(units=[unit(-33.)]))]),
                                dict(point_id='b', continuous_runs=[dict(frames=[1, 20], strokes=dict(units=[unit(-90.)]))])])
        lines, summary = look._summary(report, dict(seed=0, end=20), 'move', 'x', 'y', {1: (0., 0., 1.), 20: (-33., 0., 1.)})
        self.assertEqual(summary['camera_only_moves'], 1); self.assertEqual(len(summary['major_moves']), 1)
        self.assertIn('b 화면 왼쪽 90px', lines[0])

    def test_small_camera_shift_does_not_remove_a_move(self):
        unit = dict(major=True, start_frame=1, end_frame=20, start_seconds=.04, end_seconds=.8, dx_px=0., dy_px=-25., direction='up', distance_px=25.)
        report = dict(features=[dict(point_id='a', continuous_runs=[dict(frames=[1, 20], strokes=dict(units=[unit]))])])
        _, summary = look._summary(report, dict(seed=0, end=20), 'move', 'x', 'y', {1: (0., 0., 1.), 20: (0., -12., 1.)})
        self.assertEqual(summary['camera_only_moves'], 0); self.assertEqual(len(summary['major_moves']), 1)

if __name__ == '__main__': unittest.main()


class Ahead(unittest.TestCase):
    def test_next_shot_is_seeded_once_and_its_failure_stays_in_place(self):
        import json, tempfile
        from image_lens import paths
        from tests.test_check import pan_video
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            paths.configure(d)
            video = Path(d)/'pan.avi'; pan_video(video, frames=60)
            segments = [dict(segment_id=f'shot-{k:04d}', start_frame=20*k, end_frame=20*k+19) for k in range(3)]
            person = [dict(track_id='t', bbox=[100, 50, 300, 350], score=.9, keypoints=[[150+j%20*5, 60+j//20*40] for j in range(133)], scores=[.9]*133)]
            calls = []
            def step(command, arguments):
                calls.append((command, *arguments[arguments.index('--frame')+1:][:1]) if command == 'seed' else (command,))
                out = Path(arguments[arguments.index('--out')+1]) if '--out' in arguments else None
                if command == 'scan':
                    out.write_text(json.dumps(dict(fps=10, source_frame_count=60, source_sha256='0', analysis_segments=segments)))
                elif command == 'pose':
                    out.write_text(json.dumps(dict(frames=[dict(frame=n, tracks=person) for n in range(60)], flags=dict(frames=[]))))
                elif command == 'seed' and arguments[arguments.index('--frame')+1] == 40:
                    raise ValueError('no seed')
                return dict(jobs=[])
            plain = look._step, look.fit
            look._step, look.fit = step, lambda source, root: (source, False)
            try:
                look._run(video, 'det', 'pose', 'ckpt', Path(d)/'look')
            finally:
                look._step, look.fit = plain
            self.assertEqual([c for c in calls if c[0] == 'seed'], [('seed', 0), ('seed', 20), ('seed', 40)])
            record = json.loads((Path(d)/'look'/'look.json').read_text())
            self.assertEqual([len(s['errors']) for s in record['shots']], [0, 0, 1])
            self.assertEqual(record['shots'][2]['errors'][0]['step'], 'seed')
            text = (Path(d)/'look'/'look-full.md').read_text()
            self.assertLess(text.index('## shot-0002'), text.index('점 배치 실패: no seed'))
