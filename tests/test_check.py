import json
import tempfile
import unittest
from pathlib import Path
from image_lens import camera, check, digest, paths

REPO = Path(__file__).resolve().parents[1]


def pan_video(path, frames=40, step=20):
    """A smooth random texture seen through a 640x360 view sliding right `step` px per frame."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(3)
    world = cv2.GaussianBlur(rng.integers(0, 255, (360, 640+frames*step, 3), dtype=np.uint8), (5, 5), 0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 360))
    for k in range(frames):
        out.write(np.ascontiguousarray(world[:, k*step:k*step+640]))
    out.release()


class Path_(unittest.TestCase):
    def test_a_pan_longer_than_the_view_stays_known(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            video = Path(d)/'pan.avi'; pan_video(video)
            found = camera.path(video, 0, 39)
            # The view moved 780 px, more than its 640 px width: the start frame shares nothing with the last one.
            self.assertIn(39, found); self.assertAlmostEqual(found[39][0], -780, delta=30); self.assertAlmostEqual(found[39][1], 0, delta=10)


class Describe(unittest.TestCase):
    def test_pan_then_hold_has_speeds(self):
        cam = {n: (-6.*min(n, 20), 0., 1.) for n in range(41)}
        found = camera.describe(cam, 0, 40, 24)
        self.assertEqual(found['net'], [-120., 0.]); self.assertEqual(found['known'], 1.)
        self.assertEqual(found['speeds'][0], 6.); self.assertEqual(found['speeds'][-1], 0.)
        self.assertNotIn('back', found)
        self.assertEqual(digest.camera(found, 10), '카메라 왼쪽 120px(속도 6→6→3→0→0 px/f)')

    def test_out_and_back_and_zoom(self):
        cam = {n: (8.*(20-abs(20-n)), 0., 1.+.01*n) for n in range(41)}
        found = camera.describe(cam, 0, 40, 24)
        # The farthest point is read from a running median of five, so the 160 px peak frame reads 152 px.
        self.assertEqual(found['back']['frame'], 19); self.assertEqual(found['zoom'], 1.4)
        text = digest.camera(found, 10)
        self.assertIn('오른쪽 152px까지 갔다 돌아옴(1.90s)', text); self.assertIn('줌 인 ×1.40', text)

    def test_few_matched_frames_are_unknown_not_still(self):
        found = camera.describe({0: (0., 0., 1.), 1: (0., 0., 1.)}, 0, 40, 24)
        self.assertTrue(found['unknown'])
        self.assertTrue(digest.camera(found, 10).startswith('카메라 모름(5%만 맞춤)'))

    def test_old_look_json_reads_the_net_shift(self):
        self.assertEqual(digest.camera(dict(net=[0., 30.], frames=12), 10), '카메라 아래 30px')
        self.assertIsNone(digest.camera(dict(net=[2., 0.], frames=12), 10))


def person(name):
    return dict(name=name, move=dict(direction='left', widths=.1), seen=1)


def window(seed, end, **more):
    return dict(window=dict(seed=seed, end=end), groups=[dict(group='arms', moves=[], warnings=0)], **more)


class Spans(unittest.TestCase):
    def record(self):
        return dict(source_video='v.mp4', shots=[
            dict(segment_id='shot-0000', frames=[0, 99], seconds=[0, 4.], skipped=[], errors=[],
                 windows=[window(0, 99, skeleton=False)]),
            dict(segment_id='shot-0001', frames=[100, 299], seconds=[4., 12.], skipped=[dict(frames=[260, 299], reason='no unflagged pose track')], errors=[],
                 windows=[window(100, 259, people=dict(people=[person('P1'), person('P2')], relations=[dict(pair=['P1', 'P2'], distance=None, touch=[], overlap=True)]),
                                 events=dict(hand_face=[], shoulders=[], width=[], hands_seen={}, turns=[[230, 233], [240, 244]], lost_frames=0))]),
        ] + [dict(segment_id=f'shot-{k:04d}', frames=[300+3*k, 302+3*k], seconds=[0, 0], windows=[], errors=[],
                  skipped=[dict(frames=[300+3*k, 302+3*k], reason='fewer than 12 frames after the seed')]) for k in range(2, 6)])

    def test_ranked_clipped_and_in_time_order(self):
        found = check.spans(self.record(), 25)
        self.assertEqual([c['frames'] for c in found], [[141, 240], [260, 299], [306, 317]])
        self.assertIn('사람 상자 겹침: 포즈 섞였을 수 있음', found[0]['reasons'])
        self.assertEqual(found[2]['reasons'], ['짧은 컷 4개: 사진만, 움직임 모름'])
        # The low skeleton alone (score 1) stays out; a 6.4 s window is cut to 4 s around its turn candidates.
        self.assertTrue(all(c['frames'][1]-c['frames'][0]+1 <= 100 for c in found))
        text = digest.markdown(dict(self.record(), check=found), dict(fps=25, source_frame_count=320), 'look-full.md')
        self.assertIn('## 먼저 확인할 곳', text); self.assertIn('- 10.40–12.00s: 포즈 없어 글에 없음', text)

    def test_at_collects_one_span(self):
        from tests.test_zoom import Run
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            paths.configure(d)
            look = Path(d)/'look'; look.mkdir()
            video = look/'v.avi'; Run._video(None, video, frames=320)
            record = dict(self.record(), source_video=str(video))
            (look/'look.json').write_text(json.dumps(record), encoding='utf-8')
            (look/'shots.json').write_text(json.dumps(dict(fps=10, source_frame_count=320)), encoding='utf-8')
            result = check.at(look, '11-14', Path(d)/'at')
            text = Path(result['output']).read_text(encoding='utf-8')
            self.assertIn('# 11.00–14.10s (f110..140)', text); self.assertIn('shot-0001', text); self.assertNotIn('shot-0000', text)
            self.assertIn('상자가 많이 겹침', text); self.assertTrue((Path(d)/'at'/'at-frames-1.jpg').is_file())
            self.assertEqual(len(result['frames']), 12)
            with self.assertRaises(ValueError):
                check.at(look, '40-42', Path(d)/'late')
