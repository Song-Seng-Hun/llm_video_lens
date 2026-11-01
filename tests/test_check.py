import json
import math
import tempfile
import unittest
from pathlib import Path
from image_lens import camera, check, digest, frames, paths

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
            # The view moved 780 px, more than its width. Check chaining both on the whole frame and on an admitted
            # background, so refusing a failed background fit cannot disable a genuinely coherent camera pan.
            for person in (None, [200, 100, 400, 250]):
                with self.subTest(person=person):
                    found = camera.path(video, 0, 39, person)
                    self.assertIn(39, found); self.assertAlmostEqual(camera.centre_path(found)[39][0], -780, delta=30); self.assertAlmostEqual(camera.centre_path(found)[39][1], 0, delta=10)

    def test_a_failed_rich_background_cannot_borrow_the_subject_motion(self):
        """A textured narrow UI strip and a different moving actor: feature count does not establish a global fit."""
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            rng = np.random.default_rng(12)
            strip = cv2.GaussianBlur(rng.integers(0, 255, (150, 680, 3), dtype=np.uint8), (3, 3), 0)
            actor = cv2.GaussianBlur(rng.integers(0, 255, (295, 480, 3), dtype=np.uint8), (3, 3), 0)
            video = Path(d)/'layers.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
            for n in range(10):
                image = np.full((480, 640, 3), 128, np.uint8)
                image[:150] = strip[:, 2*n:640+2*n]
                image[180:475, 80+6*n:560+6*n] = actor
                out.write(image)
            out.release()
            camera._LAST.clear()
            found = camera.path(video, 0, 9, [80, 180, 560, 475])
            self.assertEqual(camera.centre_path(found), {0: (0., 0., 1.)})

    def test_a_shorter_request_is_cut_from_the_kept_path(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            video = Path(d)/'pan.avi'; pan_video(video)
            camera.path(video, 0, 39)
            kept = camera.path(video, 0, 20); fresh = camera._path(video, 0, 20)
            self.assertEqual(kept, fresh); self.assertEqual(max(kept), 20)
            kept[20] = None  # callers get their own copy
            self.assertEqual(camera.path(video, 0, 20), fresh)


    def test_a_big_figure_over_a_still_background_leaves_the_camera_still(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            # A still room and a figure covering half the view, sliding 6 px a frame.
            rng = np.random.default_rng(5)
            room = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (3, 3), 0)
            figure = rng.integers(0, 255, (260, 300, 3), dtype=np.uint8)
            video = Path(d)/'figure.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 360))
            for k in range(20):
                image = room.copy(); image[50:310, 100+6*k:400+6*k] = figure; out.write(image)
            out.release()
            found = camera.path(video, 0, 19, [100, 50, 400, 310])
            self.assertEqual(len(found), 20); self.assertTrue(all(math.hypot(dx, dy) < 1 for dx, dy, _ in camera.centre_path(found).values()))
            self.assertLess(len(camera.path(video, 0, 19)), 20)  # on the whole frame the figure's moves leave it unknown


class Frames(unittest.TestCase):
    def test_a_kept_window_gives_the_frames_of_a_fresh_read(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            video = Path(d)/'pan.avi'; pan_video(video)
            frames.window(video, 0, 39)
            kept = frames.window(video, 5, 20); frames._LAST.clear(); fresh = frames.window(video, 5, 20)
            self.assertEqual(len(kept), 16); self.assertTrue(all((a == b).all() for a, b in zip(kept, fresh)))
            self.assertEqual(len(frames.checked(video, 0, 39, 640, 360, 'bad')), 40)
            with self.assertRaises(ValueError):
                frames.checked(video, 30, 45, 640, 360, 'past the end')

    def test_the_window_before_the_last_is_still_kept(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            video = Path(d)/'pan.avi'; pan_video(video)
            first = frames.window(video, 0, 9); frames.window(video, 20, 29); frames.window(video, 25, 25)
            self.assertIs(frames.window(video, 0, 9)[3], first[3])
            camera.path(video, 0, 9); camera.path(video, 20, 29)
            self.assertEqual([k[0][3] for k in camera._LAST], [20, 0])  # the window before the last is still kept
            self.assertEqual(camera.path(video, 0, 5), camera._path(video, 0, 5))

    def test_a_rewritten_source_is_read_again(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            video = Path(d)/'pan.avi'; pan_video(video)
            old = frames.window(video, 0, 9)
            pan_video(video, step=40); new = frames.window(video, 0, 9)
            self.assertFalse((old[5] == new[5]).all())


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

    def test_a_plain_wall_is_told_from_a_textured_room(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            # A pale wall with a textured figure in the middle: only the figure has points.
            rng = np.random.default_rng(7)
            wall = np.full((360, 640, 3), 210, np.uint8); wall[80:300, 260:380] = rng.integers(0, 255, (220, 120, 3), dtype=np.uint8)
            video = Path(d)/'wall.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 360))
            out.write(wall); out.release()
            self.assertTrue(camera.plain(video, 0, [260, 80, 380, 300]))
            room = Path(d)/'pan.avi'; pan_video(room, frames=2)
            self.assertFalse(camera.plain(room, 0, [260, 80, 380, 300]))

    def test_unknown_everywhere_gets_a_reason_in_the_header(self):
        def shot(cams):
            return dict(segment_id='shot-0000', frames=[0, 99], seconds=[0, 4.], skipped=[], errors=[],
                        windows=[window(20*k, 20*k+19, camera=c) for k, c in enumerate(cams)])
        plain = [dict(net=None, frames=1, known=.05, unknown=True, plain=True), dict(net=None, frames=1, known=.01, unknown=True, plain=True)]
        record = dict(source_video='v.mp4', shots=[shot(plain)])
        self.assertIn('카메라: 모든 구간에서 모름(배경이 거의 민무늬라 맞출 점이 없음, 구간마다 1–5% 프레임만 맞춤)', digest.clip_camera(record))
        text = digest.markdown(dict(record, check=[]), dict(fps=25, source_frame_count=100), 'look-full.md')
        self.assertEqual(text.splitlines()[3], digest.clip_camera(record))
        self.assertIn('카메라 모름(5%만 맞춤)', text)  # the windows keep their own
        # One window placed: no header line, and the plain reason needs every window plain.
        self.assertIsNone(digest.clip_camera(dict(source_video='v.mp4', shots=[shot([plain[0], dict(net=[0., 0.], frames=20, known=1.)])])))
        self.assertIn('배경이 프레임끼리 맞지 않음', digest.clip_camera(dict(source_video='v.mp4', shots=[shot([plain[0], dict(plain[1], plain=False)])])))

    # Anime music video: 17 of 41 windows had no trusted body and only move-a... points; a reader can name them with mark.
    def test_unnamed_windows_point_to_mark(self):
        from image_lens import look
        packets = [dict(frames=[dict(frame=n, points=[dict(point_id='move-a', xy=[100.4+n, 50.6], visible=True),
                                                      dict(point_id='move-b', xy=[9., 9.], visible=n != 30)]) for n in (30, 31)])]
        hint = dict(frame=30, end=59, points=look._seed_xy(packets, ['move-a', 'move-b'], 30))
        self.assertEqual(hint['points'], {'move-a': [130, 51]})  # move-b hidden at the seed
        line = look._mark_hint(hint, 'W/v.mp4', 'W/look/shots.json')
        self.assertIn('mark --video W/v.mp4 --shots W/look/shots.json --frame 30 --end 59', line); self.assertIn('move-a 130,51', line)
        shot = dict(segment_id='shot-0000', frames=[0, 99], seconds=[0, 4.], skipped=[], errors=[],
                    windows=[window(30, 59, mark_hint=hint), window(60, 79)])
        text = digest.markdown(dict(source_video='v.mp4', shots=[shot], check=[]), dict(fps=25, source_frame_count=100), 'look-full.md')
        self.assertIn('mark로 이름 붙인 점을 새로 찍어', text); self.assertIn('1구간', text)
        shot['windows'] = [window(60, 79)]
        self.assertNotIn('mark로', digest.markdown(dict(source_video='v.mp4', shots=[shot], check=[]), dict(fps=25, source_frame_count=100), 'look-full.md'))

    def test_old_look_json_reads_the_net_shift(self):
        self.assertEqual(digest.camera(dict(net=[0., 30.], frames=12), 10), '카메라 아래 30px')
        self.assertIsNone(digest.camera(dict(net=[2., 0.], frames=12), 10))


def person(name):
    return dict(name=name, move=dict(direction='left', widths=.1), seen=1)


def window(seed, end, **more):
    return dict(window=dict(seed=seed, end=end), groups=[dict(group='arms', moves=[], warnings=0)], **more)


class Spans(unittest.TestCase):
    def test_actor_skips_do_not_multiply_canonical_short_cuts(self):
        def shot(start):
            return dict(segment_id=f'shot-{start:04d}', frames=[start, start+4], windows=[],
                        skipped=[dict(frames=[start, start+4], reason='fewer than 12 frames after the seed',
                                      source_track_id=f'shot-{start:04d}/{n}', actor_label=f'A{n}') for n in range(1, 4)])
        record = dict(max_people=3, shots=[shot(0)])
        self.assertEqual(check.signs(record, 25), [])
        self.assertIn('짧은 컷 3개', check.signs(dict(record, max_people=1), 25)[0][3][0])
        record['shots'] += [shot(5), shot(10)]
        self.assertEqual(check.signs(record, 25)[0][3], ['짧은 컷 3개: 사진만, 움직임 모름'])

    def test_actor_full_frame_spans_deduplicate_before_cap_with_max_and_reasons(self):
        def done(a, b, n):
            return dict(window=dict(seed=a, end=b), skeleton=False, camera=dict(unknown=True),
                        source_track_id=f'shot-0000/{n}', actor_label=f'A{n}')
        record = dict(max_people=4, shots=[dict(skipped=[], windows=[done(0, 29, n) for n in range(1, 5)]+[done(30, 59, 1)])])
        found = check.spans(record, 25)
        self.assertEqual([span['frames'] for span in found], [[0, 29], [30, 59]])
        self.assertEqual([span['score'] for span in found], [2, 2])
        self.assertEqual([span['frames'] for span in check.spans(dict(record, max_people=1), 25)], [[0, 29]]*4)
        overlap = dict(window=dict(seed=0, end=199), people=dict(relations=[dict(overlap=True)]))
        record['shots'][0]['windows'] = [done(0, 99, 1), overlap, done(100, 129, 2)]
        found = check.spans(record, 25)
        self.assertEqual([span['frames'] for span in found], [[0, 99], [100, 129]])
        self.assertEqual(found[0]['score'], 3)
        self.assertEqual(set(found[0]['reasons']), {'사람 상자 겹침: 포즈 섞였을 수 있음', '골격 신뢰 낮음', '카메라 모름'})

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
