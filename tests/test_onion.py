from pathlib import Path
import unittest
from image_lens import onion


class Lines(unittest.TestCase):
    def test_bands_read_as_still_moving_or_unknown(self):
        result = [dict(band='맨 위', y=[0, 10], dx=20., dy=0., features=30), dict(band='위', y=[10, 20], dx=3., dy=-2., features=30),
                  dict(band='가운데', y=[20, 30], dx=None, dy=None, features=2)]
        line, picture = onion.lines(result, 'x-onion.jpg')
        self.assertIn('맨 위 오른쪽 20px', line); self.assertIn('위 그대로', line); self.assertIn('가운데 모름', line)
        self.assertIn('x-onion.jpg', picture)

    def test_box_is_padded_and_clamped(self):
        self.assertEqual(onion._box((0, 100, 100, 200), 150, 205), (0, 90, 110, 205))


class GapSeeds(unittest.TestCase):
    def _video(self, path, arms):
        import cv2
        import numpy as np
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
        for k in range(10):
            frame = np.full((240, 320, 3), 255, np.uint8)
            cv2.rectangle(frame, (140, 60), (180, 200), (0, 0, 0), -1)  # still body
            for y in arms:  # arms drawn with an inner line, dropping 3 px a frame
                cv2.rectangle(frame, (180, y+3*k), (240, y+3*k+20), (0, 0, 0), -1)
                cv2.line(frame, (185, y+3*k+10), (235, y+3*k+10), (255, 255, 255), 2)
            out.write(frame)
        out.release()

    def test_candidates_sit_on_moving_arms_only(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            video = pathlib.Path(d)/'arms.avi'; self._video(video, [60, 140])
            seeds = onion.gap_seeds(video, 0, 9, (130, 50, 280, 210), [f'move_{c}' for c in 'abcdefghijkl'])
            self.assertTrue(seeds)
            self.assertTrue(all(x >= 180 for x, _ in (s['xy'] for s in seeds.values())), seeds)  # none on the still body


class Parts(unittest.TestCase):
    @staticmethod
    def _tracks(paths):
        return dict(frames=[dict(frame=t, points=[dict(point_id=pid, xy=list(p[t]), visible=True) for pid, p in paths.items()])
                            for t in range(len(next(iter(paths.values()))))])

    def test_points_moving_together_keep_one(self):
        arm = [(100+5*t, 100) for t in range(10)]
        tracks = self._tracks({'move-a': arm, 'move-b': [(x+30, y+2) for x, y in arm], 'move-c': [(300, 100+6*t) for t in range(10)],
                               'move-d': [(400, 400)]*10})
        self.assertEqual(onion.parts(tracks, 9), ['move-a', 'move-c'])

    def test_small_moves_next_to_a_big_one_are_dropped(self):
        tracks = self._tracks({'move-a': [(100+40*t, 100) for t in range(10)], 'move-b': [(500, 100+2*t) for t in range(10)]})
        self.assertEqual(onion.parts(tracks, 9), ['move-a'])


class Trails(unittest.TestCase):
    def packet(self):
        return dict(frames=[dict(frame=n, points=[dict(point_id='wrist-l', xy=[100.+5*n, 50.], visible=True),
                                                  dict(point_id='knee-r', xy=[40., 200.+n], visible=n != 7),
                                                  dict(point_id='nose', xy=[0., 0.], visible=True)]) for n in range(11)])

    def test_one_trail_per_part_and_one_dot_per_frame(self):
        found = onion.trails([self.packet()], ['wrist-l', 'knee-r'], 0, 10)
        self.assertEqual([t['point'] for t in found], ['wrist-l', 'knee-r'])
        self.assertEqual(len(found[0]['xy']), 10)  # f1..f10: the seed frame is the anchor, not a prediction
        self.assertEqual(len(found[1]['xy']), 9)   # f7 not visible

    def test_camera_shift_is_taken_out(self):
        camera = {n: (5.*n, 0., 1.) for n in range(11) if n != 4}
        found = onion.trails([self.packet()], ['wrist-l'], 0, 10, camera)[0]['xy']
        self.assertEqual({x for _, x, _ in found}, {100.}); self.assertNotIn(4, [n for n, _, _ in found])

    def test_drawn_on_the_onion(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            video = Path(d)/'v.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            for k in range(11):
                out.write(np.full((240, 320, 3), 128, np.uint8))
            out.release()
            image = Path(d)/'o.jpg'
            onion.bands(video, 0, 10, (20, 20, 300, 230), None, image, onion.trails([self.packet()], ['wrist-l'], 0, 10))
            picture = cv2.imread(str(image)).astype(int)
            yellow = (abs(picture-np.array(onion.TRAILS['colors'][0])).sum(2) < 90)
            self.assertGreater(yellow.sum(), 100)


    def test_arrow_head_fixed_and_kept_inside(self):
        import cv2
        import numpy as np
        calls = []
        class Spy:
            LINE_AA = cv2.LINE_AA
            def arrowedLine(self, picture, start, end, color, thickness, line_type, tipLength):
                calls.append((start, end, tipLength))
        picture = np.zeros((200, 100, 3), np.uint8)
        onion._arrow(Spy(), picture, (50, 100), (400, 0))  # far past the right edge
        start, end, tip = calls[-1]
        self.assertTrue(0 <= end[0] < 100 and 0 <= end[1] < 200)
        self.assertAlmostEqual(tip*abs(end[0]-start[0]), onion.ONION['arrow_head'], delta=1)
        calls.clear(); onion._arrow(Spy(), picture, (50, 100), (0, 90))  # long arrow inside: same head size
        start, end, tip = calls[-1]
        self.assertAlmostEqual(tip*abs(end[1]-start[1]), onion.ONION['arrow_head'], delta=1)
        calls.clear(); onion._arrow(Spy(), picture, (50, 100), (3, 2))  # shorter than the head: left out
        self.assertEqual(calls, [])


    # Row-preservation tests above miss the renderer reconnecting f1 to f3.
    def test_rendered_trail_breaks_at_missing_native_frame(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            image=np.full((200,320,3),128,np.uint8);out=Path(d)/'gapped.png'
            onion._image(cv2,np,image,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(0,0,320,200),[],out,
                         [dict(point='wrist-l',xy=[(1,40,100),(3,260,100),(4,280,100)])])
            rendered=cv2.imread(str(out))
            self.assertEqual(rendered[100,160].tolist(),[128]*3)  # no inferred f1->f3 path
            self.assertEqual(rendered[100,270].tolist(),list(onion.TRAILS['colors'][0]))  # real f3->f4 remains

    # A large person box shrinks more in the onion sheet; fixed 3 px dots read as single markers there.
    def test_trail_marks_grow_with_the_person_box(self):
        import tempfile
        import cv2
        import numpy as np
        def coloured(width):
            with tempfile.TemporaryDirectory() as d:
                image=np.full((200,width,3),128,np.uint8);out=Path(d)/'o.png'
                onion._image(cv2,np,image,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(0,0,width,200),[],out,
                             [dict(point='move-a',xy=[(1,100,100)])])
                return int((abs(cv2.imread(str(out)).astype(int)-np.array(onion.TRAILS['colors'][0])).sum(2) < 90).sum())
        self.assertEqual(coloured(220), coloured(440))
        self.assertGreater(coloured(1760), 2*coloured(440))

    # Red grid numbers on a pink anime frame were nearly invisible: every label is white with a black rim.
    def test_labels_are_white_with_a_black_rim(self):
        import cv2
        import numpy as np
        from image_lens import frames
        image = np.full((60, 120, 3), (180, 60, 240), np.uint8)
        frames.label(cv2, image, '400', (10, 40), .9, 2)
        pixels = image.reshape(-1, 3).astype(int)
        self.assertTrue((pixels.max(1) < 30).any()); self.assertTrue((pixels.min(1) > 240).any())  # edges are anti-aliased

    # mark's points have no fixed colour order: each trail carries its name, kept inside the picture at the right edge.
    def test_named_trails_label_inside_the_picture(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            image=np.full((200,320,3),128,np.uint8);out=Path(d)/'o.png'
            onion._image(cv2,np,image,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(0,0,320,200),[],out,
                         [dict(point='knee-bent',xy=[(1,310,100),(2,312,100)])],named=True)
            white=(cv2.imread(str(out)).min(2) > 240)
            self.assertGreater(white[:, :300].sum(), 30)  # the name went left of the point, not off the edge
        self.assertEqual(len(onion.trails([dict(frames=[dict(frame=1, points=[dict(point_id=f'p{i}', xy=[i, i], visible=True) for i in range(8)])])],
                                          [f'p{i}' for i in range(8)], 0, 1, limit=8)), 8)

    # Anime clip f180..209: the whole screen panned 60 px, the onion showed it as motion while the trails (camera taken
    # out) stayed short, so a correct track looked slipped.
    def test_onion_takes_the_camera_out(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            video = Path(d)/'v.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            for k in range(11):  # a dark square panned 3 px left per frame with its background stripes
                f = np.full((240, 320, 3), 200, np.uint8)
                for x in range(0, 400, 40):
                    cv2.line(f, (x-3*k, 0), (x-3*k, 240), (60, 60, 60), 2)
                cv2.rectangle(f, (150-3*k, 90), (190-3*k, 130), (0, 0, 0), -1); out.write(f)
            out.release()
            def coloured(camera):
                image = Path(d)/'o.png'
                onion.bands(video, 0, 10, (100, 60, 240, 200), camera, image)
                picture = cv2.imread(str(image)).astype(int)
                return int((abs(picture[..., 0]-picture[..., 2]) > 80).sum())  # cyan or red pixels
            self.assertGreater(coloured(None), 4*coloured((-30., 0.)))


class Peak(unittest.TestCase):
    def test_out_and_back_shows_the_far_frame(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            video = Path(d)/'v.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            for k in range(21):  # a block goes right and comes back: start and end are the same picture
                frame = np.full((240, 320, 3), 255, np.uint8); x = 60+8*min(k, 20-k)
                cv2.rectangle(frame, (x, 80), (x+40, 160), (0, 0, 0), -1); out.write(frame)
            out.release()
            bbox = (40, 60, 280, 180); image = Path(d)/'o.jpg'
            onion.bands(video, 0, 20, bbox, None, image); width = cv2.imread(str(image)).shape[1]
            self.assertIn(onion.peak(video, 0, 20, bbox, image), (9, 10, 11))
            self.assertGreater(cv2.imread(str(image)).shape[1], 2*width)
            self.assertEqual(onion.still(video, 3, None, Path(d)/'s.jpg'), 's.jpg')
            self.assertEqual(onion.strip(video, 0, 20, bbox, Path(d)/'t.jpg')['frames'][0], 0)
            onion.still(video, 5, None, Path(d)/'s2.jpg')
            sheets = onion.sheets(Path(d), [(0, .5, 'o.jpg', ''), (.5, .6, 's.jpg', 'still'), (.6, .7, 's2.jpg', 'still'), (0, .5, 't.jpg', 'strip')])
            self.assertEqual(cv2.imread(str(Path(d)/sheets[0])).shape[1], onion.SHEET['tile']*onion.SHEET['columns'])


if __name__ == '__main__': unittest.main()
