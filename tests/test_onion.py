from pathlib import Path
import tempfile
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

    def test_timing_dots_every_tenth_of_a_second_from_each_run_start(self):
        self.assertEqual(onion.ticks([0,1,2,3,4,5,6,7], 30), {3,6})
        self.assertEqual(onion.ticks([0,1,2,3,4,5,6,7], 24), {2,4,6})
        self.assertEqual(onion.ticks([0,1,2,3,10,11,12,13], 30), {3,7})  # the gap restarts the count
        self.assertEqual(onion.ticks([0,1,2,3], None), set())

    def test_screen_basis_is_named_in_the_onion_lines(self):
        bands=[dict(band='맨 위',dx=20.,dy=0.)]
        self.assertIn('카메라 모름: 화면 좌표 그대로',onion.lines(bands,'x-onion.jpg',screen=True)[0])
        self.assertTrue(all('평행이동 제외' not in l for l in onion.lines(bands,'x-onion.jpg',screen=True)))
        self.assertIn('카메라 추정 시 평행이동 제외',onion.lines(bands,'x-onion.jpg')[0])

    def test_spacing_rows_stretch_the_path_and_ignore_held_jitter(self):
        slowing=dict(point='hand',xy=[(n,x,0.) for n,x in enumerate((0,20,38,52,62,68,71))])
        jitter=dict(point='hip',xy=[(n,10+.5*(n%2),0.) for n in range(20)])
        small=dict(point='toe',xy=[(n,4.*n,0.) for n in range(3)])
        rows,still=onion.spacing_rows([slowing,jitter,small],30)
        self.assertEqual([r['point'] for r in rows],['hand']); self.assertEqual(still,['hip','toe'])
        hand=rows[0]; self.assertAlmostEqual(hand['length'],71.)
        self.assertEqual([round(f,3) for f,timed in hand['ticks'] if timed],[0.,round(52/71,3),1.])
        held=dict(point='arm',xy=[(n,(0,0,.3,.6,.9,30,60)[n],0.) for n in range(7)])
        self.assertEqual(onion.spacing_rows([held],30)[0][0]['ticks'][4][0],0.)  # sub-threshold steps add nothing
        lost=dict(point='corner',xy=[(0,0.,0.),(1,2.,0.),(9,300.,0.)])
        self.assertEqual(onion.spacing_rows([lost],30),([],['corner']))  # a jump over missing frames is not travel

    def test_camera_and_body_rows_lead_and_drop_out_when_still(self):
        camera={n:(4.*n,0.,1.) for n in range(11)}; body={n:(100.+4.*n,50.) for n in range(11)}  # body rides the pan only
        charted,colors=onion.references([],0,10,camera,body)
        self.assertEqual([t['point'] for t in charted],['camera','body']); self.assertEqual(charted[1]['xy'][-1],(10,100.,50.))
        rows,still=onion.spacing_rows(charted,30)
        self.assertEqual(([r['point'] for r in rows],still),(['camera'],[]))  # a still body is no row and not 'still'
        self.assertEqual(colors[:2],list(onion.REFERENCE['colors']))
        self.assertEqual([t['point'] for t in onion.references([],0,10,None,body)[0]],['body'])  # no camera: screen basis

    def test_timing_dot_is_drawn_larger(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            image=np.full((200,320,3),128,np.uint8);out=Path(d)/'ticks.png'
            onion._image(cv2,np,image,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(0,0,320,200),[],out,
                         [dict(point='wrist-l',xy=[(n,40+30*n,100) for n in range(8)])],fps=30)
            rendered=cv2.imread(str(out)).astype(int);yellow=np.array(onion.TRAILS['colors'][0])
            def wide(x):return int((abs(rendered[90:111,x-12:x+13]-yellow).sum(2)<90).sum())
            self.assertGreater(wide(40+30*3),wide(40+30*2))  # f3 is a timing dot, f2 is not

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
            self.assertIsNone(onion.peak(video, 0, 20, bbox, image))  # off by default: the picture is left alone
            self.assertEqual(cv2.imread(str(image)).shape[1], width)
            self.assertIn(onion.peak(video, 0, 20, bbox, None), (9, 10, 11))
            onion.PEAK['draw'] = True
            try:
                self.assertIn(onion.peak(video, 0, 20, bbox, image), (9, 10, 11))
            finally:
                onion.PEAK['draw'] = False
            self.assertGreater(cv2.imread(str(image)).shape[1], 2*width)
            self.assertEqual(onion.still(video, 3, None, Path(d)/'s.jpg'), 's.jpg')
            self.assertEqual(onion.strip(video, 0, 20, bbox, Path(d)/'t.jpg')['frames'][0], 0)
            onion.still(video, 5, None, Path(d)/'s2.jpg')
            sheets = onion.sheets(Path(d), [(0, .5, 'o.jpg', ''), (.5, .6, 's.jpg', 'still'), (.6, .7, 's2.jpg', 'still'), (0, .5, 't.jpg', 'strip')])
            self.assertEqual(cv2.imread(str(Path(d)/sheets[0])).shape[1], onion.SHEET['width'])


    def test_sheet_tiles_share_one_height(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            cv2.imwrite(str(Path(d)/'wide.jpg'), np.full((1247, 1716, 3), 90, np.uint8))  # onion + spacing chart
            cv2.imwrite(str(Path(d)/'tall.jpg'), np.full((1242, 1080, 3), 160, np.uint8))  # a single-frame still
            name, = onion.sheets(Path(d), [(0, .03, 'tall.jpg', 'still'), (.03, 4, 'wide.jpg', '')])
            sheet = cv2.imread(str(Path(d)/name))
            self.assertEqual(sheet.shape[:2], (onion.SHEET['tile_height']+onion.SHEET['label'], onion.SHEET['width']))
            row = sheet[onion.SHEET['label']+10]
            cols = np.flatnonzero(np.abs(row.astype(int)-90).max(1) < 8)
            self.assertGreater(cols.size, 1.3*onion.SHEET['tile_height'])  # beside the still at full height, wider than tall


    def test_picture_view_has_a_floor_and_the_same_ratio_limit_both_ways(self):
        a = onion.VIEW['max_aspect']
        for bbox in ((300, 900, 800, 1050), (500, 300, 560, 1500), (0, 0, 40, 40), (1040, 1880, 1080, 1920)):
            x0, y0, x1, y1 = onion._view(bbox, 1080, 1920)
            vw, vh = x1-x0, y1-y0
            self.assertTrue(0 <= x0 < x1 <= 1080 and 0 <= y0 < y1 <= 1920)
            self.assertGreaterEqual(min(vw, vh), onion.VIEW['min_side']*1080-1)
            self.assertLessEqual(max(vw/vh, vh/vw), a+.01)
        bx = onion._box((300, 900, 800, 1050), 1080, 1920); v = onion._view((300, 900, 800, 1050), 1080, 1920)
        self.assertTrue(v[0] <= bx[0] and v[1] <= bx[1] and v[2] >= bx[2] and v[3] >= bx[3])  # never cuts the person


    def test_context_frame_sits_in_the_chart_panel_and_stills_are_whole(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            video = Path(d)/'v.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            for k in range(6):
                frame = np.full((240, 320, 3), 255, np.uint8); cv2.rectangle(frame, (100+4*k, 80), (140+4*k, 160), (0, 0, 0), -1)
                out.write(frame)
            out.release()
            image = Path(d)/'o.jpg'; bbox = (100, 80, 160, 160)
            onion.bands(video, 0, 5, bbox, None, image); width = cv2.imread(str(image)).shape[1]
            self.assertEqual(onion.spacing(video, image, [], context=(0, bbox)), dict(rows=[], still=[]))
            picture = cv2.imread(str(image)); panel = picture[:, width+6:]
            yellow = (panel[..., 0] < 80) & (panel[..., 1] > 150) & (panel[..., 2] > 200)
            self.assertGreater(int(yellow.sum()), 20)  # the view outline on the small whole frame
            self.assertIsNone(onion.spacing(video, Path(d)/'missing.jpg', []))
            onion.still(video, 2, bbox, Path(d)/'s.jpg')
            self.assertEqual(cv2.imread(str(Path(d)/'s.jpg')).shape[:2], (240, 320))


    def test_body_basis_only_for_a_whole_figure_move(self):
        import cv2
        import numpy as np
        def packets(feet_move):
            frames = []
            for n in range(0, 31):
                dx = 4*n  # 120 px right over the window
                pts = {'shoulder-l': (100+dx, 60), 'shoulder-r': (140+dx, 60), 'wrist-l': (90+dx, 100), 'wrist-r': (150+dx, 100),
                       'ankle-l': (110+(dx if feet_move else 0), 200), 'ankle-r': (130+(dx if feet_move else 0), 200)}
                frames.append(dict(frame=n, points=[dict(point_id=k, xy=list(v), visible=True) for k, v in pts.items()]))
            return [dict(frames=frames)]
        with tempfile.TemporaryDirectory() as d:
            video = Path(d)/'v.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            out.write(np.full((240, 320, 3), 255, np.uint8)); out.release()
            bbox = (80, 40, 160, 210)
            slide = packets(True); path = onion.body_path(slide, 0, 30)
            shift = onion.body_basis(video, bbox, path, slide)
            self.assertIsNotNone(shift); self.assertAlmostEqual(shift[0], 116, delta=6)
            trail, = onion.trails(slide, ['wrist-l'], 0, 30, body=path)
            self.assertLess(max(x for _, x, _ in trail['xy'])-min(x for _, x, _ in trail['xy']), 3)  # the wrist rode along
            lunge = packets(False)  # shoulders and hands go, feet stay: not a whole-figure move
            self.assertIsNone(onion.body_basis(video, bbox, onion.body_path(lunge, 0, 30), lunge))


if __name__ == '__main__': unittest.main()
