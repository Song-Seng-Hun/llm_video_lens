import unittest
import numpy as np
from image_lens import mark, motion_report


class Points(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(mark.parse_point('hammer-top=12,34.5'), ('hammer-top', (12., 34.5)))
        for bad in ('망치=1,2', 'a=1', 'a 1,2'):
            with self.assertRaises(ValueError):
                mark.parse_point(bad)

    def test_snap_moves_to_the_nearby_corner_only(self):
        gray = np.zeros((100, 100), np.uint8); gray[40:60, 40:60] = 255
        (x, y), moved = mark.snap(gray, (45., 45.))
        self.assertLess(abs(x-40)+abs(y-40), 3); self.assertGreater(moved, 0)
        self.assertEqual(mark.snap(gray, (90., 90.)), ((90., 90.), 0.))


class Ease(unittest.TestCase):
    def test_timing_shapes(self):
        ts = [i/10 for i in range(13)]
        for expected, f in (('ease_in', lambda t: t*t), ('ease_out', lambda t: 1-(1-t)**2),
                            ('ease_in_out', lambda t: 3*t*t-2*t**3), ('even', lambda t: t)):
            self.assertEqual(motion_report.ease(ts, [(100*f(t/1.2), 0.) for t in ts]), expected)
        self.assertIsNone(motion_report.ease(ts[:3], [(0., 0.), (1., 0.), (2., 0.)]))
        # float times whose thirds land past the last sample must not raise
        ts = [.1*i+1e-12*(i == 6) for i in range(7)]
        self.assertIn(motion_report.ease(ts, [(float(i), 0.) for i in range(7)]), ('even', 'ease_in', 'ease_out', 'ease_in_out'))


class Coordinates(unittest.TestCase):
    def test_unknown_camera_endpoint_keeps_numbers_trails_and_caption_in_screen_basis(self):
        """A partial estimate without the end must not correct the picture while numbers fall back to screen."""
        import json
        import tempfile
        import cv2
        from pathlib import Path
        from unittest.mock import patch
        from image_lens import camera, look, onion, paths, shot_boundaries
        from tests.test_look import Camera
        previous = paths.workspace_root()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();paths.configure(root)
            try:
                video=root/'source.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(640,360))
                image=np.full((360,640,3),128,np.uint8);image[100:200,180:300]=[50,180,220]
                for _ in range(26):writer.write(image)
                writer.release()
                scan=shot_boundaries.scan_video(video);scan_path=paths.write_json(root/'shots.json',scan)
                packet,_,_=Camera().view([(200+3*n,150) for n in range(26)],None)
                partial={n:(4.*n,0.,1.) for n in range(26) if n!=25}
                original=look._step
                def step(command,args):
                    if command!='track':return original(command,args)
                    args=list(map(str,args));query=json.loads(Path(args[args.index('--queries')+1]).read_text())
                    packet['job']=query['jobs'][0];packet['shot_check']=scan
                    paths.write_json(args[args.index('--out')+1],packet)
                out=root/'marked'
                with patch.object(look,'_step',step),patch.object(camera,'path',return_value=partial),\
                     patch.object(onion,'body_basis',return_value=None),\
                     patch.object(onion,'bands',wraps=onion.bands) as bands,\
                     patch.object(onion,'references',wraps=onion.references) as references:
                    mark.run(video,scan_path,0,['hand=200,150'],'recorded-unused',out,end=25,snap_points=False)
                text=(out/'mark.md').read_text();record=json.loads((out/'mark.json').read_text())
                self.assertNotIn('아래 수치·그림에서 뺌',text)
                self.assertIn('움직임 수치 기준: 화면.',text)
                self.assertIn('카메라 일부 모름: 화면 좌표 그대로',text)
                self.assertEqual(record['summary']['coordinate_scope'],'screen_2d')
                self.assertIsNone(bands.call_args.args[4])
                trail=bands.call_args.args[6][0]['xy']
                self.assertEqual(trail[-1][1]-trail[0][1],72.)
                self.assertIsNone(references.call_args.args[3])
                self.assertFalse((out/'mark.relative.json').exists())
            finally:
                paths.configure(previous)

if __name__ == '__main__': unittest.main()
