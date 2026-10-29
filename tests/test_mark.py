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

if __name__ == '__main__': unittest.main()
