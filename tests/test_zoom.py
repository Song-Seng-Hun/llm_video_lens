import tempfile
import unittest
from pathlib import Path
from image_lens import paths, zoom

REPO = Path(__file__).resolve().parents[1]


class Spans(unittest.TestCase):
    def test_spans_are_bounded(self):
        self.assertEqual(zoom.spans(['18-22', '3.5-5']), [(18., 22.), (3.5, 5.)])
        for bad in (['22-18'], ['10-15'], ['a-b'], [], ['1-2', '3-4', '5-6']):
            with self.assertRaises(ValueError):
                zoom.spans(bad)

    def test_picks_are_even_and_unique(self):
        self.assertEqual(zoom.picks(0, 11, 12), list(range(12)))
        self.assertEqual(zoom.picks(5, 7, 12), [5, 6, 7])
        self.assertEqual(zoom.picks(0, 110, 12)[:3], [0, 10, 20])


class Run(unittest.TestCase):
    def _video(self, path, frames=100):
        import cv2
        import numpy as np
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 180))
        for k in range(frames):
            out.write(np.full((180, 320, 3), k*2, np.uint8))
        out.release()

    def test_spans_and_even_use_the_same_budget(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            paths.configure(d)
            video = Path(d)/'v.avi'; self._video(video)
            spans = zoom.run(video, Path(d)/'a', at=['1-4', '6-9'])
            even = zoom.run(video, Path(d)/'b', even=True)
            self.assertEqual(spans['cost']['frames'], zoom.ZOOM['budget'])
            self.assertEqual(even['cost']['frames'], zoom.ZOOM['budget'])
            self.assertEqual(spans['sheets'][0]['frames'][0], 10)
            self.assertEqual(even['sheets'][1]['frames'][-1], 99)
            self.assertTrue((Path(d)/'a'/'zoom-2.jpg').is_file())
            with self.assertRaises(ValueError):
                zoom.run(video, Path(d)/'c', at=['1-2'], even=True)
