import tempfile
import unittest
from pathlib import Path
from image_lens import digest, paths, zoom

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
            out.write(np.full((180, 320, 3), k % 128 * 2, np.uint8))
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


class Overview(unittest.TestCase):
    def test_one_frame_per_interval_with_a_cap(self):
        self.assertEqual(zoom.overview_count(719, 23.98), 24)  # Kaguya 30 s: the budget that gained 3.5 of 20
        self.assertEqual(zoom.overview_count(24*600, 24), zoom.ZOOM['max_overview'])
        self.assertEqual(zoom.overview_count(3, 24), 1)

    def test_sheets_and_reading_line(self):
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            paths.configure(d)
            video = Path(d)/'v.avi'; Run._video(None, video, frames=200)  # 20 s: 16 frames, two sheets
            names = zoom.overview(video, Path(d))
            self.assertEqual(names, ['frame-sheet-1.jpg', 'frame-sheet-2.jpg'])
            record = dict(source_video=str(video), shots=[], frame_sheets=names)
            text = digest.markdown(record, dict(fps=10, source_frame_count=200), 'look-full.md')
            self.assertIn('전체 화면 모음: frame-sheet-1.jpg, frame-sheet-2.jpg', text)
