import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from image_lens import crowd, paths
from tests.test_zoom import Run as ZoomRun

REPO = Path(__file__).resolve().parents[1]


class Counting(unittest.TestCase):
    def test_near_equal_boxes_count_once(self):
        boxes = np.float32([[0, 0, 100, 200], [4, 2, 104, 198], [60, 0, 160, 200]])
        kept = crowd.distinct(boxes, np.float32([.5, .9, .4]))
        self.assertEqual([list(k) for k in kept], [[4, 2, 104, 198], [60, 0, 160, 200]])

    def test_ranked_takes_most_people_apart_in_time(self):
        rows = [dict(frame=f, boxes=[[0, 0, 10, 10]]*k) for f, k in ((70, 1), (10, 3), (12, 3), (40, 2), (80, 0))]
        self.assertEqual([r['frame'] for r in crowd.ranked(rows, fps=10, top=3, apart_s=2.)], [10, 40, 70])
        self.assertEqual(crowd.ranked([dict(frame=0, boxes=[])], fps=10), [])


class Run(unittest.TestCase):
    def test_samples_in_order_and_writes_the_sheet(self):
        def det(image):  # brightness k*2 at frame k: frame 30 has the most people
            k = round(float(image[0, 0, 0])/2)
            n = 4 if k == 30 else 1
            return np.float32([[i*70, 10, i*70+60, 170] for i in range(n)]), np.full(n, .9, np.float32)
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            paths.configure(d)
            video = Path(d)/'v.avi'; ZoomRun._video(None, video)
            with patch('image_lens.person_models.resolve', return_value=dict(kind='humanart', path='x')), \
                 patch('image_lens.pose.person_detector', return_value=det):
                result = crowd.run(video, 'x', Path(d)/'c')
            saved = json.loads((Path(d)/'c'/'crowd.json').read_text(encoding='utf-8'))
            self.assertEqual([f for f, _ in saved['counts']], list(range(0, 100, 5)))
            self.assertEqual(result['top'][0]['frame'], 30); self.assertEqual(result['suggested_max_people'], 4)
            self.assertTrue((Path(d)/'c'/'crowd.jpg').is_file())
            with self.assertRaises(ValueError):
                crowd.run(video, 'x', Path(d)/'e', every_s=10)


if __name__ == '__main__':
    unittest.main()
