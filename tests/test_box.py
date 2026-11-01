import unittest
import numpy as np
import cv2
from image_lens import box


def track(bbox, score=.9):
    kp = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]]*17
    kp = [list(p) for p in kp]; kp[9] = [bbox[0]+5, bbox[1]+5]; kp[10] = [bbox[2]-5, bbox[3]-5]
    return dict(bbox=list(bbox), keypoints=kp, scores=[score]*17)


class Box(unittest.TestCase):
    def test_the_box_picks_the_person_it_covers_not_the_largest(self):
        big, small = track((0, 0, 300, 400)), track((320, 100, 420, 300))
        who, overlap = box.person([big, small], (310, 90, 430, 310))
        self.assertIs(who, small); self.assertGreater(overlap, .6)
        self.assertEqual(box.person([big], (500, 0, 600, 100)), (None, 0.))  # nobody there
        names = [n for n, _ in box.joints(small, (310, 90, 430, 310), 640, 480)]
        self.assertEqual(names[:3], ['nose', 'wrist-l', 'wrist-r']); self.assertLessEqual(len(names), 8)
        self.assertEqual(box.joints(track((320, 100, 420, 300), score=.1), (310, 90, 430, 310), 640, 480), [])
        clamped = track((320, 100, 420, 479)); clamped['keypoints'][10] = [415, 479]  # an ankle off the bottom edge
        self.assertNotIn('wrist-r', [n for n, _ in box.joints(clamped, (310, 90, 430, 480), 640, 480)])

    def test_no_person_falls_back_to_spread_corners_inside_the_box(self):
        gray = np.zeros((240, 320), np.uint8)
        for x, y in ((40, 40), (120, 60), (60, 150), (200, 200)):
            cv2.rectangle(gray, (x, y), (x+20, y+20), 255, -1)
        got = box.corners(gray, (30, 30, 160, 190))
        self.assertTrue(got and len(got) <= 8 and got[0][0] == 'p1')
        self.assertTrue(all(30 <= x < 160 and 30 <= y < 190 for _, (x, y) in got))
        self.assertEqual([xy[1] for _, xy in got], sorted(xy[1] for _, xy in got))  # top to bottom

    def test_bad_boxes_are_refused(self):
        self.assertEqual(box.parse('-5,-5,700,500', 640, 480), (0, 0, 640, 480))  # clamped to the frame
        for text in ('10,10,20,300', '100,100,50,200'):
            with self.assertRaises(ValueError):
                box.parse(text, 640, 480)


if __name__ == '__main__':
    unittest.main()
