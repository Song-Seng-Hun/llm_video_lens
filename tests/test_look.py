"""Shot-window selection for the one-command look; synthetic, no models."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import look


class Windows(unittest.TestCase):
    def frames(self, n0, n1):
        return {n: [dict(track_id='t')] for n in range(n0, n1+1)}

    def test_long_shot_splits_into_reseeded_windows(self):
        found, skipped = look.windows(dict(start_frame=0, end_frame=399), self.frames(0, 399), set())
        self.assertEqual(found, [dict(seed=0, end=159), dict(seed=160, end=319), dict(seed=320, end=399)])
        self.assertEqual(skipped, [])

    def test_seed_skips_flagged_and_empty_frames(self):
        poses = self.frames(10, 99)
        found, _ = look.windows(dict(start_frame=0, end_frame=99), poses, {10, 11, 12})
        self.assertEqual(found, [dict(seed=13, end=99)])

    def test_short_tail_and_no_pose_are_reported(self):
        found, skipped = look.windows(dict(start_frame=0, end_frame=165), self.frames(0, 165), set())
        self.assertEqual(found, [dict(seed=0, end=159)])
        self.assertIn('fewer than', skipped[0]['reason'])
        found, skipped = look.windows(dict(start_frame=0, end_frame=50), {}, set())
        self.assertEqual(found, []); self.assertEqual(skipped[0]['reason'], 'no unflagged pose track')



class Summary(unittest.TestCase):
    def test_no_observation_is_unknown_not_still(self):
        report = dict(features=[dict(point_id='hip-l', continuous_runs=[])])
        lines, _ = look._summary(report, None, 'legs', 'x', 'y')
        self.assertEqual(len(lines), 1); self.assertIn('모름', lines[0])

if __name__ == '__main__': unittest.main()
