"""Readability contracts for coarse strokes, trajectory option guards and change candidates; synthetic, not quality evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import motion_report


def path(*legs):
    xy = [[100., 100.]]
    for steps, dx, dy in legs:
        for _ in range(steps):
            xy.append([xy[-1][0]+dx, xy[-1][1]+dy])
    return list(range(len(xy))), [n/24. for n in range(len(xy))], xy


class Strokes(unittest.TestCase):
    def test_move_hold_move_becomes_three_units(self):
        fs, ts, xy = path((12, 5., 0.), (12, 0., 0.), (12, 0., 4.))
        units = motion_report.strokes(fs, ts, xy)['units']
        self.assertEqual([u['direction'] for u in units], ['right', 'hold', 'down'])
        self.assertTrue(units[0]['major'] and units[2]['major'] and not units[1]['major'])
        self.assertAlmostEqual(units[0]['distance_px'], 60., delta=6.)

    def test_single_sample_spike_does_not_create_a_stroke(self):
        fs, ts, xy = path((24, 3., 0.))
        xy[12] = [xy[12][0], xy[12][1]+80.]
        units = motion_report.strokes(fs, ts, xy)['units']
        self.assertEqual([u['direction'] for u in units], ['right'])

    def test_straight_line_reversal_keeps_both_strokes(self):
        fs, ts, xy = path((20, 5., 0.), (18, -5., 0.))
        self.assertEqual([u['direction'] for u in motion_report.strokes(fs, ts, xy)['units']], ['right', 'left'])

    def test_merged_slow_drift_is_not_a_hold(self):
        legs = [(10, .5, 0.), (1, 2., 0.)]*6
        fs, ts, xy = path(*legs)
        units = motion_report.strokes(fs, ts, xy)['units']
        self.assertFalse(any(u['direction'] == 'hold' and u['distance_px'] >= motion_report.STROKE['hold_px'] for u in units))

    def test_raw_coordinates_untouched_and_units_capped(self):
        legs = [(4, 0., 6.) if i % 2 == 0 else (4, 0., -6.) for i in range(14)]
        fs, ts, xy = path(*legs)
        before = json.dumps(xy)
        result = motion_report.strokes(fs, ts, xy)
        self.assertEqual(json.dumps(xy), before)
        self.assertLessEqual(len(result['units']), motion_report.STROKE['max_strokes'])
        self.assertEqual(result['omitted_unit_count'] + len(result['units']) >= len(result['units']), True)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'numpy (core extra) required')
class ChangeCandidates(unittest.TestCase):
    def test_flags_only_the_jump(self):
        import numpy as np
        from image_lens import motion_view
        frames = [np.full((120, 160), 100, np.uint8) for _ in range(20)]
        for i in range(10, 20): frames[i] = np.full((120, 160), 160, np.uint8)
        result = motion_view.change_candidates(frames, 50)
        self.assertEqual([c['compare_frames'] for c in result['candidates']], [[59, 60]])

    def test_static_sequence_has_no_candidates(self):
        import numpy as np
        from image_lens import motion_view
        frames = [np.full((120, 160), 100, np.uint8) for _ in range(10)]
        self.assertEqual(motion_view.change_candidates(frames, 0)['candidates'], [])


if __name__ == '__main__': unittest.main()
