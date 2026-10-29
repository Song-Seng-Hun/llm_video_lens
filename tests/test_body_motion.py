"""Body-relative limb motion; synthetic tracks, no models."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import body_motion


def packet(paths):
    frames = sorted({n for p in paths.values() for n in p})
    return dict(frames=[dict(frame=n, points=[dict(point_id=k, xy=list(v[n]), visible=True) for k, v in paths.items() if n in v]) for n in frames])


class BodyMotion(unittest.TestCase):
    def body(self, bounce):
        return {'shoulder-l': {n: (300., 300.+bounce(n)) for n in range(21)}, 'shoulder-r': {n: (100., 300.+bounce(n)) for n in range(21)}}

    def test_whole_body_bounce_is_not_a_hand_move(self):
        paths = self.body(lambda n: -2*n); paths['wrist-l'] = {n: (320., 400.-2*n) for n in range(21)}
        r = body_motion.describe([packet(paths)], dict(seed=0, end=20))
        self.assertEqual(r['centre_shift']['direction'], 'up'); self.assertEqual(r['limbs'], [])

    def test_hands_spreading_from_the_face_read_as_outward(self):
        paths = self.body(lambda n: 0); paths['wrist-l'] = {n: (230.+3*n, 250.) for n in range(21)}
        paths['wrist-r'] = {n: (170.-3*n, 250.) for n in range(21)}
        r = body_motion.describe([packet(paths)], dict(seed=0, end=20))
        sides = {l['side']: l for l in r['limbs']}
        self.assertEqual(sides['화면 오른쪽']['strokes'][0]['direction'], 'right'); self.assertGreater(sides['화면 오른쪽']['axis_px'], 0)
        self.assertEqual(sides['화면 왼쪽']['strokes'][0]['direction'], 'left'); self.assertGreater(sides['화면 왼쪽']['axis_px'], 0)
        self.assertIn('몸 바깥으로', '\n'.join(body_motion.lines(r)))

    def test_down_then_up_splits_at_the_turn(self):
        paths = self.body(lambda n: 0); paths['wrist-r'] = {n: (150., 400.+(4*n if n <= 10 else 40-6*(n-10))) for n in range(21)}
        limb = body_motion.describe([packet(paths)], dict(seed=0, end=20))['limbs'][0]
        self.assertEqual([s['direction'] for s in limb['strokes']], ['down', 'up']); self.assertIn(limb['strokes'][0]['frames'][1], (9, 10, 11))

    def test_seed_frame_is_ignored(self):
        paths = self.body(lambda n: 0); paths['wrist-l'] = {n: (320., 400. if n else 340.) for n in range(21)}
        self.assertEqual(body_motion.describe([packet(paths)], dict(seed=0, end=20))['limbs'], [])

    def test_hand_crossing_the_centre_keeps_its_arm_side(self):
        paths = self.body(lambda n: 0); paths['wrist-r'] = {n: (210.+n, 400.) for n in range(21)}
        paths['wrist-r'].update({n: (210.+n, 400.-3*n) for n in range(21)})
        limb = body_motion.describe([packet(paths)], dict(seed=0, end=20))['limbs'][0]
        self.assertEqual(limb['side'], '화면 왼쪽')

if __name__ == '__main__': unittest.main()
