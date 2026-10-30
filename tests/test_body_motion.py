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

    def test_ease_is_timed_on_the_body_relative_path(self):
        # The body slides up evenly while the hand speeds up against it: the screen path is even-ish, the limb is ease-in.
        paths = self.body(lambda n: -6*n); paths['wrist-l'] = {n: (320.+.4*n*n, 400.-6*n) for n in range(21)}
        r = body_motion.describe([packet(paths)], dict(seed=0, end=20))
        stroke = r['limbs'][0]['strokes'][0]
        self.assertEqual((stroke['direction'], stroke['ease']), ('right', 'ease_in'))
        self.assertEqual(r['centre_shift']['ease'], 'even')
        self.assertTrue(stroke['from_start'])
        self.assertLess(stroke['speeds'][0], stroke['speeds'][-1])

    def test_travel_counts_back_and_forth(self):
        paths = self.body(lambda n: 0); paths['wrist-r'] = {n: (150., 400.+(4*n if n <= 10 else 40-4*(n-10))) for n in range(21)}
        limb = body_motion.describe([packet(paths)], dict(seed=0, end=20))['limbs'][0]
        self.assertGreaterEqual(limb['travel_px'], 60.)  # out 36 px and back from f1 (seed left out); net is about 4

    def test_hand_crossing_the_centre_keeps_its_arm_side(self):
        paths = self.body(lambda n: 0); paths['wrist-r'] = {n: (210.+n, 400.) for n in range(21)}
        paths['wrist-r'].update({n: (210.+n, 400.-3*n) for n in range(21)})
        limb = body_motion.describe([packet(paths)], dict(seed=0, end=20))['limbs'][0]
        self.assertEqual(limb['side'], '화면 왼쪽')


def pose(shoulders, wrists, size=100.):
    k = [[0., 0.] for _ in range(133)]; s = [0.]*133
    (k[5], k[6]), (k[9], k[10]) = shoulders, wrists
    k[7] = [(shoulders[0][0]+wrists[0][0])/2, (shoulders[0][1]+wrists[0][1])/2]
    k[8] = [(shoulders[1][0]+wrists[1][0])/2, (shoulders[1][1]+wrists[1][1])/2]
    width = shoulders[0][0]-shoulders[1][0]
    k[11], k[12] = [shoulders[0][0]-.1*width, shoulders[0][1]+1.5*width], [shoulders[1][0]+.1*width, shoulders[1][1]+1.5*width]
    for i in range(5, 13):
        s[i] = .9
    return dict(bbox=[0, 0, size, size], keypoints=k, scores=s)


class Skeleton(unittest.TestCase):
    def test_plain_body_passes_and_folded_one_fails(self):
        good = pose(([300., 300.], [100., 300.]), ([320., 420.], [80., 420.]))
        self.assertEqual(body_motion.skeleton_check(good), (True, 'ok'))
        folded = pose(([300., 300.], [100., 300.]), ([320., 420.], [80., 420.]))
        folded['keypoints'][11], folded['keypoints'][12] = [290., 330.], [110., 330.]
        self.assertFalse(body_motion.skeleton_check(folded)[0])
        weak = pose(([300., 300.], [100., 300.]), ([320., 420.], [80., 420.])); weak['scores'][9] = .1
        self.assertIn('missing', body_motion.skeleton_check(weak)[1])


class Bridge(unittest.TestCase):
    def test_arm_dropping_on_the_switch_frame_is_reported(self):
        before = pose(([300., 300.], [100., 300.]), ([320., 420.], [-60., 300.]))
        after = pose(([300., 300.], [100., 300.]), ([320., 420.], [180., 520.]))
        r = body_motion.bridge(before, after, (61, 62))
        limb = r['limbs'][0]
        self.assertEqual((limb['side'], limb['direction']), ('화면 왼쪽', 'down_right')); self.assertLess(limb['axis_px'], 0)
        self.assertIn('몸 쪽으로', '\n'.join(body_motion.bridge_lines(r)))

    def test_far_apart_or_different_size_is_not_bridged(self):
        a = pose(([300., 300.], [100., 300.]), ([320., 420.], [-60., 300.]))
        self.assertIsNone(body_motion.bridge(a, a, (10, 40)))
        small = pose(([220., 300.], [180., 300.]), ([320., 420.], [-60., 300.]))
        self.assertIsNone(body_motion.bridge(a, small, (61, 62)))


def keypoints(points, score=.9):
    """133 keypoints off-screen at score 0 except the given {index: (x, y)}."""
    k = [[-1., -1.]]*133; s = [0.]*133
    for i, xy in points.items():
        k[i] = list(xy); s[i] = score
    return dict(keypoints=k, scores=s)


class Framing(unittest.TestCase):
    UPPER = {1: (620, 200), 2: (660, 200), 5: (560, 420), 6: (720, 420), 7: (540, 600), 8: (740, 600), 9: (600, 700), 10: (700, 700)}

    def test_framing_from_seen_joints(self):
        self.assertEqual(body_motion.framing(keypoints(self.UPPER), 1280, 720), 'upper')
        full = {**self.UPPER, 11: (600, 600), 12: (680, 600), 15: (600, 900), 16: (680, 900)}
        self.assertEqual(body_motion.framing(keypoints(full), 1280, 1080), 'full')
        guessed = {**self.UPPER, 15: (600, 700), 16: (680, 700)}  # ankles guessed at the bottom edge of an upper-body shot
        self.assertEqual(body_motion.framing(keypoints(guessed, .4), 1280, 720), 'upper')
        self.assertEqual(body_motion.framing(keypoints({1: (400, 300), 2: (880, 300)}), 1280, 720), 'face')
        thigh = {**self.UPPER, 11: (590, 640), 12: (690, 640)}
        self.assertEqual(body_motion.framing(keypoints(thigh), 1280, 1080), 'thigh')

    def test_upper_body_names_arms_without_hips(self):
        track = keypoints(self.UPPER)
        self.assertFalse(body_motion.skeleton_check(track)[0])
        self.assertEqual(body_motion.upper_check(track), (True, 'upper body'))
        self.assertFalse(body_motion.upper_check(keypoints({5: (560, 420), 7: (540, 600)}))[0])


class PoseEvents(unittest.TestCase):
    FACE = {i: (600+2*(i-23), 150+(i-23)) for i in range(23, 91)}  # face box x 600..734, y 150..217

    def frames(self, joints, n=30, track='t'):
        return {f: [dict(keypoints(joints(f)), track_id=track, bbox=[400, 100, 900, 800])] for f in range(n)}

    def test_hand_brought_to_the_cheek_is_a_run(self):
        def joints(f):
            j = {**self.FACE, 1: (640, 170), 2: (700, 170), 5: (560, 420), 6: (780, 420), 9: (560, 700), 10: (780, 700)}
            if 10 <= f <= 20:
                j[10] = (740, 190)  # right wrist at the cheek height, just outside the face box
            return j
        found = body_motion.pose_events(self.frames(joints), dict(seed=0, end=29), 60)
        self.assertEqual(len(found['hand_face']), 1)
        e = found['hand_face'][0]
        self.assertEqual((e['frames'], e['side'], e['region'], e['enters'], e['exits']), ([10, 20], '화면 오른쪽', '볼', True, True))

    def test_another_detection_is_not_the_seed_person(self):
        frames = self.frames(lambda f: {1: (640, 170), 2: (700, 170), 5: (560, 420), 6: (760, 420), 9: (560, 700), 10: (760, 700)})
        for f in range(10, 20):
            frames[f] = [dict(frames[f][0], track_id='cat')]
        self.assertEqual(body_motion.pose_events(frames, dict(seed=0, end=29), 60)['lost_frames'], 10)

    def test_one_shoulder_rising_is_not_cancelled(self):
        def joints(f):
            lift = 40 if 10 <= f <= 20 else 0  # shoulder width 200: a .2 rise
            return {1: (640, 170), 2: (700, 170), 5: (560, 420-lift), 6: (760, 420), 9: (560, 700), 10: (760, 700)}
        found = body_motion.pose_events(self.frames(joints), dict(seed=0, end=29), 60)
        moves = [(e['side'], e['rise']) for e in found['shoulders']]
        self.assertEqual(moves, [('화면 왼쪽', True), ('화면 왼쪽', False)]); self.assertEqual(found['turns'], [])
        self.assertEqual(found['hands_seen'], {'화면 왼쪽': 1.0, '화면 오른쪽': 1.0})

    def test_hand_points_without_a_wrist_are_not_a_hand(self):
        joints = lambda f: {**self.FACE, 1: (640, 170), 2: (700, 170), 5: (560, 420), 6: (780, 420), 112: (740, 190), 113: (742, 195)}
        self.assertEqual(body_motion.pose_events(self.frames(joints), dict(seed=0, end=29), 60)['hand_face'], [])

    def test_turn_drops_shoulder_heights(self):
        def joints(f):
            half = 100 if f < 10 or f > 20 else 30  # shoulders collapse to a third of their width and come back
            return {1: (640, 170), 2: (700, 170), 5: (660-half, 420-(30 if f == 15 else 0)), 6: (660+half, 420)}
        found = body_motion.pose_events(self.frames(joints), dict(seed=0, end=29), 60)
        self.assertTrue(found['turns']); self.assertEqual(found['shoulders'], [])

    def test_unseen_hands_are_counted(self):
        found = body_motion.pose_events(self.frames(lambda f: {1: (640, 170), 2: (700, 170), 5: (560, 420), 6: (760, 420)}), dict(seed=0, end=29), 60)
        self.assertEqual(found['hands_seen'], {'화면 왼쪽': 0.0, '화면 오른쪽': 0.0})

    def test_zigzag_ignores_jitter(self):
        self.assertEqual(body_motion._zigzag([0, 1, 2, 3, 2, 1, 0], 1.5), [(0, 3, 3), (3, 6, -3)])
        self.assertEqual(body_motion._zigzag([0, .1, 0, .1], 1), [])



class People(unittest.TestCase):
    @staticmethod
    def person(x, track, lean=0., wrist=None):
        """Upright figure centred at x (shoulder width 100); `lean` shifts the shoulders sideways over the hips."""
        j = {5: (x-50+lean, 300), 6: (x+50+lean, 300), 11: (x-40, 450), 12: (x+40, 450), 9: (x-60, 420), 10: (x+60, 420)}
        if wrist:
            j[10] = wrist
        return dict(keypoints(j), track_id=track, bbox=[x-80, 200, x+80, 700])

    def test_two_people_named_from_the_left_with_a_hand_on_a_shoulder(self):
        frames = {n: [self.person(700, 'b', wrist=(410, 320) if n >= 10 else None), self.person(400, 'a')] for n in range(30)}
        found = body_motion.people(frames, dict(seed=0, end=29), 60)
        self.assertEqual([(p['name'], p['track_id']) for p in found['people']], [('P1', 'a'), ('P2', 'b')])
        touch = found['relations'][0]['touch']
        self.assertEqual((touch[0]['hand'], touch[0]['body'], touch[0]['frames']), ('P2', 'P1', [10, 29]))

    def test_falling_backward_is_a_candidate_and_walking_closer_is_said(self):
        frames = {n: [self.person(400, 'a'), self.person(800, 'b', lean=0 if n < 15 else 300)] for n in range(30)}
        found = body_motion.people(frames, dict(seed=0, end=29), 60)
        self.assertIn('fall', found['people'][1]); self.assertNotIn('fall', found['people'][0])
        frames = {n: [self.person(400, 'a'), self.person(800-10*n, 'b')] for n in range(30)}
        self.assertTrue(body_motion.people(frames, dict(seed=0, end=29), 60)['relations'][0]['distance']['closer'])

    def test_one_person_has_no_relations(self):
        found = body_motion.people({n: [self.person(400, 'a')] for n in range(30)}, dict(seed=0, end=29), 60)
        self.assertEqual((len(found['people']), found['relations']), (1, []))


if __name__ == '__main__': unittest.main()
