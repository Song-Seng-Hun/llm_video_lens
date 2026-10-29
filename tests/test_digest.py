import unittest
from image_lens import digest


def limb(pid, part, side, direction, distance, axis=0.):
    return dict(point_id=pid, part=part, side=side, axis_px=axis,
                strokes=[dict(direction=direction, distance_px=distance, frames=[1, 9])])


class Limbs(unittest.TestCase):
    def test_one_arm_moving_together_is_one_statement(self):
        body = dict(limbs=[limb('wrist-l', '손목', '화면 왼쪽', 'up_left', 48., 32.), limb('elbow-l', '팔꿈치', '화면 왼쪽', 'up_left', 35.),
                           limb('thumb-l', '엄지 끝', '화면 왼쪽', 'up_left', 57.), limb('move-c', '많이 움직인 곳 c', '화면 오른쪽', 'down_left', 54.)])
        found = digest.limbs(body, {'thumb-l': 'ease_out'})
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]['text'], '왼팔(손목·팔꿈치·엄지) 왼쪽 위 57px, 느려짐')
        self.assertTrue(found[1]['text'].startswith('움직인 점 c 왼쪽 아래 54px'))

    def test_different_directions_or_sizes_stay_apart(self):
        body = dict(limbs=[limb('wrist-r', '손목', '화면 오른쪽', 'left', 26.), limb('thumb-r', '엄지 끝', '화면 오른쪽', 'down_left', 28.),
                           limb('index-r', '검지 끝', '화면 오른쪽', 'left', 80.)])
        self.assertEqual(len(digest.limbs(body)), 3)


class Lines(unittest.TestCase):
    def test_whole_body_slide_is_said_once(self):
        done = dict(window=dict(seed=215, end=292), camera=dict(net=[2., 0.]),
                    body=dict(limit_px=12., centre_shift=dict(direction='left', distance_px=244.), limbs=[]),
                    groups=[dict(group='arms', moves=[dict(point='wrist-l', direction='left', distance_px=242., ease='ease_out')], warnings=1)])
        line = digest.window_line(done, 50)
        self.assertIn('전체: 몸 왼쪽 244px', line); self.assertNotIn('242', line); self.assertIn('팔다리 12px 미만', line)

    def test_bridge_says_the_path_is_unknown(self):
        line = digest.bridge_line(dict(frames=[61, 62], limbs=[dict(side='화면 오른쪽', part='손목', direction='up_left', distance_px=656.)]))
        self.assertIn('사이 경로 모름', line); self.assertIn('오른손목 왼쪽 위 656px', line)


if __name__ == '__main__': unittest.main()
