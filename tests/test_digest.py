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

    def test_bridge_keeps_the_other_hand(self):
        line = digest.bridge_line(dict(frames=[61, 62], limbs=[
            dict(side='화면 오른쪽', part='손목', direction='up_left', distance_px=656.),
            dict(side='화면 오른쪽', part='팔꿈치', direction='up', distance_px=444.),
            dict(side='화면 왼쪽', part='손목', direction='down_right', distance_px=294.)]))
        self.assertIn('왼손목 오른쪽 아래 294px', line); self.assertNotIn('444', line)

    def test_bands_going_opposite_ways_are_said(self):
        band = lambda name, dx, dy: dict(band=name, y=[0, 1], dx=dx, dy=dy, features=9)
        text = digest._bands([band('맨 위', 15., -16.), band('위', 2., 1.), band('가운데', -12., -13.),
                              band('아래', -20., -21.), band('맨 아래', 3., 2.)])
        self.assertTrue(text.endswith('(맨 위 오른쪽, 가운데·아래 왼쪽: 높이에 따라 좌우가 반대)'))
        self.assertNotIn('반대', digest._bands([band('맨 위', 15., 0.), band('아래', 12., 0.)]))


class Repeats(unittest.TestCase):
    def window(self, seed, wrist, motif='A'):
        return dict(window=dict(seed=seed, end=seed+14), motif=motif, groups=[],
                    body=dict(limit_px=12., centre_shift=dict(direction='down', distance_px=4.),
                              limbs=[limb('wrist-l', '손목', '화면 왼쪽', 'up', wrist)]))

    def test_repeat_says_only_what_differs(self):
        first = self.window(32, 26.)
        self.assertIn('자세 A 반복 (0.64s와 같음)', digest.window_line(self.window(93, 28.), 50, first))
        line = digest.window_line(self.window(339, 60.), 50, first)
        self.assertIn('다른 점: 왼팔(손목) 위 60px', line); self.assertIn('위 26px 없음', line)

    def test_label_swap_is_not_a_difference(self):
        first = self.window(32, 26.); again = self.window(93, 27.)
        again['body']['limbs'] = [limb('wrist-r', '손목', '화면 오른쪽', 'up', 27.)]
        self.assertIn('같음)', digest.window_line(again, 50, first))


    def test_neighbouring_direction_is_not_a_difference(self):
        first = self.window(32, 26.); again = self.window(93, 25.)
        again['body']['limbs'] = [limb('wrist-l', '손목', '화면 왼쪽', 'up_left', 25.)]
        self.assertIn('같음)', digest.window_line(again, 50, first))

if __name__ == '__main__': unittest.main()
