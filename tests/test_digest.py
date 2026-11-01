import unittest
from image_lens import digest


def limb(pid, part, side, direction, distance, axis=0., travel=None, speeds=None):
    return dict(point_id=pid, part=part, side=side, axis_px=axis, travel_px=distance if travel is None else travel,
                strokes=[dict(direction=direction, distance_px=distance, frames=[1, 9], speeds=speeds or [])])


class Limbs(unittest.TestCase):
    def test_one_arm_is_one_statement_led_by_the_farthest_travel(self):
        body = dict(limbs=[limb('wrist-l', '손목', '화면 왼쪽', 'up_left', 48., 32.), limb('elbow-l', '팔꿈치', '화면 왼쪽', 'up_left', 35.),
                           limb('thumb-l', '엄지 끝', '화면 왼쪽', 'up_left', 57., speeds=[12., 6., 2.]),
                           limb('move-c', '많이 움직인 곳 c', '화면 오른쪽', 'down_left', 54.)])
        found = digest.limbs(body)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]['text'], '왼팔(엄지·손목·팔꿈치) 왼쪽 위 57px(속도 12→6→2 px/f)')
        self.assertTrue(found[1]['text'].startswith('움직인 점 c 왼쪽 아래 54px'))

    def test_back_and_forth_beats_a_longer_net_move(self):
        # The wrist swings 30 px out and back three times (travel 180); the elbow drifts 40 px once.
        body = dict(limbs=[limb('wrist-r', '손목', '화면 오른쪽', 'left', 30., travel=180., speeds=[2., 9., 2.]),
                           limb('elbow-r', '팔꿈치', '화면 오른쪽', 'down', 40., speeds=[1., 4., 8.])])
        found = digest.limbs(body)
        self.assertEqual(len(found), 1); self.assertEqual(found[0]['point'], 'wrist-r')
        self.assertIn('속도 2→9→2', found[0]['text']); self.assertNotIn('1→4→8', found[0]['text'])

    def test_stroke_from_the_window_start_says_its_start_is_unseen(self):
        body = dict(limbs=[limb('wrist-r', '손목', '화면 오른쪽', 'up', 40., speeds=[20., 3.])])
        body['limbs'][0]['strokes'][0]['from_start'] = True
        self.assertIn('위 40px(속도 20→3 px/f, 앞선 동작은 모름)', digest.limbs(body)[0]['text'])


class Lines(unittest.TestCase):
    def test_whole_body_slide_is_said_once(self):
        done = dict(window=dict(seed=215, end=292), camera=dict(net=[2., 0.]),
                    body=dict(limit_px=12., centre_shift=dict(direction='left', distance_px=244.), limbs=[]),
                    groups=[dict(group='arms', moves=[dict(point='wrist-l', direction='left', distance_px=242., ease='ease_out')], warnings=1)])
        line = digest.window_line(done, 50)
        self.assertIn('전체: 몸 왼쪽 244px', line); self.assertNotIn('242', line); self.assertIn('팔다리 12px 미만', line)

    def test_points_moving_together_are_one_screen_move(self):
        moves = [dict(point='move-a', direction='right', distance_px=60., frames=[3, 12], speeds=[9., 4.]),
                 dict(point='move-b', direction='right', distance_px=55., frames=[4, 12], speeds=[2., 8.]),
                 dict(point='move-c', direction='down', distance_px=40., frames=[3, 9], ease=None)]
        done = dict(window=dict(seed=0, end=20), groups=[dict(group='a', moves=moves,
                    together=[dict(points=['move-a', 'move-b'], direction='right', frames=[4, 12])])])
        line = digest.window_line(done, 24)
        self.assertIn('화면: move-a 외 1점 함께 오른쪽 60px(속도 9→4 px/f); move-c 아래 40px', line); self.assertNotIn('2→8', line)

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
        done=dict(window=dict(seed=0,end=29),groups=[],onion_basis='screen',bands=[band('맨 위',15.,0.)])
        self.assertIn('높이별(화면 좌표, 카메라 모름)',digest.window_line(done,30))
        done['camera']=dict(known=.93,net=[200.,0.],until=28)
        line=digest.window_line(done,30)
        self.assertIn('카메라 오른쪽 200px',line)
        self.assertIn('높이별(화면 좌표, 카메라 일부 모름)',line)
        self.assertNotIn('화면 좌표, 카메라 모름',line)


    def test_framing_is_said_first(self):
        done = dict(window=dict(seed=0, end=29), framing='upper', scope='upper', groups=[],
                    body=dict(limit_px=12., centre_shift=dict(direction='down', distance_px=2.), limbs=[]))
        self.assertIn('| 상반신 (팔만) |', digest.window_line(done, 60))


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

    def test_known_repeat_centre_is_printed_when_first_was_unknown(self):
        first=self.window(0,26.);first['body']['centre_shift']=None
        again=self.window(32,26.);again['body']['centre_shift']=dict(direction='right',distance_px=200.)
        self.assertIn('몸 오른쪽 200px',digest.window_line(again,50,first))

    def test_label_swap_is_not_a_difference(self):
        first = self.window(32, 26.); again = self.window(93, 27.)
        again['body']['limbs'] = [limb('wrist-r', '손목', '화면 오른쪽', 'up', 27.)]
        self.assertIn('같음)', digest.window_line(again, 50, first))


    def test_neighbouring_direction_is_not_a_difference(self):
        first = self.window(32, 26.); again = self.window(93, 25.)
        again['body']['limbs'] = [limb('wrist-l', '손목', '화면 왼쪽', 'up_left', 25.)]
        self.assertIn('같음)', digest.window_line(again, 50, first))


class Events(unittest.TestCase):
    def test_hand_face_shoulders_and_hidden_hands_are_said(self):
        found = dict(hand_face=[dict(side='화면 오른쪽', frames=[60, 89], seconds=.5, region='볼', enters=False, exits=True)], lost_frames=12, turns=[],
                     shoulders=[dict(side='화면 왼쪽', rise=True, amount=.2, frames=[10, 20])], width=[dict(wider=True, share=.15, frames=[0, 30])],
                     hands_seen={'화면 왼쪽': .2, '화면 오른쪽': .9})
        text = ' | '.join(digest.events(found, 60))
        self.assertIn('오른손이 얼굴(볼) 처음부터 가까이 1.00–1.50s, 떨어짐', text); self.assertIn('12프레임 끊김', text); self.assertIn('왼어깨 올라감 0.20', text)
        self.assertIn('넓어짐 15%', text); self.assertIn('왼손 대부분 안 보임: 멈춤 아님, 모름', text)



class People(unittest.TestCase):
    def test_people_and_relations_are_said(self):
        found = dict(people=[dict(name='P1', move=dict(direction='left', widths=1.2), seen=1.),
                             dict(name='P2', move=dict(direction='down', widths=.1), seen=.6, fall=dict(frame=30, degrees=70))],
                     relations=[dict(pair=['P1', 'P2'], distance=dict(closer=True, widths=.8), overlap=True,
                                     touch=[dict(hand='P2', body='P1', frames=[60, 89])])])
        text = ' | '.join(digest.people(found, 60))
        self.assertIn('사람 2명', text); self.assertIn('P1 왼쪽 1.2폭', text); self.assertIn('쓰러짐 후보(0.50s', text)
        self.assertIn('P2 손이 P1 어깨·몸 쪽 1.00–1.50s', text); self.assertIn('다가감 0.8폭', text); self.assertIn('겹침', text)


if __name__ == '__main__': unittest.main()


class PhaseLines(unittest.TestCase):
    def test_steps_are_indented_under_the_window(self):
        done = dict(window=dict(seed=0, end=99), phases=[
            dict(frames=[0, 45], moved=[dict(part='wrist', side='화면 왼쪽', direction='up', amount=1., back=None)]),
            dict(frames=[45, 99], moved=[dict(part='knee', side='화면 오른쪽', direction='up', amount=.5, back=1.2)])])
        self.assertEqual(digest.phase_lines(done, 25.), ['  - 0.00–1.80s: 자세 단계(원본 화면·미보정): 왼손목 위 1.0', '  - 1.80–4.00s: 자세 단계(원본 화면·미보정): 오른무릎 위 0.5(최대 1.2 갔다 돌아옴)'])


class Posture(unittest.TestCase):
    def test_hand_touch_and_arm_posture_are_written(self):
        found = dict(hand_face=[], shoulders=[], width=[], hands_seen={}, lost_frames=0,
                     hand_body=[dict(side='화면 왼쪽', body='r', to='shoulder', to_side='화면 오른쪽', frames=[64, 74], enters=True, exits=True),
                                dict(to='hands', frames=[15, 17], enters=True, exits=True)])
        line = digest.window_line(dict(window=dict(seed=0, end=99), facing='front', events=found, groups=[],
                                       arms=[dict(frame=0, l='down', r='down', cross=[]), dict(frame=99, l='down', r='down', cross=[])]), 25.)
        self.assertIn('오른손(화면 왼쪽)이 왼어깨(화면 오른쪽) 가까이 옴 2.56–3.00s, 떨어짐', line)
        self.assertIn('두 손 모음 0.60–0.72s', line)
        self.assertIn('팔 자세(원본 화면·미보정): 양팔 허리 아래 그대로', line)

    def test_steps_end_with_the_arm_posture(self):
        done = dict(window=dict(seed=0, end=99), phases=[dict(frames=[0, 45], moved=[]), dict(frames=[45, 99], moved=[])],
                    arms=[dict(frame=0, l='down', r='down', cross=[]), dict(frame=45, l='up', r='up', cross=[]), dict(frame=99, l='mid', r=None, cross=[])])
        self.assertEqual(digest.phase_lines(done, 25.), ['  - 0.00–1.80s: 자세 단계(원본 화면·미보정): 팔다리 큰 이동 없음(몸통 길이 0.35배 미만) | 끝 팔: 양팔 어깨 위',
                                                      '  - 1.80–4.00s: 자세 단계(원본 화면·미보정): 팔다리 큰 이동 없음(몸통 길이 0.35배 미만) | 끝 팔: 왼팔 가슴 높이'])
        self.assertIn('팔 자세(원본 화면·미보정): 양팔 허리 아래', digest.window_line(dict(done, groups=[]), 25.))


class BodySides(unittest.TestCase):
    def test_model_side_comes_first_and_screen_side_follows_the_facing(self):
        arm = limb('wrist-l', '손목', '화면 왼쪽', 'up', 40.)
        arm['body'] = 'l'
        self.assertTrue(digest.limbs(dict(limbs=[arm]), 'front')[0]['text'].startswith('왼팔(화면 오른쪽: 손목) 위 40px'))
        self.assertTrue(digest.limbs(dict(limbs=[arm]), 'back')[0]['text'].startswith('왼팔(화면 왼쪽: 손목)'))
        self.assertTrue(digest.limbs(dict(limbs=[arm]))[0]['text'].startswith('왼팔(화면 왼쪽: 손목)'))  # facing unknown: where measured

    def test_back_facing_is_noted(self):
        line = digest.window_line(dict(window=dict(seed=0, end=20), facing='back', groups=[]), 25.)
        self.assertIn('뒤돈 자세', line); self.assertIn('뒤돎(사람 왼쪽=화면 왼쪽)', line)
