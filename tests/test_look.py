"""Shot-window selection for the one-command look; synthetic, no models."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import look, pose
sys.path.insert(0, str(REPO/'tests'))
from test_pose import track


class Windows(unittest.TestCase):
    def frames(self, n0, n1):
        return {n: [track()] for n in range(n0, n1+1)}

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

    def sleeve(self):
        bad = track(); bad['keypoints'][pose.BODY['wrist_l']] = [900., 600.]  # 3D clip f368: a sleeve read as a long arm
        return [bad]

    def test_a_bad_skeleton_at_the_seed_moves_it_to_the_next_good_frame(self):
        poses = self.frames(0, 99); poses[0] = self.sleeve()
        self.assertEqual(look.windows(dict(start_frame=0, end_frame=99), poses, set())[0], [dict(seed=1, end=99)])

    def test_the_seed_stays_when_no_good_frame_is_close(self):
        poses = self.frames(0, 99)
        for n in range(4):
            poses[n] = self.sleeve()
        self.assertEqual(look.windows(dict(start_frame=0, end_frame=99), poses, set())[0], [dict(seed=0, end=99)])


    def retry(self, poses, flagged=None, end=99, frame_size=(1440,1200)):
        return look.next_window(dict(start_frame=0,end_frame=end),0,poses,flagged or set(),frame_size=frame_size)

    def test_delayed_solo_body_retry_uses_strict_pose_and_reports_prefix(self):
        poses=self.frames(0,99)
        for n in range(8):poses[n]=self.sleeve()
        window,skipped=self.retry(poses)
        self.assertEqual(window,dict(seed=8,end=99));self.assertEqual(skipped[0]['frames'],[0,7])

    def test_extended_retry_preserves_next_valid_window_and_minimum_suffix(self):
        poses=self.frames(0,319)
        for n in list(range(8))+list(range(168,320)):poses[n]=self.sleeve()
        windows,skipped=look.windows(dict(start_frame=0,end_frame=319),poses,set(),frame_size=(1440,1200))
        self.assertEqual(windows,[dict(seed=8,end=159),dict(seed=160,end=319)])
        self.assertEqual(skipped[0]['frames'],[0,7])
        first,_=look.next_window(dict(start_frame=0,end_frame=319),0,poses,set(),length=12,frame_size=(1440,1200))
        self.assertEqual(first,dict(seed=0,end=11))

    def test_extended_retry_requires_full_single_id_native_support(self):
        for case in ('foreign','crowd','gap','flag','onegood','weak','upper','chibi','late','short_tail','no_dimensions'):
            poses=self.frames(0,99)
            for n in range(8):poses[n]=self.sleeve()
            flagged=set();end=99;size=(1440,1200)
            if case=='foreign':
                for n in range(8,100):poses[n][0]['track_id']='other'
            elif case=='crowd':poses[4].append(track('other'))
            elif case=='gap':poses.pop(4)
            elif case=='flag':flagged.add(4)
            elif case=='onegood':
                for n in range(9,100):poses[n]=self.sleeve()
            elif case=='weak':
                for n in range(8,100):poses[n][0]['scores'][pose.BODY['wrist_l']]=.29
            elif case=='upper':
                for rows in poses.values():
                    rows[0]['scores'][15]=rows[0]['scores'][16]=.1
            elif case=='chibi':
                for rows in poses.values():
                    rows[0]['keypoints'][1][0]=480.;rows[0]['keypoints'][2][0]=120.
            elif case=='late':
                for n in range(13):poses[n]=self.sleeve()
            elif case=='short_tail':end=18
            elif case=='no_dimensions':size=None
            with self.subTest(case=case):
                self.assertEqual(self.retry(poses,flagged,end,size)[0],dict(seed=0,end=end))



class Summary(unittest.TestCase):
    def test_no_observation_is_unknown_not_still(self):
        report = dict(features=[dict(point_id='hip-l', continuous_runs=[])])
        lines, _ = look._summary(report, None, 'legs', 'x', 'y')
        self.assertEqual(len(lines), 1); self.assertIn('모름', lines[0])

    def span(self, a, b):
        return dict(frames=[a, b], strokes=dict(units=[]))

    def test_group_cut_off_early_is_unknown_not_still(self):
        report = dict(features=[dict(point_id=p, continuous_runs=[self.span(340, 353)]) for p in ('elbow-l', 'elbow-r', 'wrist-l')])
        lines, summary = look._summary(report, dict(seed=339, end=383), 'arms', 'x', 'y')
        self.assertIn('큰 이동 없음', lines[0]); self.assertIn('f353 뒤로 점 3/3개 끊김', lines[1])
        self.assertEqual(summary['lost'], dict(count=3, total=3, after=353))

    def test_one_point_dropping_is_not_a_cut(self):
        report = dict(features=[dict(point_id='a', continuous_runs=[self.span(0, 99)]), dict(point_id='b', continuous_runs=[self.span(0, 99)]),
                                dict(point_id='c', continuous_runs=[self.span(0, 40)])])
        lines, summary = look._summary(report, dict(seed=0, end=99), 'arms', 'x', 'y')
        self.assertEqual(lines, []); self.assertIsNone(summary['lost'])


class Cut(unittest.TestCase):
    def packet(self, last_visible, seed=0, end=99):
        return dict(frames=[dict(frame=n, points=[dict(point_id=p, visible=n <= e) for p, e in last_visible.items()]) for n in range(seed, end+1)])

    def test_group_dropping_together_cuts_the_window(self):
        arms = self.packet(dict(a=40, b=41, c=99)); head = self.packet(dict(x=99, y=99))
        self.assertEqual(look.cut_frame([head, arms], dict(seed=0, end=99)), 41)

    def test_no_cut_when_points_last_to_the_end(self):
        self.assertIsNone(look.cut_frame([self.packet(dict(a=99, b=95, c=30))], dict(seed=0, end=99)))

    def test_a_cut_over_a_still_picture_is_not_a_new_drawing(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            rng = np.random.default_rng(3)
            room = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (3, 3), 0)
            video = Path(d)/'cut.avi'; out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
            for k in range(8):
                out.write(room if k < 5 else 255-room)  # the drawing changes after frame 4
            out.release()
            from image_lens import motion_view
            self.assertLess(look.changed_after(video, 1), motion_view.CHANGE['min_mean_abs_diff'])
            self.assertGreaterEqual(look.changed_after(video, 4), motion_view.CHANGE['min_mean_abs_diff'])
            self.assertGreaterEqual(look.changed_after(video, 3), motion_view.CHANGE['min_mean_abs_diff'])  # one frame early still sees it

    def test_reseed_starts_after_the_cut(self):
        poses = {n: [track()] for n in range(0, 200)}
        window, _ = look.next_window(dict(start_frame=0, end_frame=199), 42, poses, {42, 43})
        self.assertEqual(window, dict(seed=44, end=199))


class Camera(unittest.TestCase):
    def view(self, screen, shifts, missing=None):
        from image_lens import analyze_motion, motion_report
        frames=[]
        for n,xy in enumerate(screen):
            anchor=n==0; visible=n!=missing
            p=dict(point_id='hand',xy=list(xy),visible=visible,model_visible=visible,in_frame=True,
                   status='anchor_input_not_prediction' if anchor else 'tracked_unverified',
                   claim_scope='visible_surface_2d',anatomical_interpretation=None,query_index=0,
                   anchor_index=0,anchor_frame=0,tracking_metric_eligible=not anchor,
                   metric_excluded_common_anchor_frame=anchor,motion_pair_usable=not anchor and visible)
            frames.append(dict(frame=n,pts_seconds=n/25.,tracking_metric_eligible=not anchor,points=[p]))
        packet=dict(schema='team-eye.observed-tracks/1',observation_scope='visible_surface_2d',anatomical_claims=False,
                    job=dict(width=1920,height=1080,points=[dict(point_id='hand')]),frames=frames)
        binding=dict(schema='team-eye.motion-bindings/1',source_tracks_sha256='0'*64,basis='source_frame_grouping_only',
                     actors=[dict(actor_id='marked',label='marked',features=[dict(point_id='hand',label='hand')])])
        import copy
        original=copy.deepcopy(packet)
        raw=motion_report.build_report(packet,binding,0,len(screen)-1,analyze_motion.collect_observations,analyze_motion.infer)
        view=motion_report.camera_relative(raw,packet,binding,shifts,analyze_motion.collect_observations,analyze_motion.infer)
        self.assertEqual(packet,original)
        return packet,raw,view

    def test_opposite_motion_uses_the_same_coordinates_as_onion_and_mark(self):
        from image_lens import mark,onion
        screen=[(400+40*max(0,n-1)/24,300) for n in range(26)]
        shifts={n:(100*max(0,n-1)/24,0.,1.) for n in range(26)}
        packet,raw,view=self.view(screen,shifts)
        self.assertEqual(raw['features'][0]['continuous_runs'][0]['net_screen_direction'],'right')
        run=view['features'][0]['continuous_runs'][0]
        self.assertAlmostEqual(run['net_dx_px'],-60.)
        self.assertEqual(run['net_screen_direction'],'left')
        lines,summary=look._summary(view,dict(seed=0,end=25),'marked','raw.md','raw.png')
        self.assertIn('왼쪽',lines[0]);self.assertNotIn('오른쪽',lines[0])
        self.assertEqual(summary['coordinate_scope'],'camera_translation_relative_2d')
        self.assertIn('왼쪽',' '.join(mark._all_moves(view)))
        trail=onion.trails([packet],['hand'],0,25,shifts)[0]['xy']
        self.assertAlmostEqual(trail[-1][1]-trail[0][1],run['net_dx_px'])
        self.assertEqual(view['coordinate_transform']['camera_path'][-1],[25,100.,0.])

    def test_pan_and_returning_pan_do_not_reappear_in_all_moves(self):
        from image_lens import mark
        for pan in ([4*n for n in range(26)],[8*min(n,25-n) for n in range(26)]):
            with self.subTest(pan=pan):
                _,_,view=self.view([(400+x,300) for x in pan],{n:(x,0.,1.) for n,x in enumerate(pan)})
                _,summary=look._summary(view,dict(seed=0,end=25),'marked','x','y')
                self.assertEqual(summary['moves'],[])
                self.assertFalse(any(u['direction']!='hold' for r in view['features'][0]['continuous_runs'] for u in r['strokes']['units']))
                self.assertNotIn('오른쪽',' '.join(mark._all_moves(view)))

    def test_camera_gaps_and_raw_gaps_are_not_bridged_or_filled(self):
        screen=[(400+3*n,300) for n in range(26)]
        shifts={n:(n,0.,1.) for n in range(26) if n!=12}
        _,_,view=self.view(screen,shifts,missing=18)
        runs=view['features'][0]['continuous_runs']
        self.assertEqual([r['frames'] for r in runs],[[1,11],[13,17],[19,25]])
        self.assertIn('camera_translation_unknown',view['features'][0]['break_reasons'])
        self.assertEqual(view['features'][0]['excluded_anchor_ranges'],[[0,0]])
        _,_,view=self.view(screen,{0:(0.,0.,1.)})
        lines,summary=look._summary(view,dict(seed=0,end=25),'marked','x','y')
        self.assertTrue(summary['motion_unknown']);self.assertIn('모름',' '.join(lines))

    def test_partial_camera_support_cannot_claim_the_missing_motion_was_still(self):
        from image_lens import mark
        screen=[(400 if n<6 else 400+100*(n-5)/14 if n<20 else 500,300) for n in range(26)]
        shifts={n:(0.,0.,1.) for n in list(range(6))+list(range(20,26))}
        _,_,view=self.view(screen,shifts)
        lines,summary=look._summary(view,dict(seed=0,end=25),'marked','x','y')
        self.assertTrue(summary['incomplete']);self.assertIn('빈 구간 움직임은 모름',' '.join(lines))
        self.assertIn('빈 구간 움직임 모름',' '.join(mark._all_moves(view)))

    def test_real_camera_numpy_scalar_output_is_supported(self):
        import numpy as np
        _,_,view=self.view([(400+4*n,300) for n in range(26)],{n:(np.float64(4*n),np.float64(0),1.) for n in range(26)})
        self.assertEqual(view['features'][0]['usable_prediction_frames'],25)
        self.assertAlmostEqual(view['features'][0]['continuous_runs'][0]['net_dx_px'],0.)

    def test_no_camera_path_keeps_explicit_screen_scope(self):
        _,raw,view=self.view([(400+3*n,300) for n in range(26)],None)
        self.assertIs(view,raw)
        _,summary=look._summary(view,dict(seed=0,end=25),'marked','x','y')
        self.assertEqual(summary['coordinate_scope'],'screen_2d')


if __name__ == '__main__': unittest.main()


class Ahead(unittest.TestCase):
    def look(self, d, said, code=0, track=None):
        """look._run on a 60-frame pan with three 20-frame shots; the fake pose child writes parts 0..39 and 40..59 and
        pose.json, then prints `said` (a list of lines, or None for the two part lines). Returns the seed calls.
        With a `track` list, each seed queues one job, and submits and takes are logged to it (every take fails)."""
        import json, subprocess, sys
        from image_lens import paths
        from tests.test_check import pan_video
        paths.configure(d)
        video = Path(d)/'pan.avi'; pan_video(video, frames=60)
        segments = [dict(segment_id=f'shot-{k:04d}', start_frame=20*k, end_frame=20*k+19) for k in range(3)]
        person = [dict(track_id='t', bbox=[100, 50, 300, 350], score=.9, keypoints=[[150+j%20*5, 60+j//20*40] for j in range(133)], scores=[.9]*133)]
        def packet(lo, hi):
            return json.dumps(dict(native_range=[lo, hi], frames=[dict(frame=n, tracks=person) for n in range(lo, hi+1)], flags=dict(frames=[])))
        calls = []
        def step(command, arguments):
            if command == 'seed':
                calls.append((arguments[arguments.index('--frame')+1], Path(arguments[arguments.index('--pose')+1]).name))
                if track is not None:
                    track.append(('seed', arguments[arguments.index('--frame')+1]))
            out = Path(arguments[arguments.index('--out')+1]) if '--out' in arguments else None
            if command == 'scan':
                out.write_text(json.dumps(dict(fps=10, width=640, height=360, source_frame_count=60, source_sha256='0', analysis_segments=segments)))
            elif command == 'seed' and arguments[arguments.index('--frame')+1] == 40:
                raise ValueError('no seed')
            if command == 'seed' and track is not None:
                return dict(jobs=[arguments[arguments.index('--job')+1]+'-head'])
            return dict(jobs=[])
        def submit(arguments):
            track.append(('submit', arguments[arguments.index('--job')+1])); return arguments[arguments.index('--job')+1]
        def finish(handle):
            track.append(('take', handle)); raise ValueError('no tracker here')
        def pose_child(arguments):
            parts = Path(arguments[arguments.index('--parts-dir')+1]); parts.mkdir()
            lines = said
            if lines is None:
                lines = []
                for lo, hi in ((0, 39), (40, 59)):
                    (parts/f'pose-f{lo}-f{hi}.json').write_text(packet(lo, hi)); lines.append(json.dumps(dict(pose_part=str(parts/f'pose-f{lo}-f{hi}.json'))))
                Path(arguments[arguments.index('--out')+1]).write_text(packet(0, 59))
            return subprocess.Popen([sys.executable, '-c', f'import sys; print({chr(10).join(["load model"]+lines)!r}); sys.exit({code})'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        plain = look._step, look.fit, look._pose_child, look._submit, look._finish
        look._step, look.fit, look._pose_child = step, lambda source, root: (source, False), pose_child
        if track is not None:
            look._submit, look._finish = submit, finish
        try:
            look.run(video, 'det', 'pose', 'ckpt', Path(d)/'look')
        finally:
            look._step, look.fit, look._pose_child, look._submit, look._finish = plain
        return calls

    def test_shots_are_seeded_from_their_pose_part_once_and_a_failure_stays_in_place(self):
        import json, tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            calls = self.look(d, None)
            self.assertEqual(calls, [(0, 'pose-f0-f39.json'), (20, 'pose-f0-f39.json'), (40, 'pose-f40-f59.json')])
            record = json.loads((Path(d)/'look'/'look.json').read_text())
            self.assertEqual([len(s['errors']) for s in record['shots']], [0, 0, 1])
            self.assertEqual(record['shots'][2]['errors'][0]['step'], 'seed')
            text = (Path(d)/'look'/'look-full.md').read_text()
            self.assertLess(text.index('## shot-0002'), text.index('점 배치 실패: no seed'))
            self.assertIn('load model', (Path(d)/'look'/'steps.log').read_text())

    def test_the_next_shot_is_queued_before_this_window_is_taken(self):
        import json, tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            track = []; self.look(d, None, track=track)
            # Shot 2's part is already heard when shot 1 is queued: it is seeded ahead too (and fails in place).
            self.assertEqual(track, [('seed', 0), ('submit', 'shot-0000-w0-head'), ('seed', 20), ('submit', 'shot-0001-w0-head'),
                                     ('take', 'shot-0000-w0-head'), ('seed', 40), ('take', 'shot-0001-w0-head')])
            record = json.loads((Path(d)/'look'/'look.json').read_text())
            self.assertEqual([[e.get('job') for e in s['errors']] for s in record['shots']], [['shot-0000-w0-head'], ['shot-0001-w0-head'], [None]])

    def test_a_failed_pose_child_fails_the_look_with_its_error(self):
        import json, tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as d:
            with self.assertRaisesRegex(ValueError, 'pose wall budget exhausted'):
                self.look(d, [json.dumps(dict(status='failed', error='pose wall budget exhausted'))], code=1)
