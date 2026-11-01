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

    def test_extended_actor_retry_checks_unfiltered_prefix_and_preserves_real_solo(self):
        import copy
        poses = self.frames(0, 99)
        for n in range(8): poses[n] = self.sleeve()
        segment = dict(start_frame=0, end_frame=99)
        solo = look.next_window(segment, 0, poses, set(), frame_size=(1440, 1200), context_frames=poses)
        self.assertEqual(solo[0], dict(seed=8, end=99))
        context = copy.deepcopy(poses); context[4].append(track('other'))
        crowded = look.next_window(segment, 0, poses, set(), frame_size=(1440, 1200), context_frames=context)
        self.assertEqual(crowded[0], dict(seed=0, end=99))



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

    def test_selected_view_preserves_observations_without_supported_camera_endpoints(self):
        """Unknown endpoints or insufficient camera support keep the full raw screen report."""
        from unittest.mock import patch
        from image_lens import paths
        packet,raw,_=self.view([(400+3*n,300) for n in range(26)],None,missing=18)
        binding=dict(schema='team-eye.motion-bindings/1',source_tracks_sha256='0'*64,basis='source_frame_grouping_only',
                     actors=[dict(actor_id='marked',label='marked',features=[dict(point_id='hand',label='hand')])])
        full={n:(float(n),0.,1.) for n in range(26)}
        invalid=dict(full);invalid[12]=(float('nan'),0.,1.)
        partials=[('none',None),('empty',{}),('seed_only',{0:full[0]}),
                  ('sparse_endpoints',{n:full[n] for n in (0,25)}),
                  ('end_gap',{n:v for n,v in full.items() if n!=25}),('nonfinite',invalid)]
        for name,shifts in partials:
            with self.subTest(case=name),patch.object(paths,'load',side_effect=[(packet,None),(binding,None)]):
                selected=look._motion_view(raw,'tracks','binding',shifts)
                self.assertTrue(selected is raw,'unsupported camera must retain the raw screen report')
                self.assertEqual(selected['coordinate_scope'],'screen_2d')
                self.assertEqual(selected['features'][0]['usable_prediction_frames'],24)
                self.assertEqual([r['frames'] for r in selected['features'][0]['continuous_runs']],[[1,17],[19,25]])
        with patch.object(paths,'load',side_effect=[(packet,None),(binding,None)]):
            selected=look._motion_view(raw,'tracks','binding',full)
        self.assertEqual(selected['coordinate_scope'],'camera_translation_relative_2d')
        self.assertEqual(selected['features'][0]['usable_prediction_frames'],24)
        with patch.object(paths,'load',side_effect=[(packet,None),(binding,None)]):
            selected=look._motion_view(raw,'tracks','binding',{n:v for n,v in full.items() if n!=12})
        self.assertEqual(selected['coordinate_scope'],'camera_translation_relative_2d')
        self.assertEqual(selected['coordinate_transform']['unsupported_frames'],[[12,12]])
        self.assertEqual([r['frames'] for r in selected['features'][0]['continuous_runs']],[[1,11],[13,17],[19,25]])
        self.assertEqual(raw['features'][0]['usable_prediction_frames'],24)


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


class MultipleActors(unittest.TestCase):
    def test_observed_actor_ranking_cap_and_native_shot_scope(self):
        import copy
        a = track('a'); b = track('b'); c = track('ghost'); d = track('d'); late = track('late-large')
        c['bbox'] = [0, 0, 2000, 2000]
        late['bbox'] = [0, 0, 1500, 1500]
        poses = {0: [a, b, c, d], 1: [a, b, d, late], 2: [a, b, d, late], 3: [track('next-shot')]}
        before = copy.deepcopy(poses)
        selected, omitted = look.select_actors(dict(start_frame=0, end_frame=2), poses, 2)
        self.assertEqual([p['track_id'] for p in selected], ['a', 'b'])
        self.assertEqual([p['label'] for p in selected], ['A1', 'A2'])
        self.assertEqual({p['track_id']: p['reason'] for p in omitted},
                         {'ghost': 'fewer than2 actual observations', 'd': '--max-people cap', 'late-large': '--max-people cap'})
        self.assertEqual(poses, before)
        selected, _ = look.select_actors(dict(start_frame=3, end_frame=3), poses, 2)
        self.assertEqual(selected, [])

    def test_actor_subset_ignores_overlap_and_other_actor_jumps_only(self):
        import copy
        rows = []
        for n in range(16):
            a, b = track('a'), track('b')
            if n == 1:
                b['keypoints'] = [[x+1000, y] for x, y in b['keypoints']]
            rows.append(dict(frame=n, tracks=[a, b]))
        packet = dict(native_range=[0, 15], frames=rows, flags=pose.flags(rows)); before = copy.deepcopy(packet)
        segment = dict(start_frame=0, end_frame=15)
        selected = look.actor_packet(packet, segment, 'a')
        other = look.actor_packet(packet, segment, 'b')
        self.assertTrue(any(r['overlapping_people'] for r in packet['flags']['frames']))
        self.assertEqual(selected['flags']['frames'], [])
        self.assertTrue(any(r.get('skeleton_jump') for r in other['flags']['frames']))
        self.assertEqual(selected['actor_selection']['context_pose_flags'], packet['flags'])
        frames = {r['frame']: r['tracks'] for r in selected['frames']}
        self.assertEqual(look.next_window(segment, 1, frames, set(), min_frames=2)[0]['seed'], 1)
        self.assertTrue(all([p['track_id'] for p in ps] == ['a'] for ps in frames.values()))
        self.assertEqual(packet, before)

    def run_people(self, directory, fail_second=False, weak_prefix=False, budget_after_first=False, short_second=False):
        """Actual look.run routing with fake public inference steps; spies reject geometry borrowing."""
        import copy, json, subprocess
        from contextlib import ExitStack
        from unittest.mock import patch
        from image_lens import paths, body_motion, camera, onion, motif, zoom, check, motion_report
        clock = dict(expired=False)
        paths.configure(directory); root = Path(directory); video = root/'source.bin'; video.write_bytes(b'synthetic')
        segments = [dict(segment_id=f'shot-{k:04d}', start_frame=16*k, end_frame=16*k+15) for k in range(2)]
        rows = []
        for n in range(32):
            a, b = track('a'), track('b')  # Deliberately reuse strings across cuts: no reassociation/bridge allowed.
            a['bbox'] = [0, 0, 300, 500]; b['bbox'] = [100, 0, 350, 500]
            if weak_prefix and n%16 < 4:
                a['keypoints'][pose.BODY['wrist_l']] = [900., 600.]
                b['keypoints'][pose.BODY['wrist_l']] = [900., 600.]
            rows.append(dict(frame=n, tracks=[a]+([b] if not short_second or n%16 >= 10 else [])))
        packet = dict(schema='image-lens.pose/1', source_video=str(video), source_sha256='synthetic',
                      width=1440, height=1200, fps=10, native_range=[0, 31], frames=rows, flags=pose.flags(rows))
        original = copy.deepcopy(packet); seed_calls = []; geometry = []; onion_boxes = []
        def value(arguments, flag): return arguments[arguments.index(flag)+1]
        def step(command, arguments):
            out = Path(value(arguments, '--out')) if '--out' in arguments else None
            if command == 'scan':
                out.write_text(json.dumps(dict(fps=10, width=1440, height=1200, source_frame_count=32,
                                               source_sha256='synthetic', analysis_segments=segments)))
            elif command == 'seed':
                identity = value(arguments, '--track'); selected = json.loads(Path(value(arguments, '--pose')).read_text())
                self.assertTrue(all(all(p['track_id'] == identity for p in r['tracks']) for r in selected['frames']))
                seed_calls.append((identity, value(arguments, '--frame')))
                if fail_second and identity == 'b': raise ValueError('no usable seeds for b')
                out.write_text(json.dumps(dict(seed_source=dict(track_id=identity))))
                return dict(jobs=[value(arguments, '--job')+'-'+group for group in ('arms', 'move')], skeleton=dict(plausible=True))
            elif command == 'analyze':
                selected = json.loads(Path(value(arguments, '--pose')).read_text())
                self.assertEqual(len({p['track_id'] for r in selected['frames'] for p in r['tracks']}), 1)
                out.write_text(json.dumps(dict(features=[])))
            elif command == 'bind': out.write_text(json.dumps(dict(actors=[dict(features=[dict(point_id='wrist-l')])])))
            return dict(jobs=[])
        def child(arguments):
            Path(value(arguments, '--out')).write_text(json.dumps(packet))
            return subprocess.Popen([sys.executable, '-c', 'pass'], stdout=subprocess.PIPE, text=True)
        def submit(arguments):
            job = value(arguments, '--job'); lo = 0 if 'shot-0000' in job else 16
            raw = dict(frames=[dict(frame=n, points=[dict(point_id='wrist-l', visible=True, xy=[100., 100.])]) for n in range(lo, lo+16)])
            Path(value(arguments, '--out')).write_text(json.dumps(raw))
            if budget_after_first and job == 'shot-0000-w0-move':
                clock['expired'] = True
            return job
        def exact_view(view, window):
            identities = {p['track_id'] for ps in view.values() for p in ps}
            self.assertEqual(len(identities), 1)
            self.assertEqual(identities, {window['track_id']})
            geometry.append((next(iter(identities)), window['seed']))
        def describe(packets, window, view, **kwargs): exact_view(view, window)
        def events(view, window, fps): exact_view(view, window)
        def phases(view, window, fps): exact_view(view, window); return []
        def facing(view, window): exact_view(view, window)
        def arms(view, window, frames): exact_view(view, window); return []
        def reconcile(raw, view, identity, window, *args):
            exact_view(view, window); self.assertEqual(identity, window['track_id']); return raw
        def bands(source, seed, end, bbox, *args, **kwargs): onion_boxes.append((seed, bbox)); return []
        with ExitStack() as stack:
            replacements = [(look, '_step', step), (look, '_pose_child', child), (look, '_submit', submit),
                            (look, '_finish', lambda handle: None), (look, 'fit', lambda source, root: (source, None)),
                            (look, '_motion_view', lambda found, *args: found),
                            (look, '_summary', lambda *args: ([], dict(group='arms', moves=[], major_moves=[], together=[], warnings=0, lost=None))),
                            (look.joint_tracking, 'reconcile', reconcile), (body_motion, 'describe', describe),
                            (body_motion, 'pose_events', events), (body_motion, 'phases', phases), (body_motion, 'facing', facing),
                            (body_motion, 'arm_poses', arms), (body_motion, 'geometry_policy', lambda *args: {}),
                            (body_motion, 'geometry_note', lambda *args: None),
                            (body_motion, 'people', lambda *args: dict(people=[])),
                            (camera, 'path', lambda *args: None), (camera, 'plain', lambda *args: False),
                            (motion_report, 'markdown', lambda *args: 'raw report'),
                            (onion, 'body_path', lambda *args: None), (onion, 'body_basis', lambda *args: None),
                            (onion, 'parts', lambda *args: ['wrist-l']),
                            (onion, 'trails', lambda *args: []), (onion, 'bands', bands), (onion, 'peak', lambda *args: None),
                            (onion, 'references', lambda *args, **kwargs: ([], [])), (onion, 'spacing', lambda *args, **kwargs: None),
                            (onion, 'lines', lambda *args, **kwargs: []), (onion, 'sheets', lambda *args: []),
                            (motif, 'thumb', lambda *args: None), (zoom, 'overview', lambda *args: []),
                            (check, 'sheets', lambda *args: None)]
            for module, name, replacement in replacements: stack.enter_context(patch.object(module, name, replacement))
            if budget_after_first:
                stack.enter_context(patch.object(look.time, 'monotonic', side_effect=lambda: 1801. if clock['expired'] else 0.))
            bridge = stack.enter_context(patch.object(body_motion, 'bridge', wraps=body_motion.bridge))
            look.run(video, 'det', 'pose', 'checkpoint', root/'look', max_people=2)
            bridge.assert_not_called()
        self.assertEqual(packet, original)
        return json.loads((root/'look/look.json').read_text()), seed_calls, geometry, onion_boxes

    def test_actor_extended_retry_does_not_treat_crowd_as_solo(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as directory:
            _, seeds, _, _ = self.run_people(directory, weak_prefix=True)
            self.assertEqual(seeds, [('a', 0), ('b', 0), ('a', 16), ('b', 16)])

    def test_actor_budget_reason_keeps_completed_window(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as directory:
            record, seeds, _, _ = self.run_people(directory, budget_after_first=True)
            self.assertEqual(seeds, [('a', 0)])
            first = record['shots'][0]
            self.assertEqual(len(first['windows']), 1)
            self.assertEqual(first['windows'][0]['source_track_id'], 'a')
            self.assertEqual(first['actor_accounting']['completed'], 1)
            second = first['actor_accounting']['actors'][1]
            self.assertEqual(second['status'], 'unknown')
            self.assertEqual(second['reason'], 'overall time budget exhausted')
            error, = first['errors']
            self.assertEqual((error['source_track_id'], error['actor_label']), ('b', 'A2'))

    def test_actor_skips_share_labels_in_summary_full_focus_and_at(self):
        import cv2
        import tempfile
        from unittest.mock import Mock, patch
        from image_lens import check, digest, zoom
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as directory:
            record, _, _, _ = self.run_people(directory, short_second=True)
            skipped = record['shots'][0]['skipped'][0]
            self.assertEqual((skipped['source_track_id'], skipped['actor_label']), ('b', 'A2'))
            line = '- 건너뜀 A2 · 모델 ID b f10..15: fewer than 12 frames after the seed'
            root = Path(directory)/'look'
            text = (root/'look.md').read_text()
            self.assertIn('한 사람 이상 추적된 구간 32/32프레임', text)
            self.assertIn('사람별 건너뜀 2곳', text)
            self.assertIn(line, text)
            self.assertIn(line, (root/'look-full.md').read_text())
            self.assertIn(line, digest.focus(root, 'shot-0000'))
            with patch.object(cv2, 'VideoCapture', return_value=Mock()), \
                 patch.object(zoom, '_write', return_value=[]):
                receipt = check.at(root, '0-1.5', Path(directory)/'at')
            self.assertIn(line, Path(receipt['output']).read_text())

    def test_actual_entry_tracks_each_id_with_own_geometry_and_context(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as directory:
            record, seeds, geometry, boxes = self.run_people(directory)
            self.assertEqual(seeds, [('a', 0), ('b', 0), ('a', 16), ('b', 16)])
            self.assertTrue(geometry)
            self.assertEqual(boxes, [(0, [0, 0, 300, 500]), (0, [100, 0, 350, 500]),
                                     (16, [0, 0, 300, 500]), (16, [100, 0, 350, 500])])
            for shot in record['shots']:
                self.assertEqual([w['source_track_id'] for w in shot['windows']], ['a', 'b'])
                self.assertEqual(shot['actor_accounting']['completed'], 2)
                self.assertTrue(all(w['context_pose_flags'] for w in shot['windows']))
                self.assertTrue(all(w['source_native_range'] == shot['frames'] for w in shot['windows']))
                for window in shot['windows']:
                    self.assertEqual(set(window['pose_diagnostic_bases']), {'events','phases','facing','arms'})
                    self.assertTrue(all(basis['coordinate_scope']=='screen_2d' and not basis['camera_corrected']
                                        for basis in window['pose_diagnostic_bases'].values()))
            text = (Path(directory)/'look/look.md').read_text()
            self.assertIn('A2 · 모델 ID b', text); self.assertNotIn('P2 · 모델 ID b', text)
            # Each actor's windows sit under its own heading, so A2 restarting at the shot's first second reads as a new person.
            self.assertEqual(text.count('### A1 · 모델 ID a (동일인 미검증)'), 2)
            self.assertEqual(text.count('### A2 · 모델 ID b (동일인 미검증)'), 2)
            self.assertIn('사람별 분석(장면마다 최대 2명): A1·A2…는', text)
            self.assertIn('P1…은 화면 기준 문맥 순서로 A와 별개', text)
            self.assertIn('32/32프레임', text)
            from image_lens import digest
            self.assertIn(digest.POSE_BASIS,text)
            self.assertIn(digest.POSE_BASIS,(Path(directory)/'look/look-full.md').read_text())
            import copy
            diagnostic=dict(window=dict(seed=0,end=15),groups=[],phases=[dict(frames=[0,14],moved=[])],
                            facing='back',arms=[dict(frame=0,l='down',r='down'),dict(frame=14,l='up',r='up')])
            before=copy.deepcopy(diagnostic)
            self.assertIn('원본 화면·미보정',digest.window_line(diagnostic,10))
            self.assertIn('자세 단계(원본 화면·미보정)',digest.phase_lines(diagnostic,10)[0])
            self.assertEqual(diagnostic,before)
            self.assertNotIn('64/32프레임', text)

    def test_missing_selected_seed_stays_unknown_and_never_substitutes_another_id(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=REPO/'workspace') as directory:
            record, seeds, _, boxes = self.run_people(directory, fail_second=True)
            self.assertEqual([identity for identity, _ in seeds], ['a', 'b', 'a', 'b'])
            self.assertEqual([bbox for _, bbox in boxes], [[0, 0, 300, 500]]*2)
            for shot in record['shots']:
                self.assertEqual(shot['actor_accounting']['completed'], 1)
                self.assertEqual(shot['actor_accounting']['actors'][1]['status'], 'unknown')
                self.assertEqual(shot['windows'][1]['source_track_id'], 'b')
                self.assertEqual(shot['windows'][1]['groups'], [])

    def test_cli_default_and_opt_in_bounds(self):
        from unittest.mock import patch
        from image_lens import cli
        args = ['--video', 'v', '--detector', 'd', '--pose-model', 'p', '--checkpoint', 'c', '--out', 'o']
        with patch.object(cli, 'models', side_effect=lambda value, *roles: value), patch.object(look, 'run', return_value={}) as run:
            cli.dispatch('look', args); self.assertEqual(run.call_args.kwargs, {})
            cli.dispatch('look', args+['--max-people', '1']); self.assertEqual(run.call_args.kwargs, {})
            cli.dispatch('look', args+['--max-people', '2']); self.assertEqual(run.call_args.kwargs, dict(max_people=2))
            for value in ('0', '9'):
                with self.assertRaisesRegex(ValueError, '--max-people'):
                    cli.dispatch('look', args+['--max-people', value])
