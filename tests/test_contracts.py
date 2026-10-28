"""Stdlib contract checks; synthetic packets are not quality evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_lens import paths, shot_boundaries, brief, jobs
(REPO/'workspace').mkdir(exist_ok=True)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=REPO/'workspace')
        self.root = Path(self.temp.name)
        paths.configure(self.root)
        self.video = self.root/'source.bin'
        self.video.write_bytes(b'authored source bytes; packet-contract test only')
        job = dict(source_video=str(self.video), source_sha256=paths.sha(self.video),
                   width=320, height=240, fps=24., start_frame=0, end_frame=47)
        self.scan = shot_boundaries.assemble(job, [n/24 for n in range(48)], [16, 17, 32])
        self.scan.update(scan_scope='full_source', source_frame_count=48)
        paths.write_json('shots.json', self.scan)

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *arguments):
        return subprocess.run([sys.executable, '-I', '-B', str(REPO/'image_lens_cli.py'),
                               '--workspace', str(self.root), *arguments],
                              cwd='/private/tmp' if Path('/private/tmp').exists() else self.root,
                              capture_output=True, text=True, timeout=10)

    def test_actual_isolated_cli_and_no_overwrite(self):
        result = self.cli('brief', '--shots', 'shots.json', '--out', 'brief.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = json.loads((self.root/'brief.json').read_text())
        self.assertEqual(packet['coverage']['images_written'], 0)
        self.assertEqual(packet['status'], 'verified_full_source')
        first = (self.root/'brief.json').read_bytes()
        again = self.cli('brief', '--shots', 'shots.json', '--out', 'brief.json')
        self.assertEqual(again.returncode, 1)
        self.assertEqual((self.root/'brief.json').read_bytes(), first)

    def test_unknown_cut_and_single_frame_semantics(self):
        result = brief.build('shots.json')
        self.assertEqual(len(result['shots']), 4)
        one = result['shots'][1]
        self.assertEqual([one['start_frame'], one['end_frame']], [16, 16])
        self.assertEqual(one['analysis_eligibility'], 'inspection_only_single_frame')
        self.assertFalse(one['identity_continues_from_previous'])
        for shot in result['shots']:
            self.assertIsNone(shot['action']); self.assertIsNone(shot['anatomy']); self.assertIsNone(shot['original_ease'])
            self.assertEqual(shot['motion_coverage'], 'unknown_no_admitted_report')
            self.assertTrue(shot['needs_fresh_source_observations'])
        self.assertFalse(self.scan['events'][0]['verified'])
        self.assertEqual(self.scan['events'][0]['kind'], 'brief_transition_ambiguous')

    def test_native_pts_and_bounded_pagination(self):
        result = brief.build('shots.json', offset=1, limit=2)
        self.assertEqual(result['coverage']['returned_segments'], 2)
        self.assertEqual(result['coverage']['omitted_segments'], 2)
        self.assertEqual(result['coverage']['next_offset'], 3)
        self.assertEqual(result['shots'][0]['start_seconds'], 16/24)
        self.assertNotIn('native_pts_seconds', result)
        with self.assertRaises(ValueError): brief.build('shots.json', limit=17)
        with self.assertRaises(ValueError): brief.build('shots.json', reports=['missing']*9)

    def test_incomplete_decode_refused(self):
        bad = copy.deepcopy(self.scan); bad['decode_complete'] = False
        paths.write_json('bad.json', bad)
        with self.assertRaisesRegex(ValueError, 'complete shot scan'): brief.build('bad.json')

    def test_partial_source_does_not_become_full(self):
        self.scan['scan_scope'] = 'partial_source'; paths.write_json('partial.json', self.scan)
        result = brief.build('partial.json')
        self.assertEqual(result['status'], 'verified_partial_source')
        self.assertFalse(result['coverage']['full_source'])

    def test_source_change_refused(self):
        self.video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source unavailable or changed'): brief.build('shots.json')

    def test_helper_provenance_change_refused(self):
        self.scan['detector_helper_sha256'] = '0'*64; paths.write_json('bad-pin.json', self.scan)
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'): brief.build('bad-pin.json')

    def test_outside_and_symlink_paths_refused(self):
        result = self.cli('brief', '--shots', 'shots.json', '--out', '../escaped.json')
        self.assertEqual(result.returncode, 1)
        link = self.root/'linked'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'): paths.confined('linked/shots.json', exists=True)
        result = self.cli('brief', '--shots', 'linked/shots.json', '--out', 'none.json')
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root/'none.json').exists())

    def test_duplicate_and_nonfinite_json_refused(self):
        paths.write('duplicate.json', '{"a":1,"a":2}'); paths.write('nan.json', '{"a":NaN}')
        for name in ['duplicate.json','nan.json']:
            with self.assertRaises(ValueError): paths.load(name)

    def test_optional_missing_or_untrusted_report_keeps_unknown(self):
        paths.write_json('untrusted.json', dict(schema='team-eye.motion-report/1', source_video_sha256='0'*64))
        result = brief.build('shots.json', reports=['missing.json', 'untrusted.json'])
        self.assertEqual(result['coverage']['admitted_report_count'], 0)
        self.assertTrue(all(r['status'] == 'not_admitted' for r in result['report_references']))
        self.assertTrue(all(s['motion_coverage'] == 'unknown_no_admitted_report' for s in result['shots']))

    def query(self, end=15):
        return dict(schema='team-eye.queries/1', jobs=[dict(name='shot0', source_video=str(self.video),
                    source_sha256=paths.sha(self.video), width=320, height=240, fps=24.,
                    start_frame=0, end_frame=end, seed_frame=0,
                    points=[dict(point_id='surface0', xy=[10., 20.]), dict(point_id='unknown0', xy=None)])])

    def test_fresh_freeze_and_mutation_refusal(self):
        paths.write_json('queries.json', self.query())
        receipt = jobs.prepare('queries.json', 'shots.json', 'job.freeze.json', 'shot0')
        self.assertTrue(receipt['validation_only']); self.assertFalse(receipt['model_loaded'])
        jobs.verify_frozen(self.root/'job.freeze.json', self.root/'queries.json')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            jobs.prepare('queries.json', 'shots.json', 'job.freeze.json', 'shot0')
        (self.root/'queries.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'query packet changed'):
            jobs.verify_frozen(self.root/'job.freeze.json', self.root/'queries.json')

    def test_cross_cut_new_job_refused(self):
        paths.write_json('queries.json', self.query(end=33))
        with self.assertRaisesRegex(ValueError, 'crosses a detected cut'):
            jobs.prepare('queries.json', 'shots.json', 'job.freeze.json', 'shot0')
        self.assertFalse((self.root/'job.freeze.json').exists())

    def test_prepare_imports_no_torch(self):
        paths.write_json('queries.json', self.query())
        script = '''import sys
sys.path.insert(0, sys.argv[1])
class NoTorch:
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch' or fullname.startswith('torch.'):
   raise RuntimeError('Torch imported by stdlib prepare')
sys.meta_path.insert(0, NoTorch())
from image_lens.cli import main
raise SystemExit(main(['--workspace', sys.argv[2], 'prepare', '--queries', 'queries.json', '--shots', 'shots.json', '--job', 'shot0', '--out', 'no-torch.freeze.json']))
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(REPO), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def observed_tracks(self, end=15, missing=None):
        job = self.query(end)['jobs'][0]
        job['points'] = [dict(point_id='surface0', observed_label='authored patch', anatomical_interpretation=None)]
        frames = []
        for n in range(end+1):
            anchor = n == 0
            usable = n != missing
            row = dict(point_id='surface0', xy=[40.+3*n, 90.], visible=usable,
                       model_visible=usable, in_frame=True,
                       status='anchor_input_not_prediction' if anchor else 'tracked_unverified',
                       claim_scope='visible_surface_2d', anatomical_interpretation=None,
                       query_index=0, anchor_index=0, anchor_frame=0,
                       tracking_metric_eligible=not anchor, metric_excluded_common_anchor_frame=anchor,
                       motion_pair_usable=not anchor and usable)
            frames.append(dict(frame=n, pts_seconds=n/24., tracking_metric_eligible=not anchor, points=[row]))
        return dict(schema='team-eye.observed-tracks/1', observation_scope='visible_surface_2d', anatomical_claims=False,
                    job=job, frames=frames, shot_check=self.scan,
                    fixture_note='Authored contract fixture; not neural inference or tracking-quality evidence.')

    def test_real_analysis_and_optional_brief_recomputation(self):
        paths.write_json('tracks.json', self.observed_tracks())
        result = self.cli('analyze', '--tracks', 'tracks.json', '--point', 'surface0',
                          '--start', '0', '--end', '15', '--out', 'timing.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root/'timing.json').read_text())
        packet = brief.build('shots.json', reports=['timing.json'])
        self.assertEqual(packet['coverage']['admitted_report_count'], 1)
        self.assertEqual(packet['report_references'][0]['observation']['label'], report['label'])
        self.assertIn('not neural inference', packet['report_references'][0]['source_fixture_note'])
        self.assertFalse(packet['report_references'][0]['model_quality_verified'])
        report['label'] = 'forged_action'
        paths.write_json('tampered.json', report)
        packet = brief.build('shots.json', reports=['tampered.json'])
        self.assertEqual(packet['coverage']['admitted_report_count'], 0)
        self.assertIn('differs from selected cached analysis', packet['report_references'][0]['reason'])

    def test_cross_cut_generation_stays_blocked_after_coordinate_crop_request(self):
        paths.write_json('cross.json', self.observed_tracks(end=47))
        result = self.cli('analyze', '--tracks', 'cross.json', '--point', 'surface0',
                          '--start', '0', '--end', '15', '--out', 'blocked.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root/'blocked.json').read_text())
        self.assertEqual(report['state'], 'requires_shot_local_retracking')
        self.assertEqual(report['label'], 'unknown')
        self.assertFalse(report['motion_inference'])
        packet = brief.build('shots.json', reports=['blocked.json'])
        self.assertEqual(packet['coverage']['admitted_report_count'], 0)

    def test_missing_observation_remains_unknown_in_brief(self):
        paths.write_json('gapped.json', self.observed_tracks(missing=8))
        result = self.cli('analyze', '--tracks', 'gapped.json', '--point', 'surface0',
                          '--start', '0', '--end', '15', '--out', 'unknown.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = brief.build('shots.json', reports=['unknown.json'])
        observation = packet['report_references'][0]['observation']
        self.assertEqual(observation['label'], 'unknown')
        self.assertIn('unusable_or_unknown_surface_observation', observation['reasons'])

    def test_actual_track_child_rejects_validation_only_freeze(self):
        paths.write_json('queries.json', self.query())
        jobs.prepare('queries.json', 'shots.json', 'job.freeze.json', 'shot0')
        result = self.cli('track', '--job', 'shot0', '--queries', 'queries.json',
                          '--frozen', 'job.freeze.json', '--out', 'tracks.json',
                          '--checkpoint', str(self.root/'missing-checkpoint.pth'))
        self.assertEqual(result.returncode, 1)
        self.assertIn('freeze permits validation only', result.stderr)
        self.assertFalse((self.root/'tracks.json').exists())
        self.assertFalse((self.root/'job.freeze.json.attempt.json').exists())

    @unittest.skipUnless(sys.platform != 'win32', 'POSIX process-group watchdog')
    def test_hard_parent_watchdog_kills_and_reaps_real_child(self):
        import os
        from image_lens import watchdog
        with self.assertRaises(watchdog.ChildTimeout) as caught:
            watchdog.run([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(10)'], .15)
        pid = caught.exception.child_pid
        with self.assertRaises(ChildProcessError): os.waitpid(pid, os.WNOHANG)
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_parent_interruption_kills_and_reaps_child(self):
        import os
        from unittest.mock import patch
        from image_lens import watchdog
        child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(10)'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        communicate = child.communicate
        interrupted = False
        def interrupt_once(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return communicate(*args, **kwargs)
        with patch.object(watchdog.subprocess, 'Popen', return_value=child), patch.object(child, 'communicate', side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                watchdog.run(['unused'], 1)
        self.assertIsNotNone(child.returncode)
        with self.assertRaises(ChildProcessError): os.waitpid(child.pid, os.WNOHANG)
        with self.assertRaises(ProcessLookupError): os.kill(child.pid, 0)

    def test_repeat_cli_invocations_select_current_workspace(self):
        from image_lens.cli import main
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--workspace', str(self.root), 'brief', '--shots', 'shots.json', '--out', 'first.json']), 0)
        self.assertEqual(shot_boundaries.workspace_root(), self.root)
        self.assertIsNone(shot_boundaries.context({'shot_check':None}))


    # Readability additions: trajectory guard and stroke recomputation in brief.
    def bindings(self):
        return dict(schema='team-eye.motion-bindings/1', source_tracks_sha256=paths.sha(self.root/'tracks.json'),
                    basis='source_frame_grouping_only',
                    actors=[dict(actor_id='A', label='authored patch', features=[dict(point_id='surface0', label='patch')])])

    def test_trajectory_dir_requires_report_bindings(self):
        paths.write_json('tracks.json', self.observed_tracks())
        result = self.cli('analyze', '--tracks', 'tracks.json', '--point', 'surface0', '--start', '0', '--end', '15',
                          '--trajectory-dir', 'traj', '--out', 'timing.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --report-bindings', result.stdout+result.stderr)
        self.assertFalse((self.root/'traj').exists() or (self.root/'timing.json').exists())

    def test_pose_requires_trajectory_dir(self):
        paths.write_json('tracks.json', self.observed_tracks())
        paths.write_json('bindings.json', self.bindings())
        paths.write_json('pose.json', dict(schema='image-lens.pose/1'))
        result = self.cli('analyze', '--tracks', 'tracks.json', '--report-bindings', 'bindings.json',
                          '--start', '0', '--end', '15', '--pose', 'pose.json', '--out', 'report.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--pose requires --trajectory-dir', result.stdout+result.stderr)
        self.assertFalse((self.root/'report.json').exists())

    def test_brief_carries_recomputed_strokes(self):
        paths.write_json('tracks.json', self.observed_tracks())
        paths.write_json('bindings.json', self.bindings())
        result = self.cli('analyze', '--tracks', 'tracks.json', '--report-bindings', 'bindings.json',
                          '--start', '0', '--end', '15', '--out', 'report.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = brief.build('shots.json', reports=['report.json'])
        strokes = packet['report_references'][0]['observation']['features'][0]['strokes']
        self.assertEqual([s['direction'] for s in strokes], ['right'])
        report = json.loads((self.root/'report.json').read_text())
        report['features'][0]['continuous_runs'][0]['strokes']['units'][0]['direction'] = 'left'
        paths.write_json('forged.json', report)
        packet = brief.build('shots.json', reports=['forged.json'])
        self.assertEqual(packet['coverage']['admitted_report_count'], 0)


if __name__ == '__main__': unittest.main()
