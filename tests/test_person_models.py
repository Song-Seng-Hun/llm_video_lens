"""Detector contract checks with synthetic data; no model inference or quality claims."""
import builtins
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from image_lens import person_models as pm, pose


class Coverage(unittest.TestCase):
    def test_primary_unchanged_and_only_original_primary_controls_coverage(self):
        import numpy as np
        primary = np.array([[0, 0, 10, 10]], np.float32); before = primary.copy()
        scores = np.array([.8], np.float32)
        # Partial overlap and containment reject; edge contact and disjoint proposals remain.
        # Two secondary proposals overlap each other: neither becomes a new primary veto.
        secondary = np.array([[9, 9, 20, 20], [2, 2, 4, 4], [10, 0, 20, 10], [11, 1, 19, 9]], np.float32)
        result, confidence = pm.add_disjoint(primary, scores, secondary, np.array([.9, .7, .6, .5], np.float32))
        np.testing.assert_array_equal(primary, before)
        np.testing.assert_array_equal(result, np.vstack((before, secondary[2:])))
        np.testing.assert_array_equal(confidence, np.array([.8, .6, .5], np.float32))

    def test_empty_primary_keeps_all_secondary(self):
        import numpy as np
        secondary = np.array([[1, 2, 3, 4]], np.float32); score = np.array([.4], np.float32)
        boxes, scores = pm.add_disjoint(np.empty((0, 4), np.float32), np.empty(0, np.float32), secondary, score)
        np.testing.assert_array_equal(boxes, secondary); np.testing.assert_array_equal(scores, score)

    def test_deepghs_threshold_native_geometry_and_stable_nms(self):
        import numpy as np
        # Exact .3 excluded; duplicate removed; the highest-score box maps to native pixels.
        rows = np.array([[320, 320, 320, 320, .9], [320, 320, 320, 320, .8], [40, 40, 20, 20, .3]], np.float32)
        boxes, scores = pm.anime_boxes(rows.T[None], 1280, 320)
        self.assertEqual(boxes.tolist(), [[320, 80, 960, 240]])
        self.assertAlmostEqual(float(scores[0]), .9, places=6)
        boxes, scores = pm.anime_boxes(np.zeros((1, 5, 0), np.float32), 640, 640)
        self.assertEqual(boxes.shape, (0, 4)); self.assertEqual(scores.shape, (0,))


class Selection(unittest.TestCase):
    def test_bad_models_and_bundles_fail_before_heavy_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'mlx', 'onnxruntime', 'rtmlib', 'PIL'}:
                raise AssertionError('heavy import before pin rejection: '+name)
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, mock.patch('builtins.__import__', side_effect=guarded):
            root = Path(folder).resolve()
            for name in ('wrong.onnx', 'model.safetensors'):
                file = root/name; file.write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                    pm.resolve(file)
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                pm.resolve(root)

    def test_all_bundle_pins_required_and_metadata_uses_actual_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with mock.patch.object(pm, '_asset', side_effect=lambda path, key: dict(path=str(path), sha256=pm.PINS[key][1], bytes=pm.PINS[key][0])) as asset:
                info = pm.resolve(root)
            self.assertEqual([call.args[1] for call in asset.call_args_list], ['sam3', 'tokenizer', 'deepghs'])
            self.assertEqual(info['sha256'], pm.PINS['sam3'][1])
            self.assertEqual(info['supplement']['sha256'], pm.PINS['deepghs'][1])
            self.assertEqual(info['tokenizer']['sha256'], pm.PINS['tokenizer'][1])
            self.assertEqual(info['recipe']['prompt'], 'person')

    def test_default_pin_and_imports_remain_compatible(self):
        self.assertEqual(pose.MODELS['detector'], pm.PINS['humanart'])
        code = "import sys; from image_lens import pose; assert not any(n.split('.')[0] in {'mlx','onnxruntime','rtmlib'} for n in sys.modules)"
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], check=True)

    def test_default_humanart_file_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            # Existing callers may use content-addressed model cache files without an extension.
            for name in ('detector.onnx', 'cached-model-blob'):
                with self.subTest(name=name):
                    file = Path(folder).resolve()/name; file.write_bytes(b'x')
                    with mock.patch.dict(pm.PINS, humanart=(1, 'hash')), mock.patch.object(pm.paths, 'sha', return_value='hash'):
                        info = pm.resolve(file)
                    self.assertEqual(info, dict(kind='humanart', path=str(file), sha256='hash', bytes=1))


class DetectorBound(unittest.TestCase):
    @staticmethod
    def detector(count):
        import numpy as np
        detector = pm.DeepGHS.__new__(pm.DeepGHS)
        rows = np.array([[8+i*18, 100, 4, 20, .8] for i in range(count)], np.float32)
        if count:
            rows[-1, 4] = .95
        output = rows.T[None]
        detector.input_name = 'images'
        detector.session = mock.Mock(run=mock.Mock(return_value=[output]))
        return detector, output

    def test_top32_stable_ties_and_explicit_counts(self):
        import numpy as np
        detector, output = self.detector(33)
        expected, confidence = pm.anime_boxes(output, 640, 640)
        boxes, scores = detector(np.zeros((640, 640, 3), np.uint8))
        np.testing.assert_array_equal(boxes, expected[:32])
        np.testing.assert_array_equal(scores, confidence[:32])
        self.assertEqual(boxes[0, 0], 8+32*18-2)
        self.assertEqual(boxes[1:, 0].tolist(), [8+i*18-2 for i in range(31)])
        self.assertEqual(detector.diagnostics[0]['original_count'], 33)
        self.assertEqual(detector.diagnostics[0]['kept_count'], 32)
        self.assertEqual(detector.diagnostics[0]['dropped_count'], 1)

    def test_at_or_below32_preserves_arrays_without_truncation_flag(self):
        import numpy as np
        for count in (1, 32):
            detector, output = self.detector(count)
            expected = pm.anime_boxes(output, 640, 640)
            actual = detector(np.zeros((640, 640, 3), np.uint8))
            for before, after in zip(expected, actual):
                self.assertEqual(before.tobytes(), after.tobytes())
            self.assertEqual(detector.diagnostics, [])
            flags = pose.flags([dict(frame=7, tracks=[], detector_diagnostics=detector.diagnostics)])
            self.assertEqual(flags['frames'], [])
            self.assertNotIn('detector_truncations', flags)

    def test_secondary_truncation_is_retained_when_primary_rejects_all_proposals(self):
        import numpy as np
        import types
        supplement, _ = self.detector(33)
        detector = pm.Sam3Coverage.__new__(pm.Sam3Coverage)
        detector.supplement = supplement
        state = dict(boxes=np.array([[0, 0, 640, 640]], np.float32), scores=np.array([.9], np.float32))
        detector.processor = mock.Mock(set_image=mock.Mock(return_value=state), set_text_prompt=mock.Mock(return_value=state))
        mlx = types.ModuleType('mlx'); core = types.ModuleType('mlx.core'); core.eval = lambda *args: None; mlx.core = core
        with mock.patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': core}):
            boxes, scores = detector(np.zeros((640, 640, 3), np.uint8))
        np.testing.assert_array_equal(boxes, state['boxes'])
        self.assertEqual(detector.diagnostics[0]['role'], 'supplement')
        self.assertEqual(detector.diagnostics[0]['dropped_count'], 1)

    def test_read_ahead_snapshot_keeps_diagnostic_on_its_native_frame(self):
        import cv2
        import numpy as np
        import onnxruntime
        import rtmlib
        import supervision
        from threading import Event
        ready = Event()
        diagnostic = dict(detector='deepghs', role='primary', original_count=33, kept_count=32, dropped_count=1, basis='synthetic cap')
        class Detector:
            diagnostics = []
            def __call__(self, image):
                if image[0, 0, 0] == 0:
                    self.diagnostics = [diagnostic]
                else:
                    diagnostic.clear(); self.diagnostics = []; ready.set()
                return np.tile([1, 1, 2, 6], (32, 1)), np.full(32, .9)
        class Capture:
            index = 0
            def isOpened(self): return True
            def set(self, prop, value): self.index = int(value)
            def read(self):
                image = np.full((8, 8, 3), self.index, np.uint8); self.index += 1
                return True, image
            def get(self, prop): return self.index if prop == cv2.CAP_PROP_POS_FRAMES else (self.index-1)*40.
            def release(self): pass
        def estimate(image, bboxes):
            if image[0, 0, 0] == 0:
                self.assertTrue(ready.wait(2), 'second detection must overwrite mutable state before first frame is consumed')
            return np.zeros((len(bboxes), 133, 2), np.float32), np.zeros((len(bboxes), 133), np.float32)
        scan = dict(scan_scope='full_source', source_video='/private/tmp/synthetic.bin', source_sha256='hash',
                    source_frame_count=2, width=8, height=8, fps=25., native_pts_seconds=[0., .04],
                    analysis_segments=[dict(segment_id='shot-0000', start_frame=0, end_frame=1)])
        estimator = mock.Mock(side_effect=estimate)
        session = mock.Mock(get_providers=mock.Mock(return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider']))
        tracker = mock.Mock(update_with_detections=mock.Mock(return_value=supervision.Detections.empty()))
        with mock.patch.object(pose.paths, 'load', return_value=(scan, Path('/private/tmp/shots.json'))), \
             mock.patch.object(pose.shot_boundaries, 'validate'), \
             mock.patch.object(pose.paths, 'confined', return_value=Path(scan['source_video'])), \
             mock.patch.object(pose.paths, 'sha', return_value='hash'), \
             mock.patch.object(pm, 'resolve', return_value=dict(kind='deepghs', path='/private/tmp/model', recipe={})), \
             mock.patch.object(pm, 'create', return_value=Detector()), \
             mock.patch.object(pose, '_model', return_value=Path('/private/tmp/pose')), \
             mock.patch.object(pose, '_coreml_cache', return_value='/private/tmp/cache'), \
             mock.patch.object(rtmlib, 'RTMPose', return_value=estimator), \
             mock.patch.object(onnxruntime, 'InferenceSession', return_value=session), \
             mock.patch.object(supervision, 'ByteTrack', return_value=tracker), \
             mock.patch.object(cv2, 'VideoCapture', return_value=Capture()):
            packet = pose.run('/private/tmp/shots.json', 'detector', 'pose')
        self.assertEqual(packet['frames'][0]['detector_diagnostics'][0]['dropped_count'], 1)
        self.assertNotIn('detector_diagnostics', packet['frames'][1])
        self.assertEqual(packet['flags']['frames'], [])
        self.assertEqual([row['frame'] for row in packet['flags']['detector_truncations']], [0])
        self.assertEqual(packet['flags']['detector_truncations'][0]['detector_diagnostics'][0]['original_count'], 33)


if __name__ == '__main__':
    unittest.main()
