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


if __name__ == '__main__':
    unittest.main()
