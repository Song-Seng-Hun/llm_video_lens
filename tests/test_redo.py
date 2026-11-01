import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from image_lens import digest, look, paths, redo


def packet(frames, kind=None):
    detector = dict(path='d.onnx', sha256='x', **(dict(kind=kind) if kind else {}))
    return dict(models=dict(detector=detector), frames=[dict(frame=n, tracks=[dict(bbox=list(b))]) for n, b in frames.items()])


class Redo(unittest.TestCase):
    def test_window_names_follow_the_file_names(self):
        record = dict(shots=[dict(segment_id='shot-0002', windows=[dict(window=dict(seed=40, end=199)), dict(window=dict(seed=200, end=230))])])
        self.assertEqual(redo.window(record, 'shot-0002-w1'), (200, 230))
        record['shots'][0]['windows'][0]['cut_after'] = 131
        self.assertEqual(redo.window(record, 'shot-0002-w0'), (40, 131))
        for bad in ('shot-0002-w2', 'shot-0003-w0', 'shot-0002'):
            with self.assertRaises(ValueError):
                redo.window(record, bad)

    def test_a_different_video_is_refused_before_look_runs(self):
        from unittest.mock import patch
        previous = paths.workspace_root()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); paths.configure(root)
            try:
                video = root/'v.avi'; video.write_bytes(b'now')
                first = root/'first'; first.mkdir()
                shots = [dict(segment_id='s', windows=[dict(window=dict(seed=0, end=9))])]
                paths.write_json(first/'look.json', dict(source_video=str(video), source_sha256=paths.sha(video), shots=shots))
                paths.write_json(first/'pose.json', dict(packet({}), source_sha256='other'))
                with patch('image_lens.look.run') as run, self.assertRaises(ValueError):
                    redo.run(first, 's-w0', 'd', 'p', 'c', root/'out')
                run.assert_not_called()
            finally:
                paths.configure(previous)

    def test_old_pose_packets_were_humanart(self):
        self.assertEqual(redo.kind(packet({})), 'humanart')
        self.assertEqual(redo.kind(packet({}, 'sam3-deepghs')), 'sam3-deepghs')
        self.assertEqual(redo.picks(200, 230), [200, 215, 230])
        self.assertEqual(redo.picks(5, 5), [5])

    def test_only_the_window_frames_are_looked_at(self):
        scan = dict(source_frame_count=300, analysis_segments=[dict(segment_id='a', start_frame=0, end_frame=99),
                                                               dict(segment_id='b', start_frame=100, end_frame=299)])
        self.assertEqual(look.clip(scan), scan['analysis_segments'])
        self.assertEqual([(s['segment_id'], s['start_frame'], s['end_frame']) for s in look.clip(scan, (120, 259))], [('b', 120, 259)])
        with self.assertRaises(ValueError):
            look.clip(scan, (250, 300))

    def test_boxes_picture_draws_both_boxes_on_each_frame(self):
        previous = paths.workspace_root()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); paths.configure(root)
            try:
                video = root/'v.avi'; writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (320, 180))
                for _ in range(5):
                    writer.write(np.full((180, 320, 3), 128, np.uint8))
                writer.release()
                before = packet({n: (10, 10, 100, 170) for n in range(5)}); after = packet({n: (150, 20, 300, 160) for n in (0, 4)})
                redo.boxes(video, before, after, [0, 2, 4], root/'b.jpg')
                image = cv2.imread(str(root/'b.jpg'))
                self.assertEqual(image.shape[1], 3*(redo.REDO['width']+4))
                k = redo.REDO['width']/320
                red, green = image[round(90*k), round(10*k)], image[round(90*k), round(150*k)]
                self.assertTrue(red[2] > 150 and red[1] < 100)
                self.assertTrue(green[1] > 150 and green[2] < 120)
                middle = image[round(90*k), redo.REDO['width']+4+round(150*k)]
                self.assertTrue(abs(int(middle[1])-128) < 20)  # no redo box on f2: none drawn
            finally:
                paths.configure(previous)

    def test_look_md_says_it_is_a_redo_of_one_window(self):
        record = dict(source_video='v.mp4', shots=[], check=[], frames=[200, 230],
                      redo=dict(look='first', window='shot-0002-w1', detector='sam3-deepghs', was='humanart', boxes='redo-boxes.jpg'))
        text = digest.markdown(record, dict(fps=25, source_frame_count=900), 'look-full.md')
        self.assertIn('first의 shot-0002-w1 구간(f200..230)만 검출기 sam3-deepghs로 다시 본 것(처음은 humanart)', text)
        self.assertIn('추적된 구간 0/31프레임', text)
        self.assertNotIn('다시 본 것', digest.markdown(dict(record, redo=None, frames=None), dict(fps=25, source_frame_count=900), 'look-full.md'))


if __name__ == '__main__':
    unittest.main()
