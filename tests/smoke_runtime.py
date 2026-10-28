"""Optional core-dependency smoke: authored video, real CLI, no neural models."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='new directory within repository/workspace')
    args = parser.parse_args()
    destination = Path(args.workspace).resolve()
    if not destination.is_relative_to(REPO/'workspace') or destination.exists():
        raise ValueError('new directory inside repository/workspace required')
    destination.mkdir(parents=True)
    import cv2
    import numpy as np
    source = destination/'control.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 24., (320, 240))
    if not writer.isOpened():
        raise RuntimeError('authored video writer unavailable')
    try:
        for n in range(48):
            image = np.zeros((240, 320, 3), np.uint8)
            image[:] = [(15,25,40), (200,210,220), (30,120,30)][n//16]
            cv2.rectangle(image, (30+n%16*3, 70), (90+n%16*3, 130), (240,50,50), -1)
            writer.write(image)
    finally:
        writer.release()
    class NoTorch:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'torch' or fullname.startswith('torch.'):
                raise RuntimeError('Torch import forbidden during core smoke')
    assert 'torch' not in sys.modules
    sys.meta_path.insert(0, NoTorch())
    from image_lens.cli import main as cli
    from image_lens import paths
    steps = []
    def run(arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli(['--workspace', str(destination), *arguments])
        steps.append(dict(arguments=arguments, returncode=status, stdout=stdout.getvalue(), stderr=stderr.getvalue()))
        if status:
            raise RuntimeError('CLI smoke failed: '+stderr.getvalue())
    run(['scan', '--video', 'control.mp4', '--out', 'shots.json'])
    scan = json.loads((destination/'shots.json').read_text())
    assert [event['after_frame'] for event in scan['events']] == [16, 32]
    run(['brief', '--shots', 'shots.json', '--out', 'brief.json'])
    run(['catalog', '--shots', 'shots.json', '--limit', '2', '--format', 'markdown', '--out', 'brief.md'])
    run(['compare', '--shots', 'shots.json', '--frames', '15,16', '--evidence-dir', 'comparison', '--out', 'comparison.json'])
    query = dict(schema='team-eye.queries/1', jobs=[dict(name='shot0', source_video=str(source), source_sha256=paths.sha(source),
                 width=320, height=240, fps=24., start_frame=0, end_frame=15, seed_frame=0,
                 points=[dict(point_id='surface0', xy=[40.,90.]), dict(point_id='unknown0', xy=None)])])
    paths.write_json('queries.json', query)
    run(['prepare', '--queries', 'queries.json', '--shots', 'shots.json', '--job', 'shot0', '--out', 'job.freeze.json'])
    run(['validate', '--queries', 'queries.json', '--frozen', 'job.freeze.json', '--job', 'shot0', '--out', 'unused-tracks.json'])
    assert 'torch' not in sys.modules and not (destination/'unused-tracks.json').exists()
    comparison = json.loads((destination/'comparison.json').read_text())
    assert comparison['automatic_redraw_decision'] is None and comparison['automatic_action_label'] is None
    assert comparison['used_for_segmentation'] is False
    brief = json.loads((destination/'brief.json').read_text())
    assert brief['coverage']['images_written'] == 0 and len(brief['shots']) == 3
    receipt = dict(status='passed', authored_video_frames=48, expected_and_detected_cuts=[16,32],
                   real_cli_steps=len(steps), steps=steps, torch_import_blocker_active=True,
                   torch_loaded=False, neural_forward_calls=0, checkpoint_required=False,
                   standalone_source_compare_panels=3, brief_images_written=0,
                   tracking_output_written=False, script_sha256=paths.sha(__file__))
    paths.write_json('smoke-results.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'steps'}))


if __name__ == '__main__':
    main()
