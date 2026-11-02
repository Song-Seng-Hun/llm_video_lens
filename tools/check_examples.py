#!/usr/bin/env python3
"""Check that look still says what the README examples say, on the two README clips.

    python tools/check_examples.py --workspace W --video dancer=W/dancer-0-20.mp4 --video live2d=W/mao-sweep2.mp4

runs `look --max-shots 64` on each clip into a new folder (or reads one given with --look name=DIR) and checks
the README's window line. The clips are not in the repository (Pexels and Live2D terms), so each machine keeps its
own copies, identified by SHA-256 (docs/provenance.md, "README example images"). Run it before merging a change to
camera, pose, tracking or the motion text: unit tests passed on #83, which turned the Live2D clip's static camera into
a 3° rotation. Exit 0 when every check holds, 1 otherwise. A legitimate change updates the README and this together.
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = {
    # README "영상 하나를 요약해서 읽기": the quoted look.md line, read from the README itself (pieces between "…").
    'dancer': dict(sha256='cfd7de8fd37b5668876f3eb846b8b8d8a17a71dfbb508c0a03345b87775831c2', window='15.44–16.32s f386..407'),
    # README Live2D paragraph: camera unknown on the flat background; the left arm goes right 124 px, then 68 px more.
    'live2d': dict(sha256='4a62d052cb2a55250f409c01d5460e23845f86e4725e6c6fb10d150310f1aced', window='0.00–1.50s f0..44',
                   expect=['카메라 모름', '왼팔(화면 오른쪽', '오른쪽 124px', '→ 오른쪽 68px']),
}


def readme_quote(text):
    """The pieces of the README's quoted look.md line, split where it was shortened with "…"."""
    block = text.split('```text\n', 1)[1].split('\n```', 1)[0]
    line = block.strip().removeprefix('- ')
    return [piece.strip() for piece in line.split('…') if piece.strip()]


def window_line(look_md, window):
    return next((line for line in look_md.splitlines() if line.startswith('- '+window+' ')), None)


def check(name, look_md, readme_text):
    """[(expected piece, found)] for one example; a missing window line fails every piece."""
    example = EXAMPLES[name]
    expect = example.get('expect') or readme_quote(readme_text)
    line = window_line(look_md, example['window'])
    return [(piece, line is not None and piece in line) for piece in expect]


def run_look(workspace, video, name):
    out = f'check-{name}-{time.strftime("%Y%m%d-%H%M%S")}'
    done = subprocess.run([sys.executable, '-B', '-m', 'image_lens', '--workspace', workspace, 'look', '--video', video,
                           '--max-shots', '64', '--out', out], cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'{name}: look failed: {done.stdout[-300:]}{done.stderr[-300:]}')
    return Path(workspace)/out


def pairs(values):
    return dict(v.split('=', 1) for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--video', action='append', default=[], help='name=path for dancer and/or live2d')
    ap.add_argument('--look', action='append', default=[], help='name=existing look folder instead of a new run')
    a = ap.parse_args()
    videos, looks = pairs(a.video), pairs(a.look)
    names = [n for n in EXAMPLES if n in videos or n in looks]
    if not names or (set(videos) | set(looks)) - set(EXAMPLES):
        ap.error(f'give --video or --look for {", ".join(EXAMPLES)}')
    readme = (ROOT/'README.md').read_text()
    failed = 0
    for name in names:
        if name in videos:
            if hashlib.sha256(Path(videos[name]).read_bytes()).hexdigest() != EXAMPLES[name]['sha256']:
                raise SystemExit(f'{name}: {videos[name]} is not the README clip (SHA-256 differs)')
            folder = run_look(a.workspace, videos[name], name)
        else:
            folder = Path(looks[name])
        results = check(name, (folder/'look.md').read_text(), readme)
        bad = [piece for piece, found in results if not found]
        failed += bool(bad)
        print(f'{name}: {"ok" if not bad else "FAIL"} ({len(results)-len(bad)}/{len(results)}) {folder}')
        for piece in bad:
            print(f'  missing: {piece}')
        if bad and (line := window_line((folder/'look.md').read_text(), EXAMPLES[name]['window'])):
            print(f'  line: {line}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
