"""Internal tracker child: one job, or with --serve one job per stdin line. The parent owns the hard watchdog."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_lens import paths


def serve(entry, module):
    """One JSON argument list per line in, one {returncode, stdout, stderr} line out. Torch and the model stay loaded."""
    import contextlib
    import io
    import json
    import os
    import signal
    reply = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)  # native code printing to fd 1 must not break the reply line
    for line in sys.stdin:
        out, err, code = io.StringIO(), io.StringIO(), 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                entry(module, json.loads(line))
            except (Exception, SystemExit) as e:
                print(json.dumps({'status': 'failed', 'type': type(e).__name__, 'error': str(e)[:1200]}), file=sys.stderr)
                code = 1
            finally:
                signal.alarm(0)
        reply.write(json.dumps(dict(returncode=code, stdout=out.getvalue(), stderr=err.getvalue()))+'\n')
        reply.flush()
        if code:
            return  # a job cut off mid-inference may leave Torch state behind; the parent starts a fresh child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--serve', action='store_true')
    args, remaining = parser.parse_known_args()
    paths.configure(args.workspace)
    from image_lens.cli import selected_entry
    from image_lens import run_point_tracker
    if args.serve:
        serve(selected_entry, run_point_tracker)
    else:
        selected_entry(run_point_tracker, remaining)


if __name__ == '__main__':
    main()
