"""Internal one-job child. The public track CLI owns its hard parent watchdog."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_lens import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    args, remaining = parser.parse_known_args()
    paths.configure(args.workspace)
    from image_lens.cli import selected_entry
    from image_lens import run_point_tracker
    selected_entry(run_point_tracker, remaining)


if __name__ == '__main__':
    main()
