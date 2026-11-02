# Memory and time

Measured on one Mac Studio (Apple M4 Max, 16 cores, 64 GB unified memory, macOS) on 2026-11-03, master 897e3c5, `look --max-shots 64`, models already on disk and CoreML caches warm. Other machines were not measured; an 8 GB or 16 GB Mac has not been tried.

## Whole `look` runs

| Clip | Options | Wall time | Peak memory |
| --- | --- | ---: | ---: |
| README dancer: 20 s, 501 frames, 1280×674, 1 shot, 13 windows | default | 76 s | 2.0 GB |
| same | `--cuts thorough` | 76 s | 3.5 GB |
| README Live2D: 1.5 s, 45 frames, 1080×1440, 1 window | default | 8 s | 1.5 GB |
| same | `--detector` SAM3 bundle | 41 s | 4.9 GB |

Peak memory is the highest sum of resident memory (RSS) over every process of the run (main, pose child, tracker child), sampled every 0.3 s. Memory the GPU (MPS) holds may be counted only in part, so leave some room above these numbers.

## Per unit of work

| Step | Unit | Time |
| --- | --- | ---: |
| pose (HumanArt detector + RTMW, CoreML) | one frame | 20–27 ms |
| pose with the SAM3 bundle | one frame | about 0.7 s (45 frames in 33 s, model load included) |
| tracking (CoTracker, MPS) | one point group of up to 160 frames | about 0.7 s; the first group of a run about 2.4 s (model load) |
| tracking | one window (5 groups: head, arms, hands, legs, move) | about 4 s |

Tracking dominates a default run: on the dancer clip 62 groups took 53 s of the 76 s, pose 10–14 s. A clip's time grows with its number of windows (about one per 1–2 s of moving person) and with `--max-people`; `--tracking-resolution detail` is about 1.7× slower in tracking (see the guide).

## Disk

- Default models: 0.4 GB (detector 97 MB, pose 219 MB, CoTracker 97 MB).
- SAM3 bundle: 3.4 GB more. AutoShot weights for `--cuts thorough`: 57 MB, downloaded on first use.
- The CoreML cache in each workspace (`.cache/coreml`): about 0.5 GB, rebuilt when missing.

## How it was measured

Each run started `python -B -m image_lens --workspace W look --video V --max-shots 64 --out O` and, every 0.3 s, summed `ps` RSS over the processes whose command line names `O`. Step times come from `steps.log` (`elapsed_seconds` of pose, tracker group lines) and the run time from `look.json` `elapsed_seconds`.
