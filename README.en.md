<p align="center"><a href="README.md">한국어</a> · <b>English</b> · <a href="README.ja.md">日本語</a></p>

# LLM Image Lens

A source-bound video inspection CLI for an LLM: locate shot boundaries, request a small original-frame contact, and read conservative 2D surface-motion reports. The bounded improvement in this release is a compact shot brief/catalog, with native frame/PTS ranges, source hashes, missing coverage and exact raw-artifact locators. No pose, action, identity or easing accuracy improvement is claimed.

Core workflows retain the selected existing engines: diagnostic frame comparison, canonical full-source hard-cut context, surface-point timing, explicit actor/feature reports, source-color regions, and optional flat-background foreground inspection. Default image output is **zero**. Explicit source contacts share an eight-frame cap; comparison uses two originals and one derived residual panel.

## Run without installing

Requires Python 3.11+ for stdlib commands. Decode/render commands additionally need `requirements.txt`; dependencies, checkpoints and media are not bundled. Python 3.12 on macOS arm64 is the verified extraction environment. Windows core operation and installed-wheel execution remain unverified; experimental tracking requires a POSIX parent watchdog.

From the repository:

```sh
python -m image_lens --help
python -m image_lens --workspace /path/to/workspace scan --video input.mp4 --out shots.json
python -m image_lens --workspace /path/to/workspace brief --shots shots.json --out brief.json
python -m image_lens --workspace /path/to/workspace catalog --shots shots.json --offset 0 --limit 8 --format markdown --out brief.md
python -m image_lens --workspace /path/to/workspace compare --shots shots.json --frames 5,6 --evidence-dir comparison --out comparison.json
```

From a different working directory, including isolated Python, use the absolute path to the thin direct launcher:

```sh
python -I /path/to/llm_image_lens/image_lens_cli.py --workspace /path/to/workspace brief --shots shots.json --out brief.json
```

An optional installation exposes `image-lens`; extras are `.[core]`, `.[tracking]` and `.[pose]`. A stdlib-only installation has no mandatory runtime dependencies. Pinned PySceneDetect source is already vendored. The public CLI is the supported entrypoint; individual engine mains are internal adapters.

All relative data/output paths are relative to the selected workspace, default `<repository>/workspace`. Source inputs must be copied there explicitly. Inputs and outputs cannot escape this boundary; symlink components and overwrites are refused. The optional checkpoint is the sole explicit external input exception. Workspaces, videos, models, generated images and results are Git-ignored.

## Read the brief

`brief` and `catalog` emit the same bounded schema, `image-lens.brief/1`. Eight shot rows are returned by default (maximum 16); `next_offset` and omitted counts make pagination explicit. Native PTS arrays stay at the raw scan locator. Source bytes and current detector helper/source pins must match before a verified brief is issued. Partial-source scans remain explicitly partial; incomplete decode packets are rejected. Hash checks bind bytes and provenance, not the correctness of detected cuts.

An optional `--report report.json` adds only supported cached motion reports that can be recomputed from their pinned source tracks and bindings under the exact canonical scan. Missing, outdated, unsupported or cross-cut reports are listed as `not_admitted`; their coverage remains unknown. Eight report references maximum. No LLM/API call, network, image generation, raw-coordinate dump or model import is performed by a brief. Labels in reports are untrusted quoted data.

Every shot retains its fresh-source-observation/retracking obligation. A single-frame shot permits inspection only. A missed cut or dissolve is possible; no detected cut does not prove continuity or identity. Do not crop coordinates from a whole-clip offline inference and call them shot-local tracking.

## Fresh jobs and optional tracking

See [the job contract](docs/jobs.md). `prepare` accepts a fresh query packet and canonical full-source scan, validates one shot-local job, and creates a new immutable freeze. Its default permits validation only. `validate` checks source bytes, queries, vendor/helper pins, decode/PTS and cuts **without importing Torch**, without a checkpoint and without producing tracks. It requires the core decode dependencies.

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --out job.freeze.json
python -m image_lens --workspace /path/to/workspace validate --queries queries.json --frozen job.freeze.json --job shot0 --out unused-tracks.json
```

Experimental inference requires a separately prepared freeze with `--allow-tracking`, an explicit local checkpoint and a frozen wall budget. Its single attempt is consumed immediately before the model import; failures never trigger automatic retries. The CLI parent kills/reaps its own child process group if the budget expires. No download occurs. CoTracker source and checkpoint have noncommercial upstream terms. No neural inference was run to validate this extraction.

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --allow-tracking --max-seconds 120 --out tracking.freeze.json
python -m image_lens --workspace /path/to/workspace track --queries queries.json --frozen tracking.freeze.json --job shot0 --max-seconds 120 --checkpoint /path/to/scaled_offline.pth --out tracks.json
```

For observed-surface tracks, selected analysis is available through the real engine:

```sh
python -m image_lens --workspace /path/to/workspace analyze --tracks tracks.json --shots shots.json --point surface0 --start 0 --end 30 --out timing.json
python -m image_lens analyze --help
```

For an LLM reader, add `--report-bindings bindings.json --format markdown --trajectory-dir traj` to a report. The Markdown then leads with coarse strokes (median-filtered path split at corners and pauses, major = at least half the largest unit), and `traj/trajectory.png` shows the whole-frame path (blue early, red late), a stroke/timeline strip and up to four uncropped frames. The same pass lists appearance-change candidates: consecutive-frame gray-difference jumps to check with `compare --frames a,b`. A major stroke that overlaps a candidate is flagged, because echo/redraw can make the point jump. Strokes are recomputed in `brief`; the picture and candidates are not.

## Whole-body seeds from pose (optional)

`pose` runs a local whole-body pose model over the canonical scan: YOLOX HumanArt person boxes, RTMW 133 keypoints (rtmlib, ONNX Runtime CPU) and ByteTrack IDs reset at every detected cut. The two ONNX files are user-supplied and pinned by size and SHA256; nothing is downloaded or bundled, and the weight license is not established. Install `requirements-pose.txt` into the same environment. The packet also flags frames where the pose layer itself is unreliable: overlapping person boxes (echo copies, afterimages) and body-skeleton jumps within one track.

`seed` turns one frame of that packet into fresh observation jobs, one per group so each fits the 12-point surface contract: `<job>-head` (nose, eyes, front/top/side hair), `<job>-arms` and `<job>-legs`. Hair points are geometry on a head frame built from the eyes (distance and the up direction away from the nose), so they are candidate surface points, not detected hair. Joints near the frame edge are dropped because off-screen joints come back clamped there. Each seed gets a marked source crop as its pinned visual reference; look at them before tracking. `bind` writes the report bindings for one tracked job.

```sh
python -m image_lens --workspace W pose --shots shots.json --detector /models/yolox_humanart.onnx --pose-model /models/rtmw133.onnx --out pose.json
python -m image_lens --workspace W seed --pose pose.json --frame 215 --job s7 --out s7.obs.json --evidence-dir s7-crops
# prepare/track each of s7-head, s7-arms, s7-legs, then:
python -m image_lens --workspace W bind --queries s7.obs.json --tracks s7-head.tracks.json --job s7-head --out s7-head.bindings.json
python -m image_lens --workspace W analyze --tracks s7-head.tracks.json --shots shots.json --report-bindings s7-head.bindings.json --start 215 --end 307 --format markdown --trajectory-dir s7-head-traj --pose pose.json --out s7-head.md
```

With `--pose`, the trajectory view marks pose-trouble frames (orange on the timeline), tags change candidates near them, and groups warnings so one line names every big stroke that coincides with a candidate or a pose-trouble range. With more than two points, only major strokes are labelled, and the Markdown leads with a one-line-per-point summary plus "moved together" lines (major strokes of several points with the same direction, overlapping time and similar size); per-point speed windows, phases and span comparisons stay in the JSON. Pupils are not modelled; eye points follow the drawn eye area.

Analysis help also exposes review selection, explicit feature bindings, source-color region reports and optional foreground inspection. Read [limitations](docs/limitations.md) before interpretation.

## Verification and provenance

```sh
python -B -m unittest discover -s tests -v
# Optional core-dependency smoke, with a new output directory:
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

[Verification](docs/verification.md) separates tested CLI/guards from unrun inference and platform coverage. [Provenance and licenses](docs/provenance.md) lists original hashes, upstream pins, the restricted-loading patch and the authored-code licensing boundary. Historical experiment media, fixtures, ledgers, results and rejected automatic classifiers are not selected runtime paths.
