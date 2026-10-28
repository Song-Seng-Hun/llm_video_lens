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

An optional installation exposes `image-lens`; extras are `.[core]` and `.[tracking]`. A stdlib-only installation has no mandatory runtime dependencies. Pinned PySceneDetect source is already vendored. The public CLI is the supported entrypoint; individual engine mains are internal adapters.

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

Analysis help also exposes review selection, explicit feature bindings, source-color region reports and optional foreground inspection. Read [limitations](docs/limitations.md) before interpretation.

## Verification and provenance

```sh
python -B -m unittest discover -s tests -v
# Optional core-dependency smoke, with a new output directory:
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

[Verification](docs/verification.md) separates tested CLI/guards from unrun inference and platform coverage. [Provenance and licenses](docs/provenance.md) lists original hashes, upstream pins, the restricted-loading patch and the authored-code licensing boundary. Historical experiment media, fixtures, ledgers, results and rejected automatic classifiers are not selected runtime paths.
