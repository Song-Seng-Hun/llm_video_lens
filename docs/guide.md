# LLM Video Lens: full guide

The [README](README.en.md) gives the overview and a quick start. This page keeps the full reference: every command, the brief format, jobs and tracking, how to read `look.md`, and pose seeds.

A source-bound video inspection CLI for an LLM: locate shot boundaries, request a small original-frame contact, and read conservative 2D surface-motion reports. The bounded improvement in this release is a compact shot brief/catalog, with native frame/PTS ranges, source hashes, missing coverage and exact raw-artifact locators. No pose, action, identity or easing accuracy improvement is claimed.

Core workflows retain the selected existing engines: diagnostic frame comparison, canonical full-source hard-cut context, surface-point timing, explicit actor/feature reports, source-color regions, and optional flat-background foreground inspection. Default image output is **zero**. Explicit source contacts share an eight-frame cap; comparison uses two originals and one derived residual panel.

## Run without installing

Requires Python 3.11+ for stdlib commands. Decode/render commands additionally need `requirements/base.txt`; dependencies, checkpoints and media are not bundled. Python 3.12 on macOS arm64 is the verified extraction environment. Windows core operation and installed-wheel execution remain unverified; experimental tracking requires a POSIX parent watchdog.

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
python -I /path/to/llm_video_lens/tools/image_lens_cli.py --workspace /path/to/workspace brief --shots shots.json --out brief.json
```

An optional installation exposes `image-lens`; extras are `.[core]`, `.[tracking]`, `.[pose]` and `.[cuts]`. A stdlib-only installation has no mandatory runtime dependencies. Pinned PySceneDetect source is already vendored. The public CLI is the supported entrypoint; individual engine mains are internal adapters.

All relative data/output paths are relative to the selected workspace, default `<repository>/workspace`. Source inputs must be copied there explicitly. Inputs and outputs cannot escape this boundary; symlink components and overwrites are refused. The optional checkpoint is the sole explicit external input exception. Workspaces, videos, models, generated images and results are Git-ignored.

## Read the brief

`brief` and `catalog` emit the same bounded schema, `image-lens.brief/1`. Eight shot rows are returned by default (maximum 16); `next_offset` and omitted counts make pagination explicit. Native PTS arrays stay at the raw scan locator. Source bytes and current detector helper/source pins must match before a verified brief is issued. Partial-source scans remain explicitly partial; incomplete decode packets are rejected. Hash checks bind bytes and provenance, not the correctness of detected cuts.

An optional `--report report.json` adds only supported cached motion reports that can be recomputed from their pinned source tracks and bindings under the exact canonical scan. Missing, outdated, unsupported or cross-cut reports are listed as `not_admitted`; their coverage remains unknown. Eight report references maximum. No LLM/API call, network, image generation, raw-coordinate dump or model import is performed by a brief. Labels in reports are untrusted quoted data.

Every shot retains its fresh-source-observation/retracking obligation. A single-frame shot permits inspection only. A missed cut or dissolve is possible; no detected cut does not prove continuity or identity. Do not crop coordinates from a whole-clip offline inference and call them shot-local tracking.

## Optional additional cuts

`scan --cuts thorough` and `look --cuts thorough` add fixed AutoShot candidates to the existing adaptive detector. The default is `--cuts fast`. Thorough uses the sorted union: it never removes an adaptive cut, merges a short shot or tracks across an added boundary. `redo` inherits the recorded cut mode. A thorough scan requires the complete source; partial `scan --start/--end` requests are refused before model loading or download. Both components see the same actual analysis video after any explicit region selection or resizing.

Thorough decodes through FFmpeg with 4 threads; fast uses OpenCV's default backend. Their adaptive boundaries matched on the tested Mac. Equivalence to a separately produced fast scan is unverified on hosts whose default decoder differs; the union retains every adaptive cut from its own decode.

```sh
# Install optional dependencies yourself in your environment.
python -m pip install '.[core,cuts]'
python -m image_lens --workspace W scan --video clip.mp4 --cuts thorough --out thorough-shots.json
python -m image_lens --workspace W look --video clip.mp4 --cuts thorough --out look-thorough
```

Only an explicitly requested thorough scan imports Torch and may download a model. First use fetches 57,243,097 bytes into `W/models/autoshot/` from the pinned third-party Hugging Face mirror `backseollgi/AutoShot` revision `9dbc41acb14de0dc32509f7c7d0a9876a6f44105`. The expected SHA-256 is `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`; size and hash are checked before publication. Model bytes are not bundled or committed. The vendored source has its MIT notice; equivalence of this checkpoint to the author's original distribution and checkpoint-specific license terms remain unverified. Missing optional dependencies, corrupt assets, failed downloads and timeouts are errors; none silently switches to fast or installs packages. Fast scans and cached readers do not download or import Torch.

A killed process can leave `W/models/autoshot/ckpt_0_200_0.pth.partial`, which later runs refuse to overwrite. Recovery is manual after confirming that no download is active.

This is additional boundary evidence, not a calibrated semantic edit classifier. AutoShot can miss abrupt pose/outfit replacements within a stable layout, so it must not veto the adaptive detector. Existing false cuts remain, and AutoShot can add false cuts on other inputs. More cuts may consume `look`'s default 16-shot cap earlier. Choose `--max-shots` explicitly for the needed coverage (maximum 256); the default cap is unchanged. A scan's last selected segment is only planned coverage, not proof of successful pose/tracking.

One integrated CPU check over 39 files / 6,645 frames (about 200 seconds of summed source spans): fast took 16.82 seconds, thorough 56.45 seconds including its first-use download. Process peak RSS was 0.47 GiB and 2.22 GiB respectively, not incremental memory measurements. The known synthetic set gained 8 exact cuts (63 to 71 of 84) at the same 18 false cuts; the 7 provisionally labelled real spans were unchanged. This is known development material, not held-out human ground truth. CPU-only, one ordered sample; MPS and new-style accuracy are unmeasured. In those six full real sources, the default 16-shot selection ended at the same native frame in both modes; this does not establish successful tracking or guarantee unchanged coverage on other clips.

Scan parameters, helper/model-source pins and canonical references bind the selected mode. Regenerate old scans/freezes with the new helper. Preparing, validating or consuming a cached thorough scan never downloads or runs the model. This option is currently for the local CLI; LAN client/server forwarding and installed-service rollout are separate work.

## Optional gradual-transition intervals

When fades or dissolves matter, use `--cuts thorough --transitions` on `scan` or `look`. The flag is off by default and requires thorough; an incompatible request fails before loading a model. It reuses the already computed AutoShot auxiliary multi-frame head and adds a small endpoint-mixture check. It adds no neural forward calls, model or dependency. The auxiliary head was trained to mark transition frames, not calibrated to certify a real edit.

```sh
python -m image_lens --workspace W scan --video clip.mp4 --cuts thorough --transitions --out transition-shots.json
python -m image_lens --workspace W look --video clip.mp4 --cuts thorough --transitions --out look-transitions
```

`transition_intervals` are inclusive native-frame ranges with native first/last-frame PTS. They are unverified gradual-transition candidates, without a reliable fade-versus-dissolve subtype. Actual hard-cut events remain intact. Analysis segments contain only the remaining contiguous frames; pose emits dense `excluded_transition` rows, and point tracking starts afresh in eligible segments. `look.md` labels excluded ranges and counts. JSON/Markdown briefs show at most eight interval rows, total/omitted counts and the complete raw-scan locator.

`--max-shots` still defaults to 16 safe segments, including existing short/untracked cases. Transition ranges consume no slot, but splitting can increase the number of safe segments and reduce later selection. `transition_coverage` separates excluded, selected-safe and cap-omitted frames; selected does not mean successfully tracked. `mark`/`box` reject excluded seeds, and `redo` inherits the selected policy. A cached inference whose original generation touched an excluded span cannot regain motion claims by cropping its output to a safe range. Raw manual analysis without a protected canonical scan remains unchecked.

Stored scans now use `team-eye.shots/2`; regenerate old scans, freezes and reports from source. Cached admission remains model/network-free. This flag is local CLI functionality; installed services and LAN clients do not forward it yet.

Known-material confirmation: eight newly authored fades/dissolves contained 152 mixed frames, all excluded. Of 2,248 clean frames, 106 were additionally excluded (4.72%); the eight no-transition controls lost 0/1,200 frames, separately measured. False ranges concentrated in black tails and 3D motion; one 150-frame example lost 43 extra frames. This is not real-transition human ground truth or evidence of uniform accuracy. Weak/truncated/long transitions can be missed; ordinary motion, exposure and blends can be ambiguous. Inspect source evidence before treating an interval as an edit. An extra interval means unknown motion, never a stationary pose.

One ordered cached-model CPU batch over sixteen clips: thorough alone 14.19 s, with exclusions 13.92 s; process peaks were both about 1.95 GiB. The small timing difference is not a speedup claim. Existing 39-source hard-cut predictions were unchanged. One actual 150-frame moving-person fade ran pose and MPS point tracking: 21 excluded pose rows, five generated track packets wholly outside those ranges, zero errors. Other styles and longer live workloads remain unmeasured.

## Fresh jobs and optional tracking

See [the job contract](jobs.md). `prepare` accepts a fresh query packet and canonical full-source scan, validates one shot-local job, and creates a new immutable freeze. Its default permits validation only. `validate` checks source bytes, queries, vendor/helper pins, decode/PTS and cuts **without importing Torch**, without a checkpoint and without producing tracks. It requires the core decode dependencies.

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --out job.freeze.json
python -m image_lens --workspace /path/to/workspace validate --queries queries.json --frozen job.freeze.json --job shot0 --out unused-tracks.json
```

Experimental inference requires a separately prepared freeze with `--allow-tracking`, an explicit local checkpoint and a frozen wall budget. Its single attempt is consumed immediately before the model import; failures never trigger automatic retries. The CLI parent kills/reaps its own child process group if the budget expires. Tracking runs on the Apple GPU (MPS) and fails without it. No download occurs. CoTracker source and checkpoint have noncommercial upstream terms. No neural inference was run to validate this extraction.

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

## One command: look

`look` turns a clip into a short per-shot motion summary in one new directory.

```sh
python -m image_lens --workspace W look --video clip.mp4 --out look-clip
```

Model paths come from `<workspace>/models.json` (see [Model paths](#model-paths)) or from `--detector`, `--pose-model` and `--checkpoint`. Exact limits and the checks behind them are in [look notes](look-notes.md).

### A video region inside a player or layout

Inspect an original frame and choose the region containing the movement you need:

```sh
python -m image_lens --workspace W look --video clip.mp4 --scene-box 140,40,714,440 --out look-region
```

`--scene-box X0,Y0,X1,Y1` uses original-input pixels, with right/bottom excluded. Width and height must be even; invalid or out-of-frame rectangles are refused. Keep the whole movement and every person you need inside the rectangle. Only that region is analyzed. This is explicit input selection: it cannot separate UI drawn over a person or recover anything outside the crop.

The new output directory holds `source-region.avi`, encoded losslessly from the decoded BGR frames after the crop and optional size reduction. Size reduction is intentional resampling; this does not preserve the source codec’s native YUV representation. The original is retained. FFmpeg-backed OpenCV with FFV1 and verifiable constant frame timing is required; VFR/unknown timing, an FPS the AVI writer cannot preserve (including 30000/1001fps in the checked runtime), unavailable lossless encoding, timeout or a copy exceeding 512 MiB is refused. Writer FPS incompatibility may be detected after conversion; the failed partial copy is removed. Crop conversion and verification add processing time and disk space. Frame indices remain those of the input. Cut detection now sees only the selected content, so its cuts may differ from a full-frame run.

`look.md`, `look-full.md`, `focus` and `at` carry the region notice. Pixel distances and raw coordinates refer to the analysis copy. `look.json.source_video` names that actual copy; `source_region` binds its hash/geometry to the parent's hash/geometry and records a pixel-centre mapping for **raw analysis coordinates only**. Camera-corrected and body-relative coordinates keep their own reference and basis.

`redo` reuses the bound analysis copy without cropping twice and validates the retained parent and analysis files. Region metadata is a local processing record, not signed provenance. Reuse checks referenced file identities, geometry and map consistency; it does not authenticate deliberately rewritten records or repeat the pixel comparison performed during creation. For independent `zoom`, `grid`, `box` or `mark` commands, use the actual `source_video` from `look.json` and coordinates in that copy. The full original remains available for inspecting excluded context.

### More than one person: run crowd first

`look` follows only the largest person in each shot unless `--max-people` (1–8) says otherwise. When a clip may show several people, check first:

```sh
python -m image_lens --workspace W crowd --video clip.mp4 --out crowd-clip
```

- `crowd` runs only the person detector, on one frame every 0.5 s (`--every`), and writes `crowd.jpg`: the four frames with the most boxes, at least 2 s apart, each box numbered. A 110 s 1080p clip took about 4 s.
- Look at `crowd.jpg` and count the people yourself. The count is rough: two people overlapping can share one box, and close-up eyes, posters or drawn figures can get boxes. `crowd.json` holds the count for every checked frame and `suggested_max_people`, the largest count capped at 8.
- Then run `look --max-people N` with your own count. The cap is per shot, so a shot with fewer people only tracks the people it has.

### What it runs

- `scan`, one `pose` pass, then for each shot `seed` / `prepare` / `track` / `bind` / `analyze`, seeded from pose. The first 16 shots by default; `--max-shots` goes up to 256.
- Cut detection uses four frames of context on either side. More detected cuts can use up the default 16-shot cap earlier; choose `--max-shots` for the needed coverage (at most 256). Short fast pans/zooms can still cause false splits, and dense edit trains can be missed. See [cut evidence and limits](look-notes.md#wider-adaptive-cut-context-2026-11-02).
- Long shots (over 160 frames) are split into windows. When half or more of a group's points stop well before the window ends, `look` treats it as a drawing change and starts a new window right after it.
- Single-frame shots, shots without a usable pose track and short tails are skipped with a reason.
- A failed step is recorded and the run goes on. A window that could not be tracked says "추적 결과 없음: 움직임 모름".
- `--max-seconds` is checked between steps, so a run can overshoot by one step.
- A source larger than 1920×1080 is first shrunk into `source-<w>x<h>.mp4`; px in the reports are of that copy.

### Output

| File | What it holds |
|---|---|
| `look.md` | Read this first. One line per tracking window. |
| `look-full.md` | Major moves per group, moved-together lines, warning counts, links to the full reports. `focus --look <dir> --shot shot-0007` prints one shot. |
| `look.json`, `steps.log` | The same record as data, and a receipt for every step. |
| `<window>-onion.jpg`, `onion-sheet-N.jpg`, `frame-sheet-N.jpg`, `check-sheet-N.jpg` | Pictures, below. |

### Reading look.md

**Check the points first.** Before reading any text or JSON numbers, open `onion-sheet-N.jpg`. In each window, the spacing chart on the right names the points drawn as coloured trails (`elbow-l`, `move-b`, …). Check that each trail sits on the body part it is named for. In the README dancer onion, the `elbow-l` trail sits on the chest and the `move-b` trail sits beside the shin, yet both still carry full distances and speeds. A trail on the background, on another part or on another person means that point slipped. For that window, read the lines it appears in (direction, distance, speed, ease) and its numbers in `look.json` / `*.report.json` as unknown. Unnamed points (`move-a`…) only need to be on the body. `look.md` repeats this check as its first line, and `image-lens --help` and `look --help` give the same order.

At the top, **먼저 확인할 곳** lists up to four spans where the tool itself is least sure: overlapping person boxes, lost tracks or poses, frames with no pose (so missing from the text) and runs of short cuts. Each has dense frames in `check-sheet-N.jpg`. Spans not listed can be wrong too: two people merged into one box (18–22 s of one test clip) was not caught, so read the frame sheets as well.

Below it, each window line starts with the shot framing (전신, 무릎 위, 상반신, 얼굴 클로즈업) and then says:

- **전체**: how the whole frame (camera) and the body centre moved, said once.
- **몸 기준**: what each arm, leg or move point did on top of that, measured against the body centre. Paths are split where they turn ("아래 → 위"); "몸 쪽으로/바깥으로" is the change in distance from the body axis. Sides are screen sides.
- **속도**: each major move has up to five speeds in time order, e.g. "속도 9→5→3→1 px/f". Judge the timing from these numbers. A move that starts on the window's first frame says its start was not seen.
- **화면 전체**: a camera move of 24 px or more, with speeds in time order, a move that went out and came back ("오른쪽 80px까지 갔다 돌아옴") and zoom ("줌 인 ×1.25"). Moves that only follow it are left out. When under half the frames could be matched it says "카메라 모름": screen moves may then include the camera.
- **높이별 흐름**: the person box split into five bands, with each band's shift. It needs no named points, so a hip pushed out under a skirt or a prop still shows.
- **많이 움직인 곳** (`move-a`…): unnamed points placed where the shape changed most.

Notes that change how to read a line:

- "전환": a drawing swap. It gives the pose difference between the two frames; the path between them was not seen.
- "f<N> 뒤로 점 k/n개 끊김": the points were lost. Read what follows as unknown, not still.
- "골격 신뢰 낮음": the pose skeleton looked wrong, so only the face is named and the body block is skipped. "(팔만)": an upper-body shot, arms only.
- "자세 A": the same pose shown again. The first showing is written in full, repeats only what differs.
- Pose events: a hand near the face (이마·눈/볼/입·턱), shoulder rises and falls, "몸 돌림 후보" (a possible turn), "대부분 안 보임: 멈춤 아님, 모름", "포즈 추적이 N프레임 끊김".
- "사람 N명": with several people, each one's movement, possible falls, touches and overlapping boxes.

### Pictures

- `<window>-onion.jpg`: the first frame in cyan and the last in red, with band shifts as arrows (doubled; shortened to stay inside the picture, left out when shorter than the arrow head; the text keeps the measured shift). Each part's lead point is drawn as white-rimmed dots, one per frame, with a slightly bigger dot every 0.1 s (big circle at the start, big square at the end): wide spacing is fast, tight spacing is slow. A spacing chart on the right stretches each trail's path (steps under 0.75 px/f left out) to one equal-length line with a thick tick every 0.1 s, so the ease of a small move reads as well as a large one; parts under 15 px are listed as still. `--peak` (look and mark) adds, on the left, a grey panel of the frame most unlike the first; it is off by default because the turn strip, the trails and the chart cover out-and-back moves.
- `onion-sheet-N.jpg`: all onion images of the run with their time ranges. Ranges too short to track appear as a single "still" frame, and the six windows most likely to show a turn get a strip of five frames.
- `frame-sheet-N.jpg`: whole frames spread evenly over the clip, about one per 1.25 s, up to 96.
- `check-sheet-N.jpg`: one per "먼저 확인할 곳" span, dense frames in time order.
- `zoom --video <clip> --out <dir> --at 18-22` shows dense whole frames for up to two spans of 4 s. Readers tend not to zoom where they were wrong, so the frame sheets come first.

### Asking about one span: at

When a question is about one span ("what happens at 18–22 s?"), ask for that span instead of rereading the whole summary.

```bash
python -m image_lens --workspace W at --look W/look-clip --span 18-22 --out W/at-18-22
```

`at.md` collects the lines, skipped cuts, transitions and "먼저 확인할 곳" signs that overlap the span, and names the pictures to open. `at-frames-1.jpg` holds 12 dense frames of the span. Spans are 4 s at most. The reader asks the question; the tool only gathers the evidence for that span.

### When pose seeds fail: grid and mark

For props, odd poses or parts the pose model has no name for, an agent can place and name points itself.

```bash
python -m image_lens --workspace W grid --video V --frame 572 --box 100,250,1000,1000 --out W/grid-f572.jpg
python -m image_lens --workspace W mark --video V --shots W/look/shots.json --frame 572 --end 599 --point hammer-stripe=430,680 --point eye=322,738 --checkpoint C --out W/mark-s19
```

- `grid` draws a pixel grid on the source frame to read positions from: lines every 25 px inside a zoom box and every 50 px otherwise, numbers on every other line.
- Place points on a corner where two outlines meet, or on a small spot (eye, button, pattern dot), not mid-edge on a straight line or inside a flat fill, where a tracker cannot see motion along the edge.
- `mark` also writes `mark-onion.jpg`. Check each trail against it: if a part's start (cyan) and end (red) outlines are far apart and its trail is short or goes another way, the tracking slipped. Read the dark outlines: a fill lighter than the background shows the colours the other way round (cyan band on the side it moved to). Onion and trails both have the camera shift taken out.
- `box --video V --look W/look --frame F [--end E] --box x0,y0,x1,y1 --out O` picks mark's points for you: the confident body joints (no edge-clamped ones) of the person the box overlaps best (IoU at least 0.3), else up to 8 spread corners inside the box named p1.. top to bottom; then runs `mark` (no snapping) and draws the box in green on the small whole frame. look draws its tracked person there in green too, so a wrong pick shows.
- `redo --look W/look --window <shot id>-w<i> --detector D --out O` re-runs one look window with another pinned person detector (e.g. the SAM3 bundle in [person-detectors.md](person-detectors.md)) when the green box is on the right person but badly shaped. It runs look on that window's frames only, writes a new look folder whose look.md says what was redone, and `redo-boxes.jpg` with the first look's largest person box in red and the new one in green on the window's first, middle and last frames. The first look is untouched. Use `box` instead when the wrong person was picked.
- `mark` moves each point to the nearest trackable corner within 15 px (`--no-snap` keeps it), saves marked crops as evidence, tracks the points and writes `mark.md` plus per-frame coordinates in `mark.tracks.json`.
- Names are ASCII, up to 8 points. They are the agent's reading, not verified anatomy.

### Model paths

Model paths left out of `look`, `pose` or `mark` are read from `<workspace>/models.json`:

```json
{"detector": "/models/yolox_humanart.onnx", "pose_model": "/models/rtmw133.onnx", "checkpoint": "/models/scaled_offline.pth"}
```

A flag on the command line wins. The workspace is Git-ignored, so each machine keeps its own file.

## Whole-body seeds from pose (optional)

`pose` runs a local whole-body pose model over the canonical scan: YOLOX HumanArt person boxes, RTMW-l 384x288 133 keypoints (rtmlib, ONNX Runtime; scores divided by `pose_score_scale` back to 0..1; boxes and keypoints on CoreML) and ByteTrack IDs reset at every detected cut. The two ONNX files are user-supplied and pinned by size and SHA256; nothing is downloaded or bundled, and the weight license is not established. Install `requirements/pose.txt` into the same environment. The packet also flags frames where the pose layer itself is unreliable: overlapping person boxes (echo copies, afterimages) and body-skeleton jumps within one track.

`seed` turns one frame of that packet into fresh observation jobs, one per group so each fits the 12-point surface contract: `<job>-head` (nose, eyes, front/top/side hair), `<job>-arms` and `<job>-legs`. Hair points are geometry on a head frame built from the eyes (distance and the up direction away from the nose), so they are candidate surface points, not detected hair. Joints near the frame edge are dropped because off-screen joints come back clamped there. Each seed gets a marked source crop as its pinned visual reference; look at them before tracking. `bind` writes the report bindings for one tracked job.

```sh
python -m image_lens --workspace W pose --shots shots.json --detector /models/yolox_humanart.onnx --pose-model /models/rtmw133.onnx --out pose.json
python -m image_lens --workspace W seed --pose pose.json --frame 215 --job s7 --out s7.obs.json --evidence-dir s7-crops
# prepare/track each of s7-head, s7-arms, s7-legs, then:
python -m image_lens --workspace W bind --queries s7.obs.json --tracks s7-head.tracks.json --job s7-head --out s7-head.bindings.json
python -m image_lens --workspace W analyze --tracks s7-head.tracks.json --shots shots.json --report-bindings s7-head.bindings.json --start 215 --end 307 --format markdown --trajectory-dir s7-head-traj --pose pose.json --out s7-head.md
```

With `--pose`, the trajectory view marks pose-trouble frames (orange on the timeline), tags change candidates near them, and groups warnings so one line names every big stroke that coincides with a candidate or a pose-trouble range. With more than two points, only major strokes are labelled, and the Markdown leads with a one-line-per-point summary plus "moved together" lines (major strokes of several points with the same direction, overlapping time and similar size); per-point speed windows, phases and span comparisons stay in the JSON. Pupils are not modelled; eye points follow the drawn eye area.

Analysis help also exposes review selection, explicit feature bindings, source-color region reports and optional foreground inspection. Read [limitations](limitations.md) before interpretation.

## Verification and provenance

```sh
python -B -m unittest discover -s tests -v
# Optional core-dependency smoke, with a new output directory:
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

[Verification](verification.md) separates tested CLI/guards from unrun inference and platform coverage. [Provenance and licenses](provenance.md) lists original hashes, upstream pins, the restricted-loading patch and the authored-code licensing boundary. Historical experiment media, fixtures, ledgers, results and rejected automatic classifiers are not selected runtime paths.
