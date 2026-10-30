# look: thresholds and evidence

The [guide](guide.md) says how to run `look` and read its output. This page keeps the exact limits and the checks behind them. Clips named here: clip A (hand-drawn dance), clip B (AI-generated cuts), clip C (covers, one with a hammer scene), clip D (live clip, 40–70 s held out).

## Windows and drawing changes

- Shots longer than 160 frames are split into consecutive windows, each reseeded on its first unflagged pose frame.
- When at least half of one group's points stop more than 12 frames before the window ends, `look` treats it as a drawing change. On clip A every such case was a costume change or a wipe inside one detected shot. The window is analyzed up to that frame and a new window is reseeded right after it.
- `look.md` has one line per tracking window: clip A gives 55 lines instead of 542.

## Repeated poses

- A window is the same pose shown again ("자세 A") when its first and last frames (person box, 24×24 grey) both match an earlier window's at correlation 0.9 or more and its length is within 80%.
- Clip A groups into its four costumes plus the long black-dress hold; different poses stay at 0.48 or below. No clip C cover is merged.
- Repeats are compared statement by statement, matched by direction and size within 8 px or a quarter, whatever their label: move-point letters are per window, and the pose model sometimes names the other arm on identical drawings.

## Body-relative block

- Limb points: wrists, elbows, thumb and index tips, thighs, knees, ankles. They are measured against the shoulder midpoint, or the face when shoulders are missing.
- Threshold: 6% of shoulder width, 8 px minimum. Sides are screen sides taken from the limb's shoulder or hip.
- Hand tips come from the pose model's hand keypoints and are rough on drawn hands. Thighs sit halfway from hip to knee, so legs cut off by the frame still get a point.
- Points of one screen-side arm or leg whose first stroke has the same direction and a size within half of each other are one statement; the largest three are shown and the rest counted.
- Each part is led by the point that travelled farthest, back-and-forth included. Points that moved together on the screen are one move led by the largest.
- The transition block ("전환 앞뒤 자세 차이") runs when a window starts within 3 frames of the previous one. It compares wrist and elbow keypoints on the two frames, body-relative. Animated moves often run through the drawing swap (an arm dropping from the side to the waist on the switch frame), which no single window can see. It is skipped when the shoulder width changes by more than a third or the body centre jumps more than one shoulder width.

## Skeleton check and framing

- Before naming arm and leg points, `seed` checks core joint scores, torso length, hip width and arm segments in shoulder widths. Props, costumes and odd poses fool the pose model; then only the face is named and `look.md` says "골격 신뢰 낮음".
- Framing: 전신 when both ankles score 0.6 or more and sit at least two shoulder widths below the shoulders; 무릎 위 when the hips pass the full check and the feet are out of the shot (all of clip A); 얼굴 클로즈업 when the eyes are 30% of the frame width apart or more, or no shoulder is seen; otherwise 상반신.
- On upper-body shots the pose model guesses hips onto the chest and ankles at the frame bottom (clip B: 37 of 43 windows were face-only). There the arms are named from both shoulders alone when every seen arm joint passes the same score and segment limits; the legs are left out and the line says "(팔만)".

## Move points (move-a…)

- The seed frame is compared with the frames at one third, two thirds and the end (before any drawing swap, camera shift removed). Up to 12 candidates go where the onion gaps are widest on an outline of the seed frame.
- All candidates are tracked. Candidates whose paths stay within 16 px or half the larger travel are one moving part and keep one point; parts travelling under 40% of the farthest are dropped. An arm-only move gets one point, six at most.
- Only those points reach the report (`<window>-move-parts.bindings.json`); every candidate stays in the tracks file.
- Gap width alone misleads: in the clip C hammer scene the widest gap moved 39 px while the hammer moved 450 px. A gap on empty start-frame background is where a part arrives and gets no point.

## Camera and height bands

- Whole-frame motion (pan, zoom, scrolling UI) comes from ORB matches against the seed frame. It counts only when the agreeing matches span at least half the width and a quarter of the height, so a large figure sliding over a still layout is not taken for the camera. At 24 px or more it gets a "화면 전체" line, and major moves that only follow it are dropped.
- The line carries speeds in time order, a move that went out and came back (farthest point of a running median of five, at least twice the net shift) and zoom from ×1.1. Under half the frames matched reads "카메라 모름", not still.
- Frames are matched to the window's first frame. When one does not match and the last matched frame is 24 px or more away, it is matched to that frame and the steps are chained, needing matches over 75% of the width and half the height. Clip D 70–100 s: a 671 px pan at 23.35 s was unknown before; where the old start-only match was known, the net shift is the same (all of clip C, clip B).
- Matching each frame to the one before was tried first: on clip A it read the dancer over the still title panel as the camera ("오른쪽 위 50px", really 0).
- A step whose matches include a still layer (at least 30 matches, a quarter of the inliers, moving 3 px or less while the transform moves more) is left unknown: on clip A seed matches locked on the dancer in 11 windows ("위 45px까지 갔다 돌아옴"). Cost: clip A now matches 61 of 462 frames (303 before), so its 406 px pan at 9.24 s reads "카메라 모름"; clip B 391 of 1413 (459 before) with the same camera lines.
- Blanking the pose boxes before matching, as a fallback for frames the full-frame match missed, was tried and dropped. Clip A gained no frame (what is left of the background is too narrow and plain), clip B 53 and clip D 22 frames with no key to check them, and on clip C the drawn figures that move with the pan are boxed too: 68 of 551 frames matched both ways disagreed by more than 8 px.
- "높이별 흐름": the person box is split into five horizontal bands; corner features in each band are followed from the first to the last frame and the median shift per band (minus the camera) is reported. Clip A pose 1: head right ~22 px, waist and skirt left ~20–29 px; the three repeats agree within 5 px.

## Speeds and timing

- Each major move carries up to five speeds in time order ("속도 9→5→3→1 px/f"), at least two frames per number so drawings held on twos do not read as 0, 31, 0, 28.
- A move that starts on the window's first frame says its start is unseen: windows begin where the drawing changes, and there the speed-up is before the cut (clip A f47: 40 px in two frames, then settling). 190 of 196 "slows down" labels started on a window's first frame.
- Four ease words (speeds up, slows down, fast middle, even; from progress in thirds of the move's time) were tried and dropped. With the words hidden, the user judged slowed-down tracked strokes and agreed on 6 of 11 screen strokes, then on 1 of 11 body-relative lead strokes that start after the window's first frame (two of those were tracking failures). The moves were 25–130 px and hard to call by eye. The words stay in the JSON only (`ease`).

## Sheets and reader scores

- Onion sheets: five rows per sheet, each row four onion images, one strip or one row of stills. On clip B a blind reader scored 5/21 of the user's moves from `look.md` alone and 9/21 with the sheets.
- A range under 12 frames gets one native still of its middle frame ("still", pose only, motion unknown); stills of back-to-back cuts share one row. The clip B spin at 1.8–2.2 s is seven cuts of 3–6 frames and was invisible until these stills.
- The grey panel is the frame most unlike the first, searched away from both ends (middle of the plateau of mean grey difference), so a move that goes out and comes back is not read as still.
- Turn strips: the six windows with the largest shoulder-width swing get five frames in time order. Giving one to all 18 turn candidates took the sheets from 8 to 14 for about the same score (13 vs 12.5 of 21, one reader each, not blind).
- After the pose events were added a fresh reader scored 12.5/21 on clip B. That checks that the changes reach the reader; it is not a blind score, because they were made after seeing the key.

## Pose events

- Hand near face: wrist seen and a hand point inside the face box padded by a quarter of its width for 0.1 s or more, with the region 이마·눈/볼/입·턱. Noisy: on clip B a cat's paws and chin-height hands count too.
- Shoulder rises and falls above the hips (or the eyes) are in shoulder widths. A shoulder-width swing of 35% or more against the eye distance is "몸 돌림 후보"; overlapping shoulder heights are dropped, since the points jump during a turn.
- A wrist seen in under half the window is "대부분 안 보임: 멈춤 아님, 모름".
- Events follow the seed person's tracker id only; frames where that id is missing are counted ("포즈 추적이 N프레임 끊김: 그동안 모름") instead of taking another detection.

## Several people

- A person counts when present in half the window or more with at least a quarter of the largest box area; up to three, named P1… from screen left by tracker id.
- Fields: body-centre shift in shoulder widths (camera removed), fall candidate (torso from 30° or less to 60° or more off vertical), and per pair distance change, a wrist inside the other's shoulder-to-hip box ("닿음 후보") and heavy box overlap ("포즈가 섞였을 수 있음").
- On clip D 40–70 s (key given after the answers were frozen) fresh readers scored 2.5/20 from `look.md` alone and 8.5/20 with the sheets; with the people fields 8/20 (not blind any more), so no gain shown: the pull-and-fall shot is one merged detection box.

- The detector's ONNX has NMS built in, so `det_nms` changes nothing: two people pressed together come out as one full-width box (0.85) and the second person only as boxes under 0.10. A box wider than tall is posed again as its left and right 60%; when both crops find a head (nose and eyes 0.45 or more), the noses are 35% of the box apart and each eye distance is 6% of the box or more, the two crops replace the box. Clip D: 169 of 719 frames split, 40 of them in 18–22 s, where `look.md` now has two people, a fall candidate and hand-on-shoulder spans instead of no people line; the fall is put on P1 (left), the person who falls is on the right, because unsplit frames keep the merged skeleton on P1's track. Clip A: 4 frames, the stacked copies at 3.8 and 8.8 s. Clip C: none. Clip B: none; before the eye-distance rule a one-face close-up at 19 s split on its two eyes (eye distance 0.05 of the box, real pairs 0.08 or more). Time: clip D 527 → 661 s, clip A, B and C within 15%.

## zoom and frame sheets

- `zoom` takes 12 frames per span, at most two spans of 4 s, and writes the frame and pixel cost to `zoom.json`; `--even` spreads the same 24 frames over the whole clip as a control.
- In one recheck on clip D (key known, one reader each) a reader that chose its own spans went from 10.5 to 11.5 of 20; a reader given the 24 even frames went from 8.5 to 12. Readers do not pick the spans they got wrong: the pull-and-fall at 18–22 s was written confidently as an arm around the shoulder and not chosen. So `look` writes `frame-sheet-N.jpg` for every clip.

## Check list and at

- "먼저 확인할 곳" ranks the tool's own warning signs per window or skipped range. Weights: overlapping person boxes, nothing tracked and 12+ frames with no pose 3 each; lost points, warnings, a lost pose track and three or more back-to-back short cuts 2 each; two or more people, a low skeleton score, turn candidates and an unknown camera 1 each. Up to four spans scoring 2 or more are kept, each cut to 4 s around its turn and lost-point frames, with 6 dense frames each in `check-sheet-N.jpg` (zoom's 24-frame budget).
- Windows touch each other, so neighbouring signs are not merged: merged, the whole clip D became one span.
- It does not catch clip D 18–22 s (the pull and fall). Before the box split that window scored 4 (warnings, low skeleton, turns), tied with four others, below five that score 5 to 9; with the split it reads as two people and scores 6, still outside the top four (6 to 8). A sign for the number of pose boxes changing within a window was tried: it fired on that window, but also on single-dancer clip A windows (false extra boxes) and still left the window out of the top four, so it was dropped. The frame sheets stay the guard for this case.
- `at --look <dir> --span a-b` gathers the window lines, skipped ranges, transitions and check signs overlapping one span of up to 4 s, names the pictures to open, and writes 12 dense frames (`at-frames-1.jpg`, about 1.7 Mpx at 1920×1080).

## mark

- On the clip C hammer shot, two points on the hammer head followed the whole swing (wind-up left 74 px, then right and down ~800 px speeding up, settling while slowing) and the eye stayed on the face. A point on the hammer's corner slid onto the hat, and one on the hand stayed behind when the drawing swapped hands.

## Run time

- CoTracker runs on MPS and the RTMW pose model on CoreML; the YOLOX detector at first stayed on CPU (CoreML failed on a frame with no box; see below). Whole look before → after (2026-10-30, M4 Max): clip A 657 → 280 s, B 1753 → 794 s, C 1063 → 593 s, D 661 → 268 s. Tracking alone took 3.5-4x less; pose 15-35% less.
- Results moved little. Tracks: median 0 px, p95 2-5 px. Pose keypoints scoring 0.5 or more: p99 4 px; only low-score points jump. look.md lines changed in small numbers (a few px, a turn-candidate count by one). Clip A 4.30 s named the right arm instead of the left once; clip D's 18-22 s fall-and-pull line is unchanged.
- A tracker job now takes about 1.6 s (clip A: 95 jobs, 148 s), of which a warm model call is under 1 s; the rest is the process start, model load and MPS warm-up. The CPU detector took about 0.1 s a frame.
- A tracker job seeks to its first frame instead of decoding from frame 0, which had made a window cost more the later it started (clip B: 0.9 s at frame 29, 2.5 s at frame 1169). Clip B tracking 295 → 175 s, whole look 593 → 472 s; tracks and look.md are identical.
- One tracker child serves every window of a look (`tracking_child.py --serve`); each job keeps its own hard deadline, and a timeout or failed job replaces the child. Starting Torch, hashing the checkpoint and loading the model had cost about 0.85 s per job. Clip B tracker time 175 → 67 s over 137 jobs, whole look 472 → 343 s, child max RSS 1.65 → 2.30 GB; tracks and look.md are identical.
- The YOLOX detector runs on CoreML as an MLProgram with static input shapes; without static shapes every CoreML format fails on the dynamic NMS input. Clip B: detector 104 → 11 ms a frame, pose step 156 → 42 s, whole look 343 → 227 s; repeat runs are identical. Output is not identical to CPU: boxes within 0.01 px, but 0.4% of keypoints move over 5 px (0.2% with score ≥ 0.3), mostly unstable low-score points; in look.md 5 of about 90 lines changed, and the 6–7 s window flipped from knee-up with limb names to upper body with low skeleton trust (137 → 134 track files).
- Work repeated inside a window is done once: analyze validates the shot context once per tracks packet (it had run for every point and pair, about 200 times a call), camera.path cuts a shorter request from the last path from the same start (seed and look both asked), and the trajectory view keeps the last window's change candidates (each group asked again; analyze now reuses the loaded motion_view while its file is unchanged). Clip B whole look 227 → 199 s; all 2111 output files are the same apart from timings, paths and hashes.
- A file of 1 MiB or more keeps its SHA-256 while its device, inode, size, mtime and ctime are unchanged (any write moves ctime), in `paths.sha`, the tracker child and `shot_boundaries`; every step had re-hashed the source. A 94 MB MJPG copy of clip B: whole look 229 → 208 s, all 2069 output files the same apart from timings, paths and hashes; the cost grows with file size (a 300 MB upload would save about a minute), and the 8 MB MP4 barely changes.
- The groups of one window no longer decode its frames again: the tracker child keeps the last window's checked RGB frames and timestamps, the trajectory view keeps the window's frames with its change candidates, and analyze parses the pose packet once while the file is unchanged (keys carry device, inode, size, mtime and ctime). Clip B whole look 199 → 173 s (tracker jobs 66 → 58 s), max RSS 2.30 → 2.44 GB; all 2111 output files the same apart from timings, paths and hashes. A lower PNG compression level was measured and left out: about 5 ms a sheet.
- camera.path finds ORB points of a window's frames in 8 threads (a fresh detector gives the same points and descriptors); matching, RANSAC and rekeying stay in frame order. Window 0.14 → 0.02 s for ORB; clip B whole look 173 → 170 s, all 2111 output files the same apart from timings, paths and hashes.
- One window is decoded once: `frames.py` keeps the last decoded window (key: device, inode, size, mtime, ctime, start, end) and seed gaps, seed crops, camera, trajectory view, onion bands and peak, and motif thumbs cut from it (they had read the same frames 8 times, each from its own seek: 21.9 s of decoding on clip B). The trajectory view shrinks its change thumbs in 8 threads, and the CoreML models are compiled once per pinned model hash into `workspace/.cache/coreml/` (fresh process: pose session 2.0 → 0.1 s, detector 0.5 → 0.1 s, same outputs). Clip B whole look 170 → 146 s (148 s on the run that fills the cache); all 2111 output files the same apart from timings, paths and hashes.
- The tracker child computes CoTracker's frame features (`fnet`, which sees the video only, never the queries) once per window and gives them to every group of that window, call by call; each group's tracking still runs alone on its own queries, so the numbers are the ones it had before. Batching the groups' queries into one call was measured and left out: points attend to each other, and on a fast head move one point lagged for a few frames. Clip B tracker inference 44.3 → 30.8 s over 134 jobs, whole look 146 → 130 s; all 2111 output files the same apart from timings, paths and hashes.
- The pose step decodes and detects the next frame in one worker thread while the main thread poses, splits and tracks the current one; frames still come in order with the same index, geometry and PTS checks. Clip B pose step 39.3 → 26.8 s (300-frame bench: 22.9 → 13.8 ms a frame, identical boxes and keypoints), whole look 130 → 118 s; all 2111 output files the same apart from timings, paths and hashes.
- look reports a window while the tracker child tracks the next one: it seeds and queues the next window's groups (`watchdog.Worker.submit`), runs the last window's bind, analyze, camera, onion and motif steps, then takes the replies in order. That window's lines and errors go where it began, so look-full.md and look.json read as before; a job queued behind a failed one goes to a fresh child. The frame cache keeps the last 3 windows and the camera cache the last 2, since the next window's seed comes in between. Clip B whole look 118 → 101.5 s (two runs: 101.4, 101.5); all 2111 output files the same apart from timings, paths and hashes.
