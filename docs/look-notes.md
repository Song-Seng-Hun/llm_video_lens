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

## zoom and frame sheets

- `zoom` takes 12 frames per span, at most two spans of 4 s, and writes the frame and pixel cost to `zoom.json`; `--even` spreads the same 24 frames over the whole clip as a control.
- In one recheck on clip D (key known, one reader each) a reader that chose its own spans went from 10.5 to 11.5 of 20; a reader given the 24 even frames went from 8.5 to 12. Readers do not pick the spans they got wrong: the pull-and-fall at 18–22 s was written confidently as an arm around the shoulder and not chosen. So `look` writes `frame-sheet-N.jpg` for every clip.

## Check list and at

- "먼저 확인할 곳" ranks the tool's own warning signs per window or skipped range. Weights: overlapping person boxes, nothing tracked and 12+ frames with no pose 3 each; lost points, warnings, a lost pose track and three or more back-to-back short cuts 2 each; two or more people, a low skeleton score, turn candidates and an unknown camera 1 each. Up to four spans scoring 2 or more are kept, each cut to 4 s around its turn and lost-point frames, with 6 dense frames each in `check-sheet-N.jpg` (zoom's 24-frame budget).
- Windows touch each other, so neighbouring signs are not merged: merged, the whole clip D became one span.
- It does not catch clip D 18–22 s (two people in one merged box). That window scores 4 (warnings, low skeleton, turns), tied with four others, below five that score 5 to 9. A sign for the number of pose boxes changing within a window was tried: it fired on that window, but also on single-dancer clip A windows (false extra boxes) and still left the window out of the top four, so it was dropped. The frame sheets stay the guard for this case.
- `at --look <dir> --span a-b` gathers the window lines, skipped ranges, transitions and check signs overlapping one span of up to 4 s, names the pictures to open, and writes 12 dense frames (`at-frames-1.jpg`, about 1.7 Mpx at 1920×1080).

## mark

- On the clip C hammer shot, two points on the hammer head followed the whole swing (wind-up left 74 px, then right and down ~800 px speeding up, settling while slowing) and the eye stayed on the face. A point on the hammer's corner slid onto the hat, and one on the hand stayed behind when the drawing swapped hands.
