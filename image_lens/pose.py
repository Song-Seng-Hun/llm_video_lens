"""Optional local whole-body pose: YOLOX HumanArt detector + RTMW133 (rtmlib, ONNX Runtime on CoreML) with per-shot ByteTrack IDs.
User-supplied weights are pinned by size and SHA256; nothing is downloaded or bundled (weight license unestablished).
Pose output is a model estimate on screen 2D: seeds and overlap flags, never verified anatomy, identity or redraw decisions."""
import math
from pathlib import Path
from image_lens import paths, shot_boundaries

MODELS = dict(detector=(101400344, '3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'),
              pose=(229320930, 'bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054'))
PARAMS = dict(det_score=.10, det_nms=.45, det_input=[640, 640], pose_input=[288, 384], track_activation=.25,
              lost_track_buffer=30, minimum_matching_threshold=.8, tracker_reset='every canonical shot',
              max_frames=9000, keypoint_schema='coco133', pose_score_scale=6.2)
FLAGS = dict(overlap_iou=.10, jump_diagonals=.15, min_score=.30)
# The detector (NMS baked into the ONNX) returns one box for two people pressed together. A box wider than tall is posed
# again as a left and a right crop: two confident heads far apart mean two people, and the crops replace the box.
# Each head's eye distance must be a face-sized share of the box: on a clip B close-up of one face (0.05 of the box)
# the two crops found the two eyes as two heads; real pairs were 0.08 or more. Fit on clip D 18-22 s, clip A's
# stacked copies and that one close-up.
SPLIT = dict(min_detection=.30, crop=.60, head_score=.45, head_gap=.35, min_eyes=.06)
PART_FRAMES = 90  # a pose part is written at the first cut after this many frames
SEED = dict(min_score=.30, max_frames=160, border_px=32, uncertainty_px=(8., 24.))
BODY = dict(nose=0, eye_l=1, eye_r=2, ear_l=3, ear_r=4, shoulder_l=5, shoulder_r=6, elbow_l=7, elbow_r=8,
            wrist_l=9, wrist_r=10, hip_l=11, hip_r=12, knee_l=13, knee_r=14, ankle_l=15, ankle_r=16)
# COCO-WholeBody hand blocks start at 91 (left) and 112 (right); +4 is the thumb tip, +8 the index tip. Rough on drawn hands.
HAND = dict(thumb_l=95, index_l=99, thumb_r=116, index_r=120)
GROUPS = dict(head=('nose', 'eye_l', 'eye_r', 'hair_front', 'head_top', 'hair_side_l', 'hair_side_r'),
              arms=('shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r'),
              hands=('thumb_l', 'index_l', 'thumb_r', 'index_r'),
              legs=('hip_l', 'hip_r', 'thigh_l', 'thigh_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r'),
              move=tuple(f'move_{c}' for c in 'abcdefghijkl'))
# When the skeleton is implausible (props, folded or cut-off bodies), arm/leg names would be confident and wrong:
# keep the face, drop named limbs, and track unnamed high-contrast corners inside the person box instead.
# Hair seeds sit on a head frame built from the two eyes: d = eye distance, up = away from the nose.
# Offsets in d units; checked by eye on clip A only, so every seed stays a candidate surface point.
HAIR = dict(front=(0., .8), top=(0., 1.7), side=(1.3, -.2))


def _model(path, role):
    path = Path(path).expanduser().absolute()
    paths._no_symlinks(path)
    size, digest = MODELS[role]
    if not path.is_file() or path.stat().st_size != size or paths.sha(path) != digest:
        raise ValueError(role+' model pin mismatch; supply the pinned local ONNX file')
    return path


def _coreml_cache(role):
    path = paths.workspace_root()/'.cache'/'coreml'/MODELS[role][1]
    path.mkdir(parents=True, exist_ok=True)
    # Compiled models are kept per pinned hash; a model switch left the old pose model's 431 MB behind for good.
    import shutil
    for old in path.parent.iterdir():
        if old.name not in {digest for _, digest in MODELS.values()} and old.is_dir() and not old.is_symlink():
            shutil.rmtree(old)
    return str(path)


def _detector_class():
    import numpy as np
    from rtmlib import YOLOX
    from rtmlib.tools.object_detection.post_processings import multiclass_nms

    class ScoredYOLOX(YOLOX):
        # Derived from rtmlib 0.0.16 YOLOX.postprocess (Apache-2.0); keeps person scores for ByteTrack.
        def postprocess(self, outputs, ratio=1.):
            outputs = outputs.copy()
            if outputs.shape[-1] == 5:
                boxes = outputs[0, :, :4]/ratio; conf = outputs[0, :, 4]; mask = conf >= self.score_thr
                return boxes[mask], conf[mask]
            grids, strides = [], []
            for s in (8, 16, 32):
                xv, yv = np.meshgrid(np.arange(self.model_input_size[1]//s), np.arange(self.model_input_size[0]//s))
                g = np.stack((xv, yv), 2).reshape(1, -1, 2); grids.append(g); strides.append(np.full((*g.shape[:2], 1), s))
            grids = np.concatenate(grids, 1); strides = np.concatenate(strides, 1)
            outputs[..., :2] = (outputs[..., :2]+grids)*strides; outputs[..., 2:4] = np.exp(outputs[..., 2:4])*strides
            v = outputs[0]; box = v[:, :4]; scores = v[:, 4:5]*v[:, 5:]
            xy = np.column_stack((box[:, 0]-box[:, 2]/2, box[:, 1]-box[:, 3]/2, box[:, 0]+box[:, 2]/2, box[:, 1]+box[:, 3]/2))/ratio
            dets, _ = multiclass_nms(xy, scores, nms_thr=self.nms_thr, score_thr=self.score_thr)
            if dets is None:
                return np.empty((0, 4), np.float32), np.empty(0, np.float32)
            dets = dets[dets[:, 5] == 0]
            return dets[:, :4], dets[:, 4]
    return ScoredYOLOX


def run(shots, detector, pose_model, start=0, end=None, part=None):
    """Pose packet for start..end. With `part`, look's pose child also hands over packets for runs of whole shots as
    they finish (track IDs and flags never cross a cut, so a part reads the same as that range of the whole)."""
    scan, scan_path = paths.load(shots)
    shot_boundaries.validate(scan)
    if scan.get('scan_scope') != 'full_source':
        raise ValueError('full-source canonical shot scan required')
    video = paths.confined(scan['source_video'], exists=True, max_bytes=512*1024*1024)
    if str(video) != scan['source_video'] or paths.sha(video) != scan['source_sha256']:
        raise ValueError('shot scan source changed or not canonical')
    last = scan['source_frame_count']-1
    end = last if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= last or end-start+1 > PARAMS['max_frames']:
        raise ValueError('bounded native range required')
    models = dict(detector=_model(detector, 'detector'), pose=_model(pose_model, 'pose'))
    import time
    import cv2
    import numpy as np
    import onnxruntime
    import supervision
    from rtmlib import RTMPose
    started = time.monotonic()
    det = _detector_class()(str(models['detector']), model_input_size=tuple(PARAMS['det_input']), score_thr=PARAMS['det_score'],
                            nms_thr=PARAMS['det_nms'], backend='onnxruntime', device='cpu')
    # The detector's CoreML part needs static input shapes; without them every CoreML format fails on the dynamic NMS
    # input. Clip B, 390 frames: 104 -> 11 ms a frame, the same box count on every frame, boxes within 0.01 px of CPU.
    # The compiled CoreML models are kept per pinned model hash (fresh process: pose 2.0 -> 0.1 s, detector 0.5 -> 0.1 s,
    # the same outputs). The pose session is rtmlib's 'mps' one (CoreML default options) plus the cache.
    det.session = onnxruntime.InferenceSession(str(models['detector']), providers=[
        ('CoreMLExecutionProvider', dict(ModelFormat='MLProgram', RequireStaticInputShapes='1', ModelCacheDirectory=_coreml_cache('detector'))),
        'CPUExecutionProvider'])
    estimator = RTMPose(str(models['pose']), model_input_size=tuple(PARAMS['pose_input']), to_openpose=False, backend='onnxruntime', device='cpu')
    estimator.session = onnxruntime.InferenceSession(str(models['pose']), providers=[
        ('CoreMLExecutionProvider', dict(ModelCacheDirectory=_coreml_cache('pose'))), 'CPUExecutionProvider'])
    # Pose on CoreML (29 -> 8 ms a crop, keypoints within 0.01).
    if any(m.session.get_providers() != ['CoreMLExecutionProvider', 'CPUExecutionProvider'] for m in (det, estimator)):
        raise ValueError('unexpected execution provider')
    # RTMW-l 384 scores run about 6.2x the earlier RTMW 256x192 ones at every quantile (seven clips, 90k body joints), so they
    # are divided back to the 0..1 scale every score threshold was calibrated on.
    def pose(image, bboxes):
        k, s = estimator(image, bboxes=bboxes)
        return k, s/PARAMS['pose_score_scale']
    segments = {n: s['segment_id'] for s in scan['analysis_segments'] for n in range(s['start_frame'], s['end_frame']+1)}
    fps = scan['fps']; frames = []; tracker = None; segment = None; parted = 0
    def packet(rows):
        return dict(schema='image-lens.pose/1', source_video=str(video), source_sha256=scan['source_sha256'],
                    width=scan['width'], height=scan['height'], fps=fps, native_range=[rows[0]['frame'], rows[-1]['frame']],
                    canonical_shots=dict(path=str(scan_path), sha256=paths.sha(scan_path)),
                    models={k: dict(path=str(v), sha256=MODELS[k][1]) for k, v in models.items()}, params=dict(PARAMS, split=SPLIT),
                    runtime=dict(onnxruntime=onnxruntime.__version__, supervision=supervision.__version__, opencv=cv2.__version__),
                    pose_module_sha256=paths.sha(__file__), elapsed_seconds=round(time.monotonic()-started, 3),
                    flags=flags(rows), frames=rows,
                    limitations=['human-trained model on drawn characters: face/hand points are rough; hair and pupils are not modelled',
                                 'overlapping copies (echo, afterimage) can merge into one wrong skeleton without any flag',
                                 'a merged box is split only in two, left and right, when both crops find a head; split skeletons see half a box'])
    def detected(n):
        ok, image = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != n+1 or image.shape[:2] != (scan['height'], scan['width']):
            raise ValueError('source decode geometry/index mismatch')
        if abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.-scan['native_pts_seconds'][n]) > .002:
            raise ValueError('decoder PTS differs from canonical scan')
        boxes, conf = det(image)
        return image, np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(conf, np.float32)
    from concurrent.futures import ThreadPoolExecutor
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('video decode failed')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # The next frame is decoded and detected while this one is posed (clip B: 22.9 -> 13.8 ms a frame, the same
        # boxes and keypoints); frames still come in order, one read at a time.
        with ThreadPoolExecutor(1) as ahead:
            following = ahead.submit(detected, start)
            for n in range(start, end+1):
                image, boxes, conf = following.result()
                if n < end:
                    following = ahead.submit(detected, n+1)
                if segments[n] != segment:
                    if part and len(frames)-parted >= PART_FRAMES:
                        part(packet(frames[parted:])); parted = len(frames)
                    # Track IDs never cross a detected cut.
                    segment = segments[n]
                    tracker = supervision.ByteTrack(track_activation_threshold=PARAMS['track_activation'], lost_track_buffer=PARAMS['lost_track_buffer'],
                                                    minimum_matching_threshold=PARAMS['minimum_matching_threshold'], frame_rate=round(fps),
                                                    minimum_consecutive_frames=1)
                # RTMPose falls back to a full-image crop for empty boxes; a detector miss stays a miss.
                if len(boxes):
                    keypoints, scores = pose(image, bboxes=boxes)
                else:
                    keypoints = np.empty((0, 133, 2), np.float32); scores = np.empty((0, 133), np.float32)
                if keypoints.shape != (len(boxes), 133, 2) or scores.shape != (len(boxes), 133):
                    raise ValueError('unexpected pose shape')
                boxes, conf, keypoints, scores, split = _split(image, boxes, conf, keypoints, scores, pose)
                if not all(np.isfinite(a).all() for a in (boxes, conf, keypoints, scores)):
                    raise ValueError('nonfinite model output')
                tracks = tracker.update_with_detections(supervision.Detections(
                    xyxy=boxes, confidence=conf, class_id=np.zeros(len(boxes), int), data=dict(keypoints=keypoints, scores=scores)))
                frames.append(dict(frame=n, pts_seconds=scan['native_pts_seconds'][n], segment_id=segment, detections=len(boxes)-split, split_boxes=split, tracks=[
                    dict(track_id=f'{segment}/{int(tracks.tracker_id[j])}', bbox=[round(float(v), 2) for v in tracks.xyxy[j]],
                         detection_score=round(float(tracks.confidence[j]), 4),
                         keypoints=[[round(float(x), 2), round(float(y), 2)] for x, y in tracks.data['keypoints'][j]],
                         scores=[round(float(s), 4) for s in tracks.data['scores'][j]]) for j in range(len(tracks))]))
    finally:
        cap.release()
    if part and parted:
        part(packet(frames[parted:]))
    return packet(frames)


def _split(image, boxes, conf, keypoints, scores, pose):
    """Replace each merged two-person box by its left and right crops; returns the arrays and how many boxes were split."""
    import numpy as np
    head = [BODY['nose'], BODY['eye_l'], BODY['eye_r']]
    out = []; split = 0
    for j, (x0, y0, x1, y1) in enumerate(boxes):
        w = x1-x0
        if conf[j] >= SPLIT['min_detection'] and w > y1-y0:
            crops = np.array([[x0, y0, x0+SPLIT['crop']*w, y1], [x1-SPLIT['crop']*w, y0, x1, y1]], np.float32)
            k, s = pose(image, bboxes=crops)
            if (k.shape == (2, 133, 2) and min(s[0][head].min(), s[1][head].min()) >= SPLIT['head_score']
                    and k[1][BODY['nose']][0]-k[0][BODY['nose']][0] >= SPLIT['head_gap']*w
                    and min(math.dist(k[i][BODY['eye_l']], k[i][BODY['eye_r']]) for i in (0, 1)) >= SPLIT['min_eyes']*w):
                out += [(crops[i], conf[j], k[i], s[i]) for i in (0, 1)]; split += 1
                continue
        out.append((boxes[j], conf[j], keypoints[j], scores[j]))
    if not split:
        return boxes, conf, keypoints, scores, 0
    return (np.array([o[0] for o in out], np.float32), np.array([o[1] for o in out], np.float32),
            np.array([o[2] for o in out], np.float32), np.array([o[3] for o in out], np.float32), split)


def _iou(a, b):
    w = min(a[2], b[2])-max(a[0], b[0]); h = min(a[3], b[3])-max(a[1], b[1])
    inter = max(0., w)*max(0., h)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def flags(frames):
    """Frames where the pose layer itself looks unreliable: overlapping people, or a body skeleton jump within one track."""
    rows = []; previous = {}
    for f in frames:
        tracks = f['tracks']
        overlap = any(_iou(a['bbox'], b['bbox']) >= FLAGS['overlap_iou'] for i, a in enumerate(tracks) for b in tracks[i+1:])
        jump = None
        for t in tracks:
            p = previous.get(t['track_id'])
            if p is None or p[0] != f['frame']-1:
                continue
            moves = sorted(math.dist(a, b) for i, (a, b) in enumerate(zip(t['keypoints'][:17], p[1]['keypoints'][:17]))
                           if min(t['scores'][i], p[1]['scores'][i]) >= FLAGS['min_score'])
            box = t['bbox']; diagonal = math.hypot(box[2]-box[0], box[3]-box[1])
            if moves and diagonal > 0 and moves[len(moves)//2]/diagonal > FLAGS['jump_diagonals']:
                jump = dict(track_id=t['track_id'], median_move_diagonals=round(moves[len(moves)//2]/diagonal, 3))
        previous.update({t['track_id']: (f['frame'], t) for t in tracks})
        if overlap or jump:
            rows.append(dict(frame=f['frame'], people=len(tracks), overlapping_people=overlap, skeleton_jump=jump))
    return dict(params=FLAGS, frames=rows,
                basis='person boxes overlapping, or median body-joint move between consecutive frames of one track; diagnostic only')


def _head_frame(k, s):
    """Eye midpoint, eye distance and an 'up' unit vector pointing away from the nose; None without both eyes and nose."""
    if min(s[BODY['eye_l']], s[BODY['eye_r']], s[BODY['nose']]) < SEED['min_score']:
        return None
    (lx, ly), (rx, ry), (nx, ny) = k[BODY['eye_l']], k[BODY['eye_r']], k[BODY['nose']]
    d = math.hypot(lx-rx, ly-ry)
    if d < 4:
        return None
    cx, cy = (lx+rx)/2, (ly+ry)/2
    ux, uy = -(ly-ry)/d, (lx-rx)/d
    if ux*(nx-cx)+uy*(ny-cy) > 0:
        ux, uy = -ux, -uy
    return (cx, cy), d, (ux, uy)


def seed_points(track, width, height):
    """Named surface seeds for one track on one frame: body joints, eyes and geometric hair points. Missing stays missing."""
    k, s = track['keypoints'], track['scores']
    points = {}
    for name, i in BODY.items():
        if s[i] >= SEED['min_score']:
            points[name] = dict(xy=list(k[i]), seed_rule=f'coco133 joint {i} score {s[i]:.2f}')
    for name, i in HAND.items():
        if s[i] >= SEED['min_score']:
            points[name] = dict(xy=list(k[i]), seed_rule=f'coco133 hand tip {i} score {s[i]:.2f}')
    for side in 'lr':
        # Knees below the frame come back clamped to the edge, but hip->knee still points down the thigh.
        hip, knee = BODY[f'hip_{side}'], BODY[f'knee_{side}']
        if min(s[hip], s[knee]) >= SEED['min_score']:
            points[f'thigh_{side}'] = dict(xy=[k[hip][0]+.5*(k[knee][0]-k[hip][0]), k[hip][1]+.5*(k[knee][1]-k[hip][1])],
                                           seed_rule=f'midway from hip {hip} to knee {knee}')
    head = _head_frame(k, s)
    if head:
        (cx, cy), d, (ux, uy) = head
        sx, sy = uy, -ux  # perpendicular to up
        for name, (side, up) in (('hair_front', HAIR['front']), ('head_top', HAIR['top'])):
            points[name] = dict(xy=[cx+ux*up*d, cy+uy*up*d], seed_rule=f'eye midpoint + {up} eye-distance up')
        for name, ear, eye in (('hair_side_l', 'ear_l', 'eye_l'), ('hair_side_r', 'ear_r', 'eye_r')):
            # Outward from the eye midpoint on the same side as that eye.
            ex, ey = k[BODY[eye]]; sign = 1 if (ex-cx)*sx+(ey-cy)*sy > 0 else -1
            side, up = HAIR['side']
            points[name] = dict(xy=[cx+sign*sx*side*d+ux*up*d, cy+sign*sy*side*d+uy*up*d],
                                seed_rule=f'eye midpoint + {side} eye-distance toward {eye} side, {up} up')
    kept = {}
    for name, p in points.items():
        x, y = p['xy']
        # Off-screen joints come back clamped to the frame edge (seen on clip A close-ups); drop them.
        b = SEED['border_px']
        if math.isfinite(x) and math.isfinite(y) and b <= x < width-b and b <= y < height-b:
            kept[name] = dict(xy=[round(x, 2), round(y, 2)], seed_rule=p['seed_rule'])
    return kept


LABELS = dict(head='머리·눈·머리카락 (포즈 기준 자동 배치, 부위 미검증)', arms='팔 (포즈 관절에서 시작, 부위 미검증)',
              hands='엄지·검지 끝 (포즈 손 키포인트, 그림 손에서는 거침, 부위 미검증)',
              move='많이 움직인 곳 (어니언 틈 후보 중 같이 움직이는 것끼리 묶은 대표, 부위 이름 없음)',
              legs='다리·허벅지 (포즈 관절에서 시작, 허벅지는 엉덩이-무릎 중간, 부위 미검증)')


def _marked_crops(video, frame, width, height, seeds, directory):
    """One marked source crop per seed: the pinned visual reference the observation contract requires."""
    from PIL import Image, ImageDraw
    from image_lens import frames
    bgr, = frames.checked(video, frame, frame, width, height, 'seed frame decode geometry/index mismatch')
    image = Image.fromarray(bgr[:, :, ::-1].copy()); half = 120; written = {}
    directory.mkdir()
    for name, s in seeds.items():
        x, y = s['xy']
        crop = image.crop((round(x)-half, round(y)-half, round(x)+half, round(y)+half))
        mark = ImageDraw.Draw(crop)
        mark.ellipse((half-10, half-10, half+10, half+10), outline='black', width=4)
        mark.ellipse((half-10, half-10, half+10, half+10), outline='yellow', width=2)
        path = directory/f'{name}.png'
        crop.save(path)
        written[name] = dict(path=str(path), sha256=paths.sha(path))
    return written


def seed(pose_path, frame, job, evidence_dir, track_id=None, length=SEED['max_frames']):
    """One observation job per seed group (the surface contract allows twelve points per job)."""
    packet, pose_file = paths.load(pose_path, max_bytes=64*1024*1024)
    if type(packet) is not dict or packet.get('schema') != 'image-lens.pose/1':
        raise ValueError('image-lens pose packet required')
    video = paths.confined(packet['source_video'], exists=True, max_bytes=512*1024*1024)
    if paths.sha(video) != packet['source_sha256']:
        raise ValueError('pose source changed')
    scan, scan_path = paths.load(packet['canonical_shots']['path'])
    if paths.sha(scan_path) != packet['canonical_shots']['sha256']:
        raise ValueError('pose shot scan changed')
    rows = {f['frame']: f for f in packet['frames']}
    if frame not in rows:
        raise ValueError('seed frame outside pose range')
    tracks = rows[frame]['tracks']
    if track_id is not None:
        tracks = [t for t in tracks if t['track_id'] == track_id]
    if not tracks:
        raise ValueError('no pose track on the seed frame; choose another frame')
    # Default: the largest person box on the seed frame.
    track = max(tracks, key=lambda t: (t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1]))
    segment = next(s for s in scan['analysis_segments'] if s['start_frame'] <= frame <= s['end_frame'])
    if type(length) is not int or not 2 <= length <= SEED['max_frames']:
        raise ValueError('job length 2..160 frames')
    end = min(segment['end_frame'], frame+length-1)
    if end == frame:
        raise ValueError('seed frame is the last frame of its shot')
    seeds = seed_points(track, packet['width'], packet['height'])
    from image_lens import body_motion
    plausible, reason = body_motion.skeleton_check(track)
    shot = body_motion.framing(track, packet['width'], packet['height']); scope = 'full' if plausible else None
    same = [t for n in range(frame, end+1) for t in rows.get(n, {}).get('tracks', []) if t['track_id'] == track['track_id']]
    head = body_motion.chibi(same)
    if head:
        # Named arm and leg seeds would be confident and wrong; the face and unnamed corners are tracked instead.
        plausible, reason = False, f'chibi proportions (eye gap {head:.2f} of eye to ankle)'
    if not plausible and shot == 'upper' and not head:
        # Hips below the frame: name the arms from the shoulders, leave the legs out.
        plausible, why = body_motion.upper_check(track)
        if plausible:
            scope = 'upper'; reason = f'upper body (full check: {reason})'
            seeds = {name: p for name, p in seeds.items() if name not in GROUPS['legs']}
    if not plausible:
        seeds = {name: p for name, p in seeds.items() if name in GROUPS['head']}
    from image_lens import onion
    seeds.update(onion.gap_seeds(video, frame, end, track['bbox'], GROUPS['move']))
    if not any(name in seeds for names in GROUPS.values() for name in names):
        raise ValueError('no usable pose seeds on this frame')
    crops = _marked_crops(video, frame, packet['width'], packet['height'], seeds, paths.confined(evidence_dir, new=True))
    jobs = []
    for group, names in GROUPS.items():
        points = [dict(point_id=name.replace('_', '-'), observed_label=f'{group} pose seed: {name}', anatomical_interpretation=None,
                       seed_rule=seeds[name]['seed_rule'],
                       visual_reference=dict(feature_id=name.replace('_', '-'), frame=frame, kind='marked-source-crop', **crops[name]),
                       anchors=[dict(frame=frame, xy=seeds[name]['xy'], status='source_image_observation',
                                     uncertainty_px=SEED['uncertainty_px'][name.startswith(('hair', 'head'))])])
                  for name in names if name in seeds]
        if points:
            jobs.append(dict(name=f'{job}-{group}', group=group, source_video=packet['source_video'], source_sha256=packet['source_sha256'],
                             width=packet['width'], height=packet['height'], fps=packet['fps'],
                             start_frame=frame, end_frame=end, seed_frame=frame, points=points, mode='single-anchor',
                             observation_scope='visible_surface_2d', anatomical_claims=False, common_metric_excluded_frames=[frame]))
    flagged = any(r['frame'] == frame for r in packet['flags']['frames'])
    return dict(schema='team-eye.observations/1', seed_source=dict(pose=str(pose_file), pose_sha256=paths.sha(pose_file),
                track_id=track['track_id'], frame=frame, seed_frame_flagged=flagged, skeleton=dict(plausible=plausible, reason=reason, scope=scope, framing=shot, chibi=head),
                basis='pose-model joints and eye-based hair geometry; candidate visible-surface seeds, not verified anatomy'),
                jobs=jobs)


def bindings(queries_path, tracks_path, job):
    """Report bindings for one seed-group job (a report shows at most eight features)."""
    queries, _ = paths.load(queries_path)
    tracks = paths.confined(tracks_path, exists=True, max_bytes=16*1024*1024)
    found = [j for j in queries.get('jobs', []) if j.get('name') == job]
    if len(found) != 1 or found[0].get('group') not in GROUPS:
        raise ValueError('pose seed job name required')
    group = found[0]['group']
    return dict(schema='team-eye.motion-bindings/1', source_tracks_sha256=paths.sha(tracks), basis='source_frame_grouping_only',
                actors=[dict(actor_id=group, label=LABELS[group],
                             features=[dict(point_id=p['point_id'], label=p['point_id']) for p in found[0]['points']])])
