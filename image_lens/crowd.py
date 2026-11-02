"""Before look: the frames where the person detector sees the most people, as one picture, so the reader can count them by
eye and pick `look --max-people`. Only the detector runs, on a frame every CROWD['every_s'] seconds read in order
(kaguya full, 110 s 1080p: reading every frame in order 2.6 s against 16.5 s seeking to each of the 221 picked ones).
Counts are rough: people overlapping into one box count once, and drawn figures or posters can count as people."""
import json

CROWD = dict(every_s=.5, min_score=.3, same_iou=.6, top=4, apart_s=2., tile=640, label=28, max_people=8)


def distinct(boxes, scores, same_iou=CROWD['same_iou']):
    """Boxes by descending score, dropping one that overlaps a kept box by IoU >= same_iou: the detector's NMS (.45, made
    for tracking) still left two near-equal boxes on one figure (kaguya full 56.55 s: two on the front girl, while the two girls
    behind her shared one box). Partly overlapping extra boxes stay (59.55 s: still 3 boxes on 2 people)."""
    def iou(a, b):
        w = max(0., min(a[2], b[2])-max(a[0], b[0])); h = max(0., min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h
        return w*h/union if union > 0 else 0.
    kept = []
    for i in sorted(range(len(boxes)), key=lambda i: -float(scores[i])):
        if all(iou(boxes[i], k) < same_iou for k in kept):
            kept.append(boxes[i])
    return kept


def ranked(rows, fps, top=CROWD['top'], apart_s=CROWD['apart_s']):
    """Up to `top` rows with the most boxes (ties: larger total box area, then earlier), at least `apart_s` apart."""
    out = []
    for row in sorted(rows, key=lambda r: (-len(r['boxes']), -sum((b[2]-b[0])*(b[3]-b[1]) for b in r['boxes']), r['frame'])):
        if not row['boxes'] or len(out) == top:
            break
        if all(abs(row['frame']-o['frame']) >= apart_s*fps for o in out):
            out.append(row)
    return out


def _tile(cv2, np, image, row, fps):
    from image_lens.frames import label
    t = CROWD['tile']; s = t/image.shape[1]
    small = cv2.resize(image, (t, max(1, round(image.shape[0]*s))), interpolation=cv2.INTER_AREA)
    for i, (x0, y0, x1, y1) in enumerate(row['boxes'], 1):
        cv2.rectangle(small, (round(x0*s), round(y0*s)), (round(x1*s), round(y1*s)), (0, 220, 255), 2)
        label(cv2, small, str(i), (round(x0*s)+4, round(y0*s)+22), .6, 1)
    tile = np.zeros((small.shape[0]+CROWD['label'], t, 3), np.uint8); tile[CROWD['label']:] = small
    label(cv2, tile, f"{row['frame']/fps:.2f}s f{row['frame']}: {len(row['boxes'])}", (4, 21), .6, 1)
    return tile


def run(video, detector, out, every_s=CROWD['every_s']):
    """Writes crowd.jpg (the top frames, boxes numbered) and crowd.json (count per sampled frame and the suggestion)."""
    import time
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from image_lens import paths, person_models, pose
    if not .1 <= every_s <= 5:
        raise ValueError('--every 0.1..5 seconds')
    root = paths.confined(out, new=True)
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    started = time.monotonic()
    selection = person_models.resolve(detector)
    det = pose.person_detector(selection)
    cap = cv2.VideoCapture(str(source))
    rows, images = [], []
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not fps or not count:
            raise ValueError('video has no frame rate or frame count')
        step = max(1, round(every_s*fps))
        def boxes(n, image):
            b, c = det(image)
            b, c = np.asarray(b, np.float32).reshape(-1, 4), np.asarray(c, np.float32)
            return dict(frame=n, boxes=[[round(float(v), 1) for v in x] for x in distinct(b[c >= CROWD['min_score']], c[c >= CROWD['min_score']])])
        # The next picked frame is decoded while this one is detected; at most one frame waits.
        with ThreadPoolExecutor(1) as worker:
            pending = None
            for n in range(count):
                if not cap.grab():
                    break
                if n % step == 0:
                    ok, image = cap.retrieve()
                    if ok:
                        if pending:
                            rows.append(pending.result())
                        pending = worker.submit(boxes, n, image)
            if pending:
                rows.append(pending.result())
        best = ranked(rows, fps)
        images = []
        for r in best:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r['frame'])
            ok, image = cap.read()
            if ok:
                images.append((image, r))
    finally:
        cap.release()
    root.mkdir(parents=True)
    if images:
        tiles = [_tile(cv2, np, image, r, fps) for image, r in images]
        height = max(x.shape[0] for x in tiles)
        tiles = [np.vstack([x, np.zeros((height-x.shape[0], x.shape[1], 3), np.uint8)]) for x in tiles]
        tiles += [np.zeros_like(tiles[0])]*(len(tiles) % 2)
        cv2.imwrite(str(root/'crowd.jpg'), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    most = len(best[0]['boxes']) if best else 0
    result = dict(status='ok', video=source.name, fps=fps, every_frames=step, sampled=len(rows),
                  detector=selection['kind'], min_score=CROWD['min_score'], image='crowd.jpg' if images else None,
                  top=[dict(frame=r['frame'], seconds=round(r['frame']/fps, 2), people=len(r['boxes']), boxes=r['boxes']) for r in best],
                  most_boxes=most, suggested_max_people=max(1, min(CROWD['max_people'], most)),
                  counts=[[r['frame'], len(r['boxes'])] for r in rows], elapsed_seconds=round(time.monotonic()-started, 2),
                  note='rough detector count: look at crowd.jpg and count the people yourself before choosing look --max-people')
    (root/'crowd.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return {k: v for k, v in result.items() if k != 'counts'}
