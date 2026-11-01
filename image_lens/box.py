"""Re-track from a box the agent gives. look follows the largest person; when its small whole frame shows the green box
on the wrong figure (or on none), the agent gives a box and this picks the points mark should track: the joints of the
person that box covers, or corners inside it when no person does."""
from image_lens import mark

# COCO body indices; names as look's trails spell them ('wrist-l' = the pose model's left). First come first kept.
JOINTS = (('nose', 0), ('wrist-l', 9), ('wrist-r', 10), ('elbow-l', 7), ('elbow-r', 8), ('ankle-l', 15), ('ankle-r', 16),
          ('knee-l', 13), ('knee-r', 14), ('shoulder-l', 5), ('shoulder-r', 6))
BOX = dict(min_iou=.3, min_score=.3, min_side_px=16, corner_quality=.02, corner_spacing=.2)


def iou(a, b):
    w = max(0., min(a[2], b[2])-max(a[0], b[0])); h = max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h
    return w*h/union if union > 0 else 0.


def person(tracks, box):
    """(pose track, overlap) of the person the box covers best, or (None, best overlap) under BOX['min_iou']."""
    scored = sorted(((iou(t['bbox'], box), i) for i, t in enumerate(tracks or ())), reverse=True)
    if not scored:
        return None, 0.
    best, i = scored[0]
    return (tracks[i] if best >= BOX['min_iou'] else None), best


def joints(track, box, w, h):
    """[(name, (x, y))] of up to mark's point limit confident joints inside the frame (and the box)."""
    from image_lens import pose
    out = []; e = pose.SEED['border_px']  # off-screen joints come back clamped to the edge, as pose's seeds found
    for name, k in JOINTS:
        x, y = track['keypoints'][k][:2]
        if track['scores'][k] >= BOX['min_score'] and e <= x < w-e and e <= y < h-e and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            out.append((name, (round(x), round(y))))
    return out[:mark.MARK['max_points']]


def corners(gray, box):
    """[(p1.., (x, y))] of up to mark's point limit spread-out corners inside the box, top to bottom."""
    import cv2
    x0, y0, x1, y1 = box
    found = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], mark.MARK['max_points'], BOX['corner_quality'],
                                    max(8., BOX['corner_spacing']*min(x1-x0, y1-y0)))
    xy = sorted((round(float(y))+y0, round(float(x))+x0) for x, y in (found.reshape(-1, 2) if found is not None else ()))
    return [(f'p{i+1}', (x, y)) for i, (y, x) in enumerate(xy)]


def parse(text, w, h):
    x0, y0, x1, y1 = (int(v) for v in text.split(','))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1-x0 < BOX['min_side_px'] or y1-y0 < BOX['min_side_px']:
        raise ValueError(f"box must be x0,y0,x1,y1 inside the frame, each side at least {BOX['min_side_px']} px")
    return x0, y0, x1, y1


def run(video, look_dir, frame, box, checkpoint, out, end=None):
    import cv2
    from image_lens import paths
    from pathlib import Path
    root = Path(look_dir)  # mark confines shots.json; pose.json is confined below
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    bgr = mark._frame(source, frame); h, w = bgr.shape[:2]
    region = parse(box, w, h)
    pose, _ = paths.load(paths.confined(root/'pose.json', exists=True), max_bytes=64*1024*1024)
    if pose['source_sha256'] != paths.sha(source):
        raise ValueError('look run belongs to another video')
    tracks = next((f['tracks'] for f in pose['frames'] if f['frame'] == frame), None)
    who, overlap = person(tracks, region)
    picked = joints(who, region, w, h) if who else []
    where = f"- 상자 {','.join(map(str, region))} (f{frame}): "
    if picked:
        note = where+f"사람 1명(상자와 겹침 {overlap:.2f})의 관절 {len(picked)}개를 추적(이름=자세 모델의 관절)"
    else:
        picked = corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), region)
        why = ('그 프레임의 자세 없음' if tracks is None else f'사람 없음(가장 많이 겹친 사람 {overlap:.2f})' if not who
               else '겹친 사람의 관절이 상자 안에 없음')
        note = where+f"{why} → 상자 안 모서리 점 {len(picked)}개를 추적(이름=위에서부터 p1..)"
    if not picked:
        raise ValueError('no joint or corner to track inside the box; give a box with visible outlines')
    return mark.run(source, root/'shots.json', frame, [f'{n}={x},{y}' for n, (x, y) in picked], checkpoint, out, end,
                    snap_points=False, subject=region, note=note)
