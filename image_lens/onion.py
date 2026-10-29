"""Onion view and per-height flow for one window. Named points miss what has no pose keypoint (a hip pushed out under a
skirt, a prop); splitting the person box into horizontal bands and following corner features in each band shows which
height moved where. The onion image overlays the first frame (cyan) and the last (red) so the motion reads in one picture."""
import math

# AKAGE pose 1 (black dress, f32..46): head right 20, chest 3, waist left 15, skirt left 8, hem right 4; the three
# repeats agree within 3 px. Below min_px a band counts as still.
ONION = dict(bands=('맨 위', '위', '가운데', '아래', '맨 아래'), features=40, quality=.02, spacing=10, min_features=5,
             min_px=8., pad=.1, arrow_scale=2)


def _box(bbox, w, h):
    x0, y0, x1, y1 = bbox
    px, py = ONION['pad']*(x1-x0), ONION['pad']*(y1-y0)
    return max(0, int(x0-px)), max(0, int(y0-py)), min(w, int(x1+px)), min(h, int(y1+py))


def bands(video, start, end, bbox, camera=None, image=None):
    """[{band, y, dx, dy, features}] of the median feature shift start->end per band, minus the camera shift.
    A band with too few surviving features gets dx=dy=None. Writes the onion image when `image` is a path."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, first = cap.read()
        if not ok:
            return []
        h, w = first.shape[:2]
        x0, y0, x1, y1 = _box(bbox, w, h)
        prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        n = len(ONION['bands']); step = (bbox[3]-bbox[1])/n
        edges = [(int(bbox[1]+i*step), int(bbox[1]+(i+1)*step)) for i in range(n)]
        seeds = []
        for top, bottom in edges:
            mask = np.zeros_like(prev); mask[max(0, top):min(h, bottom), x0:x1] = 255
            p = cv2.goodFeaturesToTrack(prev, ONION['features'], ONION['quality'], ONION['spacing'], mask=mask)
            seeds.append(np.zeros((0, 1, 2), np.float32) if p is None else p)
        sizes = [len(p) for p in seeds]
        origin = np.concatenate(seeds); cur = origin.copy(); alive = np.ones(len(origin), bool)
        last = prev
        for _ in range(start+1, end+1):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(cur):
                cur, status, _ = cv2.calcOpticalFlowPyrLK(last, gray, cur, None, winSize=(31, 31), maxLevel=3)
                alive &= status.ravel() == 1
            last = gray
    finally:
        cap.release()
    cam = camera or (0., 0.)
    shift = (cur-origin).reshape(-1, 2)
    result = []; at = 0
    for name, (top, bottom), size in zip(ONION['bands'], edges, sizes):
        keep = alive[at:at+size]; d = shift[at:at+size][keep]; at += size
        if len(d) < ONION['min_features']:
            result.append(dict(band=name, y=[top, bottom], dx=None, dy=None, features=int(len(d))))
            continue
        mx, my = np.median(d, 0)
        result.append(dict(band=name, y=[top, bottom], dx=round(float(mx)-cam[0], 1), dy=round(float(my)-cam[1], 1), features=int(len(d))))
    if image:
        _image(cv2, np, first, last, (x0, y0, x1, y1), result, image)
    return result


def _image(cv2, np, first, last, box, result, path):
    x0, y0, x1, y1 = box
    a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]; b = last[y0:y1, x0:x1]
    picture = np.dstack([b, b, a])  # dark only at the start -> cyan, dark only at the end -> red
    centre = (x1-x0)//2
    for band in result:
        top = band['y'][0]-y0
        cv2.line(picture, (0, top), (x1-x0, top), (150, 150, 150), 1)
        if band['dx'] is not None:
            mid = (band['y'][0]+band['y'][1])//2-y0; s = ONION['arrow_scale']
            cv2.arrowedLine(picture, (centre, mid), (int(centre+s*band['dx']), int(mid+s*band['dy'])), (0, 140, 0), 4, tipLength=.3)
    cv2.imwrite(str(path), picture)


def lines(result, image_name=None):
    from image_lens import motion_report
    parts = []
    for band in result:
        if band['dx'] is None:
            parts.append(f"{band['band']} 모름")
        elif math.hypot(band['dx'], band['dy']) < ONION['min_px']:
            parts.append(f"{band['band']} 그대로")
        else:
            parts.append(f"{band['band']} {motion_report.KO[motion_report.direction(band['dx'], band['dy'])].replace('화면 ', '')} {math.hypot(band['dx'], band['dy']):.0f}px")
    if not parts:
        return []
    out = ['- 높이별 흐름 (사람 상자 5등분, 카메라 뺌): ' + ' · '.join(parts)]
    if image_name:
        out.append(f"  - 어니언 그림: {image_name} (청록=시작, 빨강=끝, 초록 화살표=띠별 이동 ×{ONION['arrow_scale']})")
    return out


# Seeds where the onion gap is thickest: the band between the start outline and a later outline is widest where that
# part moved most (AKAGE pose 1: sleeves and hair tips; MikuMiku hammer: the head's lower edge). Gaps also open where
# a part arrives, so a seed must sit on an outline of the start frame inside the gap. Thirds catch moves that return.
GAPS = dict(points=6, diff=40, min_area=80, min_half_px=4., spacing_px=40., edge=(50, 150), switch_share=.3)


def gap_seeds(video, start, end, bbox, names):
    """{name: {xy, seed_rule}} at the thickest start-frame outline points of the onion gaps, one per height band first."""
    import cv2
    import numpy as np
    from image_lens import camera, motion_report
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, bgr = cap.read()
        if not ok:
            return {}
        first = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); h, w = first.shape
        x0, y0, x1, y1 = _box(bbox, w, h)
        # Stop before a drawing swap (costume change, wipe): a gap across it is the new drawing, not motion.
        kept = {start: first}; prev = first
        for n in range(start+1, end+1):
            ok, bgr = cap.read()
            if not ok:
                break
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if (cv2.absdiff(prev, image)[y0:y1, x0:x1] > GAPS['diff']).mean() > GAPS['switch_share']:
                break
            kept[n] = prev = image
    finally:
        cap.release()
    end = max(kept)
    if end == start:
        return {}
    later = {n: kept[n] for n in sorted({start+(end-start)//3, start+2*(end-start)//3, end} - {start})}
    moves = camera.path(video, start, end)
    outline = cv2.dilate(cv2.Canny(first, *GAPS['edge']), np.ones((3, 3), np.uint8)) > 0
    found = []
    for n, image in later.items():
        if image is None:
            continue
        shift = camera.between(moves, start, n)
        if shift and math.hypot(*shift) >= motion_report.STROKE['major_min_px']:
            image = cv2.warpAffine(image, np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = cv2.absdiff(first, image)[y0:y1, x0:x1] > GAPS['diff']
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        count, labels = cv2.connectedComponents(mask)
        for i in range(1, count):
            region = labels == i
            if region.sum() < GAPS['min_area']:
                continue
            on_outline = np.where(region & outline, depth, 0)
            if on_outline.max() < GAPS['min_half_px']:
                continue  # an arrival gap: nothing of the start frame to follow there
            y, x = np.unravel_index(on_outline.argmax(), on_outline.shape)
            found.append((2*float(depth[region].max()), int(x), int(y), n))
    found.sort(reverse=True)
    step = (bbox[3]-bbox[1])/len(ONION['bands'])
    def band(y):
        return min(len(ONION['bands'])-1, max(0, int((y-bbox[1])//step)))
    chosen = []
    def take(candidate):
        if len(chosen) < GAPS['points'] and all(math.dist(candidate[1:3], c[1:3]) >= GAPS['spacing_px'] for c in chosen):
            chosen.append(candidate)
    for b in range(len(ONION['bands'])):  # spread over heights first so a big arm swing does not take every seed
        best = next((c for c in found if band(c[2]) == b), None)
        if best:
            take(best)
    for candidate in found:
        take(candidate)
    chosen.sort(key=lambda c: -c[0])
    return {name: dict(xy=[float(x), float(y)], seed_rule=f'onion gap {gap:.0f} px wide by f{n}, on a start-frame outline')
            for name, (gap, x, y, n) in zip(names, chosen)}
