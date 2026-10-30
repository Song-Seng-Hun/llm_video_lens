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


# The onion overlays start and end only: a spin or a hand that goes out and back ends where it began, so the overlay
# looks still (Seedance blind test, all three spins missed). The frame most unlike the start, away from both ends,
# goes next to it.
PEAK = dict(min_frames=8, margin=.15, scale=4, plateau=.95)


def _crop(cv2, image, bbox):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _box(bbox, w, h) if bbox else (0, 0, w, h)
    return image[y0:y1, x0:x1]


def peak(video, start, end, bbox, image):
    """Frame number of the frame most unlike `start` (mean grey difference inside the padded person box), searched
    away from both window ends, drawn as a grey panel right of the onion image. None for short windows."""
    import cv2
    import numpy as np
    if end-start+1 < PEAK['min_frames']:
        return None
    lo, hi = start+max(1, int(PEAK['margin']*(end-start))), end-max(1, int(PEAK['margin']*(end-start)))
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        found = {}; first = None
        for n in range(start, hi+1):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(cv2.cvtColor(_crop(cv2, frame, bbox), cv2.COLOR_BGR2GRAY), None, fx=1/PEAK['scale'], fy=1/PEAK['scale'], interpolation=cv2.INTER_AREA).astype(np.float32)
            if first is None:
                first = small
            elif n >= lo:
                found[n] = float(np.abs(small-first).mean())
        if not found:
            return None
        # Once a part has left its start outline the difference stops growing; take the middle of that plateau.
        plateau = [n for n, d in found.items() if d >= PEAK['plateau']*max(found.values())]
        n = plateau[len(plateau)//2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        return n if ok else None
    onion = cv2.imread(str(image))
    panel = cv2.cvtColor(cv2.cvtColor(_crop(cv2, frame, bbox), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if onion is not None and onion.shape[0] == panel.shape[0]:
        cv2.putText(panel, f'f{n}', (6, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 200, 255), 2)
        cv2.imwrite(str(image), np.hstack([onion, np.full((onion.shape[0], 6, 3), 255, np.uint8), panel]))
    return n


def who(video, frame, people, image):
    """The whole frame with each person's pose box and name (P1..), so an image reader can tell who is who."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, picture = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    for p in people:
        x0, y0, x1, y1 = map(int, p['bbox'])
        cv2.rectangle(picture, (x0, y0), (x1, y1), (0, 0, 255), 6)
        cv2.putText(picture, p['name'], (x0+12, y0+70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
    cv2.imwrite(str(image), picture)
    return image.name


def still(video, frame, bbox, image):
    """One native frame (cropped to the padded person box when there is one) for a range too short to track."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, picture = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    cv2.imwrite(str(image), _crop(cv2, picture, bbox))
    return image.name


SHEET = dict(tile=360, label=20, columns=4, rows=5, strip_height=200, gap_s=.05)


def strip(video, start, end, bbox, image, count=5):
    """`count` frames from start to end in order (evenly spaced, the far frame of `peak` in place of its nearest), side
    by side, frame numbers on each: a turn shows as front, side, back phases in order, which one far frame cannot."""
    import cv2
    import numpy as np
    picks = sorted({round(start+i*(end-start)/(count-1)) for i in range(count)})
    far = peak(video, start, end, bbox, None)
    if far is not None and far not in picks:
        inner = picks[1:-1] or picks
        picks = sorted(set(picks) - {min(inner, key=lambda n: abs(n-far))} | {far})
    cap = cv2.VideoCapture(str(video)); panels = []
    try:
        for n in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            if ok:
                panel = _crop(cv2, frame, bbox)
                cv2.putText(panel, f'f{n}', (6, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 200, 255), 2)
                panels.append(panel)
    finally:
        cap.release()
    if not panels:
        return None
    cv2.imwrite(str(image), np.hstack([np.hstack([p, np.full((p.shape[0], 6, 3), 255, np.uint8)]) for p in panels]))
    return dict(image=image.name, frames=picks)


def sheets(root, items):
    """Onion images of a whole look on a few contact sheets (time label above each), so an image reader sees the
    gestures the text loses (heart hands, V signs on Seedance close-ups). items: [(start_s, end_s, image name, note)],
    note '' for an onion image, 'still' for a short cut's still, 'strip' for a frame strip. Stills of back-to-back
    cuts share one full-width row in time order (Seedance: a spin made of seven 3-6 frame cuts); a strip gets its own
    row under its window. Rows, not tiles, are counted per sheet."""
    import cv2
    import numpy as np
    t, top = SHEET['tile'], SHEET['label']; width = t*SHEET['columns']
    def read(name):
        return cv2.imread(str(root/name))
    def label(image, text):
        tile = np.zeros((image.shape[0]+top, image.shape[1], 3), np.uint8); tile[top:] = image
        cv2.putText(tile, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        return tile
    def fit(image, w, h):
        k = min(w/image.shape[1], h/image.shape[0])
        return cv2.resize(image, (max(1, int(image.shape[1]*k)), max(1, int(image.shape[0]*k))), interpolation=cv2.INTER_AREA)
    def full_row(images, texts):
        # One row across the sheet: equal height, shrunk until the row fits the sheet width.
        height = SHEET['strip_height']
        while True:
            parts = [label(fit(i, 10**6, height), x) for i, x in zip(images, texts)]
            if sum(p.shape[1] for p in parts)+4*len(parts) <= width or height < 40:
                break
            height = int(height*.85)
        row = np.hstack([np.hstack([p, np.zeros((p.shape[0], 4, 3), np.uint8)]) for p in parts])[:, :width]
        return np.hstack([row, np.zeros((row.shape[0], width-row.shape[1], 3), np.uint8)])
    units = []  # ('tile', tile) or ('row', row)
    i = 0
    while i < len(items):
        start, end, name, note = items[i]
        if note == 'still':
            run = [items[i]]
            while i+1 < len(items) and items[i+1][3] == 'still' and items[i+1][0]-run[-1][1] <= SHEET['gap_s']:
                i += 1; run.append(items[i])
            images = [(read(n), f'{s:.2f}-{e:.2f}s still') for s, e, n, _ in run]
            images = [x for x in images if x[0] is not None]
            if len(images) > 1:
                units.append(('row', full_row(*zip(*images))))
            elif images:
                units.append(('tile', label(fit(images[0][0], t, 1.5*t), images[0][1])))
        else:
            image = read(name)
            if image is not None:
                if note == 'strip':
                    units.append(('row', full_row([image], [f'{start:.2f}-{end:.2f}s strip'])))
                else:
                    units.append(('tile', label(fit(image, t, 1.5*t), f'{start:.2f}-{end:.2f}s')))
        i += 1
    rows = []; pending = []
    def flush():
        if pending:
            height = max(x.shape[0] for x in pending)
            row = np.hstack([np.vstack([x, np.zeros((height-x.shape[0], x.shape[1], 3), np.uint8)]) for x in pending])
            rows.append(np.hstack([row, np.zeros((height, width-row.shape[1], 3), np.uint8)]))
            pending.clear()
    for kind, image in units:
        if kind == 'row':
            flush(); rows.append(image)
        else:
            image = np.hstack([image, np.zeros((image.shape[0], t-image.shape[1], 3), np.uint8)])
            pending.append(image)
            if len(pending) == SHEET['columns']:
                flush()
    flush()
    written = []
    for first in range(0, len(rows), SHEET['rows']):
        path = root/f'onion-sheet-{len(written)+1}.jpg'
        cv2.imwrite(str(path), np.vstack(rows[first:first+SHEET['rows']])); written.append(path.name)
    return written


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
        out.append(f"  - 어니언 그림: {image_name} (청록=시작, 빨강=끝, 초록 화살표=띠별 이동 ×{ONION['arrow_scale']}; 오른쪽 회색=시작과 가장 다른 가운데 프레임)")
    return out


# Seeds where the onion gap is thickest: the band between the start outline and a later outline is widest where that
# part moved (AKAGE pose 1: sleeves and hair tips; MikuMiku hammer: the hammer head). Gaps also open where a part
# arrives, so a candidate must sit on an outline of the start frame inside the gap. Thirds catch moves that return.
# Gap width is not travel (the hammer scene's widest gap, 124 px, moved 39 px; the hammer moved 450 px), and joined gaps
# are not one part, so up to 12 candidates are seeded and tracked; `parts` then keeps one point per group that moved
# together. LK flow cannot follow the hammer swing (110-170 px of 450, and a face point dragged 500 px by it), so the
# grouping uses the real tracks. Together = paths never further apart than 16 px or half the larger travel: at 8 px / a
# quarter, AKAGE kept 6 points in 11 of 19 windows (drawn parts deform as they move) and the hammer got two points.
GAPS = dict(points=6, diff=40, min_area=80, min_half_px=4., spacing_px=40., edge=(50, 150), switch_share=.3,
            keep_share=.4, min_travel_px=8., together_px=16., together_share=.5)


def gap_seeds(video, start, end, bbox, names):
    """{name: {xy, seed_rule}}: candidates on start-frame outlines inside onion gaps, widest gap first; `parts` picks."""
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
    later = sorted({start+(end-start)//3, start+2*(end-start)//3, end} - {start})
    moves = camera.path(video, start, end)
    outline = cv2.dilate(cv2.Canny(first, *GAPS['edge']), np.ones((3, 3), np.uint8)) > 0
    found = []
    for n in later:
        image = kept[n]; shift = camera.between(moves, start, n)
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
    candidates = []
    for c in found:
        if len(candidates) < len(names) and all(math.dist(c[1:3], k[1:3]) >= GAPS['spacing_px'] for k in candidates):
            candidates.append(c)
    return {name: dict(xy=[float(x), float(y)], seed_rule=f'onion gap {gap:.0f} px wide by f{n}, on a start-frame outline (candidate)')
            for name, (gap, x, y, n) in zip(names, candidates)}


def parts(tracks, end, frame_motion=None):
    """Point ids to report: one per group of candidates whose tracked paths stay together, farthest first, at most 6.
    Paths are shifts from the first frame minus the camera; a group travelling under keep_share of the farthest is dropped."""
    from image_lens import camera
    frames = [f for f in tracks['frames'] if f['frame'] <= end]
    if not frames:
        return []
    origin = {p['point_id']: p['xy'] for p in frames[0]['points'] if p.get('visible') and p.get('xy')}
    paths = {pid: [] for pid in origin}
    for f in frames:
        cam = camera.between(frame_motion, frames[0]['frame'], f['frame']) or (0., 0.)
        seen = {p['point_id']: p['xy'] for p in f['points'] if p.get('visible') and p.get('xy')}
        for pid, xy in origin.items():
            paths[pid].append((seen[pid][0]-xy[0]-cam[0], seen[pid][1]-xy[1]-cam[1]) if pid in seen else None)
    travel = {pid: max((math.hypot(*d) for d in p if d), default=0.) for pid, p in paths.items()}
    def together(i, j):
        gaps = [math.dist(a, b) for a, b in zip(paths[i], paths[j]) if a and b]
        return bool(gaps) and max(gaps) < max(GAPS['together_px'], GAPS['together_share']*max(travel[i], travel[j]))
    order = sorted(travel, key=lambda pid: -travel[pid]); chosen = []
    for pid in order:
        if travel[pid] < max(GAPS['min_travel_px'], GAPS['keep_share']*travel[order[0]]) or len(chosen) >= GAPS['points']:
            break
        if not any(together(pid, other) for other in chosen):
            chosen.append(pid)
    return sorted(chosen)
