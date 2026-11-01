"""Onion view and per-height flow for one window. Named points miss what has no pose keypoint (a hip pushed out under a
skirt, a prop); splitting the person box into horizontal bands and following corner features in each band shows which
height moved where. The onion image overlays the first frame (cyan) and the last (red) so the motion reads in one picture."""
import math
from image_lens.frames import label as outlined

# Clip A pose 1 (black dress, f32..46): head right 20, chest 3, waist left 15, skirt left 8, hem right 4; the three
# repeats agree within 3 px. Below min_px a band counts as still.
ONION = dict(bands=('맨 위', '위', '가운데', '아래', '맨 아래'), features=40, quality=.02, spacing=10, min_features=5,
             min_px=8., pad=.1, arrow_scale=2, arrow_head=9, arrow_margin=4)


def _box(bbox, w, h):
    x0, y0, x1, y1 = bbox
    px, py = ONION['pad']*(x1-x0), ONION['pad']*(y1-y0)
    return max(0, int(x0-px)), max(0, int(y0-py)), min(w, int(x1+px)), min(h, int(y1+py))


# The pictures (onion, still, strip, peak panel) show at least half the frame's shorter side each way, between 1.6:1
# and 1:1.6 (wide or tall, the same limit):
# a pose box around the visible part of a character in a square card (mikumiku) cut 1372x395 strips of one body part.
# Measurements (bands, gaps, peak's difference) stay on the padded box.
VIEW = dict(min_side=.5, max_aspect=1.6)


def _view(bbox, w, h):
    x0, y0, x1, y1 = _box(bbox, w, h)
    side = VIEW['min_side']*min(w, h)
    bw, bh = max(x1-x0, side), max(y1-y0, side)
    bw, bh = min(w, max(bw, bh/VIEW['max_aspect'])), min(h, max(bh, bw/VIEW['max_aspect']))
    cx, cy = (x0+x1)/2, (y0+y1)/2
    vx, vy = int(min(max(0, cx-bw/2), w-bw)), int(min(max(0, cy-bh/2), h-bh))
    return vx, vy, vx+int(bw), vy+int(bh)


# Each part's lead point (digest.limbs) drawn frame by frame on the onion: the overlay shows only start and end, the
# trail shows how the part got there and where it bunched up (slow) or spread out (fast).
# Every frame's dot looked evenly spaced even where the speed fell (Live2D sweep: the hand went 21 -> 0 px/f in its last
# 6 frames, steps over 3 frames 61 -> 54 -> 32 -> 10 px); a bigger dot at equal times makes the spacing readable.
TRAILS = dict(parts=6, dot=2, tick=5, tick_seconds=.1, start=9, end=8, line=2, box=(440, 680),
              colors=((0, 220, 255), (255, 0, 255), (255, 80, 0), (0, 128, 255), (255, 255, 255), (160, 0, 160)),
              names=('노랑', '자홍', '파랑', '주황', '흰색', '보라'))  # BGR; not cyan, red or green, which the onion already uses


def trails(packets, points, seed, end, camera=None, limit=None, body=None):
    """[{point, xy: [(frame, x, y)]}] for up to TRAILS['parts'] point ids, visible frames after the seed, in screen
    coordinates minus the camera shift since the seed (`camera` is camera.path; None = no camera move). With a camera
    move, frames whose camera shift is unknown are left out rather than drawn in the wrong place. `body` (body_path)
    instead takes out the body centre's move since its first frame (camera included: the centre is on screen)."""
    out = []
    shift = camera
    if body:
        b0 = body[min(body)]; shift = {n: (x-b0[0], y-b0[1]) for n, (x, y) in body.items()}
    for pid in points[:limit or TRAILS['parts']]:
        xy = []
        for packet in packets:
            for frame in packet['frames']:
                n = frame['frame']
                if not seed < n <= end or (shift is not None and n not in shift):
                    continue
                for q in frame['points']:
                    if q['point_id'] == pid and q['visible']:
                        dx, dy = shift[n][:2] if shift is not None else (0., 0.)
                        xy.append((n, q['xy'][0]-dx, q['xy'][1]-dy))
        if xy:
            out.append(dict(point=pid, xy=sorted(xy)))
    return out


# A whole-body move (Live2D wave sliding 300 px, 2026-11-01) split the onion in two and buried the hand's wave in a
# long sideways line, while look.md read both well ("전체" and "몸 기준"). When the body centre moves at least
# min_share of the picture width, the picture follows the text: the end frame and the trails are moved back by the
# body centre's shift, and the move itself goes to the chart's top "body" row and the small whole frame.
# Only a move of the whole figure counts: most tracked points (agree_share), and as many of the lowest quarter of them,
# must have gone at least half the centre's shift its way. A lunge drops the shoulders while the feet stay; drawn
# against the body, the planted foot moved 117 px (dancer f54..213), and the upper-body points alone still agreed.
# Points an agent placed are not a body: a hammer swing in mikumiku s19 moved 4 of 6 points and their median 179 px,
# so the barely moving eye (16 px) was drawn as 477 px. Against their median, nearly every point must agree.
BODY_VIEW = dict(min_share=.15, agree_share=.6, subject_agree_share=.85, smooth=5, color=(60, 60, 220))


def body_path(packets, seed, end):
    """{frame: (x, y)} body centre after the seed (body_motion.reference: shoulder midpoint, else face), median of 3
    then mean of BODY_VIEW['smooth'] frames, so pose jitter does not shake every trail drawn against it."""
    from image_lens import body_motion
    window = dict(seed=seed, end=end)
    kind, centre, _ = body_motion.reference(body_motion.series(packets, window), window)
    return _smooth(centre) if kind else {}


def subject_path(packets, seed, end):
    """('body', body_path) when shoulders or a face were tracked, else ('subject', median of the visible points per
    frame, smoothed the same way): mark's and box's points need not be a body (box on a hat: corners p1..p8)."""
    from image_lens import body_motion
    path = body_path(packets, seed, end)
    if path:
        return 'body', path
    centre = {}
    for q in body_motion.series(packets, dict(seed=seed, end=end)).values():
        for n, xy in q.items():
            centre.setdefault(n, []).append(xy)
    mid = {n: (sorted(p[0] for p in v)[len(v)//2], sorted(p[1] for p in v)[len(v)//2]) for n, v in centre.items()}
    return 'subject', _smooth(mid)


def _smooth(centre):
    if not centre:
        return {}
    from image_lens import body_motion
    ns = sorted(centre); path = body_motion._median3([centre[n] for n in ns]); r = BODY_VIEW['smooth']//2
    out = {}
    for i, n in enumerate(ns):
        q = min(r, i, len(ns)-1-i)  # symmetric near the ends, so a steady slide keeps its full length
        run = [path[j] for j in range(i-q, i+q+1) if abs(ns[j]-n) <= q]
        out[n] = (sum(p[0] for p in run)/len(run), sum(p[1] for p in run)/len(run))
    return out


def body_basis(video, bbox, path, packets, share=None):
    """(dx, dy) of the body centre over the window when the whole figure moved that way far enough to draw against
    the body, else None."""
    import cv2
    from image_lens import body_motion
    if not path or not bbox:
        return None
    capture = cv2.VideoCapture(str(video)); w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); capture.release()
    x0, _, x1, _ = _view(bbox, w, h); a, b = path[min(path)], path[max(path)]
    shift = (b[0]-a[0], b[1]-a[1]); size = math.hypot(*shift)
    if size < BODY_VIEW['min_share']*(x1-x0):
        return None
    points = [q for q in body_motion.series(packets, dict(seed=min(path)-1, end=max(path))).values() if len(q) > 1]
    moves = sorted(((q[min(q)][1], ((q[max(q)][0]-q[min(q)][0])*shift[0]+(q[max(q)][1]-q[min(q)][1])*shift[1])/size**2)
                    for q in points), reverse=True)  # lowest start first
    def agree(group):
        return bool(group) and sum(v >= .5 for _, v in group) >= (share or BODY_VIEW['agree_share'])*len(group)
    return shift if agree(moves) and agree(moves[:max(1, len(moves)//4)]) else None


def bands(video, start, end, bbox, camera=None, image=None, trails=(), colors=None, named=False, view_shift=None):
    """[{band, y, dx, dy, features}] of the median feature shift start->end per band, minus the camera shift.
    A band with too few surviving features gets dx=dy=None. Writes the onion image when `image` is a path, with
    `trails` (see `trails`) drawn on it and the end frame moved back by the camera shift, so the overlay and the
    trails both leave the camera out."""
    import cv2
    import numpy as np
    from image_lens import frames
    window = frames.window(video, start, end)
    if not window:
        return []
    first = window[0]; h, w = first.shape[:2]
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
    for frame in window[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(cur):
            cur, status, _ = cv2.calcOpticalFlowPyrLK(last, gray, cur, None, winSize=(31, 31), maxLevel=3)
            alive &= status.ravel() == 1
        last = gray
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
        drawn = result
        if view_shift:  # body basis: the overlay and the arrows leave the body centre's move out; the text above does not
            last = cv2.warpAffine(last, np.float32([[1, 0, -view_shift[0]], [0, 1, -view_shift[1]]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
            drawn = [dict(b, dx=b['dx']+cam[0]-view_shift[0], dy=b['dy']+cam[1]-view_shift[1]) if b['dx'] is not None else b for b in result]
        elif camera:  # trails have the camera taken out; without this the overlay showed a pan as motion (anime clip f180..209)
            last = cv2.warpAffine(last, np.float32([[1, 0, -cam[0]], [0, 1, -cam[1]]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
        capture = cv2.VideoCapture(str(video)); fps = capture.get(cv2.CAP_PROP_FPS); capture.release()
        _image(cv2, np, first, last, _view(bbox, w, h), drawn, image, trails, colors, named, fps, (x0+x1)//2)
    return result


def _arrow(cv2, picture, start, vector):
    """Band shift arrow: fixed head size, black and white rim, kept inside the picture so the head is never cut.
    Arrows shorter than the head are left out; the shift stays in the text report."""
    head, margin = ONION['arrow_head'], ONION['arrow_margin']
    h, w = picture.shape[:2]; dx, dy = vector
    fit = 1.
    for d, lo, hi in ((dx, margin-start[0], w-1-margin-start[0]), (dy, margin-start[1], h-1-margin-start[1])):
        if d > hi > 0: fit = min(fit, hi/d)
        elif d < lo < 0: fit = min(fit, lo/d)
    dx, dy = dx*fit, dy*fit
    length = (dx*dx+dy*dy)**.5
    if length < head:
        return
    end = (int(round(start[0]+dx)), int(round(start[1]+dy))); tip = min(.5, head/length)
    cv2.arrowedLine(picture, start, end, (0, 0, 0), 6, cv2.LINE_AA, tipLength=tip)  # black then white rim: visible on light and dark
    cv2.arrowedLine(picture, start, end, (255, 255, 255), 4, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(picture, start, end, (0, 140, 0), 2, cv2.LINE_AA, tipLength=tip)


def _grow(h, w):
    """Mark and label size factor for a crop: 1 up to the solo dance clip's box, then in proportion (see TRAILS)."""
    return max(1., w/TRAILS['box'][0], h/TRAILS['box'][1])


def _frame_label(cv2, panel, n):
    k = _grow(*panel.shape[:2])
    outlined(cv2, panel, f'f{n}', (round(6*k), round(28*k)), .9*k, round(2*k))


def ticks(frames, fps):
    """Indexes of the timing dots: every TRAILS['tick_seconds'] from the start of each continuous run."""
    if not fps or fps <= 0:
        return set()
    every = max(1, round(fps*TRAILS['tick_seconds'])); out = set(); run = None
    for i, n in enumerate(frames):
        if i == 0 or n != frames[i-1]+1:
            run = n
        elif (n-run) % every == 0:
            out.add(i)
    return out


# The timing dots still merged where a part moved only a little (README examples: steps shorter than a dot). An
# animator's spacing chart stretches each trail's path to one fixed-length line, so a 20 px move reads like a 300 px one.
SPACING = dict(width=420, height=680, row=70, min_px=15., margin=30)


# The whole frame's and the body centre's own moves as the chart's top rows, so a part's ease is read against them.
REFERENCE = dict(names=('camera', 'body', 'subject'), colors=((130, 130, 130), BODY_VIEW['color']))


def references(trails, seed, end, camera=None, body=None, name='body', colors=None):
    """(trails, colours) with `camera` (camera.path) and `body` (body_path) put first as rows. The body row leaves
    the camera out when it is given, as look.md's body numbers do; frames missing from either are left out."""
    rows, out = [], []
    if camera:
        rows.append(dict(point='camera', xy=[(n, camera[n][0], camera[n][1]) for n in sorted(camera) if seed <= n <= end]))
        out.append(REFERENCE['colors'][0])
    if body:
        xy = [(n, x-camera[n][0], y-camera[n][1]) if camera else (n, x, y) for n, (x, y) in sorted(body.items())
              if not camera or n in camera]
        rows.append(dict(point=name, xy=xy)); out.append(REFERENCE['colors'][1])
    return rows+list(trails), out+list(colors or TRAILS['colors'])


def moved(camera):
    """camera (camera.path) when it travelled enough for a chart row, else None."""
    rows, _ = spacing_rows([dict(point='camera', xy=[(n, v[0], v[1]) for n, v in sorted(camera.items())])], None)
    return camera if rows else None


def spacing_rows(trails, fps):
    """[{point, length, ticks: [(fraction, timed)]}, ...] and the names of parts too still to chart. Path length grows
    only by steps of at least motion_report STROKE['hold_step_px'], so a held part's jitter does not read as motion."""
    from image_lens import motion_report
    still = motion_report.STROKE['hold_step_px']; rows, skipped = [], []
    for t in trails:
        xy = [(x, y) for _, x, y in t['xy']]; frames = [n for n, _, _ in t['xy']]
        cum = [0.]  # a jump over missing frames is not travel the chart can time (mark trial: one lost corner read 326 px)
        for (a, b), (n, m) in zip(zip(xy, xy[1:]), zip(frames, frames[1:])):
            d = math.dist(a, b) if m == n+1 else 0.; cum.append(cum[-1]+(d if d >= still else 0.))
        if cum[-1] < SPACING['min_px']:
            if t['point'] not in REFERENCE['names']:  # a still camera or body is not a still part: just no row
                skipped.append(t['point'])
            continue
        timed = ticks(frames, fps)
        rows.append(dict(point=t['point'], length=cum[-1], ticks=[(c/cum[-1], i in timed or i == 0) for i, c in enumerate(cum)]))
    return rows, skipped


def spacing(video, image, trails, colors=None, context=None):
    """Append the spacing chart right of the onion image: one equal-length line per trail, a thick tick every
    TRAILS['tick_seconds'] and a thin one per frame at the fraction of path covered; close ticks = slow.
    context=(frame, bbox): under the rows, that whole frame small with the onion's view outlined (`_context`)."""
    import cv2
    import numpy as np
    picture = cv2.imread(str(image))
    if picture is None:
        return None
    capture = cv2.VideoCapture(str(video)); fps = capture.get(cv2.CAP_PROP_FPS); capture.release()
    rows, skipped = spacing_rows(trails, fps) if trails else ([], [])
    if not rows and not context:
        return None
    h = picture.shape[0]; k = max(1., h/SPACING['height']); w = round(SPACING['width']*k)
    panel = np.full((h, w, 3), 245, np.uint8); m = round(SPACING['margin']*k); x0, x1 = m, w-m
    row = min(round(SPACING['row']*k), (h-2*m)//(len(rows)+1)); size, th = .45*k, max(1, round(k))
    if rows:
        cv2.putText(panel, f"spacing: {TRAILS['tick_seconds']:g}s ticks", (round(10*k), round(22*k)), cv2.FONT_HERSHEY_SIMPLEX, size, (60, 60, 60), th, cv2.LINE_AA)
    palette = {t['point']: c for t, c in zip(trails, colors or TRAILS['colors'])}
    for i, r in enumerate(rows):
        y = m+round(10*k)+row*i+row//2; c = palette[r['point']]; big, small = round(12*k), round(4*k)
        cv2.line(panel, (x0, y), (x1, y), (150, 150, 150), th)
        for f, timed in r['ticks']:
            x = round(x0+(x1-x0)*f)
            if timed:
                cv2.line(panel, (x, y-big), (x, y+big), (0, 0, 0), 4*th); cv2.line(panel, (x, y-big), (x, y+big), c, 2*th)
            else:
                cv2.line(panel, (x, y-small), (x, y+small), c, th)
        cv2.putText(panel, f"{r['point']} {r['length']:.0f}px", (x0, y-big-round(4*k)), cv2.FONT_HERSHEY_SIMPLEX, size, (40, 40, 40), th, cv2.LINE_AA)
    x, y = x0, m+round(10*k)+row*len(rows)+row//2
    for i, name in enumerate(skipped):  # each still name in its trail colour, outlined, so colour -> name stays readable
        text = ('still: ' if i == 0 else '')+name+(',' if i < len(skipped)-1 else '')
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (40, 40, 40), th+2, cv2.LINE_AA)
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, palette[name], th, cv2.LINE_AA)
        x += cv2.getTextSize(text+' ', cv2.FONT_HERSHEY_SIMPLEX, size, th)[0][0]
        if x > x1-round(60*k):
            x, y = x0, y+round(24*k)
    if context:
        _context(cv2, video, context, panel, (x0, y+row//2 if rows or skipped else m, x1, h-m))
    cv2.imwrite(str(image), np.hstack([picture, np.full((h, 6, 3), 255, np.uint8), panel]))
    return dict(rows=[r['point'] for r in rows], still=skipped)


def _views(camera, w, h):
    """{frame: (x0, y0, x1, y1)} of the view on each frame in the first frame's pixels. camera.path is how the
    content moved (a pan right reads dx < 0, scale > 1 = zoom in), so the view went the other way: centre c - d/s."""
    out = {}
    for n, (dx, dy, k) in camera.items():
        cx, cy = w/2-dx/k, h/2-dy/k
        out[n] = (cx-w/2/k, cy-h/2/k, cx+w/2/k, cy+h/2/k)
    return out


def _dashed(cv2, panel, a, b, color, t):
    (x0, y0), (x1, y1) = a, b; dash = 4*t
    for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(math.dist(p, q)//(2*dash)))
        for i in range(n):
            f, g = 2*i*dash/max(1., math.dist(p, q)), (2*i+1)*dash/max(1., math.dist(p, q))
            u = (round(p[0]+(q[0]-p[0])*f), round(p[1]+(q[1]-p[1])*f)); v = (round(p[0]+(q[0]-p[0])*min(g, 1)), round(p[1]+(q[1]-p[1])*min(g, 1)))
            cv2.line(panel, u, v, (255, 255, 255), t+2); cv2.line(panel, u, v, color, t)


SUBJECT_COLOR = (60, 200, 60)  # green: the tracked person, or the box given to `box`


def _context(cv2, video, context, panel, area):
    """The whole frame, fitted into `area` of the chart panel, with the onion's view outlined: a crop alone hid that
    mikumiku is a character inside a square card on a phone player (2026-11-01). Costs no width: the panel is there.
    context=(frame, bbox, body path, camera path, subject box): with a camera path the view's centre path and the last frame's
    view (dashed) are drawn too, and the picture shrinks so that view fits beside it."""
    from image_lens import frames
    frame, bbox, path, camera, subject = (tuple(context)+(None, None, None))[:5]
    got = frames.window(video, frame, frame)
    if not got:
        return
    image = got[0]; h, w = image.shape[:2]; ax0, ay0, ax1, ay1 = area
    views = _views(camera, w, h) if camera else {}
    end = views[max(views)] if views else (0, 0, w, h)
    ux0, uy0, ux1, uy1 = min(0, end[0]), min(0, end[1]), max(w, end[2]), max(h, end[3])
    k = min((ax1-ax0)/(ux1-ux0), (ay1-ay0)/(uy1-uy0))
    if k <= 0:
        return
    small = cv2.resize(image, (max(1, int(w*k)), max(1, int(h*k))), interpolation=cv2.INTER_AREA)
    ox, oy = ax0+(ax1-ax0-round((ux1-ux0)*k))//2-round(ux0*k), ay1-round((uy1-uy0)*k)-round(uy0*k)
    sx, sy = ox, oy
    panel[sy:sy+small.shape[0], sx:sx+small.shape[1]] = small
    vx0, vy0, vx1, vy1 = (_view(bbox, w, h) if bbox else (0, 0, w, h))
    t = max(2, round(small.shape[0]/200))  # 1 px of colour did not survive JPEG
    corners = ((sx+int(vx0*k), sy+int(vy0*k)), (sx+int(vx1*k)-1, sy+int(vy1*k)-1))
    cv2.rectangle(panel, *corners, (0, 0, 0), t+2); cv2.rectangle(panel, *corners, (0, 220, 255), t)
    if subject:  # who was tracked: the agent can tell a wrong pick and hand `box` a better one
        a, b = (sx+int(subject[0]*k), sy+int(subject[1]*k)), (sx+int(subject[2]*k)-1, sy+int(subject[3]*k)-1)
        cv2.rectangle(panel, a, b, (0, 0, 0), t+2); cv2.rectangle(panel, a, b, SUBJECT_COLOR, t)
    capture = cv2.VideoCapture(str(video)); fps = capture.get(cv2.CAP_PROP_FPS); capture.release()

    def line(points, color):  # a path in first-frame pixels: white rim, 0.1 s dots, square at the end
        ns = sorted(points); at = [(sx+round(points[n][0]*k), sy+round(points[n][1]*k)) for n in ns]
        for a, b in zip(at, at[1:]):
            cv2.line(panel, a, b, (255, 255, 255), t+2); cv2.line(panel, a, b, color, t)
        for i in sorted(ticks(ns, fps) | {0}):
            cv2.circle(panel, at[i], t+2, (255, 255, 255), -1); cv2.circle(panel, at[i], t+1, color, -1)
        e = t+3; cv2.rectangle(panel, (at[-1][0]-e, at[-1][1]-e), (at[-1][0]+e, at[-1][1]+e), color, t)
    if views:  # where the camera ended up, and how its centre got there
        _dashed(cv2, panel, (sx+round(end[0]*k), sy+round(end[1]*k)), (sx+round(end[2]*k)-1, sy+round(end[3]*k)-1), REFERENCE['colors'][0], t)
        line({n: ((v[0]+v[2])/2, (v[1]+v[3])/2) for n, v in views.items()}, REFERENCE['colors'][0])
    if path:  # body basis: the body centre's own move, on screen
        line(path, BODY_VIEW['color'])


def _image(cv2, np, first, last, box, result, path, trails=(), colors=None, named=False, fps=None, centre=None):
    x0, y0, x1, y1 = box
    a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]; b = last[y0:y1, x0:x1]
    picture = np.dstack([b, b, a])  # dark only at the start -> cyan, dark only at the end -> red
    centre = (x1-x0)//2 if centre is None else centre-x0  # arrows on the person, not the middle of the view
    for band in result:
        top = band['y'][0]-y0
        cv2.line(picture, (0, top), (x1-x0, top), (150, 150, 150), 1)
        if band['dx'] is not None:
            mid = (band['y'][0]+band['y'][1])//2-y0; s = ONION['arrow_scale']
            _arrow(cv2, picture, (centre, mid), (s*band['dx'], s*band['dy']))
    # The sheet fits each onion in one tile, so a big person box shrinks more: a 1440 px clip's 3 px dots came out under
    # 1 px there. Sizes are for the solo dance clip's box (marks about 1.2 px on the sheet) and grow with larger boxes.
    k = _grow(y1-y0, x1-x0)
    dot, tick, start, end, line = (max(1, round(TRAILS[n]*k)) for n in ('dot', 'tick', 'start', 'end', 'line'))
    for trail, color in zip(trails, colors or TRAILS['colors']):
        at = [(int(round(x-x0)), int(round(y-y0))) for _, x, y in trail['xy']]
        frames = [n for n, _, _ in trail['xy']]
        for i, (a, b) in enumerate(zip(at, at[1:])):
            if frames[i+1] == frames[i]+1:
                cv2.line(picture, a, b, color, line)
        timed = ticks(frames, fps)
        for i, p in enumerate(at):  # white rim: middle dots on a busy drawing were hard to find
            r = tick if i in timed else dot
            cv2.circle(picture, p, r+1, (255, 255, 255), -1); cv2.circle(picture, p, r, color, -1)
        for i, p in enumerate(at):
            if i == 0 or frames[i] != frames[i-1]+1:
                cv2.circle(picture, p, start, color, line)
            if i == len(at)-1 or frames[i+1] != frames[i]+1:
                cv2.rectangle(picture, (p[0]-end, p[1]-end), (p[0]+end, p[1]+end), color, line)
    if named:  # mark: the agent's names, since its points have no fixed order to read colours by
        firsts = [(t['point'], (int(round(t['xy'][0][1]-x0)), int(round(t['xy'][0][2]-y0))), c) for t, c in zip(trails, colors or TRAILS['colors']) if t['xy']]
        taken = [(p[0]-start, p[1]-start, p[0]+start, p[1]+start) for _, p, _ in firsts]  # labels keep off every start mark
        size, t = .6*k, max(1, round(k))
        for name, p, color in firsts:
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, size, t+3)
            x = p[0]+start+2 if p[0]+start+2+tw <= picture.shape[1] else max(0, p[0]-start-2-tw)
            y = y_first = max(th+2, p[1]-start)
            while any(x < r[2] and r[0] < x+tw and y-th < r[3] and r[1] < y for r in taken) and y < picture.shape[0]:
                y += th+4  # two corners of one foot: stack the names
            if y != y_first:  # moved away: a leader in the trail colour back to its point
                cv2.line(picture, (min(max(p[0], x), x+tw), y-th//2), p, (0, 0, 0), t+3); cv2.line(picture, (min(max(p[0], x), x+tw), y-th//2), p, color, t+1)
            taken.append((x, y-th, x+tw, y)); outlined(cv2, picture, name, (x, y), size, t)
    cv2.imwrite(str(path), picture)


# The onion overlays start and end only: a spin or a hand that goes out and back ends where it began, so the overlay
# looks still (clip B blind test, all three spins missed). The frame most unlike the start, away from both ends,
# goes next to it.
# Off by default (user 2026-11-01): the strip, trails and spacing chart now cover turns and out-and-back moves, and the
# extra panel cost a third of the picture width. `--peak` on look/mark turns it back on.
PEAK = dict(min_frames=8, margin=.15, scale=4, plateau=.95, draw=False)


def _crop(cv2, image, bbox, view=True):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = ((_view if view else _box)(bbox, w, h)) if bbox else (0, 0, w, h)
    return image[y0:y1, x0:x1]


def peak(video, start, end, bbox, image):
    """Frame number of the frame most unlike `start` (mean grey difference inside the padded person box), searched
    away from both window ends, drawn as a grey panel left of the onion image. None for short windows."""
    import cv2
    import numpy as np
    if end-start+1 < PEAK['min_frames']:
        return None
    lo, hi = start+max(1, int(PEAK['margin']*(end-start))), end-max(1, int(PEAK['margin']*(end-start)))
    from image_lens import frames
    window = frames.window(video, start, end)[:hi-start+1]
    found = {}; first = None
    for n, frame in zip(range(start, hi+1), window):
        small = cv2.resize(cv2.cvtColor(_crop(cv2, frame, bbox, view=False), cv2.COLOR_BGR2GRAY), None, fx=1/PEAK['scale'], fy=1/PEAK['scale'], interpolation=cv2.INTER_AREA).astype(np.float32)
        if first is None:
            first = small
        elif n >= lo:
            found[n] = float(np.abs(small-first).mean())
    if not found:
        return None
    # Once a part has left its start outline the difference stops growing; take the middle of that plateau.
    plateau = [n for n, d in found.items() if d >= PEAK['plateau']*max(found.values())]
    n = plateau[len(plateau)//2]; frame = window[n-start]
    if image is None:
        return n
    if not PEAK['draw']:
        return None
    onion = cv2.imread(str(image))
    panel = cv2.cvtColor(cv2.cvtColor(_crop(cv2, frame, bbox), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if onion is not None and onion.shape[0] == panel.shape[0]:
        _frame_label(cv2, panel, n)
        # Plain frame first, then the onion with its trails next to the spacing chart: read the pose, then the motion.
        cv2.imwrite(str(image), np.hstack([panel, np.full((onion.shape[0], 6, 3), 255, np.uint8), onion]))
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
        outlined(cv2, picture, p['name'], (x0+12, y0+70), 2.5, 6)
    cv2.imwrite(str(image), picture)
    return image.name


def still(video, frame, bbox, image):
    """One native frame, whole (bbox unused, kept for callers), for a range too short to track."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, picture = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    cv2.imwrite(str(image), picture)  # the whole frame: a short cut's still is context, not a close-up
    return image.name


# Tiles share one height, not one width: fitted to a 360 px width, an onion with its spacing chart (about 1.4x as
# wide as tall) shrank to 360x262 next to a 360x414 still (Live2D sliding wave, 2026-11-01). 360 high: about 1.3x
# the image tokens of the old layout on mikumiku (540 high: 2x), onions 1.3-1.5x taller as read; docs/look-notes.md.
SHEET = dict(width=1440, tile_height=360, label=24, rows=5, strip_height=200, gap_s=.05)


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
                _frame_label(cv2, panel, n)
                panels.append(panel)
    finally:
        cap.release()
    if not panels:
        return None
    cv2.imwrite(str(image), np.hstack([np.hstack([p, np.full((p.shape[0], 6, 3), 255, np.uint8)]) for p in panels]))
    return dict(image=image.name, frames=picks)


def sheets(root, items):
    """Onion images of a whole look on a few contact sheets (time label above each), so an image reader sees the
    gestures the text loses (heart hands, V signs on clip B close-ups). items: [(start_s, end_s, image name, note)],
    note '' for an onion image, 'still' for a short cut's still, 'strip' for a frame strip. Stills of back-to-back
    cuts share one full-width row in time order (clip B: a spin made of seven 3-6 frame cuts); a strip gets its own
    row under its window. Rows, not tiles, are counted per sheet."""
    import cv2
    import numpy as np
    top, width, h = SHEET['label'], SHEET['width'], SHEET['tile_height']
    def read(name):
        return cv2.imread(str(root/name))
    def label(image, text):
        tile = np.zeros((image.shape[0]+top, image.shape[1], 3), np.uint8); tile[top:] = image
        outlined(cv2, tile, text, (4, 18), .6, 1)
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
                units.append(('tile', label(fit(images[0][0], width, h), images[0][1])))
        else:
            image = read(name)
            if image is not None:
                if note == 'strip':
                    units.append(('row', full_row([image], [f'{start:.2f}-{end:.2f}s strip'])))
                else:
                    units.append(('tile', label(fit(image, width, h), f'{start:.2f}-{end:.2f}s')))
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
            # Left to right; a tile that would pass the sheet width starts the next row.
            if pending and sum(x.shape[1] for x in pending)+image.shape[1]+4 > width:
                flush()
            pending.append(np.hstack([image, np.zeros((image.shape[0], 4, 3), np.uint8)])[:, :width])
    flush()
    written = []
    for first in range(0, len(rows), SHEET['rows']):
        path = root/f'onion-sheet-{len(written)+1}.jpg'
        cv2.imwrite(str(path), np.vstack(rows[first:first+SHEET['rows']])); written.append(path.name)
    return written


def lines(result, image_name=None, screen=False, peak=False, body=None):
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
    basis = '카메라 모름: 화면 좌표 그대로' if screen else '카메라 추정 시 평행이동 제외'
    out = [f'- 높이별 흐름 (사람 상자 5등분, {basis}): ' + ' · '.join(parts)]
    if image_name:
        out.append(f"  - 어니언 그림: {image_name} (청록=시작, 빨강=끝(진한 윤곽선 기준; 배경보다 밝은 면은 띠 색이 반대(청록 띠=간 쪽); 카메라가 움직였으면 끝 프레임을 카메라만큼 되돌려 겹침), 초록 화살표=띠별 이동 ×{ONION['arrow_scale']}(그림 밖으로 나가면 줄임, 머리보다 짧으면 생략); {'왼쪽 회색=시작과 가장 다른 가운데 프레임; ' if peak else ''}"
                   f"색 점선=부위별 대표 점(몸 기준이 없으면 '화면'에 적힌 점)의 프레임마다 위치({'·'.join(TRAILS['names'])} 순, {basis}), 큰 원·네모=각 연속 관측의 시작·끝, 선 끊김=관측 누락, 이어진 점 간격이 좁으면 느림·넓으면 빠름, 조금 큰 점={TRAILS['tick_seconds']:g}초마다(그 간격이 좁아지면 느려짐); 맨 오른쪽 spacing=부위마다 지나온 길이를 같은 길이 줄로 펼친 것, 굵은 눈금={TRAILS['tick_seconds']:g}초마다·가는 눈금=프레임마다, 눈금이 모이면 느림·벌어지면 빠름(작은 이동도 같은 크기로 보임, {SPACING['min_px']:g}px 미만 부위는 still), 맨 위 camera 줄=화면 전체 이동·body 줄=몸 중심 이동(카메라 추정 시 카메라 제외; {SPACING['min_px']:g}px 미만이면 줄 없음), 그 아래 작은 그림=구간 첫 프레임 전체(노란 테두리=어니언이 잘라 보여 준 범위, 초록 테두리=추적한 사람(틀렸으면 box 명령에 상자를 주어 다시), 카메라가 움직였으면 회색 선=화면 중심의 궤적·회색 점선 사각형=구간 끝의 화면))")
    if image_name and body:
        out.append(f"  - 몸 기준 그림: 몸 중심이 {motion_report.KO[motion_report.direction(*body)].replace('화면 ', '')} {math.hypot(*body):.0f}px "
                   "움직여 끝 프레임과 궤적을 그만큼 되돌려 겹침(몸은 포개지고 팔다리 움직임만 갈라짐, 배경은 반대로 갈라짐; "
                   "높이별 흐름 숫자는 위 기준 그대로). 몸 이동 자체는 차트의 body 줄과 그 아래 작은 그림의 빨간 선(점=0.1초마다, 네모=끝)")
    return out


# Seeds where the onion gap is thickest: the band between the start outline and a later outline is widest where that
# part moved (clip A pose 1: sleeves and hair tips; clip C hammer: the hammer head). Gaps also open where a part
# arrives, so a candidate must sit on an outline of the start frame inside the gap. Thirds catch moves that return.
# Gap width is not travel (the hammer scene's widest gap, 124 px, moved 39 px; the hammer moved 450 px), and joined gaps
# are not one part, so up to 12 candidates are seeded and tracked; `parts` then keeps one point per group that moved
# together. LK flow cannot follow the hammer swing (110-170 px of 450, and a face point dragged 500 px by it), so the
# grouping uses the real tracks. Together = paths never further apart than 16 px or half the larger travel: at 8 px / a
# quarter, clip A kept 6 points in 11 of 19 windows (drawn parts deform as they move) and the hammer got two points.
GAPS = dict(points=6, diff=40, min_area=80, min_half_px=4., spacing_px=40., edge=(50, 150), switch_share=.3,
            keep_share=.4, min_travel_px=8., together_px=16., together_share=.5)


def gap_seeds(video, start, end, bbox, names):
    """{name: {xy, seed_rule}}: candidates on start-frame outlines inside onion gaps, widest gap first; `parts` picks."""
    import cv2
    import numpy as np
    from image_lens import camera, motion_report
    from image_lens import frames
    window = frames.window(video, start, end)
    if not window:
        return {}
    first = cv2.cvtColor(window[0], cv2.COLOR_BGR2GRAY); h, w = first.shape
    x0, y0, x1, y1 = _box(bbox, w, h)
    # Stop before a drawing swap (costume change, wipe): a gap across it is the new drawing, not motion.
    kept = {start: first}; prev = first
    for n, bgr in zip(range(start+1, end+1), window[1:]):
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if (cv2.absdiff(prev[y0:y1, x0:x1], image[y0:y1, x0:x1]) > GAPS['diff']).mean() > GAPS['switch_share']:
            break
        kept[n] = prev = image
    end = max(kept)
    if end == start:
        return {}
    later = sorted({start+(end-start)//3, start+2*(end-start)//3, end} - {start})
    moves = camera.path(video, start, end, bbox)
    # Gaps are found in the box plus a zero margin, not the whole frame (whole-frame masks per gap were 0.23 of 0.37 s per
    # call on clip B). The margin is wider than the opening and distance-transform steps, so they see the same zeros
    # around the box as on the whole frame; the outline is still traced on the whole frame. Same seeds on 6 clips.
    ox, oy, ex, ey = max(0, x0-4), max(0, y0-4), min(w, x1+4), min(h, y1+4)
    outline = (cv2.dilate(cv2.Canny(first, *GAPS['edge']), np.ones((3, 3), np.uint8)) > 0)[oy:ey, ox:ex]
    base = first[y0:y1, x0:x1]
    found = []
    for n in later:
        shift = camera.between(moves, start, n)
        if shift and math.hypot(*shift) >= motion_report.STROKE['major_min_px']:
            image = cv2.warpAffine(kept[n], np.float32([[1, 0, x0+shift[0]], [0, 1, y0+shift[1]]]), (x1-x0, y1-y0),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        else:
            image = kept[n][y0:y1, x0:x1]
        mask = np.zeros((ey-oy, ex-ox), np.uint8)
        mask[y0-oy:y1-oy, x0-ox:x1-ox] = cv2.absdiff(base, image) > GAPS['diff']
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, count):
            sx, sy, sw, sh, area = stats[i]
            if area < GAPS['min_area']:
                continue
            region = labels[sy:sy+sh, sx:sx+sw] == i; d = depth[sy:sy+sh, sx:sx+sw]
            on_outline = np.where(region & outline[sy:sy+sh, sx:sx+sw], d, 0)
            if on_outline.max() < GAPS['min_half_px']:
                continue  # an arrival gap: nothing of the start frame to follow there
            y, x = np.unravel_index(on_outline.argmax(), on_outline.shape)
            found.append((2*float(d[region].max()), int(x+sx+ox), int(y+sy+oy), n))
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
