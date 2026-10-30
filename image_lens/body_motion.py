"""Body-relative limb motion for look. Screen-space strokes mix a whole-body bounce or lean with the gesture itself,
so each limb point is measured against the body centre, and its path is split where it turns.
Sides are screen sides (화면 왼쪽/오른쪽), taken from where the point starts relative to the body centre."""
import math
from image_lens import motion_report

BODY_MOTION = dict(min_ratio=.06, min_px=8., min_frames=4, max_segments=3, ref_coverage=.8)
PARTS = dict(wrist='손목', elbow='팔꿈치', thumb='엄지 끝', index='검지 끝', thigh='허벅지', knee='무릎', ankle='발목', move='많이 움직인 곳')
HEAD = ('nose', 'eye-l', 'eye-r')


def series(packets, window):
    """Visible positions per point inside the window, without the seed frame (it is the pose anchor, not a prediction)."""
    result = {}
    for packet in packets:
        for frame in packet['frames']:
            n = frame['frame']
            if window['seed'] < n <= window['end']:
                for p in frame['points']:
                    if p['visible']:
                        result.setdefault(p['point_id'], {})[n] = tuple(p['xy'])
    return result


def _mean(points):
    return (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))


def reference(points, window):
    """Body centre per frame and a body size. Shoulder midpoint when both shoulders cover most of the window, else the face."""
    frames = range(window['seed']+1, window['end']+1)
    sl, sr = points.get('shoulder-l', {}), points.get('shoulder-r', {})
    both = [n for n in frames if n in sl and n in sr]
    if len(both) >= BODY_MOTION['ref_coverage']*len(frames):
        size = sorted(math.dist(sl[n], sr[n]) for n in both)[len(both)//2]
        return 'shoulders', {n: _mean([sl[n], sr[n]]) for n in both}, size
    centre = {}
    for n in frames:
        found = [points[k][n] for k in HEAD if n in points.get(k, {})]
        if found:
            centre[n] = _mean(found)
    eyes = [math.dist(points['eye-l'][n], points['eye-r'][n]) for n in frames
            if n in points.get('eye-l', {}) and n in points.get('eye-r', {})]
    if not centre or not eyes:
        return None, {}, None
    return 'face', centre, 2.5*sorted(eyes)[len(eyes)//2]


def _median3(path):
    return [path[0]]+[tuple(sorted(c)[1] for c in zip(*path[i-1:i+2])) for i in range(1, len(path)-1)]+[path[-1]] if len(path) > 2 else path


def _off_chord(p, a, b):
    """Distance from p to the segment a-b; a point beyond an end (a reversal along the line) counts too."""
    (ax, ay), (bx, by) = a, b
    length = (bx-ax)**2+(by-ay)**2
    t = 0. if not length else max(0., min(1., ((p[0]-ax)*(bx-ax)+(p[1]-ay)*(by-ay))/length))
    return math.dist(p, (ax+t*(bx-ax), ay+t*(by-ay)))


def segments(path, tolerance):
    """Split a path at the points that leave the straight line most; returns index pairs."""
    def split(lo, hi):
        best, at = 0., None
        for i in range(lo+1, hi):
            d = _off_chord(path[i], path[lo], path[hi])
            if d > best:
                best, at = d, i
        return split(lo, at)+split(at, hi) if at is not None and best >= tolerance else [(lo, hi)]
    return split(0, len(path)-1)


def describe(packets, window):
    """Body centre shift plus per-limb relative strokes (direction, size, frames) and toward/away from the body axis."""
    points = series(packets, window)
    kind, centre, size = reference(points, window)
    if not kind:
        return None
    limit = max(BODY_MOTION['min_px'], BODY_MOTION['min_ratio']*size)
    frames = sorted(centre)
    shift = (centre[frames[-1]][0]-centre[frames[0]][0], centre[frames[-1]][1]-centre[frames[0]][1])
    limbs = []
    for pid, track in sorted(points.items()):
        part = PARTS.get(pid.split('-')[0])
        common = [n for n in sorted(track) if n in centre]
        if not part or len(common) < BODY_MOTION['min_frames']:
            continue
        rel = _median3([(track[n][0]-centre[n][0], track[n][1]-centre[n][1]) for n in common])
        strokes = []
        for lo, hi in segments(rel, limit):
            dx, dy = rel[hi][0]-rel[lo][0], rel[hi][1]-rel[lo][1]
            if math.hypot(dx, dy) >= limit:
                strokes.append(dict(direction=motion_report.direction(dx, dy), distance_px=round(math.hypot(dx, dy), 1),
                                    frames=[common[lo], common[hi]]))
        if not strokes:
            continue
        spread = abs(rel[-1][0])-abs(rel[0][0])
        # Screen side of the limb, not of the point: a hand crossing to the chest still belongs to its arm's side.
        anchor = points.get(('hip-' if part in ('허벅지', '무릎', '발목') else 'shoulder-')+pid.rsplit('-', 1)[-1], {})
        at = [anchor[n][0]-centre[n][0] for n in common if n in anchor]
        x = sorted(at)[len(at)//2] if at else rel[0][0]
        if part == PARTS['move']:
            part = f"{part} {pid.rsplit('-', 1)[-1]}"  # unnamed: keep the letter so it matches the onion image and tracks
        limbs.append(dict(point_id=pid, part=part, side='화면 왼쪽' if x < 0 else '화면 오른쪽',
                          strokes=strokes[:BODY_MOTION['max_segments']], omitted=max(0, len(strokes)-BODY_MOTION['max_segments']),
                          axis_px=round(spread, 1) if abs(spread) >= limit else 0.))
    return dict(reference=kind, size_px=round(size, 1), limit_px=round(limit, 1),
                centre_shift=dict(direction=motion_report.direction(*shift), distance_px=round(math.hypot(*shift), 1),
                                  frames=[frames[0], frames[-1]]), limbs=limbs)


def lines(result):
    """look.md lines: one for the body centre, one per moving limb point."""
    if not result:
        return ['- 몸 기준: 기준점(어깨·얼굴)이 부족해 계산 못 함']
    ref = '어깨 가운데' if result['reference'] == 'shoulders' else '얼굴 가운데'
    c = result['centre_shift']
    head = f"- 몸 기준 움직임 (몸 중심={ref}, 기준 {result['limit_px']:.0f}px):"
    body = (f"  - 몸 중심: {motion_report.KO[c['direction']]} {c['distance_px']:.0f}px" if c['distance_px'] >= result['limit_px']
            else '  - 몸 중심: 거의 그대로')
    out = [head, body]
    for limb in result['limbs']:
        moves = ' → '.join(f"{motion_report.KO[s['direction']].replace('화면 ', '')} {s['distance_px']:.0f}px(f{s['frames'][0]}..{s['frames'][1]})"
                           for s in limb['strokes'])
        axis = '' if not limb['axis_px'] else f", 몸 {'바깥으로' if limb['axis_px'] > 0 else '쪽으로'} {abs(limb['axis_px']):.0f}px"
        out.append(f"  - {limb['side']} {limb['part']}: {moves}{axis}")
    if not result['limbs']:
        out.append('  - 팔다리: 몸 기준으로도 기준 미만')
    return out


# Across a costume change or cut the drawing is replaced, but an animated move often runs through it (an arm dropping
# from the side to the waist on the switch frame). Compare the same pose keypoints on the two frames around it.
BRIDGE = dict(max_gap=3, min_score=.30, size_ratio=(.75, 1.33), max_centre_shift=1.)
KEYPOINTS = dict(wrist=(9, 10), elbow=(7, 8))  # hand tips repeat the wrist and are rough on drawn hands


def largest(tracks):
    return max(tracks, key=lambda t: (t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1])) if tracks else None


def bridge(before, after, frames):
    """Body-relative jumps of limb keypoints between two nearby frames, or None when it is not plausibly the same figure."""
    if not before or not after or frames[1]-frames[0] > BRIDGE['max_gap']:
        return None
    if not skeleton_check(before)[0] or not skeleton_check(after)[0]:
        return None
    def body(t):
        k, s = t['keypoints'], t['scores']
        if min(s[5], s[6]) < BRIDGE['min_score']:
            return None
        return ((k[5][0]+k[6][0])/2, (k[5][1]+k[6][1])/2), math.dist(k[5], k[6])
    a, b = body(before), body(after)
    if not a or not b or not a[1] or not BRIDGE['size_ratio'][0] <= b[1]/a[1] <= BRIDGE['size_ratio'][1] \
            or math.dist(a[0], b[0]) > BRIDGE['max_centre_shift']*a[1]:
        return None
    size = (a[1]+b[1])/2; limit = max(BODY_MOTION['min_px'], BODY_MOTION['min_ratio']*size)
    limbs = []
    for part, pair in KEYPOINTS.items():
        for side, i in zip('lr', pair):
            if min(before['scores'][i], after['scores'][i]) < BRIDGE['min_score']:
                continue
            ra = (before['keypoints'][i][0]-a[0][0], before['keypoints'][i][1]-a[0][1])
            rb = (after['keypoints'][i][0]-b[0][0], after['keypoints'][i][1]-b[0][1])
            dx, dy = rb[0]-ra[0], rb[1]-ra[1]
            if math.hypot(dx, dy) < limit:
                continue
            shoulder = after['keypoints'][5 if side == 'l' else 6][0]-b[0][0]
            spread = abs(rb[0])-abs(ra[0])
            limbs.append(dict(point_id=f'{part}-{side}', part=PARTS[part], side='화면 왼쪽' if shoulder < 0 else '화면 오른쪽',
                              direction=motion_report.direction(dx, dy), distance_px=round(math.hypot(dx, dy), 1),
                              axis_px=round(spread, 1) if abs(spread) >= limit else 0.))
    shift = (b[0][0]-a[0][0], b[0][1]-a[0][1])
    return dict(frames=list(frames), limit_px=round(limit, 1), limbs=limbs,
                centre_shift=dict(direction=motion_report.direction(*shift), distance_px=round(math.hypot(*shift), 1)))


def bridge_lines(result):
    if not result or not result['limbs']:
        return []
    a, b = result['frames']
    out = [f"- 전환 앞뒤 자세 차이 (f{a}→f{b}, 포즈 기준; 두 프레임만 비교, 사이 경로·속도는 모름):"]
    for limb in result['limbs']:
        axis = '' if not limb['axis_px'] else f", 몸 {'바깥으로' if limb['axis_px'] > 0 else '쪽으로'} {abs(limb['axis_px']):.0f}px"
        out.append(f"  - {limb['side']} {limb['part']}: {motion_report.KO[limb['direction']].replace('화면 ', '')} {limb['distance_px']:.0f}px{axis}")
    return out


# Calibrated by eye on AKAGE (all plain frames pass; echo-copy ranges fail) and on a 39-cover MikuMiku playlist
# (6/6 good skeletons pass, 2/26 wrong ones pass). Ratios are in shoulder widths.
SKELETON = dict(min_score=.30, mean_min=.45, torso=(1., 2.6), hip_width=(.45, 1.6), max_arm=2.5)


def skeleton_check(track):
    """(plausible, reason) for naming arm and leg points from this pose. Faces are judged separately and kept."""
    k, s = track['keypoints'], track['scores']
    core = [s[i] for i in range(5, 13)]
    if min(core) < SKELETON['min_score']:
        return False, 'shoulder/elbow/wrist/hip missing'
    if sum(core)/len(core) < SKELETON['mean_min']:
        return False, f'core joint score {sum(core)/len(core):.2f}'
    width = math.dist(k[5], k[6])
    if not width:
        return False, 'zero shoulder width'
    torso = math.dist(_mean([k[5], k[6]]), _mean([k[11], k[12]]))/width
    hips = math.dist(k[11], k[12])/width
    arm = max(math.dist(k[a], k[b]) for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)))/width
    if not SKELETON['torso'][0] <= torso <= SKELETON['torso'][1]:
        return False, f'torso {torso:.2f} shoulder widths'
    if not SKELETON['hip_width'][0] <= hips <= SKELETON['hip_width'][1]:
        return False, f'hip width {hips:.2f} shoulder widths'
    if arm > SKELETON['max_arm']:
        return False, f'arm segment {arm:.2f} shoulder widths'
    return True, 'ok'


# Shot framing from which joints are on screen (Seedance: hips missing or guessed onto the chest dropped 37 of 43
# windows to face-only). Eyes farther apart than this share of the frame width is a face close-up: Seedance face
# close-ups had .37-.38, bust shots .14-.22.
# On upper-body shots the pose model still guesses ankles at the frame bottom (scores .3-.5, shoulder to ankle
# <=1.2 shoulder widths); real full-body Seedance frames had scores >=.8 and 2.8-4.6 widths.
FRAMING = dict(face_eyes=.3, min_score=.30, border=.02, ankle_score=.6, legs=2.)
FRAMING_KO = dict(full='전신', thigh='무릎 위', upper='상반신', face='얼굴 클로즈업')


def framing(track, width, height):
    """'full' when both ankles are surely on screen well below the shoulders, 'thigh' when the hips pass the full
    skeleton check without feet, 'face' when the eyes are very far apart or no shoulder is seen, else 'upper'."""
    k, s = track['keypoints'], track['scores']
    seen = lambda i: s[i] >= FRAMING['min_score'] and FRAMING['border']*height <= k[i][1] <= (1-FRAMING['border'])*height
    if min(s[1], s[2]) >= FRAMING['min_score'] and math.dist(k[1], k[2]) >= FRAMING['face_eyes']*width:
        return 'face'
    shoulders = math.dist(k[5], k[6])
    if min(s[15], s[16]) >= FRAMING['ankle_score'] and seen(15) and seen(16) and shoulders \
            and (k[15][1]+k[16][1]-k[5][1]-k[6][1])/2 >= FRAMING['legs']*shoulders:
        return 'full'
    if skeleton_check(track)[0]:
        return 'thigh'  # hips and thighs pass the full check, feet out of the shot (AKAGE)
    return 'upper' if seen(5) or seen(6) else 'face'


def upper_check(track):
    """(plausible, reason) for naming arm points when the hips are out of the shot: both shoulders, and every seen arm
    joint with segments no longer than the full check allows."""
    k, s = track['keypoints'], track['scores']
    if min(s[5], s[6]) < SKELETON['min_score']:
        return False, 'shoulders missing'
    seen = [i for i in range(5, 11) if s[i] >= SKELETON['min_score']]
    if sum(s[i] for i in seen)/len(seen) < SKELETON['mean_min']:
        return False, f'arm joint score {sum(s[i] for i in seen)/len(seen):.2f}'
    width = math.dist(k[5], k[6])
    if not width:
        return False, 'zero shoulder width'
    pairs = [(a, b) for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)) if a in seen and b in seen]
    arm = max((math.dist(k[a], k[b]) for a, b in pairs), default=0.)/width
    if arm > SKELETON['max_arm']:
        return False, f'arm segment {arm:.2f} shoulder widths'
    return True, 'upper body'



# Relations and shoulder moves straight from the per-frame pose (every frame is posed). Seedance blind test: hand to
# cheek, hand to face, shoulder rolls and shrugs were missed; the tracked-point path measures limbs against the
# shoulder midpoint, which cancels a shared shrug. Distances are in face or shoulder widths, durations in seconds.
# COCO-WholeBody: 23-90 face contour and features, 91-111 left hand, 112-132 right hand.
EVENTS = dict(min_score=.30, face_points=10, face_pad=.25, near_seconds=.1, gap_frames=3, hip_score=.5, hip_coverage=.8,
              shoulder_move=.12, width_change=.12, turn=.35, hands_seen=.5)
FACE = range(23, 91)
HANDS = dict(l=(9, range(91, 112)), r=(10, range(112, 133)))
REGION_KO = ('이마·눈', '볼', '입·턱')


def _person(tracks, track_id):
    """The seed person on this frame; None when the tracker lost that id. Another detection is never taken as the
    same person (it can be a second figure, or not a person at all)."""
    if track_id is None:
        return largest(tracks)
    return next((t for t in tracks or [] if t.get('track_id') == track_id), None)


def _runs(flags, gap):
    """[(first, last)] of frame runs where flags[n] is true, joining holes of at most `gap` frames."""
    out = []
    for n in sorted(n for n, v in flags.items() if v):
        if out and n-out[-1][1] <= gap+1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return [tuple(r) for r in out]


def _zigzag(values, limit):
    """[(i0, i1, change)] of the rises and falls of a 1-D series of at least `limit`; a turn counts once the series
    has come back by `limit`, so jitter below it never splits a move."""
    out = []; lo = hi = anchor = 0; trend = 0
    for i, v in enumerate(values):
        if trend >= 0 and v >= values[hi]:
            hi = i
        if trend <= 0 and v <= values[lo]:
            lo = i
        if trend == 0 and values[hi]-values[lo] >= limit:
            trend = 1 if hi > lo else -1; anchor = lo if trend > 0 else hi
        elif trend > 0 and values[hi]-v >= limit:
            out.append((anchor, hi, values[hi]-values[anchor])); trend = -1; anchor = hi; lo = i
        elif trend < 0 and v-values[lo] >= limit:
            out.append((anchor, lo, values[lo]-values[anchor])); trend = 1; anchor = lo; hi = i
    end = hi if trend > 0 else lo
    if trend and abs(values[end]-values[anchor]) >= limit:
        out.append((anchor, end, values[end]-values[anchor]))
    return out


def _smooth(values):
    return [v for v, _ in _median3([(v, 0.) for v in values])]


def _screen_side(person, i):
    k = person['keypoints']
    return '화면 왼쪽' if k[i][0] < (k[5][0]+k[6][0])/2 else '화면 오른쪽'


def pose_events(pose_frames, window, fps):
    """Hand-near-face runs, per-shoulder rises and falls, shoulder-width changes and the share of frames each wrist
    was seen, for the person posed on the seed frame. Sides are screen sides on the seed frame."""
    seed = largest(pose_frames.get(window['seed']))
    if not seed:
        return None
    people = {n: _person(pose_frames.get(n), seed.get('track_id')) for n in range(window['seed'], window['end']+1)}
    people = {n: p for n, p in people.items() if p}
    ok = lambda p, i: p['scores'][i] >= EVENTS['min_score']
    sides = dict(l=_screen_side(seed, 5), r=_screen_side(seed, 6))
    near = {s: {} for s in HANDS}; where = {s: {} for s in HANDS}; seen = {s: 0 for s in HANDS}
    for n, p in people.items():
        k = p['keypoints']
        face = [k[i] for i in FACE if ok(p, i)]
        for s, (wrist, hand) in HANDS.items():
            seen[s] += ok(p, wrist)
            points = [k[i] for i in (wrist, *hand) if ok(p, i)]
            # Hand points without their wrist are often something else near the face (Seedance: a cat's paws).
            if len(face) < EVENTS['face_points'] or not ok(p, wrist):
                continue
            x0, x1 = min(q[0] for q in face), max(q[0] for q in face); y0, y1 = min(q[1] for q in face), max(q[1] for q in face)
            pad = EVENTS['face_pad']*(x1-x0)
            inside = [q for q in points if x0-pad <= q[0] <= x1+pad and y0-pad <= q[1] <= y1+pad]
            near[s][n] = bool(inside)
            if inside:
                y = sorted(q[1] for q in inside)[len(inside)//2]
                where[s][n] = min(2, max(0, int(3*(y-y0)/max(1., y1-y0))))
    shortest = max(3, round(EVENTS['near_seconds']*fps))
    hand_face = []
    for s in HANDS:
        for a, b in _runs(near[s], EVENTS['gap_frames']):
            spots = sorted(where[s][n] for n in range(a, b+1) if n in where[s])
            if b-a+1 >= shortest:
                # Held from the seed frame is a pose with an unknown start, not a gesture made here; still worth saying.
                hand_face.append(dict(side=sides[s], frames=[a, b], seconds=round((b-a+1)/fps, 2), region=REGION_KO[spots[len(spots)//2]],
                                      enters=a > window['seed'], exits=b < window['end']))
    shoulders = []; width = []
    both = [n for n, p in people.items() if ok(p, 5) and ok(p, 6)]
    hips = [n for n in both if min(people[n]['scores'][11], people[n]['scores'][12]) >= EVENTS['hip_score']]
    ref = (11, 12) if both and len(hips) >= EVENTS['hip_coverage']*len(both) else (1, 2)
    rows = [n for n in both if all(ok(people[n], i) for i in ref)]
    if len(rows) >= 4:
        k = lambda n: people[n]['keypoints']
        size = sorted(math.dist(k(n)[5], k(n)[6]) for n in rows)[len(rows)//2]
        for i, s in ((5, 'l'), (6, 'r')):
            # Up is positive: the shoulder's height above the hips or eyes, in shoulder widths.
            height = _smooth([(_mean([k(n)[j] for j in ref])[1]-k(n)[i][1])/size for n in rows])
            shoulders += [dict(side=sides[s], rise=c > 0, amount=round(abs(c), 2), frames=[rows[a], rows[b]])
                          for a, b, c in _zigzag(height, EVENTS['shoulder_move'])]
        # Shoulder width against the eye distance, which does not turn with the chest: wider is a chest opening or a
        # turn toward the camera, narrower the reverse.
        scale = [n for n in rows if ok(people[n], 1) and ok(people[n], 2) and math.dist(k(n)[1], k(n)[2])]
        if len(scale) >= 4:
            ratio = _smooth([math.dist(k(n)[5], k(n)[6])/math.dist(k(n)[1], k(n)[2]) for n in scale])
            base = sorted(ratio)[len(ratio)//2]
            width = [dict(wider=c > 0, share=round(abs(c), 2), frames=[scale[a], scale[b]])
                     for a, b, c in _zigzag([r/base for r in ratio], EVENTS['width_change'])]
    # A large width swing is a turn (or a mislabelled, blurred pose): the shoulder points jump, so heights that overlap
    # it mean nothing. Only those are dropped; the rest of the window keeps its shoulder moves.
    turns = [e['frames'] for e in width if e['share'] >= EVENTS['turn']]
    shoulders = [e for e in shoulders if not any(e['frames'][0] <= b and a <= e['frames'][1] for a, b in turns)]
    lost = window['end']-window['seed']+1-len(people)
    return dict(lost_frames=lost, turns=turns, hand_face=sorted(hand_face, key=lambda e: e['frames'][0]), shoulders=sorted(shoulders, key=lambda e: e['frames'][0]),
                width=width, hands_seen={sides[s]: round(seen[s]/max(1, len(people)), 2) for s in HANDS})


# Several people (Kaguya live scene: two or three girls lean on, pull and point at each other; only the largest was
# followed, and a girl falling backward read as a camera pan). Each person is followed by the pose tracker id alone;
# relations are pose geometry in shoulder widths and are candidates. Overlapping figures often merge into one
# detection box with a mixed skeleton (Kaguya f450-f530), so heavy box overlap is said, not hidden.
PEOPLE = dict(max_people=3, presence=.5, min_area=.25, move=.5, distance=.5, touch_pad=.25, touch_seconds=.1,
              upright_deg=30., fallen_deg=60., overlap_iou=.5, overlap_share=.5)


def _iou(a, b):
    w = max(0., min(a[2], b[2])-max(a[0], b[0])); h = max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h
    return w*h/union if union > 0 else 0.


def _centre(t):
    k, s = t['keypoints'], t['scores']
    if min(s[5], s[6]) >= EVENTS['min_score']:
        return _mean([k[5], k[6]]), math.dist(k[5], k[6])
    x0, y0, x1, y1 = t['bbox']
    return ((x0+x1)/2, (y0+y1)/2), None


def people(pose_frames, window, fps, camera=None):
    """Up to three people present in most of the window, named P1.. from screen left; per person the body-centre
    shift (camera removed) and a fall candidate; per pair distance change, hand-on-body runs and box overlap."""
    frames = range(window['seed'], window['end']+1)
    tracks = {}
    for n in frames:
        for t in pose_frames.get(n) or []:
            if t.get('track_id') is not None:
                tracks.setdefault(t['track_id'], {})[n] = t
    area = lambda t: (t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1])
    median = lambda v: sorted(v)[len(v)//2]
    found = {i: seen for i, seen in tracks.items() if len(seen) >= PEOPLE['presence']*len(frames)}
    if not found:
        return dict(people=[], relations=[])
    big = max(median([area(t) for t in seen.values()]) for seen in found.values())
    found = sorted(((i, seen) for i, seen in found.items() if median([area(t) for t in seen.values()]) >= PEOPLE['min_area']*big),
                   key=lambda x: -median([area(t) for t in x[1].values()]))[:PEOPLE['max_people']]
    found.sort(key=lambda x: median([(t['bbox'][0]+t['bbox'][2])/2 for t in x[1].values()]))
    cam = camera or (0., 0.)
    out = []
    for number, (track_id, seen) in enumerate(found, 1):
        first, last = min(seen), max(seen)
        (a, wa), (b, wb) = _centre(seen[first]), _centre(seen[last])
        size = median([w for w in (_centre(t)[1] for t in seen.values()) if w] or [math.dist(seen[first]['bbox'][:2], seen[first]['bbox'][2:])/4])
        dx, dy = b[0]-a[0]-cam[0], b[1]-a[1]-cam[1]
        person = dict(name=f'P{number}', track_id=track_id, frames=[first, last], seen=round(len(seen)/len(frames), 2), size_px=round(size, 1),
                      bbox=seen[first]['bbox'], move=dict(direction=motion_report.direction(dx, dy), widths=round(math.hypot(dx, dy)/size, 2)))
        # Torso lean from vertical (shoulder midpoint over hip midpoint), only where both pairs are seen.
        lean = {}
        for n, t in seen.items():
            k, s = t['keypoints'], t['scores']
            if min(s[5], s[6], s[11], s[12]) >= EVENTS['min_score']:
                (sx, sy), (hx, hy) = _mean([k[5], k[6]]), _mean([k[11], k[12]])
                lean[n] = math.degrees(math.atan2(abs(sx-hx), max(1e-6, hy-sy))) if hy > sy else 180.
        if lean:
            start = lean[min(lean)]; worst = max(lean, key=lean.get)
            if start <= PEOPLE['upright_deg'] and lean[worst] >= PEOPLE['fallen_deg']:
                person['fall'] = dict(frame=worst, degrees=round(lean[worst]))
        out.append(person)
    shortest = max(3, round(PEOPLE['touch_seconds']*fps))
    relations = []
    for i, p in enumerate(out):
        for q in out[i+1:]:
            a, b = tracks[p['track_id']], tracks[q['track_id']]
            both = sorted(set(a) & set(b))
            if not both:
                continue
            size = (p['size_px']+q['size_px'])/2
            gap = lambda n: math.dist(_centre(a[n])[0], _centre(b[n])[0])/size
            change = gap(both[-1])-gap(both[0])
            rel = dict(pair=[p['name'], q['name']], frames=[both[0], both[-1]])
            if abs(change) >= PEOPLE['distance']:
                rel['distance'] = dict(closer=change < 0, widths=round(abs(change), 2))
            overlap = sum(_iou(a[n]['bbox'], b[n]['bbox']) >= PEOPLE['overlap_iou'] for n in both)
            rel['overlap'] = overlap >= PEOPLE['overlap_share']*len(both)
            touches = []
            for who, hand_track, body_track in ((p['name'], a, b), (q['name'], b, a)):
                for wrist in (9, 10):
                    on = {}
                    for n in both:
                        h, t = hand_track[n], body_track[n]
                        k, s = t['keypoints'], t['scores']
                        if h['scores'][wrist] < EVENTS['min_score'] or min(s[5], s[6]) < EVENTS['min_score']:
                            continue
                        width = math.dist(k[5], k[6]); pad = PEOPLE['touch_pad']*width
                        low = max(k[11][1], k[12][1]) if min(s[11], s[12]) >= EVENTS['min_score'] else max(k[5][1], k[6][1])+width
                        x, y = h['keypoints'][wrist]
                        on[n] = min(k[5][0], k[6][0])-pad <= x <= max(k[5][0], k[6][0])+pad and min(k[5][1], k[6][1])-pad <= y <= low
                    for lo, hi in _runs(on, EVENTS['gap_frames']):
                        if hi-lo+1 >= shortest:
                            touches.append(dict(hand=who, body=q['name'] if who == p['name'] else p['name'], frames=[lo, hi]))
            rel['touch'] = sorted(touches, key=lambda e: e['frames'][0])
            relations.append(rel)
    return dict(people=out, relations=relations)
