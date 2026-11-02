"""Whole-frame motion (camera pan or zoom, scrolling UI) per window, so screen strokes that only follow it can be set aside.
ORB matches at half resolution, RANSAC similarity; frames with too few inliers stay unknown."""
import math
import statistics
import copy

# A big character sliding over a static layout also yields one dominant transform; count it as the frame moving only
# when the agreeing matches span most of the frame. Clip A's sliding figure spans 32-37% of the width; the clip C
# playlist pan spans 53-70% of the width but only 27-58% of the height (the cover art is a wide band).
CAMERA = dict(scale=.5, features=2000, min_inliers=30, ransac_px=3., min_spread=(.5, .25), rekey_px=24., chain_spread=(.75, .5), still_share=.25, person_pad=.1, background_features=200, known_share=.5, zoom=1.1)


def _matrix(value):
    rows = tuple(tuple(float(v) for v in row) for row in value)
    if (len(rows) != 3 or any(len(row) != 3 for row in rows) or not all(math.isfinite(v) for row in rows for v in row)
            or rows[2] != (0., 0., 1.) or rows[0][0]**2+rows[1][0]**2 <= 0
            or not math.isclose(rows[0][0], rows[1][1], rel_tol=1e-6, abs_tol=1e-9)
            or not math.isclose(rows[0][1], -rows[1][0], rel_tol=1e-6, abs_tol=1e-9)):
        raise ValueError('finite non-reflected invertible camera similarity required')
    return rows


class CameraPath(dict):
    """Native reference→frame matrices. Missing frames are absent, never identity estimates."""
    def __init__(self, samples=(), *, reference=0, size=None, native_range=None, source=None, kind='similarity', diagnostic_scales=None):
        if kind not in ('similarity','legacy_translation'):raise ValueError('unknown camera correction kind')
        if kind=='similarity' and (size is None or len(size)!=2 or not all(math.isfinite(v) and v>0 for v in size)):
            raise ValueError('canonical camera similarity requires positive native size')
        super().__init__({n: _matrix(matrix) for n, matrix in dict(samples).items()})
        if any(type(n) is not int for n in self):
            raise ValueError('native camera frame keys required')
        self.reference = reference; self.size = tuple(size) if size is not None else None
        self.native_range = copy.deepcopy(native_range); self.source = copy.deepcopy(source); self.kind = kind
        self.diagnostic_scales = copy.deepcopy(diagnostic_scales) if diagnostic_scales is not None else {}

    def prefix(self, end):
        return CameraPath({n:m for n,m in self.items() if n <= end}, reference=self.reference, size=self.size,
            native_range=[self.reference, end], source=self.source, kind=self.kind, diagnostic_scales=self.diagnostic_scales)


def normalize(value):
    if isinstance(value, CameraPath):
        return value
    if not isinstance(value, dict):
        raise ValueError('camera path required')
    if 'schema' in value:
        if value['schema'] != 'image-lens.camera/1' or value.get('orientation') != 'reference_to_source_frame':
            raise ValueError('unknown explicit camera schema/orientation')
        if len({row['frame'] for row in value['samples']}) != len(value['samples']):raise ValueError('duplicate camera samples')
        return CameraPath({row['frame']:row['matrix'] for row in value['samples']}, reference=value['reference_frame'],
            size=value.get('native_size'), native_range=value.get('native_range'), source=value.get('source'),
            kind=value['correction_kind'], diagnostic_scales={n:k for n,k in value.get('diagnostic_scales', [])})
    samples = {}; scales = {}
    for n, v in value.items():
        if type(n) is not int or not isinstance(v, (tuple, list)) or len(v) not in (2, 3) or not all(math.isfinite(x) for x in v):
            raise ValueError('explicit legacy translation pair/triple required')
        if len(v) == 3 and v[2] <= 0:
            raise ValueError('invalid legacy scale diagnostic')
        samples[n] = ((1., 0., float(v[0])), (0., 1., float(v[1])), (0., 0., 1.))
        scales[n] = float(v[2]) if len(v) == 3 else 1.
    return CameraPath(samples, reference=min(samples, default=0), kind='legacy_translation', diagnostic_scales=scales)


def transforms(value, reference=None):
    import numpy as np
    path = normalize(value); reference = path.reference if reference is None else reference
    if reference not in path:
        return {}
    base = np.asarray(_matrix(path[reference]))
    return {n:base @ np.linalg.inv(np.asarray(_matrix(matrix))) for n,matrix in path.items()}


def apply(matrix, xy):
    return (float(matrix[0][0]*xy[0]+matrix[0][1]*xy[1]+matrix[0][2]),
            float(matrix[1][0]*xy[0]+matrix[1][1]*xy[1]+matrix[1][2]))


def warp(image, matrix, size, *, offset=(0., 0.), shift=(0., 0.), border=None):
    """Current pixels→reference crop, same matrix as points; no inverse-map flag."""
    import cv2
    import numpy as np
    matrix = np.asarray(matrix, dtype=float).copy()
    matrix[:2, 2] -= np.asarray(offset)+np.asarray(shift)
    return cv2.warpAffine(image, matrix[:2], size, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE if border is None else border)


def correct_packet(packet, value, reference=None):
    path = normalize(value);reference = path.reference if reference is None else reference
    matrices = transforms(path, reference); rows = [];basis = scope(path)
    for row in packet['frames']:
        matrix = matrices.get(row['frame']); points = []
        for point in row['points']:
            out = dict(point,source_xy=copy.deepcopy(point.get('xy')),source_visible=point.get('visible'),
                       coordinate_scope=basis if matrix is not None else 'camera_transform_unknown')
            if matrix is not None and point.get('xy') is not None:
                out['xy'] = list(apply(matrix, point['xy']))
            elif matrix is None:
                out.update(visible=False, motion_pair_usable=False, tracking_metric_eligible=False,
                           motion_geometry_usable=False, motion_geometry_exclusion='camera_transform_unknown')
            points.append(out)
        rows.append(dict(row, points=points))
    return dict(packet, frames=rows, coordinate_scope=basis, coordinate_reference_frame=reference,
                coordinate_provenance='derived camera view; source_xy/source_visible retain native observations')


def correct_pose(frames, value, reference=None):
    matrices = transforms(value, reference)
    return {n:[dict(person, keypoints=[list(apply(matrices[n], xy)) for xy in person['keypoints']]) for person in people]
            for n,people in frames.items() if n in matrices}


def centre_path(value, reference=None):
    """Forward centre displacement/scale relative to the same declared frame as correction."""
    import numpy as np
    path = normalize(value);reference = path.reference if reference is None else reference
    if reference not in path:return {}
    centre = (path.size[0]/2,path.size[1]/2) if path.size else (0.,0.)
    inverse = np.linalg.inv(np.asarray(_matrix(path[reference])));out = {}
    for n,m in path.items():
        relative = np.eye(3) if n==reference else np.asarray(_matrix(m)) @ inverse
        xy = apply(relative,centre)
        scale = (path.diagnostic_scales.get(n,1.)/path.diagnostic_scales.get(reference,1.) if path.kind=='legacy_translation'
                 else math.hypot(relative[0,0],relative[1,0]))
        out[n] = (xy[0]-centre[0],xy[1]-centre[1],float(scale))
    return out


def scope(value):
    return 'camera_similarity_relative_2d' if normalize(value).kind == 'similarity' else 'camera_translation_relative_2d'


def record(value, source_sha256=None):
    path = normalize(value)
    return dict(schema='image-lens.camera/1', orientation='reference_to_source_frame', correction_kind=path.kind,
                reference_frame=path.reference, native_size=list(path.size) if path.size else None, native_range=copy.deepcopy(path.native_range),
                source=copy.deepcopy(path.source), source_sha256=source_sha256, samples=[dict(frame=n,matrix=path[n]) for n in sorted(path)],
                diagnostic_scales=copy.deepcopy(sorted(path.diagnostic_scales.items())))


def view_corners(value, width, height, reference=None):
    return {n:[apply(matrix, xy) for xy in ((0.,0.),(width,0.),(width,height),(0.,height))]
            for n,matrix in transforms(value, reference).items()}


def displacement(matrix, width, height):
    return max(math.dist(xy, apply(matrix, xy)) for xy in ((0.,0.),(width,0.),(width,height),(0.,height),(width/2,height/2)))


def _step(cv2, np, matcher, base, keys, desc, shape, spread_min=CAMERA['min_spread']):
    """2x3 similarity from `base` (keys, desc) to this frame, or None."""
    if desc is None or base[1] is None:
        return None
    matches = matcher.match(base[1], desc)
    if len(matches) < CAMERA['min_inliers']:
        return None
    a = np.float32([base[0][m.queryIdx].pt for m in matches]); b = np.float32([keys[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(a, b, ransacReprojThreshold=CAMERA['ransac_px'])
    if M is None or inliers.sum() < CAMERA['min_inliers']:
        return None
    # A still layer (an overlay panel, a static background) next to a big moving figure: the agreeing matches are the
    # figure, not the camera (clip A: the title panel stays while the dancer moves 30-50 px).
    shift = math.hypot(M[0, 2]+(M[0, 0]-1)*shape[1]/2+M[0, 1]*shape[0]/2, M[1, 2]+M[1, 0]*shape[1]/2+(M[1, 1]-1)*shape[0]/2)
    still = int((np.hypot(*(b-a).T) <= CAMERA['ransac_px']).sum())
    if shift > CAMERA['ransac_px'] and still >= max(CAMERA['min_inliers'], CAMERA['still_share']*inliers.sum()):
        return None
    agree = b[inliers.ravel() == 1]
    spread = (agree[:, 0].max()-agree[:, 0].min())/shape[1], (agree[:, 1].max()-agree[:, 1].min())/shape[0]
    if spread[0] < spread_min[0] or spread[1] < spread_min[1]:
        return None
    return M


def path(video, start, end, person=None):
    """CameraPath of native reference→frame similarities; unmatched frames are left out.
    With `person` (the main person's box on the start frame) a frame is matched on the background around it first.
    When the seed has enough background features, only that background may establish the camera path: a failed
    background fit stays unknown, rather than borrowing the moving subject's full-frame fit.
    Frames are matched to the start frame. When one does not match and the last matched frame is CAMERA['rekey_px'] or
    more away, that frame becomes a retained local reference. Later frames under that same gate try it before the
    last matched frame; only a successful replacement adds another chained step. A direct start-frame match clears the local reference,
    so a pan stays known after the start frame has left the view. A chained step needs matches over
    CAMERA['chain_spread'] of the frame (ORB leaves a 31 px border, so about .94 is the most at 1920 px): frame to frame, a big figure over a still layout moves as one piece and read
    as the camera (clip A 2.76 s: 50 px, really 0; the figure spans about 70% of the width).
    A frame's value depends only on the frames before it, so the last path is kept and a shorter or equal request from
    the same start is cut from it (seed and look both ask for each window: clip B 42 x 0.4 s). Two are kept: look
    reports a window after seeding the next."""
    import os
    stat = os.stat(video); key = (str(video), stat.st_size, stat.st_mtime_ns, start, tuple(person) if person else None, 'image-lens.camera/1')
    for kept_key, kept_end, kept in _LAST:
        if kept_key == key and kept_end >= end:
            return normalize(kept).prefix(end)
    result = _path(video, start, end, person)
    result.source = dict(video=str(video), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    _LAST[:] = [(key, end, result)]+_LAST[:1]
    return normalize(result).prefix(end)


_LAST = []


def _path(video, start, end, person=None):
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    from image_lens import frames
    window = frames.window(video, start, end)
    grays = [cv2.cvtColor(cv2.resize(bgr, None, fx=CAMERA['scale'], fy=CAMERA['scale']), cv2.COLOR_BGR2GRAY) for bgr in window]
    # ORB per frame is independent and gives the same points from a fresh detector, so it runs in threads (clip B
    # window: 0.14 -> 0.02 s); matching and RANSAC stay in frame order.
    # A big textured figure over a plain room takes most of the 2000 points, and its moves outvote the background
    # (3D dance clip, fixed camera: 101 of 130 frames unknown); with the person masked out the room alone gives
    # 300-900 agreeing matches at 0-1 px. Use the whole frame only when the seed background has too few features;
    # once a background was admitted, a failed fit must not promote the moving figure to a camera estimate.
    def detect(mask):
        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(lambda gray: cv2.ORB_create(CAMERA['features']).detectAndCompute(gray, mask), grays))
    kinds = [detect(None)]
    if person is not None and grays:
        h, w = grays[0].shape; x0, y0, x1, y1 = person; px, py = CAMERA['person_pad']*(x1-x0), CAMERA['person_pad']*(y1-y0)
        mask = np.full((h, w), 255, np.uint8)
        mask[max(0, int((y0-py)*CAMERA['scale'])):max(0, int((y1+py)*CAMERA['scale'])),
             max(0, int((x0-px)*CAMERA['scale'])):max(0, int((x1+px)*CAMERA['scale']))] = 0
        background = detect(mask)
        if len(background[0][0]) >= CAMERA['background_features']:
            kinds = [background]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    size = (window[0].shape[1],window[0].shape[0]) if window else None
    result = CameraPath(reference=start, size=size, native_range=[start,end], source=dict(video=str(video)))
    start_ref = last = local_ref = None
    for i, (n, gray) in enumerate(zip(range(start, end+1), grays)):
        here = [kind[i] for kind in kinds]
        if start_ref is None:
            start_ref = last = (here, np.eye(3), 0.); centre = (gray.shape[1]/2, gray.shape[0]/2)
            result[n] = _matrix(np.eye(3))
            continue
        found = None
        for (keys, desc), base in zip(here, start_ref[0]):
            M = _step(cv2, np, matcher, base, keys, desc, gray.shape)
            if M is not None:
                found = (M, start_ref); break
        if found is not None:
            local_ref = None
        elif last is not start_ref and last[2] >= CAMERA['rekey_px']:
            references = [local_ref] if local_ref is not None else []
            if last is not local_ref:
                references.append(last)
            for reference in references:
                for (keys, desc), base in zip(here, reference[0]):
                    M = _step(cv2, np, matcher, base, keys, desc, gray.shape, CAMERA['chain_spread'])
                    if M is not None:
                        found = (M, reference); local_ref = reference; break
                if found is not None:
                    break
        if found is None:
            continue
        total = np.vstack([found[0], [0, 0, 1]]) @ found[1][1]
        D = np.diag([CAMERA['scale'],CAMERA['scale'],1.])
        native = np.linalg.inv(D) @ total @ D
        result[n] = _matrix(native)
        last = (here, total, displacement(native, *size))
    return result


def plain(video, frame, person=None):
    """True when the background on `frame` (around `person`, or the whole frame) has too few ORB points to place the
    camera: a plain studio wall says nothing about a pan, so unknown there means "cannot see", not "matches failed"
    (solo dance clip on a white cyclorama: 0 background points with the dancer masked, 224 on the dancer; five of the seven
    sample clips 2000)."""
    import cv2
    import numpy as np
    from image_lens import frames
    bgr = next(iter(frames.window(video, frame, frame)))
    gray = cv2.cvtColor(cv2.resize(bgr, None, fx=CAMERA['scale'], fy=CAMERA['scale']), cv2.COLOR_BGR2GRAY)
    mask = None
    if person is not None:
        h, w = gray.shape; x0, y0, x1, y1 = person; px, py = CAMERA['person_pad']*(x1-x0), CAMERA['person_pad']*(y1-y0)
        mask = np.full((h, w), 255, np.uint8)
        mask[max(0, int((y0-py)*CAMERA['scale'])):max(0, int((y1+py)*CAMERA['scale'])),
             max(0, int((x0-px)*CAMERA['scale'])):max(0, int((x1+px)*CAMERA['scale']))] = 0
    return len(cv2.ORB_create(CAMERA['features']).detect(gray, mask)) < CAMERA['background_features']


def supported(camera, start, end):
    """A correction needs known endpoints and sufficient camera evidence; interior gaps stay unsupported."""
    if not camera:
        return False
    try:
        path = normalize(camera)
        if start not in path or end not in path:return False
        for n,matrix in path.items():
            if start <= n <= end:_matrix(matrix)
    except (ValueError,TypeError,OverflowError):return False
    return not describe(camera, start, end, 0).get('unknown', False)


def between(camera, a, b):
    """Camera displacement from frame a to b, or None when either end is unknown."""
    if not camera:
        return None
    summary = centre_path(camera,a)
    if a not in summary or b not in summary:return None
    return summary[b][:2]


def describe(camera, start, end, min_px):
    """What the whole frame did over [start, end]: net shift, speeds over time, the farthest point of a move that came
    back, and zoom. `known` is the share of frames matched; under CAMERA['known_share'] the camera is unknown, which
    is not stillness."""
    from image_lens import motion_report
    path = normalize(camera or {}); summary = centre_path(path,start)
    frames = sorted(n for n in path if start <= n <= end)
    out = dict(net=between(camera, start, end), frames=len(frames), known=round(len(frames)/(end-start+1), 2),transform=record(path),
               diagnostic_reference_frame=start)
    if out['known'] < CAMERA['known_share'] or len(frames) < 2 or start not in path:
        out['unknown'] = True
        return out
    last = frames[-1]
    shift = [(summary[n][0]-summary[frames[0]][0], summary[n][1]-summary[frames[0]][1]) for n in frames]
    net = shift[-1]
    # A frame or two that matched the figure instead of the background jumps 30-50 px and back (clip A): the farthest
    # point is taken from the running median of five.
    half = 2
    smooth = [(statistics.median(p[0] for p in shift[max(0, i-half):i+half+1]), statistics.median(p[1] for p in shift[max(0, i-half):i+half+1]))
              for i in range(len(shift))]
    far = max(range(len(smooth)), key=lambda i: math.hypot(*smooth[i]))
    if last != end:
        out['until'] = last
    if math.hypot(*net) >= min_px:
        out['speeds'] = motion_report.speeds(frames, shift)
    # Out and back: the farthest point is well past where it ended.
    if math.hypot(*smooth[far]) >= max(min_px, 2*math.hypot(*net)) and math.hypot(*smooth[far])-math.hypot(*net) >= min_px:
        out['back'] = dict(frame=frames[far], dx=smooth[far][0], dy=smooth[far][1])
    scales = [summary[n][2] for n in frames]
    if max(scales[-1], 1/scales[-1]) >= CAMERA['zoom']:
        out['zoom'] = round(scales[-1], 2)
    elif max(max(scales), 1/min(scales)) >= CAMERA['zoom']:
        k = max(range(len(scales)), key=lambda i: max(scales[i], 1/scales[i]))
        out['zoom_back'] = dict(frame=frames[k], scale=round(scales[k], 2))
    out['net'] = list(net) if out['net'] is None else list(out['net'])
    if path.kind == 'similarity':
        import numpy as np
        relative = np.linalg.inv(transforms(path,start)[last])
        out['rotation_clockwise_degrees'] = math.degrees(math.atan2(relative[1,0],relative[0,0]))
    return out
