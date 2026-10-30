"""Whole-frame motion (camera pan or zoom, scrolling UI) per window, so screen strokes that only follow it can be set aside.
ORB matches at half resolution, RANSAC similarity; frames with too few inliers stay unknown."""
import math
import statistics

# A big character sliding over a static layout also yields one dominant transform; count it as the frame moving only
# when the agreeing matches span most of the frame. Clip A's sliding figure spans 32-37% of the width; the clip C
# playlist pan spans 53-70% of the width but only 27-58% of the height (the cover art is a wide band).
CAMERA = dict(scale=.5, features=2000, min_inliers=30, ransac_px=3., min_spread=(.5, .25), rekey_px=24., chain_spread=(.75, .5), still_share=.25, known_share=.5, zoom=1.1)


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


def path(video, start, end):
    """{frame: (dx, dy, scale)} of the frame centre relative to `start`; frames that could not be matched are left out.
    Frames are matched to the start frame. When one does not match and the last matched frame is CAMERA['rekey_px'] or
    more away, it is matched to that frame instead and the steps are chained, so a pan stays known after the start
    frame has left the view (clip D 70-100 s: a 680 px pan at 23.35 s was unknown). A chained step needs matches over
    CAMERA['chain_spread'] of the frame (ORB leaves a 31 px border, so about .94 is the most at 1920 px): frame to frame, a big figure over a still layout moves as one piece and read
    as the camera (clip A 2.76 s: 50 px, really 0; the figure spans about 70% of the width).
    A frame's value depends only on the frames before it, so the last path is kept and a shorter or equal request from
    the same start is cut from it (seed and look both ask for each window: clip B 42 x 0.4 s)."""
    import os
    stat = os.stat(video); key = (str(video), stat.st_size, stat.st_mtime_ns, start)
    if _LAST and _LAST[0] == key and _LAST[1] >= end:
        return {n: v for n, v in _LAST[2].items() if n <= end}
    result = _path(video, start, end)
    _LAST[:] = [key, end, result]
    return dict(result)


_LAST = []


def _path(video, start, end):
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    grays = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for n in range(start, end+1):
            ok, bgr = cap.read()
            if not ok:
                break
            grays.append(cv2.cvtColor(cv2.resize(bgr, None, fx=CAMERA['scale'], fy=CAMERA['scale']), cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    # ORB per frame is independent and gives the same points from a fresh detector, so it runs in threads (clip B
    # window: 0.14 -> 0.02 s); matching and RANSAC stay in frame order.
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda gray: cv2.ORB_create(CAMERA['features']).detectAndCompute(gray, None), grays))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    result = {}
    start_ref = last = None
    for n, gray, (keys, desc) in zip(range(start, end+1), grays, found):
        if start_ref is None:
            start_ref = last = (keys, desc, np.eye(3), 0.); centre = (gray.shape[1]/2, gray.shape[0]/2)
            result[n] = (0., 0., 1.)
            continue
        ref = start_ref
        M = _step(cv2, np, matcher, ref, keys, desc, gray.shape)
        if M is None and last is not start_ref and last[3] >= CAMERA['rekey_px']:
            ref = last
            M = _step(cv2, np, matcher, ref, keys, desc, gray.shape, CAMERA['chain_spread'])
        if M is None:
            continue
        total = np.vstack([M, [0, 0, 1]]) @ ref[2]
        x, y = total[0, 0]*centre[0]+total[0, 1]*centre[1]+total[0, 2], total[1, 0]*centre[0]+total[1, 1]*centre[1]+total[1, 2]
        result[n] = ((x-centre[0])/CAMERA['scale'], (y-centre[1])/CAMERA['scale'], math.hypot(total[0, 0], total[1, 0]))
        last = (keys, desc, total, math.hypot(*result[n][:2]))
    return result


def between(camera, a, b):
    """Camera displacement from frame a to b, or None when either end is unknown."""
    if not camera or a not in camera or b not in camera:
        return None
    return camera[b][0]-camera[a][0], camera[b][1]-camera[a][1]


def describe(camera, start, end, min_px):
    """What the whole frame did over [start, end]: net shift, speeds over time, the farthest point of a move that came
    back, and zoom. `known` is the share of frames matched; under CAMERA['known_share'] the camera is unknown, which
    is not stillness."""
    from image_lens import motion_report
    frames = sorted(n for n in (camera or {}) if start <= n <= end)
    out = dict(net=between(camera, start, end), frames=len(frames), known=round(len(frames)/(end-start+1), 2))
    if out['known'] < CAMERA['known_share'] or len(frames) < 2:
        out['unknown'] = True
        return out
    last = frames[-1]
    shift = [(camera[n][0]-camera[frames[0]][0], camera[n][1]-camera[frames[0]][1]) for n in frames]
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
    scales = [camera[n][2] for n in frames]
    if max(scales[-1], 1/scales[-1]) >= CAMERA['zoom']:
        out['zoom'] = round(scales[-1], 2)
    elif max(max(scales), 1/min(scales)) >= CAMERA['zoom']:
        k = max(range(len(scales)), key=lambda i: max(scales[i], 1/scales[i]))
        out['zoom_back'] = dict(frame=frames[k], scale=round(scales[k], 2))
    out['net'] = list(net) if out['net'] is None else list(out['net'])
    return out
