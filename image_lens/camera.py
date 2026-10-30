"""Whole-frame motion (camera pan or zoom, scrolling UI) per window, so screen strokes that only follow it can be set aside.
ORB matches against the window's seed frame at half resolution, RANSAC similarity; frames with too few inliers stay unknown."""
import math

# A big character sliding over a static layout also yields one dominant transform; count it as the frame moving only
# when the agreeing matches span most of the frame. AKAGE's sliding figure spans 32-37% of the width; the MikuMiku
# playlist pan spans 53-70% of the width but only 27-58% of the height (the cover art is a wide band).
CAMERA = dict(scale=.5, features=2000, min_inliers=30, ransac_px=3., min_spread=(.5, .25))


def path(video, start, end):
    """{frame: (dx, dy, scale)} of the frame centre relative to `start`; frames that could not be matched are left out."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    orb = cv2.ORB_create(CAMERA['features']); matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    result = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        base = None
        for n in range(start, end+1):
            ok, bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(bgr, None, fx=CAMERA['scale'], fy=CAMERA['scale']), cv2.COLOR_BGR2GRAY)
            keys, desc = orb.detectAndCompute(gray, None)
            if base is None:
                base = keys, desc, (gray.shape[1]/2, gray.shape[0]/2)
                result[n] = (0., 0., 1.)
                continue
            if desc is None or base[1] is None:
                continue
            matches = matcher.match(base[1], desc)
            if len(matches) < CAMERA['min_inliers']:
                continue
            a = np.float32([base[0][m.queryIdx].pt for m in matches]); b = np.float32([keys[m.trainIdx].pt for m in matches])
            M, inliers = cv2.estimateAffinePartial2D(a, b, ransacReprojThreshold=CAMERA['ransac_px'])
            if M is None or inliers.sum() < CAMERA['min_inliers']:
                continue
            agree = b[inliers.ravel() == 1]
            spread = (agree[:, 0].max()-agree[:, 0].min())/gray.shape[1], (agree[:, 1].max()-agree[:, 1].min())/gray.shape[0]
            if spread[0] < CAMERA['min_spread'][0] or spread[1] < CAMERA['min_spread'][1]:
                continue
            cx, cy = base[2]
            x, y = M[0, 0]*cx+M[0, 1]*cy+M[0, 2], M[1, 0]*cx+M[1, 1]*cy+M[1, 2]
            result[n] = ((x-cx)/CAMERA['scale'], (y-cy)/CAMERA['scale'], math.hypot(M[0, 0], M[1, 0]))
    finally:
        cap.release()
    return result


def between(camera, a, b):
    """Camera displacement from frame a to b, or None when either end is unknown."""
    if not camera or a not in camera or b not in camera:
        return None
    return camera[b][0]-camera[a][0], camera[b][1]-camera[a][1]
