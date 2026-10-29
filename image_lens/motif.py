"""Repeated poses: windows whose start and end drawings match an earlier window's are one pose shown again (AKAGE
cycles four costumes, each with the same move). look.md then says the pose once and only how a repeat differs."""
import math

MOTIF = dict(size=24, pad=.1, same=.9, length_share=.8)


def thumb(video, frame, bbox):
    """The person box of one frame as a small zero-mean, unit-length grey vector (position and scale removed)."""
    import cv2
    import numpy as np
    from image_lens import onion
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = onion._box(bbox, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    small = cv2.resize(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (MOTIF['size'],)*2, interpolation=cv2.INTER_AREA)
    v = small.astype(np.float32).ravel(); v -= v.mean(); n = float(np.linalg.norm(v))
    return None if n == 0 else (v/n).tolist()


def similarity(a, b):
    return sum(x*y for x, y in zip(a, b))


def assign(windows):
    """windows: [{start, end, frames}] with start/end thumbs (or None). Returns one label per window: 'A', 'B', ... for
    a pose seen at least twice, None for one seen once. A repeat matches the first window of its pose."""
    firsts = []; group = []
    for w in windows:
        match = None
        if w['start'] and w['end']:
            for k, f in enumerate(firsts):
                if min(w['frames'], f['frames']) >= MOTIF['length_share']*max(w['frames'], f['frames']) \
                        and similarity(w['start'], f['start']) >= MOTIF['same'] and similarity(w['end'], f['end']) >= MOTIF['same']:
                    match = k
                    break
            if match is None:
                firsts.append(w); match = len(firsts)-1
        group.append(match)
    counts = {g: group.count(g) for g in group if g is not None}
    names = {}
    for g in group:
        if g is not None and counts[g] > 1 and g not in names:
            names[g] = chr(ord('A')+len(names)) if len(names) < 26 else f'P{len(names)+1}'
    return [names.get(g) for g in group]
