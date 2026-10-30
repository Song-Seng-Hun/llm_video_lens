"""The last decoded frame window of one source, shared by the per-window steps.
Seed points, crops, camera, trajectory sheet, onion, peak and motif each read the same window from its own seek
(clip B: 8 decodes of every window); now the first reads it and the rest are cut from it. The arrays are shared:
callers must not write into them."""
import os

_LAST = []


def window(video, start, end):
    """BGR frames start..end read in order after one seek, cut at the first failed read."""
    return [bgr for bgr, _ in _read(video, start, end)]


def checked(video, start, end, width, height, error):
    """window() that raises ValueError(error) unless every frame was read, at its index and with this geometry."""
    got = _read(video, start, end)
    if len(got) != end-start+1 or any(pos != n+1 or bgr.shape[:2] != (height, width) for n, (bgr, pos) in zip(range(start, end+1), got)):
        raise ValueError(error)
    return [bgr for bgr, _ in got]


def _read(video, start, end):
    s = os.stat(video)
    key = (str(video), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if not (_LAST and _LAST[0] == key and _LAST[1] <= start and end <= _LAST[2]):
        import cv2
        cap = cv2.VideoCapture(str(video)); got = []
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, end+1):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    got.append((bgr, round(cap.get(cv2.CAP_PROP_POS_FRAMES))))
        finally:
            cap.release()
        _LAST[:] = [key, start, end, got]
    return _LAST[3][start-_LAST[1]:end-_LAST[1]+1]
