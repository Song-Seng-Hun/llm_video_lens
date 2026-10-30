"""Second read. After reading a look, the reader names at most two short spans it could not tell; zoom shows each as
dense whole frames in time order (who pulls whom, a fall: clip D 18-22 s was one merged pose box, so pose lines and
person crops cannot show it). The frame budget is fixed and counted, so a gain can be checked against the same number
of frames spread evenly over the clip (--even). There the even frames gained more (clip D 8.5 -> 12 of 20 against
10.5 -> 11.5 for the reader's own spans), so `look` also writes such frames for the whole clip (`overview`)."""
import json

ZOOM = dict(spans=2, span_s=4., frames=12, budget=24, columns=4, tile=480, label=22, every_s=1.25, max_overview=96)


def spans(texts):
    """'a-b' seconds -> [(a, b)]; at most ZOOM['spans'], each longer than 0 and at most ZOOM['span_s']."""
    out = []
    for text in texts:
        try:
            a, b = (float(v) for v in text.split('-'))
        except ValueError:
            raise ValueError(f'span {text!r}: use start-end seconds, e.g. 18-22')
        if not 0 <= a < b or b-a > ZOOM['span_s']:
            raise ValueError(f"span {text!r}: start < end, at most {ZOOM['span_s']:g} s")
        out.append((a, b))
    if not 1 <= len(out) <= ZOOM['spans']:
        raise ValueError(f"1..{ZOOM['spans']} spans")
    return out


def picks(first, last, count):
    """`count` frame numbers from first to last inclusive, evenly spaced, no repeats."""
    if count == 1 or first == last:
        return [first]
    return sorted({round(first+i*(last-first)/(count-1)) for i in range(count)})


def sheet(frames, fps, path):
    """Frames (number, image) as a grid of tiles in time order, each labelled with its time in seconds."""
    import cv2
    import numpy as np
    t, top, columns = ZOOM['tile'], ZOOM['label'], ZOOM['columns']
    tiles = []
    for n, image in frames:
        small = cv2.resize(image, (t, max(1, round(image.shape[0]*t/image.shape[1]))), interpolation=cv2.INTER_AREA)
        tile = np.zeros((small.shape[0]+top, t, 3), np.uint8); tile[top:] = small
        cv2.putText(tile, f'{n/fps:.2f}s f{n}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        tiles.append(tile)
    height = max(x.shape[0] for x in tiles)
    tiles = [np.vstack([x, np.zeros((height-x.shape[0], t, 3), np.uint8)]) for x in tiles]
    tiles += [np.zeros((height, t, 3), np.uint8)]*(-len(tiles) % columns)
    rows = [np.hstack(tiles[i:i+columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows))
    return path.name


def _write(cap, fps, groups, root, prefix):
    """One sheet per (name, frame numbers) group as <prefix>-N.jpg; [{span, image, frames, pixels}]."""
    import cv2
    written = []
    for i, (name, numbers) in enumerate(groups, 1):
        frames = []
        for n in numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, image = cap.read()
            if ok:
                frames.append((n, image))
        if frames:
            path = root/f'{prefix}-{i}.jpg'
            sheet(frames, fps, path)
            h, w = cv2.imread(str(path)).shape[:2]
            written.append(dict(span=name, image=path.name, frames=[n for n, _ in frames], pixels=w*h))
    return written


def overview_count(count, fps):
    return max(1, min(ZOOM['max_overview'], round(count/fps/ZOOM['every_s'])))


def overview(video, root):
    """Whole frames spread evenly over the whole clip, about one per ZOOM['every_s'] seconds up to ZOOM['max_overview']
    (longer clips get wider gaps), 12 per sheet, as
    frame-sheet-N.jpg: what people do to each other and falls, which one merged pose box cannot show."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not count:
            return []
        numbers = picks(0, count-1, overview_count(count, fps))
        k = ZOOM['frames']
        return [s['image'] for s in _write(cap, fps, [('all', numbers[i:i+k]) for i in range(0, len(numbers), k)], root, 'frame-sheet')]
    finally:
        cap.release()


def run(video, out, at=(), even=False):
    """Writes zoom-N.jpg (one per span, or two halves for --even) and zoom.json with the frames and their cost."""
    import cv2
    from image_lens import paths
    if bool(at) == even:
        raise ValueError('give --at spans or --even, not both')
    root = paths.confined(out, new=True)
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    cap = cv2.VideoCapture(str(source))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not count:
            raise ValueError('video has no frame rate or frame count')
        if even:
            numbers = picks(0, count-1, ZOOM['budget']); half = len(numbers)//2
            groups = [('even', numbers[:half]), ('even', numbers[half:])]
        else:
            groups = []
            for a, b in spans(at):
                first, last = round(a*fps), min(round(b*fps), count-1)
                if first > count-1:
                    raise ValueError(f'span {a:g}-{b:g}s starts after the video ends ({count/fps:.2f} s)')
                # A span running past the end is cut there; the name says the frames actually shown.
                groups.append((f'{a:g}-{last/fps:.2f}s' if last < round(b*fps) else f'{a:g}-{b:g}s', picks(first, last, ZOOM['frames'])))
        root.mkdir(parents=True)
        written = _write(cap, fps, groups, root, 'zoom')
    finally:
        cap.release()
    result = dict(status='ok', video=source.name, fps=fps, mode='even' if even else 'spans', sheets=written,
                  cost=dict(images=len(written), frames=sum(len(s['frames']) for s in written),
                            pixels=sum(s['pixels'] for s in written)))
    (root/'zoom.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result
