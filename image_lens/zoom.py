"""Second read. After reading a look, the reader names at most two short spans it could not tell; zoom shows each as
dense whole frames in time order (who pulls whom, a fall: Kaguya 18-22 s was one merged pose box, so pose lines and
person crops cannot show it). The frame budget is fixed and counted, so a gain can be checked against the same number
of frames spread evenly over the clip (--even)."""
import json

ZOOM = dict(spans=2, span_s=4., frames=12, budget=24, columns=4, tile=480, label=22)


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
            groups = [(f'{a:g}-{b:g}s', picks(min(round(a*fps), count-1), min(round(b*fps), count-1), ZOOM['frames']))
                      for a, b in spans(at)]
        root.mkdir(parents=True)
        written = []
        for i, (name, numbers) in enumerate(groups, 1):
            frames = []
            for n in numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                ok, image = cap.read()
                if ok:
                    frames.append((n, image))
            if frames:
                path = root/f'zoom-{i}.jpg'
                sheet(frames, fps, path)
                h, w = cv2.imread(str(path)).shape[:2]
                written.append(dict(span=name, image=path.name, frames=[n for n, _ in frames], pixels=w*h))
    finally:
        cap.release()
    result = dict(status='ok', video=source.name, fps=fps, mode='even' if even else 'spans', sheets=written,
                  cost=dict(images=len(written), frames=sum(len(s['frames']) for s in written),
                            pixels=sum(s['pixels'] for s in written)))
    (root/'zoom.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result
