"""Re-run one look window with another person detector. A whole look with SAM3 runs about 23 s per second of video and
lost track frames elsewhere on the Miku clip, so the agent re-runs only the window whose green box is wrong: the first
look stays as it was, and the new folder is a look of that window's frames only."""
from image_lens import paths

REDO = dict(picture='redo-boxes.jpg', frames=3, width=480, was=(0, 0, 220), now=(60, 200, 60))


def window(record, name):
    """(first, last) native frames of look window `name` ('<shot id>-w<i>', as its file names spell it)."""
    sid, _, i = name.rpartition('-w')
    shot = next((s for s in record['shots'] if s['segment_id'] == sid), None)
    if shot is None or not i.isdigit() or int(i) >= len(shot['windows']):
        raise ValueError(f'no window {name} in that look; names are <shot id>-w<i> as in its file names')
    done = shot['windows'][int(i)]; w = done['window']
    return w['seed'], min(done.get('cut_after', w['end']), w['end'])  # a cut ends the window early, as look reported it


def kind(pose):
    """The detector a pose packet used; packets from before selectable detectors were HumanArt."""
    return pose['models']['detector'].get('kind', 'humanart')


def picks(first, last, count=REDO['frames']):
    return sorted({first+round(k*(last-first)/max(1, count-1)) for k in range(count)})


def boxes(video, before, after, numbers, out):
    """One row of frames, each with the largest person box of the first look (red) and of the redo (green)."""
    import cv2
    import numpy as np
    from image_lens import body_motion, mark
    from image_lens.frames import label
    def box(packet, n):
        tracks = next((f['tracks'] for f in packet['frames'] if f['frame'] == n), None)
        person = body_motion.largest(tracks)
        return person and person['bbox']
    tiles = []
    for n in numbers:
        image = mark._frame(video, n)
        for bbox, colour in ((box(before, n), REDO['was']), (box(after, n), REDO['now'])):
            if bbox:
                x0, y0, x1, y1 = (round(v) for v in bbox)
                cv2.rectangle(image, (x0, y0), (x1, y1), colour, max(2, image.shape[1]//320))
        k = REDO['width']/image.shape[1]
        tile = cv2.resize(image, (REDO['width'], round(image.shape[0]*k)), interpolation=cv2.INTER_AREA)
        label(cv2, tile, f'f{n}', (8, 28), .7, 1)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    row = np.hstack([np.pad(t, ((0, height-t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in tiles])
    if not cv2.imwrite(str(paths.confined(out, new=True)), row):
        raise ValueError('picture write failed')


def run(look_dir, name, detector, pose_model, checkpoint, out, max_seconds=1800):
    from pathlib import Path
    from image_lens import look, person_models
    root = Path(look_dir)
    record, _ = paths.load(paths.confined(root/'look.json', exists=True), max_bytes=64*1024*1024)
    first, last = window(record, name)
    sid, _, index = name.rpartition('-w')
    done = next(shot for shot in record['shots'] if shot['segment_id'] == sid)['windows'][int(index)]
    if record.get('max_people', 1) > 1 or done.get('source_track_id') is not None:
        raise ValueError('redo cannot preserve the selected actor across detectors; multi-person/source_track_id windows require a new explicit actor selection')
    before, _ = paths.load(paths.confined(root/'pose.json', exists=True), max_bytes=64*1024*1024)
    video = paths.confined(record['source_video'], exists=True, max_bytes=512*1024*1024)
    if not paths.sha(video) == record['source_sha256'] == before['source_sha256']:
        raise ValueError('look run, its pose.json and the video on disk differ; redo needs the video that look saw')
    now = person_models.resolve(detector)['kind']
    result = look.run(video, detector, pose_model, checkpoint, out, max_seconds=max_seconds, frames=(first, last),
                      redo=dict(look=root.name, window=name, detector=now, was=kind(before), boxes=REDO['picture']))
    after, _ = paths.load(paths.confined(Path(out)/'pose.json', exists=True), max_bytes=64*1024*1024)
    boxes(video, before, after, picks(first, last), Path(out)/REDO['picture'])
    return dict(result, window=name, frames=[first, last], detector=now, was=kind(before), boxes=str(Path(out)/REDO['picture']))
