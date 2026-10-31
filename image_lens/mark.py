"""Agent-placed points. Where pose seeds fail (props, odd poses, unnamed points), the agent reads coordinates off a grid
image, names the points itself and tracks them with the same engine. Names are the agent's reading, not verified anatomy;
the marked crops are the evidence."""
import json
import math
import re
from image_lens.frames import label as outlined

MARK = dict(grid_px=100, zoom_grid_px=50, snap_radius_px=15, max_points=8, uncertainty_px=10.)


def _frame(video, frame):
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError('frame decode failed')
    return bgr


def grid(video, frame, out, box=None):
    """Frame (or a box of it) with a labelled coordinate grid in source pixels, for reading point positions by eye."""
    import cv2
    from image_lens import paths
    bgr = _frame(paths.confined(video, exists=True, max_bytes=512*1024*1024), frame)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box or (0, 0, w, h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = bgr[y0:y1, x0:x1].copy(); step = MARK['zoom_grid_px'] if box else MARK['grid_px']
    scale = max(1, round(900/max(x1-x0, y1-y0))) if box else 1
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for x in range((x0//step+1)*step, x1, step):
        cv2.line(crop, ((x-x0)*scale, 0), ((x-x0)*scale, crop.shape[0]), (0, 200, 255), 1)
        outlined(cv2, crop, str(x), ((x-x0)*scale+2, 18), .5, 1)
    for y in range((y0//step+1)*step, y1, step):
        cv2.line(crop, (0, (y-y0)*scale), (crop.shape[1], (y-y0)*scale), (0, 200, 255), 1)
        outlined(cv2, crop, str(y), (2, (y-y0)*scale-3), .5, 1)
    target = paths.confined(out, new=True)
    cv2.imwrite(str(target), crop)
    return dict(status='written', output=str(target), frame=frame, box=[x0, y0, x1, y1], grid_px=step, zoom=scale)


def parse_point(text):
    match = re.fullmatch(r'([A-Za-z0-9_-]{1,64})=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)', text)
    if not match:
        raise ValueError(f'point must be name=x,y with an ASCII name: {text!r}')
    return match.group(1), (float(match.group(2)), float(match.group(3)))


def snap(gray, xy, radius=MARK['snap_radius_px']):
    """The nearest trackable corner within `radius` of the eyeballed position, or the position itself."""
    import cv2
    import numpy as np
    mask = np.zeros_like(gray); cv2.circle(mask, (int(round(xy[0])), int(round(xy[1]))), radius, 255, -1)
    found = cv2.goodFeaturesToTrack(gray, 20, .05, 3, mask=mask)
    if found is None:
        return xy, 0.
    x, y = min(((float(a), float(b)) for a, b in found.reshape(-1, 2)), key=lambda c: math.dist(xy, c))
    return (round(x, 1), round(y, 1)), round(math.dist(xy, (x, y)), 1)


def run(video, shots, frame, points, checkpoint, out, end=None, snap_points=True):
    import cv2
    from image_lens import camera, look, motion_report, paths, pose
    root = paths.confined(out, new=True)
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    scan, scan_path = paths.load(paths.confined(shots, exists=True))
    if paths.sha(source) != scan['source_sha256']:
        raise ValueError('shot scan belongs to another video')
    parsed = [parse_point(p) for p in points]
    if not 1 <= len(parsed) <= MARK['max_points'] or len({n for n, _ in parsed}) != len(parsed):
        raise ValueError(f"1..{MARK['max_points']} uniquely named points required")
    segment = next((s for s in scan['analysis_segments'] if s['start_frame'] <= frame <= s['end_frame']), None)
    if segment is None:
        raise ValueError('frame outside the scanned shots')
    from image_lens.run_point_tracker import PARAMS
    stop = min(segment['end_frame'], frame+PARAMS['max_frames']-1) if end is None else end
    if not frame < stop <= segment['end_frame'] or stop-frame+1 > PARAMS['max_frames']:
        raise ValueError(f"end must stay in the shot (f{segment['start_frame']}..{segment['end_frame']}) and within {PARAMS['max_frames']} frames")
    bgr = _frame(source, frame); h, w = bgr.shape[:2]; gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    seeds = {}; placed = []
    for name, xy in parsed:
        if not (0 <= xy[0] < w and 0 <= xy[1] < h):
            raise ValueError(f'{name} is outside the frame')
        at, moved = snap(gray, xy) if snap_points else (xy, 0.)
        seeds[name] = dict(xy=[at[0], at[1]]); placed.append(dict(point_id=name, given=list(xy), seed=list(at), snapped_px=moved))
    root.mkdir(parents=True)
    look.LOG = root/'steps.log'
    crops = pose._marked_crops(source, frame, w, h, seeds, root/'crops')
    job = dict(name='mark', group='marked', source_video=str(source), source_sha256=scan['source_sha256'], width=w, height=h,
               fps=scan['fps'], start_frame=frame, end_frame=stop, seed_frame=frame, mode='single-anchor',
               observation_scope='visible_surface_2d', anatomical_claims=False, common_metric_excluded_frames=[frame],
               points=[dict(point_id=name, observed_label=f'agent-marked: {name}', anatomical_interpretation=None,
                            seed_rule='agent read from grid image' + (', snapped to nearest corner' if snap_points else ''),
                            visual_reference=dict(feature_id=name, frame=frame, kind='marked-source-crop', **crops[name]),
                            anchors=[dict(frame=frame, xy=seeds[name]['xy'], status='source_image_observation',
                                          uncertainty_px=MARK['uncertainty_px'])]) for name, _ in parsed])
    queries = paths.write_json(root/'mark.obs.json', dict(schema='team-eye.observations/1', jobs=[job],
                               seed_source=dict(basis='agent-placed points; names are the agent reading, not verified anatomy')))
    look._step('prepare', ['--queries', queries, '--shots', scan_path, '--job', 'mark', '--out', root/'mark.freeze.json',
                           '--allow-tracking', '--max-seconds', look.LOOK['track_seconds']])
    look._step('track', ['--job', 'mark', '--queries', queries, '--frozen', root/'mark.freeze.json', '--out', root/'mark.tracks.json',
                         '--checkpoint', checkpoint, '--max-seconds', look.LOOK['track_seconds']])
    tracks = root/'mark.tracks.json'
    bindings = paths.write_json(root/'mark.bindings.json', dict(
        schema='team-eye.motion-bindings/1', source_tracks_sha256=paths.sha(tracks), basis='source_frame_grouping_only',
        actors=[dict(actor_id='marked', label='찍은 점', features=[dict(point_id=n, label=n) for n, _ in parsed])]))
    look._step('analyze', ['--tracks', tracks, '--shots', scan_path, '--report-bindings', bindings, '--start', frame, '--end', stop,
                           '--format', 'json', '--out', root/'mark.report.json', '--trajectory-dir', root/'mark-traj'])
    report, _ = paths.load(root/'mark.report.json')
    paths.write(root/'mark.report.md', motion_report.markdown(report))
    frame_motion = camera.path(source, frame, stop)
    cam = camera.between(frame_motion, frame, stop)
    text = [f'# 찍은 점 추적 f{frame}..{stop}', '',
            '이름은 찍은 에이전트가 붙인 것이다(부위 미검증). 근거 그림: crops/<이름>.png, 프레임별 좌표: mark.tracks.json.', '']
    text += [f"- {p['point_id']}: 준 위치 {p['given'][0]:.0f},{p['given'][1]:.0f} → 놓은 위치 {p['seed'][0]:.0f},{p['seed'][1]:.0f}"
             + (f" ({p['snapped_px']:.0f}px 옮김)" if p['snapped_px'] else '') for p in placed]
    if cam and math.hypot(*cam) >= motion_report.STROKE['major_min_px']:
        text.append(f"- 화면 전체: {motion_report.KO[motion_report.direction(*cam)]} {math.hypot(*cam):.0f}px (카메라·배경 이동 추정; 이것만 따라간 이동은 뺌)")
    lines, summary = look._summary(report, dict(seed=frame, end=stop), 'marked', 'mark.report.md', 'mark-traj/trajectory.png', frame_motion)
    text += lines or ['- 찍은 점: 큰 이동 없음']
    text += ['', '## 점별 전체 움직임 (멈춤 제외, 완급 포함)'] + _all_moves(report)
    output = paths.write(root/'mark.md', '\n'.join(text)+'\n')
    paths.write_json(root/'mark.json', dict(schema='image-lens.mark/1', source_video=str(source), frames=[frame, stop], points=placed,
                                            camera=cam, summary=summary))
    return dict(status='written', output=str(output), points=len(placed), frames=[frame, stop])


def _all_moves(report):
    from image_lens import motion_report
    out = []
    for f in report['features']:
        units = [u for run in f['continuous_runs'] for u in run['strokes']['units'] if u['direction'] != 'hold']
        if not f['continuous_runs']:
            out.append(f"- {f['point_id']}: 쓸 수 있는 연속 관측 없음(가려짐·추적 끊김)")
            continue
        moves = [f"{motion_report.KO[u['direction']].replace('화면 ', '')} {u['distance_px']:.0f}px(f{u['start_frame']}..{u['end_frame']}"
                 + (f", {motion_report.EASE_KO[u['ease']]}" if u.get('ease') else '') + ')' for u in units]
        out.append(f"- {f['point_id']}: " + (' → '.join(moves) if moves else '거의 그대로'))
    return out
