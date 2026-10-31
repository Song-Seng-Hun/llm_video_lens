"""One-command look at a video: scan, pose, pose-seeded tracking per shot window, grouped reports and one short look.md.
Runs the existing public steps in order with their own guards; a failed shot is recorded and the next one continues."""
import json
import math
import os
import signal
import time
from pathlib import Path
from image_lens import paths, joint_tracking

LOOK = dict(max_shots=16, window_frames=160, min_frames=12, seed_shift=3, track_seconds=300, strips=6, groups=('head', 'arms', 'hands', 'legs', 'move'))
NAMES = dict(head='머리', arms='팔', hands='손끝', legs='다리', move='많이 움직인 곳', marked='찍은 점')
STEP_ERRORS = (ValueError, OSError, KeyError, TypeError, ImportError, TimeoutError, OverflowError)


LOG = None


def _step(command, arguments):
    import contextlib
    from image_lens import cli
    try:
        # Step receipts and child output go to steps.log; stdout keeps only the final receipt.
        with LOG.open('a') as log, contextlib.redirect_stdout(log):
            print(json.dumps(dict(step=command, arguments=[str(a) for a in arguments])), flush=True)
            return cli.dispatch(command, [str(a) for a in arguments])
    finally:
        # Each step may arm its own SIGALRM budget; never let it leak into the next one.
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def _submit(arguments):
    """Log and queue one track step on the shared child; _finish(handle) logs its output and raises as `track` does."""
    import contextlib
    from image_lens import cli
    try:
        with LOG.open('a') as log, contextlib.redirect_stdout(log):
            print(json.dumps(dict(step='track', arguments=[str(a) for a in arguments])), flush=True)
            return cli.submit_track([str(a) for a in arguments])
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def _finish(handle):
    import contextlib
    from image_lens import cli, watchdog
    with LOG.open('a') as log, contextlib.redirect_stdout(log):
        return cli.finish_track(watchdog.shared.result(handle))


def _pose_child(arguments):
    """Start the pose step as its own process: it announces each part packet on stdout while it poses the rest."""
    import subprocess
    import sys
    with LOG.open('a') as log:
        print(json.dumps(dict(step='pose', arguments=[str(a) for a in arguments])), file=log, flush=True)
    return subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).resolve().parents[1]/'tools'/'image_lens_cli.py'),
                             '--workspace', str(paths.workspace_root()), 'pose', *[str(a) for a in arguments]],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


POSER = None


def _stop_pose():
    global POSER
    poser, POSER = POSER, None
    if poser is not None and poser.poll() is None:
        try:
            os.killpg(poser.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if poser is not None:
        poser.communicate()


def next_window(segment, start, pose_frames, flagged, length=LOOK['window_frames'], min_frames=LOOK['min_frames']):
    """The next tracking window at or after `start`, seeded on the first frame with a pose track that is not pose-flagged."""
    skipped = []; end = segment['end_frame']
    while start <= end:
        seed = next((n for n in range(start, end+1) if pose_frames.get(n) and n not in flagged), None)
        if seed is None:
            skipped.append(dict(frames=[start, end], reason='no unflagged pose track'))
            break
        # One bad pose frame at the seed names no limbs for the whole window (3D clip f368: a sleeve read as a 3.16
        # shoulder-width arm, f369 fine): seed on the next frame that passes when one is this close.
        from image_lens import body_motion
        if not body_motion.skeleton_check(body_motion.largest(pose_frames[seed]))[0]:
            seed = next((n for n in range(seed+1, min(end, seed+LOOK['seed_shift'])+1) if pose_frames.get(n) and n not in flagged
                         and body_motion.skeleton_check(body_motion.largest(pose_frames[n]))[0]), seed)
        stop = min(end, seed+length-1)
        if stop-seed+1 >= min_frames:
            return dict(seed=seed, end=stop), skipped
        skipped.append(dict(frames=[seed, stop], reason=f'fewer than {min_frames} frames after the seed'))
        start = stop+1
    return None, skipped


def windows(segment, pose_frames, flagged, length=LOOK['window_frames'], min_frames=LOOK['min_frames']):
    """Consecutive tracking windows inside one shot when nothing cuts them short."""
    result = []; skipped = []; start = segment['start_frame']
    while True:
        window, more = next_window(segment, start, pose_frames, flagged, length, min_frames)
        skipped += more
        if window is None:
            return result, skipped
        result.append(window); start = window['end']+1


def cut_frame(track_packets, window, min_frames=LOOK['min_frames']):
    """Last frame before the drawing likely changed (costume, wipe): some group lost at least half its points
    more than min_frames before the window end. None when no group did."""
    cuts = []
    for packet in track_packets:
        last = {}
        for frame in packet['frames']:
            for point in frame['points']:
                last.setdefault(point['point_id'], None)
                if point['visible'] and frame['frame'] > window['seed']:
                    last[point['point_id']] = frame['frame']
        gone = [e for e in last.values() if e is None or e < window['end']-min_frames]
        if len(gone) >= max(2, (len(last)+1)//2):
            seen = [e for e in gone if e is not None]
            cuts.append(max(seen) if seen else window['seed'])
    return min(cuts) if cuts else None


def changed_after(video, frame):
    """Largest downscaled grey difference over the two frame steps after `frame`, on motion_view's change scale. Points
    lost together over a still picture were hidden or moved too fast (3D clip, clip B: <=7); a new drawing jumps (anime
    clip on twos, f73: 22)."""
    import cv2
    import numpy as np
    from image_lens import frames, motion_view
    grey = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), motion_view.CHANGE['thumb'], interpolation=cv2.INTER_AREA).astype(np.int16)
            for f in frames.window(video, frame, frame+2)]
    return max((float(np.mean(np.abs(b-a))) for a, b in zip(grey, grey[1:])), default=0.)


def lost(report, window, min_frames=LOOK['min_frames']):
    """Points whose last usable run ends well before the window does. Several at once usually means the drawing changed (costume, wipe) or left the frame."""
    if not window:
        return None
    ends = [max((r['frames'][1] for r in f['continuous_runs']), default=None) for f in report['features']]
    gone = [e for e in ends if e is None or e < window['end']-min_frames]
    if len(gone) < max(2, (len(ends)+1)//2):
        return None
    seen = [e for e in gone if e is not None]
    return dict(count=len(gone), total=len(ends), after=max(seen) if seen else window['seed'])


def _summary(report, window, group, md_path, traj, frame_motion=None):
    """Short per-group lines for look.md, built from the recomputed report JSON. Major strokes that only follow the
    whole frame (camera pan, scrolling UI) are set aside and counted."""
    import math
    from image_lens import camera, motion_report
    moves = []; facts = []; with_camera = 0; kept = set()
    for f in report['features']:
        for run in f['continuous_runs']:
            for u in run['strokes']['units']:
                if not u['major']:
                    continue
                cam = camera.between(frame_motion, u['start_frame'], u['end_frame'])
                # A camera shift below the stroke threshold is noise (a big figure's own matches leak into it); keep the move.
                if cam and math.hypot(*cam) >= motion_report.STROKE['major_min_px'] \
                        and math.hypot(u['dx_px']-cam[0], u['dy_px']-cam[1]) < motion_report.STROKE['major_min_px']:
                    with_camera += 1
                    continue
                timing = f", {motion_report.speed_text(u['speeds'])}" if u.get('speeds') else ''
                moves.append(f"{f['point_id']} {motion_report.KO[u['direction']]} {u['distance_px']:.0f}px(f{u['start_frame']}..{u['end_frame']}{timing})")
                facts.append(dict(point=f['point_id'], direction=u['direction'], distance_px=u['distance_px'],
                                  frames=[u['start_frame'], u['end_frame']], ease=u.get('ease'), speeds=u.get('speeds')))
                kept.add(f['point_id'])
    together = [g for g in motion_report.together(report) if 2*len(kept & set(g['points'])) >= len(g['points'])]
    lines = [f"- {NAMES[group]}: " + ('; '.join(moves) if moves else '큰 이동 없음')]
    lines += [f"  - 함께 움직임 {motion_report.quote(g['points'])}: {motion_report.KO[g['direction']]} 약 {g['distance_px']:.0f}px, f{g['frames'][0]}..{g['frames'][1]}"
              for g in together]
    view = report.get('trajectory_view')
    warnings = [l for l in motion_report.trajectory_lines(report, view) if l.startswith('- 주의')] if view else []
    if warnings:
        lines.append(f"  - 경고 {len(warnings)}건: 큰 이동이 그림 변경 후보나 포즈 겹침 구간과 겹친다. 원본 비교 먼저.")
    cut = lost(report, window)
    if cut:
        lines.append(f"  - f{cut['after']} 뒤로 점 {cut['count']}/{cut['total']}개 끊김(그림 바뀜·가려짐·화면 밖). 그 뒤는 정지가 아니라 모름.")
    if not moves and not any(f['continuous_runs'] for f in report['features']):
        lines = [f"- {NAMES[group]}: 쓸 수 있는 연속 관측 없음(가려짐·화면 밖·추적 끊김). 정지가 아니라 모름."]
    return (lines if moves or cut or len(lines) == 1 and '모름' in lines[0] else []), dict(group=group, major_moves=moves, moves=facts, together=together, warnings=len(warnings),
                       lost=cut, camera_only_moves=with_camera, report=str(md_path), trajectory=str(traj))


def fit(source, root, max_pixels=1920*1080):
    """(video, scaled) where video fits the point tracker's pixel cap. A larger source is shrunk once into the output
    directory (aspect kept, even sides); px in every report are then of that copy. scaled is None or {from, to}."""
    import cv2
    cap = cv2.VideoCapture(str(source))
    try:
        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        if w*h <= max_pixels:
            return source, None
        k = math.sqrt(max_pixels/(w*h)); size = (int(w*k)//2*2, int(h*k)//2*2)
        copy = root/f'source-{size[0]}x{size[1]}.mp4'
        writer = cv2.VideoWriter(str(copy), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        finally:
            writer.release()
    finally:
        cap.release()
    return copy, {'from': [w, h], 'to': list(size)}


def run(*args, **kwargs):
    from image_lens import watchdog
    # One tracker child serves every window (clip B: 137 jobs each paid about 0.85 s to start Torch and load the model).
    watchdog.shared = watchdog.Worker()
    try:
        return _run(*args, **kwargs)
    finally:
        watchdog.shared.close(); watchdog.shared = None
        _stop_pose()


def _run(video, detector, pose_model, checkpoint, out, max_shots=LOOK['max_shots'], max_seconds=1800):
    started = time.monotonic()
    def remaining():
        return max_seconds-(time.monotonic()-started)
    root = paths.confined(out, new=True)
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    root.mkdir(parents=True)
    global LOG
    LOG = root/'steps.log'
    source, scaled = fit(source, root)
    _step('scan', ['--video', source, '--out', root/'shots.json'])
    # Pose runs in its own process and hands over runs of whole shots as it goes, so the first shots are seeded and
    # tracked while it poses the rest (clip B side by side: pose 28 -> 38 s, tracking 35 -> 41 s, 63 -> 41 s in all).
    global POSER
    POSER = _pose_child(['--shots', root/'shots.json', '--detector', detector, '--pose-model', pose_model, '--out', root/'pose.json',
                         '--max-seconds', max(1, min(1800, int(remaining()))), '--parts-dir', root/'pose-parts'])
    # A thread reads the child's lines, so look can take a part that has arrived without waiting for the next one: the
    # next shot is seeded ahead only once its part is taken (clip B: every third shot missed it, the child idled 0.6-1 s).
    # Named apart from the local queue() below: in this scope that def would shadow the module (only hit when the
    # pose child falls behind tracking, as with the 384x288 pose model on clip B).
    import queue as queue_mod, threading
    heard = queue_mod.Queue()
    def listen(stream):
        try:
            for line in stream:
                heard.put(line)
        except (OSError, ValueError):
            pass  # _stop_pose closed the pipe
        heard.put('')
    threading.Thread(target=listen, args=(POSER.stdout,), daemon=True).start()
    scan, _ = paths.load(root/'shots.json')
    pose_frames = {}; flagged = set(); posed = []; failed = {}  # posed: (first, last, packet path)
    def take(path):
        packet, _ = paths.load(path, max_bytes=64*1024*1024)
        pose_frames.update({f['frame']: f['tracks'] for f in packet['frames']})
        flagged.update(r['frame'] for r in packet['flags']['frames'])
        posed.append((*packet['native_range'], path))
    def covered(frame):
        return next((p for lo, hi, p in posed if lo <= frame <= hi), None)
    def pose_for(frame, wait=True):
        """The pose packet holding `frame`: a part, or pose.json when the clip fits in one; waits for the child's line,
        or with wait=False takes only the lines already heard and returns None if none holds it."""
        while covered(frame) is None:
            try:
                line = heard.get(block=wait)
            except queue_mod.Empty:
                return None
            if not line:
                heard.put('')  # the end stays heard for later calls
                if POSER.wait() or failed or not (root/'pose.json').is_file():
                    raise ValueError(failed.get('error', 'pose step failed'))
                if posed:
                    raise ValueError('pose parts do not cover the clip')
                take(root/'pose.json')
                continue
            with LOG.open('a') as log:
                log.write(line)
            try:
                said = json.loads(line)
            except ValueError:
                continue  # model loaders print plain lines
            if type(said) is dict and 'pose_part' in said:
                take(Path(said['pose_part']))
            elif type(said) is dict and said.get('status') == 'failed':
                failed.update(said)
        return covered(frame)
    shots = []; lines = []; last_end = None; looks = []; stripping = []
    pending = None
    def report(window):
        """Everything after tracking for one window; look runs it while the child tracks the next window. Its lines and
        errors go where the window began, so look-full.md and look.json read as if it had run in order."""
        nonlocal last_end
        entry, sid, w, end, base, people, receipt, tracked, done, slot, at = window
        errors = []; quiet = []; fps = scan['fps']
        from image_lens import body_motion
        if last_end is not None:
            done['bridge'] = body_motion.bridge(body_motion.largest(pose_frames.get(last_end)), body_motion.largest(pose_frames.get(w['seed'])),
                                                (last_end, w['seed']))
            slot += body_motion.bridge_lines(done['bridge'])
        slot.append(f"- 구간 f{w['seed']}..{end} (점 배치 f{w['seed']}" + (f", {people}명 중 가장 큰 사람" if people > 1 else '') + ')')
        skeleton = receipt.get('skeleton') or dict(plausible=True); done['skeleton'] = skeleton['plausible']
        done['framing'] = skeleton.get('framing'); done['scope'] = skeleton.get('scope')
        done['chibi'] = skeleton.get('chibi')
        if done['chibi']:
            slot.append(f"  - SD 체형(머리가 큼, 눈 간격이 눈~발목의 {done['chibi']:.2f}): 팔·다리 이름 없이 얼굴과 많이 움직인 곳(move-a…)만 봄")
        elif not skeleton['plausible']:
            slot.append(f"  - 골격 신뢰 낮음({skeleton['reason']}): 팔·다리 이름 없이 얼굴과 많이 움직인 곳(move-a…)만 봄")
        from image_lens import camera, motion_report
        try:
            main = body_motion.largest(pose_frames.get(w['seed']))
            frame_motion = camera.path(source, w['seed'], end, main['bbox'] if main else None)
        except Exception as error:  # OpenCV raises its own error type; a missing estimate only disables the filter
            frame_motion = None; errors.append(dict(window=w, step='camera', error=str(error)[:300]))
        cam = camera.between(frame_motion, w['seed'], end)
        done['camera'] = camera.describe(frame_motion, w['seed'], end, motion_report.STROKE['major_min_px'])
        from image_lens import digest
        camera_text = digest.camera(done['camera'], fps)
        if camera_text:
            slot.append(f"  - 화면 전체: {camera_text} (카메라·배경 이동 추정; 카메라만 따라간 이동은 아래에서 뺌)")
        # Move candidates: report one per group that moved together (see onion.parts); the rest stay in the tracks file.
        from image_lens import onion
        for k, (name, group, stem, tracks) in enumerate(tracked):
            if group == 'move':
                keep = set(onion.parts(tracks, end, frame_motion))
                done['move_candidates'] = len(tracks['frames'][0]['points']) if tracks['frames'] else 0
                tracked[k] = (name, group, stem, dict(tracks, frames=[dict(f, points=[p for p in f['points'] if p['point_id'] in keep])
                                                                      for f in tracks['frames']]))
        # Preserve filtered non-body candidates in the raw geometry view.
        raw_packets = [paths.load(joint_tracking.track_path(t[2], t[1]))[0] if t[1] in joint_tracking.GROUPS else t[3] for t in tracked]
        for name, group, stem, tracks in tracked:
            raw_tracks = joint_tracking.track_path(stem, group)
            if group == 'move' and not (tracks['frames'] and tracks['frames'][0]['points']):
                quiet.append(NAMES[group]); continue
            try:
                _step('bind', ['--queries', base.with_suffix('.obs.json'), '--tracks', raw_tracks,
                               '--job', name, '--out', stem.with_suffix('.bindings.json')])
                binding = stem.with_suffix('.bindings.json')
                if group == 'move':
                    keep = {p['point_id'] for p in tracks['frames'][0]['points']}
                    bindings, _ = paths.load(binding)
                    for actor in bindings['actors']:
                        actor['features'] = [f for f in actor['features'] if f['point_id'] in keep]
                    binding = paths.write_json(stem.with_name(stem.name+'-parts.bindings.json'), bindings)
                _step('analyze', ['--tracks', raw_tracks, '--shots', root/'shots.json',
                                  '--report-bindings', binding, '--start', w['seed'], '--end', end,
                                  '--format', 'json', '--out', stem.with_suffix('.report.json'),
                                  '--trajectory-dir', stem.with_name(stem.name+'-traj'), '--pose', pose_for(w['seed'])])
                found, _ = paths.load(stem.with_suffix('.report.json'))
                from image_lens import motion_report
                md = paths.write(stem.with_suffix('.report.md'), motion_report.markdown(found))
                group_lines, summary = _summary(found, dict(w, end=end), group, md.relative_to(root), stem.with_name(stem.name+'-traj').relative_to(root)/'trajectory.png', frame_motion)
                slot += ['  '+l for l in group_lines]; done['groups'].append(summary)
                if not group_lines:quiet.append(NAMES[group])
            except STEP_ERRORS as error:
                errors.append(dict(window=w, job=name, error=str(error)[:300]))
                slot.append(f'  - {group} 실패: {str(error)[:120]}')
        if quiet:slot.append(f"  - {'·'.join(quiet)}: 화면 기준 큰 이동({motion_report.STROKE['major_min_px']:.0f}px 이상) 없음")
        if tracked and skeleton['plausible']:
            done['body_geometry'] = body_motion.geometry_policy([t[3] for t in tracked], dict(w, end=end), raw_packets)
            note = body_motion.geometry_note(done['body_geometry'])
            if note:
                slot.append('  - '+note)
            done['body'] = body_motion.describe([t[3] for t in tracked], dict(w, end=end), pose_frames,
                                               timing_packets=raw_packets)
            slot += ['  '+l for l in body_motion.lines(done['body'])]
        done['events'] = body_motion.pose_events(pose_frames, dict(w, end=end), fps)
        done['phases'] = body_motion.phases(pose_frames, dict(w, end=end), fps)
        done['facing'] = body_motion.facing(pose_frames, dict(w, end=end))
        done['people'] = body_motion.people(pose_frames, dict(w, end=end), fps, cam)
        if len(done['people']['people']) > 1:
            from image_lens import onion
            try:
                done['who'] = onion.who(source, w['seed'], done['people']['people'], base.with_name(base.name+'-who.jpg'))
            except Exception as error:  # OpenCV raises its own error type; the picture is optional
                errors.append(dict(window=w, step='who', error=str(error)[:300]))
        from image_lens import digest
        slot += ['  - '+f for f in digest.events(done['events'], fps)]
        person = body_motion.largest(pose_frames.get(w['seed']))
        if tracked and person:
            from image_lens import onion
            picture = base.with_name(base.name+'-onion.jpg')
            try:
                moving = cam if cam and math.hypot(*cam) >= motion_report.STROKE['major_min_px'] else None
                from image_lens import digest
                leads = [f['point'] for f in digest.limbs(done['body'])] if done.get('body') else []
                done['trails'] = onion.trails(raw_packets, leads, w['seed'], end, frame_motion if moving else None)
                done['bands'] = onion.bands(source, w['seed'], end, person['bbox'], moving, picture, done['trails'])
                done['onion'] = picture.name
                done['peak'] = onion.peak(source, w['seed'], end, person['bbox'], picture)
                turns = [e['share'] for e in (done.get('events') or {}).get('width', []) if e['share'] >= body_motion.EVENTS['turn']]
                if turns:
                    done['turn_share'] = max(turns); stripping.append((done, end, person['bbox'], base))
                slot += ['  '+l for l in onion.lines(done['bands'], picture.name)]
            except Exception as error:  # OpenCV raises its own error type; the view is optional
                errors.append(dict(window=w, step='onion', error=str(error)[:300]))
        if end < w['end']:
            from image_lens import motion_view
            try:
                done['cut_redrawn'] = changed_after(source, end) >= motion_view.CHANGE['min_mean_abs_diff']
                why = '그림 바뀜·화면 전환 추정' if done['cut_redrawn'] else '가려짐·빠른 동작 추정, 그림은 그대로'
            except Exception as error:  # OpenCV raises its own error type; without the frames the cause stays open
                errors.append(dict(window=w, step='cut_diff', error=str(error)[:300])); why = '그림 바뀜·가려짐·화면 밖'
            slot.append(f"  - f{end} 뒤 점이 한꺼번에 끊김({why}): f{end+1}부터 점을 다시 놓음")
        if done['groups']:slot.append(f"  - 자세히: {base.name}-<head|arms|legs>.report.md, 그림 {base.name}-<묶음>-traj/trajectory.png")
        last_end = end if done['groups'] else None
        if person:
            from image_lens import motif
            try:
                looks.append((done, dict(start=motif.thumb(source, w['seed'], person['bbox']), end=motif.thumb(source, end, person['bbox']),
                                         frames=end-w['seed']+1)))
            except Exception as error:  # OpenCV raises its own error type; without thumbs the window is never a repeat
                errors.append(dict(window=w, step='motif', error=str(error)[:300]))
        entry['errors'][at:at] = errors
    def seed(sid, w, i):
        base = root/f'{sid}-w{i}'
        return _step('seed', ['--pose', pose_for(w['seed']), '--frame', w['seed'], '--job', f'{sid}-w{i}', '--length', w['end']-w['seed']+1,
                              '--out', base.with_suffix('.obs.json'), '--evidence-dir', base.with_name(base.name+'-crops')])
    def queue(base, receipt):
        queued = []
        for name in receipt['jobs']:
            group = name.rsplit('-', 1)[1]; stem = base.with_name(f'{base.name}-{group}')
            try:
                _step('prepare', ['--queries', base.with_suffix('.obs.json'), '--shots', root/'shots.json', '--job', name,
                                  '--out', stem.with_suffix('.freeze.json'), '--allow-tracking', '--max-seconds', LOOK['track_seconds']])
                queued.append((name, group, stem, _submit(['--job', name, '--queries', base.with_suffix('.obs.json'), '--frozen', stem.with_suffix('.freeze.json'),
                                                           '--out', joint_tracking.track_path(stem, group), '--checkpoint', checkpoint,
                                                           '--max-seconds', max(1, min(LOOK['track_seconds'], int(remaining())))]), None))
            except STEP_ERRORS as error:
                queued.append((name, group, stem, None, error))
        return queued
    # (sid, window, receipt, error, queued): the next shot's first window, seeded and queued while the child tracks
    ahead = None
    segments = scan['analysis_segments']; shown = segments[:max_shots]
    for order, segment in enumerate(shown):
        sid = segment['segment_id']; s0, s1 = segment['start_frame'], segment['end_frame']
        if pending and pose_for(s1, wait=False) is None:
            report(pending); pending = None  # report the last window while this shot is still being posed
        pose_for(s1)
        fps = scan['fps']
        entry = dict(segment_id=sid, frames=[s0, s1], seconds=[round(s0/fps, 2), round((s1+1)/fps, 2)], windows=[], skipped=[], errors=[])
        shots.append(entry)
        lines.append(f"## {sid} {entry['seconds'][0]:.2f}–{entry['seconds'][1]:.2f}s (f{s0}..{s1})")
        def still(k):
            # A cut too short to track still shows its pose (clip B: 83 skipped ranges, many whole cuts under 0.2 s).
            from image_lens import body_motion, onion
            mid = (k['frames'][0]+k['frames'][1])//2; person = body_motion.largest(pose_frames.get(mid))
            try:
                k['still'] = onion.still(source, mid, person['bbox'] if person else None, root/f'{sid}-still-f{mid}.jpg')
            except Exception as error:  # OpenCV raises its own error type; the still is optional
                entry['errors'].append(dict(frames=k['frames'], step='still', error=str(error)[:300]))
            return f" (사진 {k['still']}: 자세만, 움직임 모름)" if k.get('still') else ''
        if s1 == s0:
            k = dict(frames=[s0, s1], reason='single-frame shot'); entry['skipped'].append(k)
            lines += ['- 건너뜀: 1프레임 장면'+still(k), '']
            continue
        start = s0; i = 0
        while start <= s1:
            w, skipped = next_window(segment, start, pose_frames, flagged)
            entry['skipped'] += skipped
            lines += [f"- 건너뜀 f{k['frames'][0]}..{k['frames'][1]}: {k['reason']}" + ('' if entry['windows'] else still(k)) for k in skipped]
            if w is None:
                break
            if remaining() < 60:
                entry['errors'].append(dict(window=w, error='overall time budget exhausted')); lines.append('- 시간 예산 소진: 이후 장면 생략')
                break
            base = root/f'{sid}-w{i}'; i += 1; start = w['end']+1
            people = len(pose_frames[w['seed']])
            done = dict(window=w, groups=[]); tracked = []
            queued = None
            try:
                if ahead and ahead[:2] == (sid, w):
                    receipt, error, queued = ahead[2:]; ahead = None
                    if error:
                        raise error
                else:
                    receipt = seed(sid, w, i-1)
            except STEP_ERRORS as error:
                entry['errors'].append(dict(window=w, step='seed', error=str(error)[:300]))
                lines += [f"- 구간 f{w['seed']}..{w['end']}", f'  - 점 배치 실패: {str(error)[:120]}']
                entry['windows'].append(done)
                continue
            if queued is None:
                queued = queue(base, receipt)
            # The child tracks this window while the last one is reported (clip B: 44 s tracking, 45 s of CPU work around it).
            if pending:
                report(pending); pending = None
            # A cut moves this shot's next window, but the next shot always starts at its first frame (clip B: 39 of 42
            # windows): seed it while the child still tracks (25 s of waiting, 23 s of seeding), and queue its jobs so the
            # child goes on to them while this window is reported (clip B: 11 s idle between short shots). A window after
            # a cut in this shot is queued behind them; the worker keeps their replies until they are taken.
            after = shown[order+1] if order+1 < len(shown) else None
            if not ahead and after and after['end_frame'] > after['start_frame'] and pose_for(after['end_frame'], wait=False) and remaining() >= 60:
                first, _ = next_window(after, after['start_frame'], pose_frames, flagged)
                if first:
                    try:
                        receipt_ahead = seed(after['segment_id'], first, 0)
                        ahead = (after['segment_id'], first, receipt_ahead, None, queue(root/f"{after['segment_id']}-w0", receipt_ahead))
                    except STEP_ERRORS as error:
                        ahead = (after['segment_id'], first, None, error, None)
            for name, group, stem, handle, error in queued:
                try:
                    if error:
                        raise error
                    _finish(handle)
                    tracked.append((name, group, stem, paths.load(joint_tracking.track_path(stem, group))[0]))
                except STEP_ERRORS as error:
                    entry['errors'].append(dict(window=w, job=name, error=str(error)[:300]))
                    lines.append(f'  - {group} 실패: {str(error)[:120]}')
            # A drawing change (costume, wipe) drops many points at once: analyze up to it and reseed right after.
            # Hand tips flicker on drawn hands, so only head, arms and legs decide a cut.
            cut = cut_frame([t[3] for t in tracked if t[1] != 'hands'], w)
            end = w['end'] if cut is None else min(w['end'], max(cut, w['seed']+LOOK['min_frames']-1))
            if end < w['end']:
                done['cut_after'] = end; start = end+1
            # Cuts/reseeding remain based on raw model visibility. The selected joint stream is separate;
            # bind/analyze use the raw files so fusion never manufactures a surface ease measurement.
            observations = None
            for k, (name, group, stem, packet) in enumerate(tracked):
                if group in joint_tracking.GROUPS:
                    if observations is None:
                        observations, _ = paths.load(base.with_suffix('.obs.json'))
                    selected = joint_tracking.reconcile(packet, pose_frames, observations['seed_source']['track_id'],
                                                       dict(w, end=end), joint_tracking.track_path(stem, group), pose_for(w['seed']))
                    paths.write_json(stem.with_suffix('.tracks.json'), selected)
                    tracked[k] = (name, group, stem, selected)
            slot = []; lines.append(slot); entry['windows'].append(done)
            pending = (entry, sid, w, end, base, people, receipt, tracked, done, slot, len(entry['errors']))
        lines.append('')
    if pending:
        report(pending)
    pose_for(scan['source_frame_count']-1)  # pose.json covers the whole clip, past --max-shots too
    if posed[-1][2] != root/'pose.json':
        with LOG.open('a') as log:
            line = heard.get()
            while line:  # its receipt: the child writes pose.json after its last part
                log.write(line); line = heard.get()
        if POSER.wait() or not (root/'pose.json').is_file():
            raise ValueError('pose step failed after its last part')
    if len(segments) > max_shots:
        lines.append(f'생략된 장면 {len(segments)-max_shots}개 (--max-shots {max_shots}).')
    head = ['# 영상 움직임 자세히 (요약은 look.md)', '',
            f"영상 {source.name}: {scan['source_frame_count']}프레임, {scan['fps']:g}fps, 장면 {len(segments)}개. 점은 포즈 기준 자동 배치(머리카락은 눈 기준 기하), 부위·동일인 미검증.",
            '경고는 "원본부터 비교하라"는 뜻이다. 가짜 움직임이라는 판정이 아니다. 자세한 수치는 장면별 report.json에 있다.', '']
    paths.write(root/'look-full.md', '\n'.join(head+[l for l in lines for l in (l if isinstance(l, list) else [l])]).rstrip('\n')+'\n')
    from image_lens import onion
    # Frame strips for the largest shoulder-width swings only: on clip B all 18 turn candidates got one, the sheets went
    # from 8 to 14 and a fresh reader scored about the same (13 vs 12.5 of 21); hair and a cat on the shoulder swing too.
    for done, end, bbox, base in sorted(stripping, key=lambda s: -s[0]['turn_share'])[:LOOK['strips']]:
        try:
            done['strip'] = onion.strip(source, done['window']['seed'], end, bbox, base.with_name(base.name+'-strip.jpg'))
        except Exception as error:  # OpenCV raises its own error type; the strip is optional
            done.setdefault('errors', []).append(dict(step='strip', error=str(error)[:300]))
    fps = scan['fps']
    sheets = onion.sheets(root, sorted([(d['window']['seed']/fps, (d.get('cut_after', d['window']['end'])+1)/fps, d['onion'], '')
                                        for s in shots for d in s['windows'] if d.get('onion')]
                                       + [(d['window']['seed']/fps, (d.get('cut_after', d['window']['end'])+1)/fps, d['who'], 'who')
                                          for s in shots for d in s['windows'] if d.get('who')]
                                       + [(d['window']['seed']/fps, (d.get('cut_after', d['window']['end'])+1)/fps, d['strip']['image'], 'strip')
                                          for s in shots for d in s['windows'] if d.get('strip')]
                                       + [(k['frames'][0]/fps, (k['frames'][1]+1)/fps, k['still'], 'still')
                                          for s in shots for k in s['skipped'] if k.get('still')]))
    from image_lens import zoom
    errors = []
    try:
        frame_sheets = zoom.overview(source, root)
    except Exception as error:  # OpenCV raises its own error type; the overview is optional
        frame_sheets = []; errors.append(dict(step='frame_sheets', error=str(error)[:300]))
    from image_lens import motif
    for (done, _), label in zip(looks, motif.assign([l for _, l in looks])):
        if label:
            done['motif'] = label
    record = dict(schema='image-lens.look/1', source_video=str(source), source_sha256=scan['source_sha256'], params=LOOK,
                  max_shots=max_shots, scaled=scaled, onion_sheets=sheets, frame_sheets=frame_sheets, errors=errors, shots=shots,
                  joint_tracking=joint_tracking.PARAMS)
    from image_lens import check
    record['check'] = check.spans(record, fps)
    try:
        check.sheets(source, root, record['check'])
    except Exception as error:  # OpenCV raises its own error type; the sheets are optional
        errors.append(dict(step='check_sheets', error=str(error)[:300]))
    record['elapsed_seconds'] = round(time.monotonic()-started, 1)
    paths.write_json(root/'look.json', record)
    from image_lens import digest
    look_md = paths.write(root/'look.md', digest.markdown(record, scan, 'look-full.md'))
    return dict(status='written', output=str(look_md), shots=len(shots), windows=sum(len(s['windows']) for s in shots),
                errors=sum(len(s['errors']) for s in shots)+len(errors), elapsed_seconds=record['elapsed_seconds'])
