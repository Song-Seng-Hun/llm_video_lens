"""One-command look at a video: scan, pose, pose-seeded tracking per shot window, grouped reports and one short look.md.
Runs the existing public steps in order with their own guards; a failed shot is recorded and the next one continues."""
import json
import signal
import time
from pathlib import Path
from image_lens import paths

LOOK = dict(max_shots=16, window_frames=160, min_frames=12, track_seconds=300, groups=('head', 'arms', 'legs'))
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


def next_window(segment, start, pose_frames, flagged, length=LOOK['window_frames'], min_frames=LOOK['min_frames']):
    """The next tracking window at or after `start`, seeded on the first frame with a pose track that is not pose-flagged."""
    skipped = []; end = segment['end_frame']
    while start <= end:
        seed = next((n for n in range(start, end+1) if pose_frames.get(n) and n not in flagged), None)
        if seed is None:
            skipped.append(dict(frames=[start, end], reason='no unflagged pose track'))
            break
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


def _summary(report, window, group, md_path, traj):
    """Short per-group lines for look.md, built from the recomputed report JSON."""
    from image_lens import motion_report
    names = {'head': '머리', 'arms': '팔', 'legs': '다리'}
    moves = [f"{f['point_id']} {motion_report.KO[u['direction']]} {u['distance_px']:.0f}px(f{u['start_frame']}..{u['end_frame']})"
             for f in report['features'] for run in f['continuous_runs'] for u in run['strokes']['units'] if u['major']]
    together = motion_report.together(report)
    lines = [f"- {names[group]}: " + ('; '.join(moves) if moves else '큰 이동 없음')]
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
        lines = [f"- {names[group]}: 쓸 수 있는 연속 관측 없음(가려짐·화면 밖·추적 끊김). 정지가 아니라 모름."]
    return (lines if moves or cut or len(lines) == 1 and '모름' in lines[0] else []), dict(group=group, major_moves=moves, together=together, warnings=len(warnings),
                       lost=cut, report=str(md_path), trajectory=str(traj))


def run(video, detector, pose_model, checkpoint, out, max_shots=LOOK['max_shots'], max_seconds=1800):
    started = time.monotonic()
    def remaining():
        return max_seconds-(time.monotonic()-started)
    root = paths.confined(out, new=True)
    source = paths.confined(video, exists=True, max_bytes=512*1024*1024)
    root.mkdir(parents=True)
    global LOG
    LOG = root/'steps.log'
    _step('scan', ['--video', source, '--out', root/'shots.json'])
    _step('pose', ['--shots', root/'shots.json', '--detector', detector, '--pose-model', pose_model, '--out', root/'pose.json',
                   '--max-seconds', max(1, min(1800, int(remaining())))])
    scan, _ = paths.load(root/'shots.json')
    pose_packet, _ = paths.load(root/'pose.json', max_bytes=64*1024*1024)
    pose_frames = {f['frame']: f['tracks'] for f in pose_packet['frames']}
    flagged = {r['frame'] for r in pose_packet['flags']['frames']}
    shots = []; lines = []
    segments = scan['analysis_segments']
    for segment in segments[:max_shots]:
        sid = segment['segment_id']; s0, s1 = segment['start_frame'], segment['end_frame']
        fps = scan['fps']
        entry = dict(segment_id=sid, frames=[s0, s1], seconds=[round(s0/fps, 2), round((s1+1)/fps, 2)], windows=[], skipped=[], errors=[])
        shots.append(entry)
        lines.append(f"## {sid} {entry['seconds'][0]:.2f}–{entry['seconds'][1]:.2f}s (f{s0}..{s1})")
        if s1 == s0:
            entry['skipped'].append(dict(frames=[s0, s1], reason='single-frame shot')); lines += ['- 건너뜀: 1프레임 장면', '']
            continue
        start = s0; i = 0
        while start <= s1:
            w, skipped = next_window(segment, start, pose_frames, flagged)
            entry['skipped'] += skipped
            lines += [f"- 건너뜀 f{k['frames'][0]}..{k['frames'][1]}: {k['reason']}" for k in skipped]
            if w is None:
                break
            if remaining() < 60:
                entry['errors'].append(dict(window=w, error='overall time budget exhausted')); lines.append('- 시간 예산 소진: 이후 장면 생략')
                break
            base = root/f'{sid}-w{i}'; job = f'{sid}-w{i}'; i += 1; start = w['end']+1
            people = len(pose_frames[w['seed']])
            done = dict(window=w, groups=[]); still = []; tracked = []
            try:
                receipt = _step('seed', ['--pose', root/'pose.json', '--frame', w['seed'], '--job', job, '--length', w['end']-w['seed']+1,
                                         '--out', base.with_suffix('.obs.json'), '--evidence-dir', base.with_name(base.name+'-crops')])
            except STEP_ERRORS as error:
                entry['errors'].append(dict(window=w, step='seed', error=str(error)[:300]))
                lines += [f"- 구간 f{w['seed']}..{w['end']}", f'  - 점 배치 실패: {str(error)[:120]}']
                entry['windows'].append(done)
                continue
            for name in receipt['jobs']:
                group = name.rsplit('-', 1)[1]; stem = base.with_name(f'{base.name}-{group}')
                try:
                    _step('prepare', ['--queries', base.with_suffix('.obs.json'), '--shots', root/'shots.json', '--job', name,
                                      '--out', stem.with_suffix('.freeze.json'), '--allow-tracking', '--max-seconds', LOOK['track_seconds']])
                    _step('track', ['--job', name, '--queries', base.with_suffix('.obs.json'), '--frozen', stem.with_suffix('.freeze.json'),
                                    '--out', stem.with_suffix('.tracks.json'), '--checkpoint', checkpoint,
                                    '--max-seconds', max(1, min(LOOK['track_seconds'], int(remaining())))])
                    tracked.append((name, group, stem, paths.load(stem.with_suffix('.tracks.json'))[0]))
                except STEP_ERRORS as error:
                    entry['errors'].append(dict(window=w, job=name, error=str(error)[:300]))
                    lines.append(f'  - {group} 실패: {str(error)[:120]}')
            # A drawing change (costume, wipe) drops many points at once: analyze up to it and reseed right after.
            cut = cut_frame([t[3] for t in tracked], w)
            end = w['end'] if cut is None else min(w['end'], max(cut, w['seed']+LOOK['min_frames']-1))
            lines.append(f"- 구간 f{w['seed']}..{end} (점 배치 f{w['seed']}" + (f", {people}명 중 가장 큰 사람" if people > 1 else '') + ')')
            for name, group, stem, _ in tracked:
                try:
                    _step('bind', ['--queries', base.with_suffix('.obs.json'), '--tracks', stem.with_suffix('.tracks.json'),
                                   '--job', name, '--out', stem.with_suffix('.bindings.json')])
                    _step('analyze', ['--tracks', stem.with_suffix('.tracks.json'), '--shots', root/'shots.json',
                                      '--report-bindings', stem.with_suffix('.bindings.json'), '--start', w['seed'], '--end', end,
                                      '--format', 'json', '--out', stem.with_suffix('.report.json'),
                                      '--trajectory-dir', stem.with_name(stem.name+'-traj'), '--pose', root/'pose.json'])
                    report, _ = paths.load(stem.with_suffix('.report.json'))
                    from image_lens import motion_report
                    md = paths.write(stem.with_suffix('.report.md'), motion_report.markdown(report))
                    group_lines, summary = _summary(report, dict(w, end=end), group, md.relative_to(root), stem.with_name(stem.name+'-traj').relative_to(root)/'trajectory.png')
                    lines += ['  '+l for l in group_lines]; done['groups'].append(summary)
                    if not group_lines:still.append({'head': '머리', 'arms': '팔', 'legs': '다리'}[group])
                except STEP_ERRORS as error:
                    entry['errors'].append(dict(window=w, job=name, error=str(error)[:300]))
                    lines.append(f'  - {group} 실패: {str(error)[:120]}')
            if still:lines.append(f"  - {'·'.join(still)}: 큰 이동 없음")
            if end < w['end']:
                done['cut_after'] = end; start = end+1
                lines.append(f"  - f{end} 뒤 점이 한꺼번에 끊김(그림 바뀜·화면 전환 추정): f{end+1}부터 점을 다시 놓음")
            if done['groups']:lines.append(f"  - 자세히: {base.name}-<head|arms|legs>.report.md, 그림 {base.name}-<묶음>-traj/trajectory.png")
            entry['windows'].append(done)
        lines.append('')
    if len(segments) > max_shots:
        lines.append(f'생략된 장면 {len(segments)-max_shots}개 (--max-shots {max_shots}).')
    head = ['# 영상 움직임 한눈 보기', '',
            f"영상 {source.name}: {scan['source_frame_count']}프레임, {scan['fps']:g}fps, 장면 {len(segments)}개. 점은 포즈 기준 자동 배치(머리카락은 눈 기준 기하), 부위·동일인 미검증.",
            '경고는 "원본부터 비교하라"는 뜻이다. 가짜 움직임이라는 판정이 아니다. 자세한 수치는 장면별 report.json에 있다.', '']
    text = '\n'.join(head+lines).rstrip('\n')+'\n'
    look_md = paths.write(root/'look.md', text)
    record = dict(schema='image-lens.look/1', source_video=str(source), source_sha256=scan['source_sha256'], params=LOOK,
                  max_shots=max_shots, elapsed_seconds=round(time.monotonic()-started, 1), shots=shots)
    paths.write_json(root/'look.json', record)
    return dict(status='written', output=str(look_md), shots=len(shots), windows=sum(len(s['windows']) for s in shots),
                errors=sum(len(s['errors']) for s in shots), elapsed_seconds=record['elapsed_seconds'])
