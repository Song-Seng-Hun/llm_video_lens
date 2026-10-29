"""Unified portable entry points over the selected analysis engines."""
import argparse
import json
from pathlib import Path
import sys
from image_lens import paths

COMMANDS = ('look', 'scan', 'compare', 'brief', 'catalog', 'pose', 'seed', 'bind', 'prepare', 'validate', 'track', 'analyze', 'grid', 'mark', 'focus')


def parser(command, description):
    return argparse.ArgumentParser(prog='image-lens '+command, description=description)


def models(a, *names):
    """Fill model paths left out on the command line from <workspace>/models.json (per machine, Git-ignored), e.g.
    {"detector": "...onnx", "pose_model": "...onnx", "checkpoint": "...pth"}. A flag given on the command line wins."""
    missing = [n for n in names if getattr(a, n) is None]
    if not missing:
        return a
    path = paths.workspace_root()/'models.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    for n in missing:
        if not saved.get(n):
            raise ValueError(f"--{n.replace('_', '-')} not given and not in {path}")
        setattr(a, n, saved[n])
    return a


def selected_entry(module, arguments):
    """Use the existing selected main; only normalize its external path boundary."""
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *arguments]
        module.main()
    finally:
        sys.argv = previous


def analyze(arguments):
    from image_lens import analyze_motion, shot_boundaries
    if '--help' in arguments or '-h' in arguments:
        return selected_entry(analyze_motion, arguments)
    arguments = list(arguments)
    path_flags = {'--tracks', '--shots', '--report-bindings', '--out', '--evidence-dir', '--trajectory-dir', '--pose'}
    parsed_paths = {}
    for i, token in enumerate(arguments):
        # One canonical spelling avoids ambiguous validation of --key=value.
        if any(token.startswith(k+'=') for k in path_flags):
            raise ValueError('use space-separated path arguments')
        if token in path_flags:
            if i+1 == len(arguments):
                raise ValueError('missing path argument')
            output = token in ('--out', '--evidence-dir', '--trajectory-dir')
            value = paths.confined(arguments[i+1], exists=not output, new=output,
                                   max_bytes=None if output else 64*1024*1024 if token == '--pose' else 16*1024*1024)
            arguments[i+1] = str(value)
            parsed_paths[token] = value
    if '--tracks' not in parsed_paths or '--out' not in parsed_paths:
        raise ValueError('--tracks and --out required')
    packet, _ = paths.load(parsed_paths['--tracks'])
    source = paths.confined(packet['job']['source_video'], exists=True, max_bytes=512*1024*1024)
    if paths.sha(source) != packet['job']['source_sha256']:
        raise ValueError('track source changed or unavailable')
    scan = packet.get('shot_check')
    if '--shots' in parsed_paths:
        scan, _ = paths.load(parsed_paths['--shots'])
    if scan is None:
        raise ValueError('canonical shot context required; unchecked historical caches are not selected')
    shot_boundaries.validate(scan, packet['job'], packet['frames'])
    if '--max-frames' in arguments and int(arguments[arguments.index('--max-frames')+1]) > 8:
        raise ValueError('one shared evidence cap is eight frames')
    if any(t.startswith('--max-frames=') for t in arguments):
        raise ValueError('use space-separated --max-frames')
    paths.confined(parsed_paths['--out']).parent.mkdir(parents=True, exist_ok=True)
    selected_entry(analyze_motion, arguments)


def dispatch(command, arguments):
    if command == 'analyze':
        return analyze(arguments)
    if command == 'scan':
        from image_lens import shot_boundaries
        ap = parser(command, 'Native-PTS hard-cut candidates; no images unless requested.')
        ap.add_argument('--video', required=True); ap.add_argument('--out', required=True)
        ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int)
        ap.add_argument('--evidence-dir'); ap.add_argument('--event-index', type=int, default=0)
        a = ap.parse_args(arguments)
        source = paths.confined(a.video, exists=True, max_bytes=512*1024*1024)
        paths.confined(a.out, new=True)
        if a.event_index and not a.evidence_dir:
            raise ValueError('--event-index requires --evidence-dir')
        if a.evidence_dir:
            evidence = paths.confined(a.evidence_dir, new=True)
            evidence.parent.mkdir(parents=True, exist_ok=True)
        import signal
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('scan wall budget exhausted')))
            signal.alarm(120)
        scan = shot_boundaries.scan_video(source, a.start, a.end)
        if a.evidence_dir:
            scan['evidence_contact'] = shot_boundaries.render(scan, evidence, a.event_index)
        output = paths.write_json(a.out, scan)
        return dict(status='written', output=str(output), frames=scan['frame_count'],
                    events=len(scan['events']), images_written=1 if a.evidence_dir else 0, model_loaded=False)
    if command == 'compare':
        from image_lens import brief, shot_boundaries
        ap = parser(command, 'Two native originals + one unverified residual panel; no automatic redraw decision.')
        ap.add_argument('--shots', required=True); ap.add_argument('--frames', required=True, help='native a,b')
        ap.add_argument('--out', required=True); ap.add_argument('--evidence-dir', required=True)
        a = ap.parse_args(arguments)
        paths.confined(a.out, new=True)
        destination = paths.confined(a.evidence_dir, new=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        scan, scan_path, _ = brief.verified_scan(a.shots)
        frames = a.frames.split(',')
        if len(frames) != 2 or any(not t.isascii() or not t.isdecimal() for t in frames):
            raise ValueError('two increasing native frame integers a,b required')
        import signal
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('comparison wall budget exhausted')))
            signal.alarm(60)
        report = shot_boundaries.render_comparison(scan, destination, [int(t) for t in frames])
        report.update(schema='image-lens.appearance-comparison/1', scan_path=str(scan_path), scan_sha256=paths.sha(scan_path))
        return dict(status='written', output=str(paths.write_json(a.out, report)), image_panels=3, model_loaded=False)
    if command in ('brief', 'catalog'):
        from image_lens import brief
        ap = parser(command, 'Small verified shot rows; default zero images and no raw coordinate dump.')
        ap.add_argument('--shots', required=True); ap.add_argument('--report', action='append', default=[])
        ap.add_argument('--offset', type=int, default=0); ap.add_argument('--limit', type=int, default=8)
        ap.add_argument('--format', choices=('json', 'markdown'), default='json'); ap.add_argument('--out', required=True)
        a = ap.parse_args(arguments)
        paths.confined(a.out, new=True)
        result = brief.build(a.shots, a.report, a.offset, a.limit)
        output = paths.write(a.out, brief.markdown(result)) if a.format == 'markdown' else paths.write_json(a.out, result)
        return dict(status='written', output=str(output), source_status=result['status'],
                    returned_shots=result['coverage']['returned_segments'], images_written=0, model_loaded=False)
    if command == 'grid':
        from image_lens import mark
        ap = parser(command, 'One frame (or a box of it) with a labelled source-pixel grid, for reading point positions by eye.')
        ap.add_argument('--video', required=True); ap.add_argument('--frame', type=int, required=True)
        ap.add_argument('--box', help='x0,y0,x1,y1 zoom box in source pixels'); ap.add_argument('--out', required=True)
        a = ap.parse_args(arguments)
        return mark.grid(a.video, a.frame, a.out, tuple(int(v) for v in a.box.split(',')) if a.box else None)
    if command == 'mark':
        from image_lens import mark
        ap = parser(command, 'Track agent-placed, agent-named points from one frame (names are the agent reading, not verified).')
        ap.add_argument('--video', required=True); ap.add_argument('--shots', required=True, help='shots.json from scan or look')
        ap.add_argument('--frame', type=int, required=True); ap.add_argument('--end', type=int)
        ap.add_argument('--point', action='append', required=True, help='name=x,y in source pixels; repeat up to 8 times')
        ap.add_argument('--no-snap', action='store_true', help='keep the given positions instead of the nearest corner')
        ap.add_argument('--checkpoint', help='default: models.json'); ap.add_argument('--out', required=True, help='new output directory')
        a = models(ap.parse_args(arguments), 'checkpoint')
        return mark.run(a.video, a.shots, a.frame, a.point, a.checkpoint, a.out, a.end, not a.no_snap)
    if command == 'focus':
        from image_lens import digest
        ap = parser(command, 'Print one shot of a look run in full detail (look-full.md).')
        ap.add_argument('--look', required=True, help='look output directory'); ap.add_argument('--shot', required=True)
        a = ap.parse_args(arguments)
        return dict(status='ok', shot=a.shot, text=digest.focus(a.look, a.shot))
    if command == 'look':
        from image_lens import look
        ap = parser(command, 'One-command look: scan, pose, pose-seeded tracking per shot window, one short look.md.')
        ap.add_argument('--video', required=True); ap.add_argument('--detector', help='default: models.json')
        ap.add_argument('--pose-model', help='default: models.json'); ap.add_argument('--checkpoint', help='default: models.json')
        ap.add_argument('--out', required=True, help='new output directory inside the workspace')
        ap.add_argument('--max-shots', type=int, default=look.LOOK['max_shots']); ap.add_argument('--max-seconds', type=int, default=1800)
        a = models(ap.parse_args(arguments), 'detector', 'pose_model', 'checkpoint')
        if not 1 <= a.max_shots <= 256 or not 60 <= a.max_seconds <= 7200:
            raise ValueError('--max-shots 1..256 and --max-seconds 60..7200')
        return look.run(a.video, a.detector, a.pose_model, a.checkpoint, a.out, a.max_shots, a.max_seconds)
    if command == 'pose':
        from image_lens import pose
        ap = parser(command, 'Optional local whole-body pose (pinned user-supplied ONNX weights); per-shot track IDs and overlap flags.')
        ap.add_argument('--shots', required=True); ap.add_argument('--detector', help='default: models.json')
        ap.add_argument('--pose-model', help='default: models.json'); ap.add_argument('--out', required=True)
        ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int)
        ap.add_argument('--max-seconds', type=int, default=300)
        a = models(ap.parse_args(arguments), 'detector', 'pose_model')
        paths.confined(a.out, new=True)
        if not 1 <= a.max_seconds <= 1800:
            raise ValueError('wall budget must be 1..1800 seconds')
        import signal
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('pose wall budget exhausted')))
            signal.alarm(a.max_seconds)
        result = pose.run(str(paths.confined(a.shots, exists=True)), a.detector, a.pose_model, a.start, a.end)
        # Compact: indented keypoints are ~4x larger and would hit the 64 MiB read cap on long clips.
        output = paths.write(a.out, json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        return dict(status='written', output=str(output), frames=len(result['frames']), flagged_frames=len(result['flags']['frames']),
                    elapsed_seconds=result['elapsed_seconds'], model_loaded=True, images_written=0)
    if command == 'seed':
        from image_lens import pose
        ap = parser(command, 'Pose-derived seed queries (joints, eyes, hair geometry) for one shot job; candidates, not verified anatomy.')
        ap.add_argument('--pose', required=True); ap.add_argument('--frame', type=int, required=True)
        ap.add_argument('--job', required=True, help='job name prefix; one job per group: <job>-head, -arms, -legs')
        ap.add_argument('--track'); ap.add_argument('--length', type=int, default=160)
        ap.add_argument('--out', required=True); ap.add_argument('--evidence-dir', required=True, help='new dir for one marked crop per seed')
        a = ap.parse_args(arguments)
        paths.confined(a.out, new=True)
        packet = pose.seed(str(paths.confined(a.pose, exists=True, max_bytes=64*1024*1024)), a.frame, a.job, a.evidence_dir, a.track, a.length)
        return dict(status='written', output=str(paths.write_json(a.out, packet)),
                    jobs={j['name']: [p['point_id'] for p in j['points']] for j in packet['jobs']},
                    frames=[packet['jobs'][0]['start_frame'], packet['jobs'][0]['end_frame']],
                    skeleton=packet['seed_source']['skeleton'],
                    seed_frame_flagged=packet['seed_source']['seed_frame_flagged'], images_written=sum(len(j['points']) for j in packet['jobs']),
                    model_loaded=False)
    if command == 'bind':
        from image_lens import pose
        ap = parser(command, 'Report bindings for one pose seed job (<job>-head, -arms or -legs).')
        ap.add_argument('--queries', required=True); ap.add_argument('--tracks', required=True)
        ap.add_argument('--job', required=True); ap.add_argument('--out', required=True)
        a = ap.parse_args(arguments)
        paths.confined(a.out, new=True)
        packet = pose.bindings(str(paths.confined(a.queries, exists=True)), str(paths.confined(a.tracks, exists=True)), a.job)
        return dict(status='written', output=str(paths.write_json(a.out, packet)), features=len(packet['actors'][0]['features']))
    if command == 'prepare':
        from image_lens.jobs import prepare
        ap = parser(command, 'Freeze a new workspace query packet against a canonical full-source shot scan.')
        ap.add_argument('--queries', required=True); ap.add_argument('--shots', required=True)
        ap.add_argument('--job', required=True); ap.add_argument('--out', required=True)
        ap.add_argument('--allow-tracking', action='store_true', help='explicitly permit one optional experimental inference attempt')
        ap.add_argument('--max-seconds', type=int, default=120)
        a = ap.parse_args(arguments)
        return prepare(a.queries, a.shots, a.out, a.job, a.allow_tracking, a.max_seconds)
    if command in ('validate', 'track'):
        from image_lens import run_point_tracker
        ap = parser(command, 'Source/PTS/cut preflight without Torch.' if command == 'validate' else 'Experimental optional pinned local CoTracker; no automatic download or retry.')
        ap.add_argument('--job', required=True); ap.add_argument('--queries', required=True)
        ap.add_argument('--frozen', required=True); ap.add_argument('--out', required=True, help='new artifact path; validation does not create tracks')
        ap.add_argument('--max-seconds', type=int, default=120)
        if command == 'track':
            ap.add_argument('--checkpoint', required=True)
        a = ap.parse_args(arguments)
        args = ['--job', a.job, '--queries', str(paths.confined(a.queries, exists=True)),
                '--frozen', str(paths.confined(a.frozen, exists=True)), '--out', str(paths.confined(a.out, new=True)),
                '--max-seconds', str(a.max_seconds)]
        paths.confined(a.out).parent.mkdir(parents=True, exist_ok=True)
        if command == 'validate':
            args.append('--validate-only')
        else:
            args.extend(['--checkpoint', a.checkpoint])
        if command == 'validate':
            return selected_entry(run_point_tracker, args)
        # Native Torch code may defer Python signal delivery; the parent owns
        # an independent hard deadline and reaps only this job's process group.
        from image_lens import watchdog
        child = Path(__file__).with_name('tracking_child.py')
        try:
            returncode, stdout, stderr = watchdog.run(
                [sys.executable, '-I', '-B', str(child), '--workspace', str(paths.workspace_root()), *args],
                a.max_seconds)
        except watchdog.ChildTimeout as error:
            if error.stdout: print(error.stdout, end='')
            if error.stderr: print(error.stderr, end='', file=sys.stderr)
            raise
        if stdout: print(stdout, end='')
        if stderr: print(stderr, end='', file=sys.stderr)
        if returncode:
            raise ValueError('experimental tracking child failed; inspect its error above; no retry')
        return None
    raise ValueError('unsupported command')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='image-lens', description='Source-bound video inspection and compact LLM shot briefs. Default zero images. Optional tracking is experimental.')
    ap.add_argument('--workspace', help='explicit data/output boundary; default repository/workspace')
    ap.add_argument('command', choices=COMMANDS)
    ap.add_argument('arguments', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    try:
        paths.configure(a.workspace)
        result = dispatch(a.command, a.arguments)
        if a.command == 'focus':
            print(result['text'], end='')  # plain Markdown for the reader, not a JSON receipt
        elif result is not None:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, TimeoutError, OverflowError) as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__, error=str(error)[:1200],
                              no_automatic_retry=True), ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        import signal
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
