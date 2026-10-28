"""One pinned CoTracker3 job. Explicit local queries only; no GT, fallback, training or network."""
import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
import runpy
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
CHECKPOINT_SHA = '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
PARAMS = {'device': 'cpu', 'torch_threads': 8, 'interop_threads': 1, 'offline': True,
          'v2': False, 'window_len': 60, 'backward_tracking': False,
          'max_frames': 160, 'max_queries': 24, 'max_pixels': 2073600,
          'feature_chunk_size': 16, 'query_seed_frame_scored_as_tracking': False, 'visibility': 'upstream >0.9 AND in-frame',
          'mixed_initialization_queries': False, 'seed': 0}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def spatial_shape(job):
    shape = job.get('inference_resolution', [384, 512])
    if (type(shape) is not list or len(shape) != 2 or any(type(x) is not int for x in shape)
            or tuple(shape) not in ((384, 512), (576, 768))):
        raise ValueError('only default384x512 or explicit detail576x768 V3 shapes supported')
    return tuple(shape)


def configure_spatial_shape(predictor, shape):
    predictor.interp_shape = shape
    predictor.model.model_resolution = shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', help='explicit local pinned checkpoint, required only for inference')
    ap.add_argument('--job', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-seconds', type=int, required=True)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    from image_lens import paths
    from image_lens.jobs import verify_frozen
    out = paths.confined(args.out, new=True)
    queries_path = paths.confined(args.queries, exists=True, max_bytes=16*1024*1024)
    frozen_path = paths.confined(args.frozen, exists=True, max_bytes=16*1024*1024)
    args.queries, args.frozen = str(queries_path), str(frozen_path)
    task_root = paths.workspace_root()
    if not 1 <= args.max_seconds <= 600:
        raise ValueError('wall budget must be 1..600 seconds')
    started = time.monotonic()
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('job wall budget')))
        signal.alarm(args.max_seconds)
    elif not args.validate_only:
        raise ValueError('experimental tracking requires POSIX SIGALRM wall-budget enforcement')
    frozen = verify_frozen(frozen_path, queries_path)
    if args.job != frozen.get('job_name') or args.max_seconds > frozen.get('max_seconds', 0):
        raise ValueError('job or wall budget differs from frozen scope')
    if frozen.get('validation_only') and not args.validate_only:
        raise ValueError('this freeze permits validation only; prepare a separate explicit tracking freeze')
    if frozen['runner_sha256'] != sha(__file__) or frozen['queries_sha256'] != sha(args.queries) or frozen['params'] != PARAMS:
        raise ValueError('frozen runner/query/parameter mismatch')
    source = HERE / 'vendor/source'
    if frozen.get('schema') == 'image-lens.freeze/1':
        if (frozen.get('source_manifest_sha256') != sha(HERE / 'source-manifest.json')
                or frozen.get('local_patch_sha256') != sha(HERE / 'vendor/local-patch.json')):
            raise ValueError('frozen vendor manifest/patch mismatch')
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    patch = json.loads((HERE / 'vendor/local-patch.json').read_text())
    source_pins = {}
    for item in manifest['files']:
        expected = patch['after_sha256'] if item['path'] == patch['file'] else item['sha256']
        vendor_file = source / item['path']
        if vendor_file.is_symlink() or not vendor_file.resolve().is_relative_to(source):
            raise ValueError('vendor source escapes pinned directory')
        actual = sha(vendor_file)
        if actual != expected:
            raise ValueError('vendor source pin mismatch: ' + item['path'])
        source_pins[item['path']] = actual
    checkpoint = None
    if not args.validate_only:
        if not args.checkpoint:
            raise ValueError('tracking requires an explicit local --checkpoint; no download is performed')
        checkpoint = Path(args.checkpoint).expanduser().absolute()
        paths._no_symlinks(checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size != 101890938 or sha(checkpoint) != CHECKPOINT_SHA:
            raise ValueError('checkpoint pin mismatch')
    packet = json.loads(Path(args.queries).read_text())
    observation_mode = packet.get('schema') == 'team-eye.observations/1'
    observation_api = None
    if observation_mode:
        module = HERE / 'observations.py'
        if module.is_symlink() or frozen.get('observations_sha256') != sha(module):
            raise ValueError('observation contract pin mismatch')
        observation_api = runpy.run_path(str(module))
    elif packet.get('schema') != 'team-eye.queries/1':
        raise ValueError('unknown query schema')
    selected = [j for j in packet['jobs'] if j['name'] == args.job]
    if len(selected) != 1:
        raise ValueError('job must match exactly once')
    job = selected[0]
    from image_lens.jobs import check_job
    ref = frozen['canonical_shots']
    scan_packet, _ = paths.load(ref['path'])
    check_job(packet, args.job, scan_packet)
    resolution = spatial_shape(job)
    start, end, seed = job['start_frame'], job['end_frame'], job['seed_frame']
    if any(type(x) is not int for x in (start, end, seed)) or not 0 <= start <= seed <= end or end - start + 1 > PARAMS['max_frames']:
        raise ValueError('invalid frame interval or seed')
    width, height = job['width'], job['height']
    if type(width) is not int or type(height) is not int or min(width, height) <= 1 or width * height > PARAMS['max_pixels']:
        raise ValueError('image dimension cap')
    video_path = paths.confined(job['source_video'], exists=True, max_bytes=512*1024*1024)
    if video_path.is_symlink() or sha(video_path) != job['source_sha256']:
        raise ValueError('video source pin mismatch')
    region_api = None
    if 'regions' in job:
        module = HERE / 'part_regions.py'
        if module.is_symlink() or frozen.get('part_regions_sha256') != sha(module):
            raise ValueError('region observation helper pin mismatch')
        region_api = runpy.run_path(str(module))
        region_api['validate'](job)
    points = job['points']
    if not 1 <= len(points) <= PARAMS['max_queries'] or len({p['point_id'] for p in points}) != len(points):
        raise ValueError('invalid point count or duplicate point IDs')
    queries, point_indices, anchor_mapping = [], {}, None
    if observation_mode:
        queries, anchor_mapping = observation_api['compile_observations'](job)
        visual_references = observation_api['verify_visual_references'](job, task_root)
    else:
        for i, p in enumerate(points):
            xy = p['xy']
            if xy is None:
                continue  # Unknown semantic anchors remain missing, never supplied as fake coordinates.
            if (not isinstance(xy, list) or len(xy) != 2 or not all(finite(x) for x in xy)
                    or not 0 <= xy[0] < width or not 0 <= xy[1] < height):
                raise ValueError('nonfinite/out-of-frame query')
            point_indices[i] = len(queries)
            queries.append([seed - start, float(xy[0]), float(xy[1])])
    if not queries:
        raise ValueError('no identifiable seed points')
    shot_module=HERE/'shot_boundaries.py'
    if (shot_module.is_symlink() or frozen.get('shot_boundaries_sha256')!=sha(shot_module)
            or frozen.get('shot_detector_pins_sha256')!=sha(HERE/'shot-detector-pins.json')):
        raise ValueError('shot preflight helper/detector freeze mismatch')
    shot_api=runpy.run_path(str(shot_module))
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    images, timestamps = [], []
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - job['fps']) > 1e-6:
        raise ValueError('decode/fps mismatch')
    try:
        for n in range(end + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError('incomplete source decode')
            if n < start:
                continue
            if image.shape != (height, width, 3):
                raise ValueError('decoded frame shape mismatch')
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            if not finite(pts) or abs(pts - n / job['fps']) > .002:
                raise ValueError('selected decoder PTS mismatch')
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamps.append(pts)
    finally:
        cap.release()
    cached_scan=None
    if 'canonical_shots' in frozen:
        ref=frozen['canonical_shots']
        if type(ref)is not dict or set(ref)!={'path','sha256'} or type(ref['path'])is not str:
            raise ValueError('canonical shot reference requires path and SHA256')
        scan_path=paths.confined(ref['path'], exists=True, max_bytes=16*1024*1024)
        if sha(scan_path)!=ref['sha256']:
            raise ValueError('frozen canonical shot reference mismatch')
        cached_scan=json.loads(scan_path.read_text())
    shot_check=shot_api['canonical_scan'](job,timestamps,cached_scan)
    internal_events=[e for e in shot_check['events']if start<e['after_frame']<=end]
    if internal_events:
        print(json.dumps(dict(status='needs_shot_split',model_loaded=False,forward_calls=0,
                              events=internal_events,analysis_segments=shot_check['analysis_segments'])),file=sys.stderr)
        raise ValueError('detected edit/brief-transition boundary; choose one segment and fresh source observations')
    if args.validate_only:
        print(json.dumps({'status':'valid','job':job['name'],'model_loaded':False,'forward_calls':0,
                          'queries':len(queries),'shot_preflight':'no_cut_detected','shot_scan_scope':'full_source',
                          'visual_references':visual_references if observation_mode else [],
                          'semantic_identity_verified_by_model':False}))
        if hasattr(signal, 'SIGALRM'):signal.alarm(0)
        return
    # Consume one explicit inference attempt only after all source/decode/cut preflight.
    paths.write_json(str(frozen_path)+'.attempt.json',
                     dict(status='attempt_started', job=args.job, frozen_sha256=sha(frozen_path),
                          max_seconds=args.max_seconds, automatic_retry=False))
    import torch
    if torch.__version__ != '2.9.0':raise ValueError('unexpected Torch version')
    torch.set_num_threads(PARAMS['torch_threads']);torch.set_num_interop_threads(PARAMS['interop_threads'])
    torch.manual_seed(PARAMS['seed']);np.random.seed(PARAMS['seed'])
    if 'cotracker' in sys.modules and Path(sys.modules['cotracker'].__file__).resolve()!=source/'cotracker/__init__.py':
        raise ValueError('another cotracker module is already loaded')
    sys.path.insert(0,str(source))
    from cotracker.predictor import CoTrackerPredictor
    print(json.dumps({'stage': 'decoded', 'job': args.job, 'frames': len(images), 'queries': len(queries)}), file=sys.stderr, flush=True)
    region_observations = region_api['track'](images, job) if region_api else None
    video = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).float()
    del images
    q = torch.tensor(queries, dtype=torch.float32).unsqueeze(0)
    if not torch.isfinite(video).all() or not 0 <= video.min() <= video.max() <= 255:
        raise ValueError('invalid RGB tensor')
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60).to('cpu').eval()
    configure_spatial_shape(model, resolution)
    model.model.forward = partial(model.model.forward, fmaps_chunk_size=PARAMS['feature_chunk_size'])
    print(json.dumps({'stage': 'inference_started', 'job': args.job, 'device': 'cpu'}), file=sys.stderr, flush=True)
    inference_started = time.monotonic()
    with torch.inference_mode():
        coords, vis = model(video, queries=q, backward_tracking=False)
    inference_seconds = time.monotonic() - inference_started
    if coords.shape != (1, end - start + 1, len(queries), 2) or vis.shape != coords.shape[:-1] or vis.dtype != torch.bool:
        raise ValueError('unexpected model output shape/dtype')
    if not torch.isfinite(coords).all():
        raise ValueError('nonfinite model output')
    coords, vis = coords.cpu().numpy()[0], vis.cpu().numpy()[0]
    frames = []
    previous_rows = None
    for t, pts in enumerate(timestamps):
        rows = []
        if observation_mode:
            rows = observation_api['observed_frame'](points, anchor_mapping, coords, vis,
                       start + t, t, width, height, previous_rows)
            previous_rows = rows
            common_excluded = start + t in job['common_metric_excluded_frames']
            for row in rows:
                row['tracking_metric_eligible'] = bool(row['tracking_metric_eligible'] and not common_excluded)
                row['metric_excluded_common_anchor_frame'] = common_excluded
        else:
            for i, p in enumerate(points):
                qi = point_indices.get(i)
                xy = None if qi is None else coords[t, qi].astype(float).tolist()
                model_visible = None if qi is None else bool(vis[t, qi])
                in_frame = xy is not None and 0 <= xy[0] < width and 0 <= xy[1] < height
                rows.append({'point_id': p['point_id'], 'xy': xy, 'model_visible': model_visible,
                             'visible': bool(model_visible and in_frame), 'in_frame': bool(in_frame),
                             'unknown_seed': qi is None})
        is_anchor = (any(row['status'] == 'anchor_input_not_prediction' for row in rows)
                     if observation_mode else start + t == seed)
        metric_eligible = (any(row['tracking_metric_eligible'] for row in rows)
                           if observation_mode else start + t > seed)
        frames.append({'frame': start + t, 'pts_seconds': pts, 'seed_frame_overwritten': is_anchor,
                       'tracking_metric_eligible': metric_eligible, 'points': rows})
    result = {'schema': 'team-eye.point-tracks/1', 'job': job, 'params': PARAMS, 'inference_resolution': list(resolution), 'frame_count': len(frames),
              'frames': frames, 'shot_check':shot_check, 'inference_seconds': inference_seconds, 'elapsed_seconds': time.monotonic() - started,
              'max_rss_bytes': (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)) if resource else None,
              'provenance': {'runner_sha256': sha(__file__), 'queries_sha256': sha(args.queries),
                             'frozen_sha256': sha(args.frozen), 'checkpoint_sha256': CHECKPOINT_SHA,
                             'source_commit': manifest['commit'], 'source_pins': source_pins,
                             'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__},
              'ground_truth_read': False, 'training': False, 'network_used': False, 'images_written': 0,
              'limits': f'Offline whole-clip context, internal{resolution[0]}x{resolution[1]} spatial resize, uncalibrated visibility; seed positions/visibility forced by upstream; assisted seeds are not automatic pose.'}
    if observation_mode:
        result['schema'] = 'team-eye.observed-tracks/1'
        result['params'] = dict(PARAMS, mixed_initialization_queries=job['mode'] == 'bounded-reanchor')
        result['observation_scope'] = 'visible_surface_2d'
        result['anatomical_claims'] = False
        result['assisted_source_observations'] = True
        result['anchor_mapping'] = anchor_mapping
        result['visual_references'] = visual_references
        result['raw_query_coordinates'] = coords.astype(float).tolist()
        result['raw_query_model_visible'] = vis.tolist()
        result['provenance']['observations_sha256'] = sha(HERE / 'observations.py')
        result['limits'] += ' Piecewise latest-anchor selection; no blending or motion derivative across re-anchor. Source observations are AI estimates, not anatomical GT or model verification.'
    if region_observations is not None:
        result['region_observations'] = region_observations
        result['provenance']['part_regions_sha256'] = sha(HERE / 'part_regions.py')
    with out.open('x') as f:
        json.dump(result, f, allow_nan=False, separators=(',', ':'))
        f.write('\n')
    if hasattr(signal, 'SIGALRM'):signal.alarm(0)
    print(json.dumps({k: result[k] for k in ('frame_count', 'inference_seconds', 'elapsed_seconds', 'max_rss_bytes')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps({'status': 'failed', 'type': type(e).__name__, 'error': str(e)[:1200]}), file=sys.stderr)
        raise SystemExit(1)
