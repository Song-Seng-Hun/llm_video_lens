"""Fresh source-bound freezes. Preparation and static validation never import Torch."""
import math
import re
from pathlib import Path
from image_lens import paths

HERE = Path(__file__).resolve().parent
HELPERS = ('run_point_tracker.py', 'observations.py', 'shot_boundaries.py',
           'shot-detector-pins.json', 'part_regions.py', 'source-manifest.json',
           'vendor/local-patch.json', 'jobs.py', 'paths.py', 'cli.py',
           'tracking_child.py', 'watchdog.py')


def helper_pins():
    return {name: paths.sha(HERE/name) for name in HELPERS}


def verify_frozen(frozen_path, queries_path):
    frozen, _ = paths.load(frozen_path)
    if type(frozen) is not dict:
        raise ValueError('freeze packet must be an object')
    if frozen.get('schema') != 'image-lens.freeze/1' or frozen.get('helper_pins') != helper_pins():
        raise ValueError('freeze schema/helper pins changed; prepare a new freeze')
    if frozen.get('queries_sha256') != paths.sha(queries_path):
        raise ValueError('query packet changed after freeze')
    if type(frozen.get('validation_only')) is not bool:
        raise ValueError('explicit freeze validation policy required')
    return frozen


def check_job(packet, name, scan):
    from image_lens.run_point_tracker import PARAMS, spatial_shape
    from image_lens import observations, shot_boundaries
    if type(packet) is not dict:
        raise ValueError('query packet must be an object')
    if packet.get('schema') not in ('team-eye.queries/1', 'team-eye.observations/1'):
        raise ValueError('supported query or surface-observation packet required')
    jobs = packet.get('jobs')
    if type(jobs) is not list or not 1 <= len(jobs) <= 32:
        raise ValueError('bounded jobs list required')
    found = [j for j in jobs if type(j) is dict and j.get('name') == name]
    if len(found) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('job name must match exactly once and be bounded')
    job = found[0]
    spatial_shape(job)
    start, end, seed = (job[k] for k in ('start_frame', 'end_frame', 'seed_frame'))
    if any(type(n) is not int for n in (start, end, seed)) or not 0 <= start <= seed <= end or end-start+1 > PARAMS['max_frames']:
        raise ValueError('invalid bounded job range or seed')
    if any(type(job.get(k)) is not int or job[k] < 2 for k in ('width', 'height')) or job['width']*job['height'] > PARAMS['max_pixels']:
        raise ValueError('tracking source dimension cap')
    video = paths.confined(job['source_video'], exists=True, max_bytes=512*1024*1024)
    if str(video) != job['source_video'] or paths.sha(video) != job['source_sha256']:
        raise ValueError('query needs canonical workspace source path and matching hash')
    shot_boundaries.validate(scan, job)
    if scan.get('scan_scope') != 'full_source' or scan['native_range'] != [0, scan.get('source_frame_count', 0)-1]:
        raise ValueError('full-source canonical scan required')
    if any(start < e['after_frame'] <= end for e in scan['events']):
        raise ValueError('job crosses a detected cut; choose one shot and fresh source observations')
    points = job.get('points')
    if type(points) is not list or not 1 <= len(points) <= PARAMS['max_queries']:
        raise ValueError('bounded points required')
    ids = [p.get('point_id') for p in points if type(p) is dict]
    if len(ids) != len(points) or any(type(i) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('bounded unique point IDs required')
    if packet['schema'] == 'team-eye.observations/1':
        queries, _ = observations.compile_observations(job)
        for point in points:
            paths.confined(point['visual_reference']['path'], exists=True, max_bytes=16*1024*1024)
        observations.verify_visual_references(job, paths.workspace_root())
        expected = sorted({a['frame'] for p in points for a in p['anchors']})
        if job.get('common_metric_excluded_frames') != expected:
            raise ValueError('explicit shared anchor exclusions must match all anchor frames')
    else:
        queries = []
        for point in points:
            xy = point.get('xy')
            if xy is None:
                continue
            if (type(xy) is not list or len(xy) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                    or not 0 <= xy[0] < job['width'] or not 0 <= xy[1] < job['height']):
                raise ValueError('nonfinite or out-of-frame seed')
            queries.append(xy)
    if not queries:
        raise ValueError('at least one observed seed required; unknown seeds remain missing')
    return job, len(queries)


def prepare(queries, shots, out, name, allow_tracking=False, max_seconds=120):
    from image_lens.run_point_tracker import PARAMS, CHECKPOINT_SHA
    packet, query_path = paths.load(queries)
    scan, scan_path = paths.load(shots)
    _, count = check_job(packet, name, scan)
    if type(max_seconds) is not int or not 1 <= max_seconds <= 600:
        raise ValueError('wall budget must be 1..600 seconds')
    pins = helper_pins()
    frozen = dict(schema='image-lens.freeze/1', job_name=name, validation_only=not allow_tracking,
                  max_seconds=max_seconds, helper_pins=pins,
                  runner_sha256=pins['run_point_tracker.py'], queries_sha256=paths.sha(query_path), params=PARAMS,
                  observations_sha256=pins['observations.py'], part_regions_sha256=pins['part_regions.py'],
                  shot_boundaries_sha256=pins['shot_boundaries.py'], shot_detector_pins_sha256=pins['shot-detector-pins.json'],
                  source_manifest_sha256=pins['source-manifest.json'], local_patch_sha256=pins['vendor/local-patch.json'],
                  checkpoint_sha256=CHECKPOINT_SHA,
                  canonical_shots=dict(path=str(scan_path), sha256=paths.sha(scan_path)))
    output = paths.write_json(out, frozen)
    return dict(status='prepared', freeze=str(output), job=name, queries=count,
                validation_only=not allow_tracking, model_loaded=False, forward_calls=0)
