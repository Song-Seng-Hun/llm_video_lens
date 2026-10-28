"""Explicit visible-surface queries. No anatomical inference or coordinate smoothing."""
import math
import hashlib
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def compile_observations(job):
    """Validate the observed-target contract and flatten independent anchor queries."""
    start, end = job['start_frame'], job['end_frame']
    width, height = job['width'], job['height']
    if job.get('observation_scope') != 'visible_surface_2d' or job.get('anatomical_claims') is not False:
        raise ValueError('explicit visible-surface scope required')
    if job.get('mode') not in ('single-anchor', 'bounded-reanchor'):
        raise ValueError('unknown observation mode')
    points = job['points']
    if not 1 <= len(points) <= 12 or len({p['point_id'] for p in points}) != len(points):
        raise ValueError('duplicate or invalid observed targets')
    queries, mapping = [], []
    for point in points:
        if (not isinstance(point.get('observed_label'), str) or not point['observed_label'].strip()
                or point.get('anatomical_interpretation') is not None):
            raise ValueError('surface label required; anatomy must remain unknown')
        identity = point.get('visual_reference')
        if (not isinstance(identity, dict) or identity.get('feature_id') != point['point_id']
                or identity.get('frame') != start
                or identity.get('kind') != 'marked-source-crop'
                or not isinstance(identity.get('sha256'), str) or len(identity['sha256']) != 64
                or not isinstance(identity.get('path'), str)):
            raise ValueError('text-only target is ambiguous; pinned marked source crop required')
        anchors = point['anchors']
        maximum = 1 if job['mode'] == 'single-anchor' else 2
        if not 1 <= len(anchors) <= maximum:
            raise ValueError('anchor count exceeds explicit confirmation budget')
        frames = [a['frame'] for a in anchors]
        if (any(type(f) is not int or not start <= f <= end for f in frames)
                or frames != sorted(set(frames)) or frames[0] != start):
            raise ValueError('anchors must be ordered/unique and start at first frame')
        entries = []
        for index, anchor in enumerate(anchors):
            xy = anchor['xy']
            if xy is None:
                if anchor.get('status') != 'unknown':
                    raise ValueError('missing observation must say unknown')
                entries.append({'anchor_index': index, 'frame': anchor['frame'], 'query_index': None})
                continue
            if (anchor.get('status') != 'source_image_observation'
                    or not isinstance(xy, list) or len(xy) != 2 or not all(finite(v) for v in xy)
                    or not 0 <= xy[0] < width or not 0 <= xy[1] < height
                    or not finite(anchor.get('uncertainty_px')) or anchor['uncertainty_px'] <= 0):
                raise ValueError('invalid observed point or uncertainty')
            entries.append({'anchor_index': index, 'frame': anchor['frame'], 'query_index': len(queries)})
            queries.append([anchor['frame'] - start, float(xy[0]), float(xy[1])])
        mapping.append(entries)
    if not queries or len(queries) > 24:
        raise ValueError('no identifiable observations or query cap exceeded')
    return queries, mapping


def verify_visual_references(job, task_root):
    """Verify frozen identity evidence; this does not certify the semantic choice."""
    root = Path(task_root).resolve()
    checked = []
    for point in job['points']:
        identity = point['visual_reference']
        path = Path(identity['path'])
        resolved = path.resolve()
        if (path.is_symlink() or not resolved.is_relative_to(root)
                or not resolved.is_file() or resolved.stat().st_size > 16 * 1024 * 1024):
            raise ValueError('visual reference must be a task-local regular PNG')
        data = resolved.read_bytes()
        if (not data.startswith(b'\x89PNG\r\n\x1a\n')
                or hashlib.sha256(data).hexdigest() != identity['sha256']):
            raise ValueError('visual reference pin mismatch')
        checked.append({'feature_id': point['point_id'], 'path': str(resolved),
                        'sha256': identity['sha256'], 'source_frame': identity['frame']})
    return checked


def observed_frame(points, mapping, coords, vis, frame, relative_frame, width, height, previous=None):
    """Piecewise selection, not blending. A fresh anchor breaks motion derivatives."""
    rows = []
    previous = {} if previous is None else {r['point_id']: r for r in previous}
    for point, anchors in zip(points, mapping):
        eligible = [a for a in anchors if a['frame'] <= frame]
        active = eligible[-1] if eligible else None
        qi = None if active is None else active['query_index']
        xy = None if qi is None else [float(v) for v in coords[relative_frame, qi]]
        model_visible = None if qi is None else bool(vis[relative_frame, qi])
        in_frame = xy is not None and 0 <= xy[0] < width and 0 <= xy[1] < height
        is_anchor = active is not None and active['frame'] == frame
        if qi is None:
            state = 'unknown_anchor'
        elif is_anchor:
            state = 'anchor_input_not_prediction'
        elif not in_frame:
            state = 'outside_frame'
        elif not model_visible:
            state = 'model_not_visible'
        else:
            state = 'tracked_unverified'
        prior = previous.get(point['point_id'])
        pair_ok = bool(prior and prior['visible'] and prior['query_index'] == qi
                       and qi is not None and not is_anchor and model_visible and in_frame)
        rows.append({'point_id': point['point_id'], 'observed_label': point['observed_label'],
                     'anatomical_interpretation': None, 'claim_scope': 'visible_surface_2d',
                     'xy': xy, 'model_visible': model_visible, 'in_frame': bool(in_frame),
                     'visible': bool(model_visible and in_frame), 'status': state,
                     'anchor_index': None if active is None else active['anchor_index'],
                     'anchor_frame': None if active is None else active['frame'],
                     'query_index': qi, 'tracking_metric_eligible': qi is not None and not is_anchor,
                     'motion_pair_usable': pair_ok,
                     'semantic_identity_verified_by_model': False})
    return rows
