"""Identity-bound body-joint estimates from pose and point tracks; raw surface tracks stay separate.

This is model fusion, not a better CoTracker visibility score or verified anatomy. No interpolation,
extra neural call or label access. Hair, fingertips, unnamed surfaces and seed anchors stay raw.
"""
import copy
import math
from image_lens import paths

PARAMS = dict(pose_score=.5, shoulder_score=.3, recovery_shoulder_widths=.25, temporal_frames=3)
BODY = dict(shoulder_l=5, shoulder_r=6, elbow_l=7, elbow_r=8, wrist_l=9, wrist_r=10,
            hip_l=11, hip_r=12, knee_l=13, knee_r=14, ankle_l=15, ankle_r=16)
GROUPS = ('arms', 'legs')


def track_path(stem, group):
    """Raw inference destination; selected body estimates use the ordinary .tracks.json name."""
    return stem.with_suffix('.raw-tracks.json' if group in GROUPS else '.tracks.json')



def _resolve_excursions(selected, raw, samples, window, inside):
    """One-frame excursions may use actual raw xy. Longer disputes remain positional
    estimates, with motion unknown: stuck raw tracks and genuine return gestures are ambiguous.
    Both cases require compact native raw support and two agreeing confident pose flanks.
    """
    from image_lens.body_motion import BODY_MOTION
    visible_sources = {}; selected_rows = {}; raw_rows = {}
    for frame in selected['frames']:
        n = frame['frame']
        if window['seed'] < n <= window['end']:
            for row in frame['points']:
                if row['visible'] and row['status'] != 'anchor_input_not_prediction':
                    pid = row['point_id']
                    visible_sources.setdefault(pid, set()).add(row.get('coordinate_source', 'point_tracker'))
                    selected_rows[n, pid] = row
    for frame in raw['frames']:
        for row in frame['points']:
            raw_rows[frame['frame'], row['point_id']] = row
    corrected = conflicts = spans = 0
    for pid, sources in visible_sources.items():
        index = BODY.get(pid.replace('-', '_'))
        if index is None or sources != {'pose'}:
            continue
        valid = {}
        for n in range(window['seed']+1, window['end']+1):
            row = selected_rows.get((n,pid)); observation = raw_rows.get((n,pid)); pose = samples.get(n)
            if (row is None or observation is None or not observation['visible']
                    or observation['status'] == 'anchor_input_not_prediction' or not inside(observation.get('xy'))
                    or pose is None or pose['scores'][index] < PARAMS['pose_score']
                    or min(pose['scores'][5:7]) < PARAMS['shoulder_score']
                    or not all(inside(pose['keypoints'][i]) for i in (index,5,6))):
                continue
            width = math.dist(pose['keypoints'][5],pose['keypoints'][6])
            if width > 0:
                valid[n] = row, observation, pose, width
        def support(ns):
            a,b = ns[0]-1,ns[-1]+1
            if a not in valid or b not in valid:
                return None
            context = [valid[n] for n in range(a,b+1)]
            width = min(v[3] for v in context)
            resolution = max(BODY_MOTION['min_px'],BODY_MOTION['min_ratio']*width)
            points = [v[1]['xy'] for v in context]+[valid[n][2]['keypoints'][index] for n in (a,b)]
            if any(math.dist(x,y) > resolution for i,x in enumerate(points) for y in points[i+1:]):
                return None
            return a,b,width,resolution
        # Preserve the original isolated decision before classifying longer runs.
        # Neighbouring shoulder widths must not absorb an otherwise repairable frame.
        for n, (row, observation, _, _) in valid.items():
            bounded = support([n])
            if bounded is None:
                continue
            a,b,width,resolution = bounded
            if math.dist(row['xy'],observation['xy']) <= PARAMS['recovery_shoulder_widths']*width:
                continue
            evidence = dict(rejected_xy=list(row['xy']),rejected_pose_score=row.pop('pose_score',None),
                            support_frames=[a,b],agreement_px=round(resolution,3))
            row.update(xy=list(observation['xy']),coordinate_source='point_tracker',visibility_source='point_tracker',
                       status='pose_excursion_corrected_raw_unverified',claim_scope=observation['claim_scope'],motion_pair_usable=False,
                       motion_geometry_usable=False,motion_geometry_exclusion='isolated_pose_excursion',pose_excursion=evidence)
            corrected += 1
        runs = []
        for n, (row, observation, _, width) in valid.items():
            if (row.get('coordinate_source') == 'pose'
                    and math.dist(row['xy'],observation['xy']) > PARAMS['recovery_shoulder_widths']*width):
                if not runs or n != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(n)
        for ns in runs:
            if len(ns) < 2:
                continue
            bounded = support(ns)
            if bounded is None:
                continue
            a,b,_,resolution = bounded
            evidence = dict(frames=[ns[0],ns[-1]],support_frames=[a,b],agreement_px=round(resolution,3),
                            interpretation='pose_or_stuck_raw_unresolved')
            for n in ns:
                valid[n][0].update(motion_geometry_usable=False,motion_geometry_exclusion='pose_tracker_conflict',
                                   pose_tracker_conflict=evidence)
            conflicts += len(ns); spans += 1
    return corrected, conflicts, spans


def reconcile(packet, pose_frames, track_id, window, raw_path, pose_path):
    """Keep every raw visible prediction. A confident same-ID pose supplies a joint location; an
    invisible raw candidate requires in-frame pose/tracker agreement relative to shoulders, or three
    adjacent confident same-ID poses with consistent positions. An ID miss stays a miss; never substitute the largest/nearest other person. Cuts use raw tracks.
    """
    result = copy.deepcopy(packet)
    width, height = packet['job']['width'], packet['job']['height']
    def inside(xy):
        return (type(xy) is list and len(xy) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in xy)
                and 0 <= xy[0] < width and 0 <= xy[1] < height)
    counts = dict(pose_coordinates=0, recovered_predictions=0, raw_visible_predictions=0)
    for frame in result['frames']:
        n = frame['frame']
        if not window['seed'] < n <= window['end']:
            continue
        person = next((t for t in pose_frames.get(n, []) if t['track_id'] == track_id), None)
        for row in frame['points']:
            index = BODY.get(row['point_id'].replace('-', '_'))
            if index is None or row['status'] == 'anchor_input_not_prediction':
                continue
            counts['raw_visible_predictions'] += bool(row['visible'])
            row['coordinate_source'] = 'point_tracker'
            row['visibility_source'] = 'point_tracker'
            if person is None or not inside(row.get('xy')):
                continue
            xy, scores = person['keypoints'][index], person['scores']
            if scores[index] < PARAMS['pose_score'] or not inside(xy):
                continue
            recover = False
            if not row['visible']:
                shoulder = math.dist(person['keypoints'][5], person['keypoints'][6])
                recover = (min(scores[5], scores[6]) >= PARAMS['shoulder_score'] and shoulder > 0
                           and math.dist(row['xy'], xy) <= PARAMS['recovery_shoulder_widths']*shoulder)
                if not recover:
                    continue
            row.update(xy=list(xy), visible=True, in_frame=True, status='pose_assisted_joint_unverified',
                       claim_scope='model_joint_2d', coordinate_source='pose', pose_score=scores[index],
                       visibility_source='pose_tracker_agreement' if recover else 'point_tracker',
                       motion_pair_usable=False)
            # model_visible remains the original CoTracker boolean, including false for recovered points.
            counts['pose_coordinates'] += 1
            counts['recovered_predictions'] += recover
    # A recovery must not turn an old raw-only point into a mixed-source window: the
    # motion consumer would discard the added poses through its whole-joint raw fallback.
    sources = {}
    for frame in result['frames']:
        if window['seed'] < frame['frame'] <= window['end']:
            for row in frame['points']:
                if row['visible'] and row['status'] != 'anchor_input_not_prediction':
                    sources.setdefault(row['point_id'], set()).add(row.get('coordinate_source', 'point_tracker'))
    samples = {n: next((p for p in pose_frames.get(n, []) if p['track_id'] == track_id), None)
               for n in range(window['seed']+1, window['end']+1)}
    counts['temporal_recoveries'] = 0
    for frame in result['frames']:
        n = frame['frame']
        adjacent = [samples.get(f) for f in (n-1,n,n+1)]
        if not all(adjacent):
            continue
        for row in frame['points']:
            index = BODY.get(row['point_id'].replace('-', '_'))
            if (index is None or row['visible'] or row['status'] == 'anchor_input_not_prediction'
                    or sources.get(row['point_id']) == {'point_tracker'}):
                continue
            if any(p['scores'][index] < PARAMS['pose_score'] or min(p['scores'][5:7]) < PARAMS['shoulder_score']
                   or not all(inside(p['keypoints'][i]) for i in (index,5,6)) for p in adjacent):
                continue
            limit = PARAMS['recovery_shoulder_widths'] * min(math.dist(p['keypoints'][5],p['keypoints'][6]) for p in adjacent)
            points = [p['keypoints'][index] for p in adjacent]
            if limit <= 0 or any(math.dist(a,b) > limit for a,b in zip(points,points[1:])):
                continue
            row.update(xy=list(points[1]), visible=True, in_frame=True, status='pose_assisted_joint_unverified',
                       claim_scope='model_joint_2d', coordinate_source='pose', pose_score=adjacent[1]['scores'][index],
                       visibility_source='pose_temporal_consistency', motion_pair_usable=False)
            counts['pose_coordinates'] += 1
            counts['recovered_predictions'] += 1
            counts['temporal_recoveries'] += 1
    counts['pose_excursion_corrections'], counts['pose_conflict_frames'], counts['pose_conflict_spans'] = _resolve_excursions(result, packet, samples, window, inside)
    counts['pose_coordinates'] -= counts['pose_excursion_corrections']
    result.update(schema='image-lens.joint-tracks/1', observation_scope='joint_and_surface_estimates_2d',
                  joint_tracking=dict(params=PARAMS, track_id=track_id, window=window, counts=counts,
                                      raw_tracks=dict(path=str(raw_path), sha256=paths.sha(raw_path)),
                                      pose=dict(path=str(pose_path), sha256=paths.sha(pose_path)),
                                      module_sha256=paths.sha(__file__)))
    result['limits'] += ' Selected body joints are pose-assisted estimates; raw CoTracker files remain separate. Temporal support is not verified visibility. Not surface timing, GT, or proof of visibility/anatomy.'
    return result
