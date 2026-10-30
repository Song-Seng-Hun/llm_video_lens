"""Identity-bound body-joint estimates from pose and point tracks; raw surface tracks stay separate.

This is model fusion, not a better CoTracker visibility score or verified anatomy. No interpolation,
extra neural call or label access. Hair, fingertips, unnamed surfaces and seed anchors stay raw.
"""
import copy
import math
from image_lens import paths

PARAMS = dict(pose_score=.5, shoulder_score=.3, recovery_shoulder_widths=.25)
BODY = dict(shoulder_l=5, shoulder_r=6, elbow_l=7, elbow_r=8, wrist_l=9, wrist_r=10,
            hip_l=11, hip_r=12, knee_l=13, knee_r=14, ankle_l=15, ankle_r=16)
GROUPS = ('arms', 'legs')


def track_path(stem, group):
    """Raw inference destination; selected body estimates use the ordinary .tracks.json name."""
    return stem.with_suffix('.raw-tracks.json' if group in GROUPS else '.tracks.json')


def reconcile(packet, pose_frames, track_id, window, raw_path, pose_path):
    """Keep every raw visible prediction. A confident same-ID pose supplies a joint location; an
    invisible raw candidate is admitted only with in-frame pose/tracker agreement relative to shoulders.
    An ID miss stays a miss; never substitute the largest/nearest other person. Cuts use raw tracks.
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
    result.update(schema='image-lens.joint-tracks/1', observation_scope='joint_and_surface_estimates_2d',
                  joint_tracking=dict(params=PARAMS, track_id=track_id, window=window, counts=counts,
                                      raw_tracks=dict(path=str(raw_path), sha256=paths.sha(raw_path)),
                                      pose=dict(path=str(pose_path), sha256=paths.sha(pose_path)),
                                      module_sha256=paths.sha(__file__)))
    result['limits'] += ' Selected body joints are pose-assisted estimates; raw CoTracker files remain separate. Not surface timing, GT, or proof of visibility/anatomy.'
    return result
