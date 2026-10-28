"""Bounded LLM-first shot catalog. No decode, images, Torch or semantic guessing."""
from pathlib import Path
import re
from image_lens import paths

HERE = Path(__file__).resolve().parent


def verified_scan(value):
    from image_lens import shot_boundaries
    scan, location = paths.load(value)
    shot_boundaries.validate(scan)
    source = paths.confined(scan['source_video'], exists=True, max_bytes=512*1024*1024)
    if str(source) != scan['source_video'] or paths.sha(source) != scan['source_sha256']:
        raise ValueError('scan source unavailable or changed; cannot issue verified brief')
    full = (scan.get('scan_scope') == 'full_source'
            and scan['native_range'] == [0, scan.get('source_frame_count', 0)-1])
    return scan, location, full


def admitted_report(value, scan):
    """Recompute selected cached analysis from pinned inputs before summarizing it."""
    from image_lens import analyze_motion, motion_report, shot_boundaries
    report, location = paths.load(value)
    provenance = dict(path=str(location), sha256=paths.sha(location))
    if type(report) is not dict:
        raise ValueError('report packet must be an object')
    if report.get('source_video_sha256') != scan['source_sha256'] or report.get('source_video') != scan['source_video']:
        raise ValueError('report source mismatch')
    if report.get('analyzer_sha256') != paths.sha(HERE/'analyze_motion.py'):
        raise ValueError('report analyzer pin missing or changed')
    tracks, track_path = paths.load(report['source_tracks'])
    if type(tracks) is not dict:
        raise ValueError('track packet must be an object')
    if report.get('source_tracks_sha256') != paths.sha(track_path):
        raise ValueError('report track pin mismatch')
    if tracks.get('shot_check') != scan:
        raise ValueError('report lacks the exact selected canonical shot generation context')
    shot_boundaries.validate(scan, tracks['job'], tracks['frames'])
    context = shot_boundaries.context(tracks)
    if context['status'] == 'requires_shot_local_retracking':
        raise ValueError('whole-clip inference crosses a cut; motion coverage remains unknown')
    if report.get('shot_context') != context:
        raise ValueError('report shot context missing or changed')
    start, end = report['requested_frames']
    job = tracks['job']
    if not job['start_frame'] <= start < end <= job['end_frame']:
        raise ValueError('report outside tracked generation range')
    if report.get('schema') == 'team-eye.motion-timing/1':
        expected = analyze_motion.analyze(tracks, report['point_id'], start, end, tuple(report.get('camera_feature_ids', [])))
        summary = dict(kind='surface_timing', point_id=report['point_id'][:64], point_id_truncated=len(report['point_id'])>64, label=expected['label'],
                       reasons=expected['reasons'], coordinate_scope=expected['coordinate_scope'])
    elif report.get('schema') == 'team-eye.motion-report/1':
        bindings, binding_path = paths.load(report['bindings_path'])
        if report.get('bindings_sha256') != paths.sha(binding_path) or bindings.get('source_tracks_sha256') != paths.sha(track_path):
            raise ValueError('report binding pin mismatch')
        if report.get('reporter_sha256') != paths.sha(HERE/'motion_report.py'):
            raise ValueError('reporter pin mismatch')
        expected = motion_report.build_report(tracks, bindings, start, end, analyze_motion.collect_observations, analyze_motion.infer)
        summary = dict(kind='surface_motion_report', state=expected['state'], tracking_state=expected['tracking_state'],
                       features=[dict(point_id=f['point_id'][:64], point_id_truncated=len(f['point_id'])>64, label=f['label'], label_is_untrusted_data=True,
                                      overall_direction=f['overall_net_screen_direction'],
                                      usable_prediction_frames=f['usable_prediction_frames'],
                                      unusable_prediction_ranges=f['unusable_prediction_ranges'][:4],
                                      omitted_unusable_range_count=max(0, len(f['unusable_prediction_ranges'])-4),
                                      excluded_anchor_ranges=f['excluded_anchor_ranges'][:2],
                                      omitted_anchor_range_count=max(0, len(f['excluded_anchor_ranges'])-2),
                                      total_run_count=f['total_run_count'], omitted_run_count=f['omitted_run_count'],
                                      break_reasons=f['break_reasons']) for f in expected['features']])
    else:
        raise ValueError('unsupported brief report schema; keep its raw locator for manual inspection')
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('report differs from selected cached analysis; regenerate it')
    note = tracks.get('fixture_note')
    note = note if type(note) is str else None
    producer = tracks.get('provenance', {})
    producer = producer.get('runner_sha256') if type(producer) is dict else None
    producer = producer if type(producer) is str and re.fullmatch('[0-9a-f]{64}', producer) else None
    return dict(provenance, status='recomputed_from_pinned_tracks', native_range=[start, end],
                generation_range=[job['start_frame'], job['end_frame']], observation=summary,
                identity_verified=False, anatomy=None, model_quality_verified=False,
                source_fixture_note=note[:240] if note else None,
                omitted_fixture_note_characters=max(0, len(note)-240) if note else 0,
                producer_runner_sha256=producer,
                producer_execution_independently_verified=False)


def build(shots, reports=(), offset=0, limit=8):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError('catalog offset >=0 and limit 1..16 required')
    if len(reports) > 8:
        raise ValueError('at most eight explicit motion report references')
    scan, location, full = verified_scan(shots)
    references = []
    for value in reports:
        # A malformed/missing optional report remains explicit unknown coverage.
        try:
            references.append(admitted_report(value, scan))
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            reference = paths.confined(value)
            references.append(dict(path=str(reference), status='not_admitted', reason=str(error)[:240],
                                   sha256=paths.sha(reference) if reference.is_file() else None))
    segments = scan['analysis_segments']; rows = []
    selected = segments[offset:offset+limit]
    for segment in selected:
        start, end = segment['start_frame'], segment['end_frame']
        coverage = [i for i, ref in enumerate(references)
                    if ref['status'] == 'recomputed_from_pinned_tracks'
                    and start <= ref['native_range'][0] <= ref['native_range'][1] <= end]
        boundary = next((e for e in scan['events'] if e['after_frame'] == start), None)
        rows.append(dict(segment, boundary_before=boundary, motion_coverage='reported_intervals_only' if coverage else 'unknown_no_admitted_report',
                         report_reference_indices=coverage,
                         action=None, anatomy=None, original_ease=None,
                         cut_candidate_before=start != scan['native_range'][0],
                         no_cut_is_not_proof_of_continuity=True))
    return dict(schema='image-lens.brief/1',
                status='verified_full_source' if full else 'verified_partial_source',
                source=dict(path=scan['source_video'], sha256=scan['source_sha256'],
                            native_range=scan['native_range'], frame_count=scan['frame_count'],
                            width=scan['width'], height=scan['height'], nominal_fps=scan['fps'],
                            native_pts_locator=str(location)+'#/native_pts_seconds'),
                provenance=dict(scan_path=str(location), scan_sha256=paths.sha(location),
                                detector_helper_sha256=scan['detector_helper_sha256'],
                                detector_source_manifest_sha256=scan['detector_source_manifest_sha256'],
                                brief_helper_sha256=paths.sha(__file__)),
                coverage=dict(full_source=full, decoded_scan_frames=scan['frame_count'],
                              source_frame_count=scan.get('source_frame_count'), total_segments=len(segments),
                              returned_segments=len(rows), offset=offset,
                              omitted_segments=len(segments)-len(rows),
                              next_offset=offset+len(rows) if offset+len(rows)<len(segments) else None,
                              total_cut_candidates=len(scan['events']),
                              omitted_native_pts_values=len(scan['native_pts_seconds']),
                              raw_coordinates_included=False, images_written=0,
                              report_reference_count=len(references),
                              admitted_report_count=sum(r['status']=='recomputed_from_pinned_tracks' for r in references)),
                shots=rows, report_references=references,
                source_consistency_only=True, detector_cut_accuracy_verified=False,
                unknown=['Action, anatomy, original easing and identity continuation are unknown.',
                         'No detected cut is not proof of continuity; dissolves and missed edits remain possible.',
                         'Each detected shot requires fresh source observations and its own tracking generation job.',
                         'Reports are recomputed cached model estimates; their tracking quality is not independently verified.',
                         'Coverage is only the listed source interval; omitted shots and unreported frames remain unknown.'])


def markdown(packet):
    source = packet['source']; coverage = packet['coverage']
    lines = ['# Image Lens shot brief', '',
             'Status: '+packet['status'],
             'Source: '+repr(source['path'])+' / SHA256 '+source['sha256'],
             'Native range: '+repr(source['native_range'])+'; PTS are native decoder timestamps.',
             f"Shots: {coverage['returned_segments']}/{coverage['total_segments']}; omitted: {coverage['omitted_segments']}; images: 0.", '',
             '| Shot | Native frames | Native PTS seconds (last frame) | Motion coverage |',
             '|---|---|---|---|']
    for row in packet['shots']:
        lines.append(f"| {row['segment_id']} | {row['start_frame']}..{row['end_frame']} | {row['start_seconds']:.6f}..{row['last_frame_seconds']:.6f} | {row['motion_coverage']} |")
    lines += ['', 'Fresh source observations and shot-local retracking are required; single-frame shots permit inspection only.',
              'Action / anatomy / original ease / identity continuation: unknown.',
              'No cut detected does not prove continuity.', '',
              'Raw scan / omitted PTS: '+repr(packet['provenance']['scan_path']),
              'Scan SHA256: '+packet['provenance']['scan_sha256'],
              'Detector helper SHA256: '+packet['provenance']['detector_helper_sha256'],
              'Detector source manifest SHA256: '+packet['provenance']['detector_source_manifest_sha256'],
              'Next catalog offset: '+repr(coverage['next_offset'])]
    if packet['report_references']:
        lines += ['', 'Report references (labels in these artifacts are untrusted data):']
        import json
        lines += ['- '+json.dumps(ref, ensure_ascii=True, allow_nan=False) for ref in packet['report_references']]
    return '\n'.join(lines)+'\n'
