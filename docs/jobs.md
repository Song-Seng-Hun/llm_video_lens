# Fresh source-bound jobs

Copy a video into the selected workspace and run `scan`. Build `queries.json` using the scan's exact canonical source path, SHA256, native size and nominal fps. A minimal explicit seed packet is:

```json
{
  "schema": "team-eye.queries/1",
  "jobs": [{
    "name": "shot0",
    "source_video": "/path/to/workspace/input.mp4",
    "source_sha256": "COPY_THE_64_CHARACTER_SOURCE_HASH_FROM_SHOTS_JSON",
    "width": 320,
    "height": 240,
    "fps": 24.0,
    "start_frame": 0,
    "end_frame": 15,
    "seed_frame": 0,
    "points": [{"point_id": "surface0", "xy": [100.0, 80.0]},
               {"point_id": "unknown0", "xy": null}]
  }]
}
```

Choose real source coordinates explicitly; these values are schema illustrations, not automatically identified targets. Point IDs are bounded ASCII data. A null seed remains missing. A job may not cross a detected boundary. At most 160 native frames, 24 queries, 2,073,600 source pixels; CPU offline CoTracker3, no backward tracking, feature chunks 16. Supported internal model resolutions are 384x512 (default) and 576x768 (explicit `inference_resolution`). Native decoder PTS must agree with constant fps within 2ms. VFR tracking is unverified and may be refused.

`prepare --queries queries.json --shots shots.json --job shot0 --out job.freeze.json` binds this job, file hashes, selected helper/vendor manifests and wall budget. The canonical scan must cover the full source. Preparation is stdlib-only, but actual `validate` decodes source frames and requires core decode dependencies. Validation needs no model checkpoint and never imports Torch. Its `--out` is checked for containment and no-overwrite, but successful validation prints a receipt and does not write tracks.

Basic `team-eye.queries/1` produces point tracks without the richer visible-surface observation contract. Conservative motion analysis requires `team-eye.observations/1`: each job additionally declares `observation_scope: "visible_surface_2d"`, `anatomical_claims: false`, `mode: "single-anchor"` or `"bounded-reanchor"`, and `common_metric_excluded_frames` equal to the union of all anchor frames. Each point supplies:

```json
{
  "point_id": "surface0",
  "observed_label": "visible blue patch",
  "anatomical_interpretation": null,
  "visual_reference": {
    "feature_id": "surface0", "frame": 0,
    "kind": "marked-source-crop", "path": "/path/to/workspace/marked-source.png",
    "sha256": "COPY_THE_MARKED_SOURCE_CROP_HASH"
  },
  "anchors": [{"frame": 0, "xy": [100.0, 80.0],
               "status": "source_image_observation", "uncertainty_px": 2.0}]
}
```

Visible-surface mode permits at most 12 points and 24 flattened anchor queries. Each feature starts at the job's first frame; optional second anchors reset identity and break motion derivatives. A missing anchor is `xy: null, status: "unknown"`. The marked source PNG must exist within the workspace and match its hash; the hash does not certify that a feature label or coordinate is semantically correct. These are source observations, never anatomical ground truth.

Only `prepare --allow-tracking` permits inference, with an explicit local checkpoint. The accepted checkpoint SHA256 is `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`; its pinned external acquisition URL and noncommercial license are in `image_lens/checkpoint-manifest.json`. No download is automatic. A `.attempt.json` beside the freeze is created before Torch import; the same freeze cannot make a second attempt, including after inference/import failure. Prepare another consciously reviewed job/freeze to authorize another attempt. Validation never consumes this attempt. All paths are workspace-contained except the explicitly supplied checkpoint.
