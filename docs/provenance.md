# Provenance and licensing boundaries

The selected authored engines originated in the team's local eye research: diagnostic comparison generation 037, canonical source context 035, shot detector 033 and foreground inspection 027. The historical working directory number was 004; it is not the selected algorithm version. Original authored hashes are recorded in `import-provenance.json`. This extraction changes workspace/CLI/job plumbing and adds the bounded brief, not the detector or timing thresholds. Historical outputs were preserved read-only and were not relabeled with current helper hashes.

The portable engine set is `shot_boundaries`, `appearance_inspection`, `analyze_motion`, `motion_report`, `part_regions`, `foreground_regions`, `observations`, plus experimental `run_point_tracker`. A thin parent launcher replaces the historical task-specific controller's process watchdog. Historical evaluation/renderer utilities, fixed fixtures and all consumed experiment ledgers are not active entrypoints. No rejected 036/037 automatic redraw/action classifier is included.

## PySceneDetect

28 pinned source/license files at version 0.6.7.1 are under `image_lens/vendor/shot-detector`. Their upstream wheel URL, SHA256 and individual hashes are in `shot-detector-pins.json`. The BSD 3-Clause notice is preserved at `scenedetect-0.6.7.1.dist-info/licenses/LICENSE`. The embedded `scenedetect/_thirdparty/simpletable.py` retains its MIT notice. No wheel/archive is bundled.

## CoTracker

33 manifested source/license files are under `image_lens/vendor/source`. The upstream commit is `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; archive URL/hash and original file hashes are in `source-manifest.json`. This source uses **CC-BY-NC 4.0**, preserved in `vendor/source/LICENSE.md`. Respect its noncommercial restriction and attribution requirements. The checkpoint carries the same upstream terms and is not included.

`vendor/local-patch.json` records the sole vendor patch: `torch.load(..., weights_only=True)`. The before/after hashes are preserved. Tracking checks the patched hash rather than silently weakening restricted loading. Checkpoint acquisition source/hash/size/license are in `checkpoint-manifest.json`, with no personal machine path. No automatic model or source download exists.

## Authored code and excluded assets

No blanket redistribution license has been granted to the user's authored code by this extraction. No root MIT/Apache license is asserted. Upstream notices apply to their own files; they do not relicense the rest of the repository. The repository's owner must choose authored-code terms separately if needed.

Source videos, generated crops/contacts (except the README example below), model weights, environments, downloaded archives/wheels, credentials, personal host paths and the research history are excluded from the release. A private repository does not create redistribution rights for those assets. `docs/release-manifest.json` records the new source-only file set; it is not a license clearance or model-quality certificate.

## README example image

`assets/example-onion.jpg` is a `look` onion picture of 15.44–16.32 s of "Video of a Man Dancing" by Antoni Shkraba (https://www.pexels.com/video/video-of-a-man-dancing-7571381/), under the Pexels License (https://www.pexels.com/license/). Seconds 0–20 were cut and scaled to 1280×674 before `look`; the video itself is not in the repository.
