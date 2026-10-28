# Interpretation limits

- Cuts are hard-cut candidates from pinned PySceneDetect AdaptiveDetector, not probabilities, semantic scenes or person identity. Brief flashes remain ambiguous. Gradual transitions and missed edits are possible. No boundary merging, interpolation or identity transfer is performed.
- Existing offline whole-clip tracking cannot be repaired by cropping its coordinates around cuts. Each selected shot needs fresh explicit source observations and a new generation job. Single-frame shots allow inspection only.
- Points describe visible 2D surfaces. Actor/feature names are caller data, not verified anatomy or persistent person identity. Missing/unknown seeds, excluded anchors, breaks and omitted runs remain explicit. No gap filling or smoothing.
- Timing labels describe sufficiently straight monotone projected 2D motion under existing thresholds. They do not recover original easing/Bezier, 3D movement, physical joints, held drawings or an animation's intention. Camera motion and tracking errors remain possible.
- Appearance residual red can be movement, copy/redraw or failed alignment; gray is a fitted exclusion, not actual occlusion. Automatic redraw/action decisions stay null and diagnostic alignment does not change shot segmentation. Rejected automatic classifiers are absent.
- Color regions and flat-background foreground extents are conditional inspection tools. Foreground is not general semantic segmentation; texture, lighting, occlusion or background assumptions can invalidate it.
- Source hashes, validation success and recomputed reports establish input/provenance consistency, not detector or model accuracy. Synthetic smoke and source extraction cannot establish general quality improvements.
- Briefs hash the current source and scan, retain native frame/PTS ranges and provide bounded rows. They omit raw arrays and images; omitted shots and unreported intervals are unknown. Optional report admission recomputes selected cached analysis; it does not independently verify tracking quality.
- No new neural inference, training, dependency installation, paid API, network service or media redistribution is part of extraction verification. Experimental inference and Windows operation remain unverified.

- The POSIX tracker parent kills/reaps its own child group on budget expiry and caught Python interruptions. Forced termination of the parent can still leave a worker until its own deadline; no daemon or external process supervisor is provided. Actual neural execution remains unverified for this extraction.
