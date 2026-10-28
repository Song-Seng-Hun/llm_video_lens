# Extraction verification

Verified on 2026-10-29 KST using the existing CPython 3.12 macOS arm64 research runtime. No dependency/model installation, neural inference, training or network request was performed.

## Focused standard-library checks

```sh
python -B -m unittest discover -s tests -v
```

20 tests passed. They cover the actual isolated direct CLI, workspace escape and symlink refusal, no-overwrite, duplicate/nonfinite JSON, source/helper mutation, incomplete versus partial/full scans, native PTS and pagination, single-frame/unknown/cut semantics, fresh freezes and cross-cut refusal, no-Torch preparation, selected cached-analysis recomputation and forged-label refusal, missing observations, and cross-cut generation remaining blocked after an interval crop request. The actual track child refuses a validation-only freeze before any neural import. A real sleeping child is killed and reaped by the hard parent watchdog on both timeout and caller interruption; this is process-control evidence, not a model-run result.

Authored track packets used by contract tests are explicitly labeled fixtures. Their deterministic coordinates test reporting contracts only; they do not establish tracking/timing accuracy.

## Core runtime smoke

With the existing core dependencies available, from the repository:

```sh
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

The standalone smoke creates a new 48-frame 320x240/24fps video with authored boundaries at native frames 16 and 32. Six real CLI steps passed: scan, JSON brief, paginated Markdown catalog, explicit comparison, fresh prepare, and validate. The scan detected those two candidates. The brief returned three shot rows and zero images. Comparison wrote two original panels plus one explicitly unverified residual panel; automatic action/redraw decisions remained null and segmentation remained disabled. The saved comparison contact was visually inspected for its three panels and warnings.

A Torch import blocker remained active across all six steps. Torch was absent from loaded modules, neural forward calls were zero, validation required no checkpoint, and validation produced no tracks. The earlier direct-launcher smoke also ran its CLI subprocesses from a different working directory under isolated Python. `python -m image_lens --help` passed separately.

Local-only evidence is saved under Git-ignored `workspace/final-smoke/smoke-results.json`, `workspace/final-smoke/source-verification.json`, and the earlier `workspace/smoke` artifacts. They are not part of the source release. New smoke directories are required; previous artifacts are never overwritten.

## Source and failure coverage

77 Python source files parsed successfully without writing bytecode. All 28 PySceneDetect files and all 33 CoTracker files matched their existing pins, using the documented patched hash for restricted checkpoint loading. Vendor directories contained exactly those manifested file sets. All eight original authored source hashes still matched the import record, confirming historical inputs were preserved.

An initial `compileall` attempt failed because the host's configured cache directory was outside the sandbox; it did not establish a source failure. AST parsing supplied the syntax check without cache writes. An initial mistaken test-file path produced zero discovered tests and was not counted; corrected discovery ran the full 19 checks. A shell hook rejected harmless ignore-file patterns; those patterns were omitted. No secret files were accessed.

## Unverified boundaries

Actual CoTracker inference, inference numerical behavior and model quality were not run. The hard watchdog was tested on a sleeping process, not a model. Windows core execution, optional tracking on non-POSIX systems, installed-wheel runtime, and broad real-world detector/foreground robustness remain unverified. Optional tracking is experimental and explicitly refuses a non-POSIX parent watchdog. No performance/pose/action/easing accuracy improvement is claimed. Source-consistency status does not independently verify cut accuracy or semantic identity.

## Root integration acceptance

The integrator independently ran nine CLI checks from a different working directory and an explicit workspace outside the checkout, with neural imports blocked. The authored 48-frame control retained native timestamps and yielded cut candidates 16/32; pagination, zero default images, explicit three-panel comparison, prepare/validate, missing-report handling, overwrite refusal and path-escape refusal passed. The comparison was visually inspected. This is portability/contract evidence, not a tracking or cut-accuracy benchmark.

One requested no-tool Claude review returned a provider execution error and no answer; it is not counted as completed review. Imported vendor code, optional neural execution, installed-wheel execution and Windows remain outside that review attempt.

A separate native reviewer read the six new integration modules, independently verified all 61 vendor pins and checked package-data inclusion. No concrete code regression was found in that scope. Its manifest-staleness finding was resolved by regenerating the release manifest after the final documented changes. This static review did not execute neural models, build/install a wheel, or verify Windows.
