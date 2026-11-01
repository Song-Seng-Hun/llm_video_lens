# Local person detector selection

The existing `--detector` argument on `pose` and `look`, or `models.json`'s `detector` value, selects an explicit pinned local model. HumanArt remains the default configured recipe. Nothing downloads models or guesses an input style.

| Selection | Local argument | Recipe |
| --- | --- | --- |
| HumanArt | existing pinned `.onnx` file | Existing fast HumanArt/CoreML path, unchanged |
| DeepGHS illustration detector | pinned `.onnx` file | RGB, PIL bicubic stretch640², /255 CHW, confidence >.3, inclusive pixel-area NMS .5, native resize/round/clip; CPU four threads |
| SAM3 + DeepGHS coverage | bundle directory, or its `model.safetensors` | SAM3 prompt `person`, resolution1008, score >.5, native clipped boxes; add only DeepGHS proposals having zero positive-area intersection with **every original SAM3 box** |

SAM3 keeps the original image pixels for downstream pose. No mask filter or extra NMS is applied. Supplement proposals do not veto one another after the fixed DeepGHS NMS. This rule recovers disjoint missed subjects; it cannot recover a missed person whose rectangle overlaps a primary person box. The existing RTMW pose estimator, split heuristic, ByteTrack IDs, shot resets and geometry remain the downstream path.

DeepGHS retains at most32 valid boxes per frame after its fixed NMS and native resize/round/clip. If more remain, it keeps the highest-score32; score ties preserve the original proposal order. Each affected native pose frame carries `detector_diagnostics` with `original_count`, `kept_count`, `dropped_count`, the detector role and selection basis; `pose.json` also lists those native frames under `flags.detector_truncations`. In the SAM3 route this records **supplement** truncation before overlap rejection, even when some retained proposals are subsequently rejected. Frames with32 or fewer proposals have no truncation diagnostic. This resource bound may omit people in crowded frames. Truncation alone does not mark a skeleton as unreliable or disqualify a `look` seed; the existing `flags.frames` remains overlap/jump-only.

Install the existing pose dependencies and optional SAM3 dependencies on Apple Silicon:

```sh
pip install -r requirements/pose.txt -r requirements/sam3.txt
python -m image_lens --workspace W pose --shots shots.json --detector /models/sam3-bundle --pose-model /models/rtmw133.onnx --start 0 --end 1 --out pose.json
```

`look` accepts the same detector argument. Its tracking dependencies and checkpoint are also required. Keep normal model paths in `models.json` unless explicitly switching recipes.

The SAM3 bundle must contain these three external files:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `model.safetensors` | 3402867661 | `0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173` |
| `bpe_simple_vocab_16e6.txt.gz` | 1356917 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| `anime-person.onnx` | 44583231 | `6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90` |

All selected file sizes/SHA256 values are checked before heavy model imports. Symlinks, unknown models and incomplete bundles fail explicitly. A standalone SAM3 file resolves the other two required assets beside it. No weights, vocabulary, media or wheels are included in the package.

Asset provenance: SAM3 [checkpoint revision](https://huggingface.co/mlx-community/sam3-image/tree/b72a14d8127e17e6f2a3d2e075bbbf4307ba146e), [MLX source/tokenizer revision](https://github.com/Deekshith-Dade/mlx_sam3/tree/9eacae749049e2b864fe63f90f3dd17c689bee03); DeepGHS [checkpoint revision](https://huggingface.co/deepghs/anime_person_detection/tree/e39c744c22432ad01f91dd254fe2b02c8d878b8c), file `person_detect_v1.3_s/model.onnx`. Obtain assets under their applicable terms. The vendored SAM3 implementation is covered by the separate [Meta custom SAM license](../image_lens/_sam3_mlx/LICENSE), not the enclosing project's MIT license; acknowledge SAM3 in research as required. See [adaptations/provenance](../image_lens/_sam3_mlx/PROVENANCE.md). The publisher declares Python >=3.13; the measured local compatibility variant used Python3.12/MLX0.32.4.

The combined route is an explicit bbox/cost choice: pilot timing was about .65s SAM3 plus .08s DeepGHS per frame on Apple Silicon, substantially slower than HumanArt. Fixed bbox checks improved target-mean IoU from .59346 to .91646 on18 development frames and .60319 to .89609 on9 fresh frames, with fresh hits13/13 versus10/13 and false positives0 versus2. Labels were provisional AI-vision annotations; the small sample and development-driven recipe do not establish broad generalization. These are bbox findings, not evidence of better joints, identity tracking or easing analysis. Pose packets record the actual primary, supplement, tokenizer and recipe for reproducibility.

The development set still had one balloon decoration detected as a person and one small inset person missed because its rectangle overlapped a primary box. Echo/copy detections count visible instances, not independent physical actors or verified identities. Downstream joint accuracy has not been measured for this route.

The combination was frozen after inspecting the18 development frames. SAM3 outputs for the9 additional frames already existed and two object-count log lines had been seen, but their metrics and label comparisons were not inspected before fixing the rule. This was an additional-frame metric check, not a blind held-out evaluation.

Two complete `look` acceptance runs used the original recipe, all shots and a1800s budget each, with no tuning: a39s phone-UI illustration clip (1170frames) took891.7s wall and an18s animated short (433frames) took375.8s; both exited0 with no reported step errors. Frames containing a track changed1033→924 and419→427 respectively against saved HumanArt outputs with the same source bytes and pre-change pose code. Track availability is not ground-truth accuracy. Three inspected illustration frames with clear figures had no new detections; rotated poses and strong colour treatments remain failure cases. The first clip's hair/UI box improved in some windows, while an inverted pose still lost protruding limbs. Largest-box subject selection can also pick a different valid person in a multi-person scene. These checks do not support a universal/default replacement. Historical HumanArt times (69.9s and99.4s) were not rerun as a controlled timing comparison.
