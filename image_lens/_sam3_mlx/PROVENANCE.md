# Private SAM3 MLX image implementation

Source: https://github.com/Deekshith-Dade/mlx_sam3/tree/9eacae749049e2b864fe63f90f3dd17c689bee03
Commit: 9eacae749049e2b864fe63f90f3dd17c689bee03.
The copied core is governed by the adjacent Meta custom SAM LICENSE (2025-11-19), separately from the enclosing project's MIT license. Research using SAM should acknowledge Segment Anything 3 as required by that license. No model weights or tokenizer asset are included.

Adaptations: trailing whitespace normalized (AST unchanged); private import namespace; unused testing Torch/torchvision imports removed; tokenizer uses local stdlib open instead of iopath; profiling stdout removed; converter, hub download and application/UI code omitted; explicit local paths required; permissive upstream checkpoint loading replaced by the audited exact-key/shape strict loader. The loader excludes exactly 22 inactive auxiliary SAM2-neck tensors and supplies exactly five constructor-generated buffers (four finite position caches and the exact causal attention mask). All checkpoint tensors must be finite. Model math and image processing are unchanged.

The publisher declares Python >=3.13. The isolated measured compatibility path used Python 3.12 and MLX 0.32.4; this does not change upstream support policy. Runtime dependencies are optional and imported only for the explicit SAM3 route.
