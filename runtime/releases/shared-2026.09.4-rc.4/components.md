# Component sources and licenses

SparkRing integration code is Apache-2.0. Bundled components retain their own
terms; the image is not universally Apache-2.0. Model weights are not bundled.

This release composes SparkRing integrations with the ARM64
`eugr/spark-vllm-b12x:nightly-20260923` foundation, pinned to manifest
`sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e`.
The inherited framework binaries are preserved. Python, Triton and CuTe source
changes are recorded separately; runtime compilation is required on first start.

| Component | Source or version | Terms |
| --- | --- | --- |
| vLLM | Integrated `91f94783190ce716e2238f899b25d44e84e2a525`; complete archive and patch against foundation source `57fdda71b06063203ef0a82cac888a7c9b5f8f7a` | Apache-2.0 |
| B12X | Integrated `a0a0c425e20f45e041de241afd6e76dcfc52d281`; complete archive and patch against foundation source `8a99d639410e39d5f39cb4037675331beceea1d4` | Apache-2.0 |
| SparkRing, SparkCache and runtime-status plugin | Source-bound integration assets; status plugin 0.1.0 with wheel and complete source archive | Apache-2.0 |
| PyTorch | Foundation 2.13.0+cu130 | Upstream package licenses and notices |
| FlashInfer | Foundation 0.7.0 | Upstream package licenses and notices |
| CUTLASS DSL | Foundation 4.7.0 | Upstream package licenses and notices |
| CUDA Python and CUDA bindings | Foundation 13.4.x | NVIDIA Software License and retained package notices |
| NCCL | Retained SparkRing dual-domain runtime, observed version 2.31.2+cuda13.3 | Component license and retained notices |
| RoCEnante | Retained adaptive prepared transport, rebound to the exact composed B12X interfaces | Component license and retained notices |

`source-manifest.json` identifies archive and patch bytes and records independent
Git-tree reconstruction. `composition.json` identifies every changed runtime
file, inherited inventory, transport binding, cache contract and status artifact.
`pr-disposition-manifest.json` records the audited upstream inventory and
distinguishes integrated, equivalent, conditional and out-of-scope changes.
These records establish provenance; serving and hardware qualification are
reported separately.

The image retains the external foundation's own licenses and the selected
SparkRing component notices. It does not copy the previous SparkRing image's
entire environment or its unrelated SGLang/Mia payload.
