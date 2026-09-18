# SGLang bounded dense prefill overlays

Status: **implemented**. CPU tests cover row partitioning, candidate masks,
request offsets, empty requests and source admission. They do not qualify CUDA
kernels, numerical equivalence on GB10, serving capacity or performance.

These overlays target SGLang source revision
`e087e662ba1ac4ef7747537e2a9141085efd4561`. Apply them before installing the Mia
adapter so that source hashes identify the SGLang distribution unambiguously:

```bash
python3 patches/apply.py --source-root /sgl-workspace/sglang \
  --receipt /opt/sparkring/sglang-overlay.json
```

`--check` verifies an installed overlay without changing the source tree. The
installer rejects differing source or patch bytes, validates both overlays in a
temporary tree, and checks all sixteen resulting file hashes before copying.
Repeated installation accepts only a completely matching output inventory.

- `sglang-39068.patch` preserves the merged upstream patch for compressed-state
  verification and related kernels, commit
  `0d5e663b8f8d80a6caec2a7f7ce4eed6394756b7` from
  [SGLang #39068](https://github.com/sgl-project/sglang/pull/39068).
- `bounded-prefill.patch` adapts Khoa Pham's dense-indexer allocation change,
  commit `cb8dd033ab54af8904733199dbae94270c3395ce` from
  [SGLang #39187](https://github.com/sgl-project/sglang/pull/39187).
  It retains the pinned backend's candidate-mask storage and score-mask helper.
  Both upstream contributions use the Apache-2.0 license; their authorship
  remains in the patch headers.

The dense indexer scores each request in row chunks with a 2 GiB fp32 logits
budget. At least one row is processed even if a single row exceeds that budget;
the budget does not cap total process memory. Candidate-source masks are
preallocated instead of concatenated. Decoder sliding-window bounded replay
publishes only the rows consumed by the late layers. Context-parallel requests
retain full candidate masks and the existing tail selection.

The patch preserves the torch fallback and graph indexer paths. It does not
establish that any particular hardware selects the dense FP4 path. Hardware
qualification must verify the selected backend, output correctness, memory use
and the requested context limit. `manifest.json` records every changed source
file before either overlay, after the verification overlay, and after both.
