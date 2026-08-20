# GLM-5.2 runtime port map

This directory answers a narrow question: which source is still needed to turn
the public-functional image into the GLM-5.2 reference runtime?

The machine-readable answer is
[`glm52-port-manifest.json`](glm52-port-manifest.json). It pins public source
commits and file hashes, maps each capability to its vLLM integration files,
and names the acceptance test that closes it.

## Short answer

The missing runtime is not an unknown 13,000-line fork. The production wheel
was a stock build of `vllm-project/vllm@fcc6141`, followed by a Python overlay.
To reproduce the GLM-5.2 reference runtime from public sources, that overlay
reduces to four work packages:

| Order | Package | Current state |
|---:|---|---|
| 1 | SM121 sparse MLA | Exact public implementations located in CosmicRaisins and davidsyoung; needs a pinned-vLLM port |
| 2 | packed `nvfp4_ds_mla` KV | Captured integration delta; the coupled cache-layout/scale ABI needs porting as one unit |
| 3 | hybrid MXFP4 checkpoint loader | Small Apache-2.0 reference implementation; needs registration and loader tests |
| 4 | adaptive MTP2/4 | Public controller/hooks located; SparkRing's reset/status and graph-width contracts are already public |

The order is deliberate. Sparse MLA and packed KV are startup blockers.
Hybrid loading is the next model-load blocker. Adaptive MTP affects speed, but
the model can first be brought up with fixed-K or speculation disabled.

## What was intentionally left out

The captured reference overlay also contained DSpark/DFlash architectures,
pipeline-parallel fixes, InstantTensor experiments, warmup coverage, and an
optional FP8 language-model head. None is required for the pinned TP4 GLM-5.2
acceptance matrix. Carrying them into the first port would increase failure
surface without getting users to a working server sooner.

## Source and license notes

- `CosmicRaisins/glm-5.2-gb10@6008487` has a root Apache-2.0 license.
- The audited davidsyoung files carry Apache-2.0 SPDX headers and its README
  describes the serving code as Apache-2.0 lineage, but that commit has no root
  license file. Preserve its file headers and attribution.
- `local-inference-lab/sparkinfer@284a2ea` is the exact kernel-library pin used
  by the reference runtime. The audited commit has no root license file. Keep
  its notices intact and treat license clarification as a follow-up rather
  than silently relicensing it.

These notes are provenance, not legal advice.

## Contributor workflow

1. Pick one capability from the manifest.
2. Fetch the exact source commit and verify every listed SHA-256.
3. Rebase the minimum target files onto the vLLM commit in
   `runtime/runtime-lock.json`.
4. Add the patch and its preimage hashes under `runtime/patches/vllm/`.
5. Add focused CPU tests wherever the logic can be isolated.
6. Run an eager SM121 probe before enabling CUDA graphs.
7. Promote the capability only after every listed acceptance condition has
   evidence.

Do not bypass `runtime/public-capability-gate.py`. It exists so a partially
ported image refuses to advertise a complete GLM-5.2 reference runtime.
