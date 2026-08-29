# GLM-5.3 Flash with BF16 DFlash2 on a four-Spark cycle

Status: **qualified** with and without the SparkCache connector for the exact
OCI image and configuration recorded below.

## Serving contract

| Property | Qualified value |
|---|---|
| Hardware | four NVIDIA DGX Spark systems; direct RoCE cycle |
| Parallelism | TP4, DCP1, PP1 |
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16, seven tokens, TP4 |
| Limits | 524,288 model tokens; 8,192 batched tokens; 32 sequences |
| GPU KV | 12,884,901,888 FP8 bytes per rank; measured 549,950-token capacity |
| Execution | B12X target attention/MoE/linear; Triton KDA prefill; async scheduling; chunked prefill; native prefix caching |
| Graphs | `FULL_AND_PIECEWISE` target; FULL DFlash; capture sizes 8–256 |
| External cache | SparkCache maximum 48 GiB and low watermark 40 GiB per rank, or connector omitted |

The qualified image is
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`
with local image ID
`sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`
on all four ranks. Its source-built parent is
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.

## Evidence

With SparkCache enabled, every rank committed the same 8,192-token context,
the four containers were replaced, and every rank restored the context in
156.8–171.8 ms. vLLM attributed 8,192 prompt tokens to external KV transfer.
The restored request completed in 1.902 seconds. DFlash emitted 504 tokens
from 72 drafts, the semantic canary passed, every rank retained 24 RTS worker
QPs, and no preemption, restart, OOM, or fatal-log match occurred.

With the connector omitted, the semantic canary passed in 2.703 seconds,
DFlash emitted 231 tokens from 33 drafts, external-cache queries stayed at
zero, and the same process and RDMA health conditions passed.

The detailed conditions and measurements are in
[`performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md`](../../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md).

## Provenance

The target repository attributes its mixed-precision artifact to NVIDIA
ModelOpt `0.39.0.dev290+gf9d9a71de.d20260407`: NVFP4 routed experts in target
layers 3–44 and MXFP8 in embedded MTP layer 45. The repository does not record
the unquantized base-checkpoint revision. The BF16 DFlash2 checkpoint is
published by Inco AI under CC BY-NC-ND 4.0.

The runtime uses
`local-inference-lab/vllm` `dev/jovian-judgement@da4d7be6c97434f6942292ed8abbf4b32dc44355`,
including merged pull requests 486, 489, 493, 494, 497, and 499, and
`local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
NCCL is built from NVIDIA commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97` with SparkRing's original
`nccl-2.30.7-switchless-cycle.patch`; patched tree
`abdeb053b94c3f6d472cd55ae2b79ca821299009`.

SparkCache source is
`FujitsuPolycom/sparkcache@3860a2250193a6679ac6bac857af53e0757841f8`
on branch `codex/glm53-public-image`, source-tree SHA-256
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.
[`runtime/glm53-flash/pins.json`](../../runtime/glm53-flash/pins.json) is the
complete machine-readable source, patch, image, SBOM, and license record.

## Limitations

The record does not qualify throughput, soak behavior, restored spans larger
than 8,192 tokens, MTP drafting, other checkpoints, other topologies, native
direct restore, or streaming snapshots. Stock safetensors loading and Triton
KDA prefill are implemented; InstantTensor loading and FlashKDA prefill are
unsupported by this image. The optional `deep_ep` import can report a
duplicate-NCCL warning before vLLM selects the source-built NCCL library.

Use the [SparkCache quickstart](../GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md)
or the [cache-disabled quickstart](../GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md).
Report runtime and transport defects at
<https://github.com/FujitsuPolycom/sparkring/issues> and connector defects at
<https://github.com/FujitsuPolycom/sparkcache/issues>.
