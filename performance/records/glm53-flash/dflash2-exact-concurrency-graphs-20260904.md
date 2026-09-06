# GLM-5.3 Flash exact DFlash request-batch graphs

Status: **implemented**. The performance comparison is **research-only**.

DFlash2 at depth seven verifies eight target rows per active request. The
GLM-5.3 launcher therefore captures every eight-row CUDA-graph shape from 8
through 128 for a 16-sequence deployment. This gives C1 through C16 an exact
request-batch graph. The launcher also warms every supported concurrency
before reporting the service ready.

The sparse five-shape configuration captured rows 8, 16, 32, 64, and 128.
Intermediate concurrency levels used larger graph buckets. The exact-shape
configuration removes that padding.

## Conditions

- Four directly connected NVIDIA GB10 systems, TP4/DCP4.
- Target checkpoint:
  `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark@df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.
- BF16 DFlash2 depth seven, FP8 KV, 16 sequences, 8,192 batched tokens,
  scheduler interval two, and asynchronous scheduling.
- B12X attention, MoE, linear, and KDA-prefill backends.
- Dual-rail SIRCL for admitted collectives with patched NCCL fallback.
- SparkCache read-write mode with `tail-cow-v2` publication.
- Operator image:
  `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f`.
- Benchmark harness: `local-inference-lab/llm-inference-bench` version 0.4.32,
  commit `a3a24a6631f011716be732bbf148f95e1d0cf716`, using 20-second sustained
  decode cells at temperature 1.0.
- The harness computed each prompt once and measured sustained decode after
  the prompt context was available. Concurrent streams shared that context.

The image already accepts an explicit CUDA-graph capture list. The measured
deployment used the launcher change documented here; no image bytes changed.

## Result

At a 16,384-token context, the exact-shape configuration produced the
following changes against the same target checkpoint and serving topology:

| Concurrency | Five-shape tok/s | Exact-shape tok/s | Change | Target-step change |
|---:|---:|---:|---:|---:|
| 6 | 91.0 | 113.6 | +24.8% | +26.9% |
| 10 | 106.9 | 143.6 | +34.3% | +33.7% |
| 12 | 126.6 | 157.3 | +24.2% | +23.5% |
| 14 | 154.5 | 172.0 | +11.3% | +10.9% |
| 16 | 184.4 | 187.0 | +1.4% | -2.0% |

Target steps per second remove workload-dependent DFlash acceptance from the
comparison. Their improvement at C6, C10, C12, and C14 shows that exact graph
selection reduced execution work rather than merely observing higher draft
acceptance. C1 through C16 had zero request errors, readiness timeouts,
capacity-limit markers, or scheduler queueing.

The source result files are not published because they contain a client
hostname and private site address. Their identities are:

| Configuration | Source file SHA-256 |
|---|---|
| Five graph shapes | `6b081de09d64f528457459cd248a518344983832a9395e003ca92fb2a3d652d0` |
| Exact shapes, C1 through C10 | `eec5052232cea1a1c05a4f4c91d602ba7d7da05a6e6f871ed14a5612b95d8c18` |
| Exact shapes, C11 through C16 | `e1e9bf733fc8229628910fcc22a20c46bd29a937ff7e2e7c4450aeaee89e783d` |

## Limits

The exact-shape measurements cover 8,192- and 16,384-token decode contexts.
Longer-context spot checks remain useful but are not required to establish the
request-batch padding mechanism. The target checkpoint is the smaller
NVFP4-Spark quantization; graph row counts depend on speculative depth and
active request count rather than checkpoint tensor values.

`fused_recurrent_kda_fwd_kernel` compiled while the C5 8K stream was becoming
ready and before its measured window. The recorded C5 value is not affected,
but a longer C5 startup request is still needed to prevent that first-use
compilation in ordinary serving.
