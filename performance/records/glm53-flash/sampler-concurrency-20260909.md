# Concurrent sampler warmup observation

Status: **research-only**. Six sequential sampler requests completed, but the
first subsequent filtered C2 sweep produced a `_topk_topp_kernel` warning on
each worker. This supports adding concurrent filtered requests to the readiness
recipe; it does not close all of issue #214.

## Conditions

Four DGX Spark GB10 systems served GLM-5.3-Flash-NVFP4-Spark with TP4/DCP1,
native MTP3, the V2 runner and InstantTensor loading. The model target in the
source lock is `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. The API was already ready and its
JIT namespace already existed; this was not a controlled cold-cache test.

| Identity | Value |
|---|---|
| Local image config digest | `sha256:db4053abcf1c82ce2da5633abd672abcaa32ca77eef5b03d567b1fa24a04b671` |
| vLLM composition | `1556c3587ee84cc3c91848893d5079d0a7ff6271` |
| B12X | `2883a5df65a7ea3cb6e82abb63d1448dd3154887` |
| SparkCache | `d0cf7296062ec8b4d17d65cd05a416d509e80bd8` |
| Initial stream validator | SparkRing `9a3d7e416a5502eac1462aa3c2cf30dca53094ba` |
| Complete recipe helper SHA-256 | `2104e7d83e84d7e2ccbb19e331b487cafea4515ac75eda2dc0082cfc7187a841` |

The image digest identifies local bytes, not a registry release. The
[JSON record](sampler-concurrency-20260909.json) includes installed rank-zero
sampler, state, rejection-sampler, Triton-kernel and monitor file hashes. The
vLLM composition comes from the source lock and image label; inherited package
version text is not used as that identity.

## Measurement

Six C1 streamed requests first exercised unfiltered, temperature-scaled,
top-k, top-p, combined-filter and seeded combined-filter settings. Each completed
32 generated tokens with validated SSE termination and usage.

Two subsequent sweeps each sent three homogeneous C2 pairs: top-k only, top-p
only, and both filters, at temperature one with no explicit seed. Each request
used `max_tokens=128` and `ignore_eos=true`; those sweeps omitted `min_tokens`.
The public record retains each request body, reported usage and client monotonic
start/end interval. HTTP overlap is the smaller end time minus the larger start
time. Warning lines were collected from all four ranks for each sweep.

The complete helper recipe was then exercised externally against the same ready
API, including `min_tokens=128` on its C2 requests. No installed startup helper
or readiness marker was changed by these API probes.

## Result

| Observation | Result |
|---|---|
| Initial six C1 requests | All completed with 32 generated tokens each |
| First three C2 pairs | All six requests completed with 128 generated tokens each |
| `_topk_topp_kernel` warnings in first C2 sweep | One observed warning on each of four ranks |
| Repeated three C2 pairs | All six requests completed with 128 generated tokens each |
| Observed warnings in repeated sweep | Zero on each rank |
| Complete six-C1/three-C2 helper replay | All twelve requests passed strict SSE and usage checks |

Each pair had positive measured HTTP overlap. These intervals are evidence of
concurrent client requests, not a model-throughput comparison.

## Conclusion

The first filtered C2 sweep reached a kernel initialization/load that the six
C1 requests had not covered in this process. A readiness recipe restricted to
those C1 requests was insufficient for this observed workload. The extended
helper completed its bounded request recipe on this already-ready API.

## Limitations

The monitor used `warning_once`, so zero repeat warnings does not prove zero
JIT events or complete filter-variant coverage. A warning also does not separate
disk-cache loading from compilation. No compile-time or speedup claim follows
from these observations.

HTTP overlap does not prove that requests shared a GPU batch or identify every
worker's execution path. Other valid configurations can route probability
filtering through FlashInfer, so three particular Triton cache entries are not
a general readiness contract. The helper retains `jit_coverage_verified=false`.

This native-MTP3 result does not qualify the original DFlash2 report, a rebuilt
image's startup readiness, controlled cold-JIT behavior, mixed long/short
prefill, or arbitrary concurrency. Issue #214 remains open for those gaps.
