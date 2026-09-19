# SparkRing 2026.09.3

Shared ARM64 serving image for NVIDIA GB10 clusters, with vLLM, SparkCache and
an isolated SGLang runtime. CUDA 13.3; PyTorch 2.13.0. Model weights are separate.

Status: **qualified for bounded GLM and Qwen correctness and restart checks**
on the selections below. The [qualification record](qualification.json) and
[correctness summary](correctness.json) identify the image, checkpoints,
physical cache-restore evidence and limits. Other model selections are not
qualified by this release.

## Image and profiles

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6
```

The version tag is `ghcr.io/fujitsupolycom/sparkring:shared-2026.09.3`.
An anonymous digest pull verified the image. Follow the selected profile's
complete quickstart; changing an image alone does not update its settings.

| Selection | Cache | Guide |
|---|---|---|
| GLM-5.3-Flash NVFP4-Spark, TP2/DCP1 | SparkCache | [Two-node GLM](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| GLM-5.3-Flash NVFP4 QAD, TP2/DCP1 | SparkCache | [Two-node GLM, QAD selection](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| GLM-5.3-Flash NVFP4-Spark, TP4/DCP1 | SparkCache | [Four-node GLM](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| GLM-5.3-Flash NVFP4 QAD, TP4/DCP1 | SparkCache | [Four-node GLM, QAD selection](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| Qwen3.8-Flash-Next QAD, TP2/DCP1 | Disabled | [Two-node Qwen](../../../profiles/qwen38-flash-next-tp2/README.md) |
| Qwen3.8-Flash-Next QAD, TP2/DCP1 | SparkCache | [Two-node persistent cache](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) |
| Qwen3.8-Flash-Next QAD, TP4/DCP1 | Disabled | [Four-node Qwen](../../../profiles/qwen38-flash-next-qad-tp4/README.md) |
| Qwen3.8-Flash-Next QAD, TP4/DCP1 | SparkCache | [Four-node persistent cache](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

NVFP4-Spark is not a Spark-QAD checkpoint. GLM's QAD selection uses its own
pinned weights and Humming for the MXFP8 MTP experts; the target loader remains
B12X. Checkpoint identities are separate.

## Runtime changes

GLM kernel-preparation inputs are released after preparation instead of remaining
allocated throughout serving. The image includes the preparation-lifetime subset
of LIL vLLM PR #803, not its subsequent convolution/output-storage reuse changes.
Native binaries are unchanged and were verified before reuse.

The runtime retains B12X loading, B12X #394, checkpoint coalescing, applicable
hyperconnection optimizations, communication integrations and the startup-settings
audit. Each profile selects its applicable features. Native GLM launchers verify
the installed inventory, source-matched SparkCache contract and checkpoint metadata.

## Qualification limits

Checks cover short/16K text, finite scores, synthetic three-image/one-red-video
requests and retained-container restarts. Each cache-enabled selection restored
two fixtures with API cache credits matched to physical evidence on every rank.
Configured maximum context and sequence counts are not full-limit qualification.
Arbitrary media accuracy, C16 pressure, sustained-load stability and performance
are not claimed. No benchmark results are published here.

A GLM QAD TP4 run on 2026.09.2 encountered a native all-reduce timeout. It did
not recur during these exact-image checks, but its cause is unresolved; this is
not a proven transport correction. Idle SparkCache publication gauges may retain
their final worker sample; physical restore checks establish persistence.

The tested native GLM settings leave custom memory-kill guards disabled. Linux
memory handling remains unchanged. Avoid competing GPU workloads and monitor
host memory. Cold kernel preparation can take many minutes; native GLM readiness
allows 30 minutes.

GLM cache-disabled/DCP4 and DeepSeek profiles retain independently qualified
image selections. Included components do not imply qualification.

## Sources and rollback

[Publication identity](publication.json), [source inputs](sources/README.md) and
[component licenses](components.md) describe the payload. The
[versioned GitHub Release](https://github.com/FujitsuPolycom/sparkring/releases/tag/shared-2026.09.3)
provides the source archives and qualification records; GHCR stores container
layers. Preserve 2026.09.2 and its matching deployment
settings/cache roots for rollback. Release-specific cache directories and the
source-bound connector contract prevent accidental reuse of unverified state;
they do not change checkpoint wire format or establish cross-image compatibility.
