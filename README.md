# SparkRing

> SparkRing is experimental. Use the immutable image digest and source
> revisions in each quickstart when you need a repeatable deployment.

SparkRing is a vLLM-based inference-serving stack with low-latency collective
communication for switchless clusters of NVIDIA DGX Spark systems, powered
by the GB10 Grace Blackwell Superchip.

SparkRing supports GB10 pairs, four-node rings, and six-node rings
(profiles pending).

Models run across multiple systems using tensor parallelism. Depending on the
profile, communication uses [SIRCL](docs/SIRCL.md),
[RoCEnante](third_party/b12x_roce/README.md), and
[patched NCCL](spark_transport/nccl/README.md). The four-node virtual-mesh
profile adds hardware-forwarded paths between opposite nodes over the existing
ring cables. The high-speed data fabric needs no external Ethernet or
InfiniBand switch; administration uses a separate management network.

The repository provides setup guides, launch tooling, model profiles,
reproducible benchmarks, and [test results](performance/).

## Setup

1. Choose a [model profile](#profiles) for your number of Sparks and review
   the [hardware and software prerequisites](docs/PREREQUISITES.md).
2. For a ring, follow the [bootstrap guide](docs/BOOTSTRAP.md) from an SSH
   session on rank 0. It covers SSH enrollment, interface discovery, fabric
   configuration, and Ring Doctor checks. Pair profiles use their own
   quickstart's direct-link setup.
3. Follow the selected quickstart to download its image and weights, launch
   the model, and verify readiness. Use the
   [validation runbook](docs/PROFILE_VALIDATION.md) to measure performance,
   accuracy, and restart behavior.

The [virtual-mesh prerequisites](docs/PREREQUISITES.md#four-spark-managed-hardware-forwarded-mesh)
extend the shared setup with the additional ConnectX configuration required
by that profile.

## Resources

- [Profile validation: performance, accuracy, and restart checks](docs/PROFILE_VALIDATION.md)
- [Model profiles](#profiles)
- [Container image roles](#container-images)
- [Benchmark results](#benchmark-results)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below

## Container images

| Image family / profile | Purpose | Start here |
|---|---|---|
| [GLM-5.3 DFlash2/SIRCL](runtime/glm53-flash-jj-r8-gb10/README.md) | Linux/ARM64 vLLM image with B12X kernels, DFlash2, SIRCL, and optional SparkCache. | [DFlash2 quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| [GLM-5.3 native-MTP3 mesh](runtime/glm53-spark-mtp3-mesh/public-image.json) | Linux/ARM64 image for NVFP4-Spark, native MTP3, hybrid mesh transport, and SparkCache. | [Mesh quickstart](docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md) |
| [`sparkring-glm53-runtime`](https://github.com/users/FujitsuPolycom/packages/container/package/sparkring-glm53-runtime) | Pinned GLM-5.3 bases used to build serving images. | Use the digest named by the source-build guide. |
| [`gb10-vllm-serving`](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) | Profile-specific GB10 images, including DeepSeek. | Use the image named by the selected model quickstart. |

Both GLM-5.3 serving images are published in the
[`sparkring-glm53-sparkcache` package](https://github.com/users/FujitsuPolycom/packages/container/package/sparkring-glm53-sparkcache).
They have different digests. Copy the immutable digest from the quickstart for
your selected profile rather than substituting another image from the package.

## Profiles

**Context** is the per-request token limit. **Seqs** is the maximum active
sequence count. **Batch** is the scheduled-token budget per model step. These
are separate limits; available KV memory also constrains which requests fit
together. In the tables, 1M means 1,048,576 tokens.

### GLM-5.3 Flash

These three **DFlash2/SIRCL profiles** use the GLM-5.3 Flash NVFP4 target,
an external BF16 DFlash2 predictor at depth seven, FP8 KV, B12X kernels, and
the same ARM64 image. Target verification captures use rows 8 through 128 in
eight-row increments, covering full request batches from C1 through C16.

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Approx. recorded KV capacity | Start here |
|---|---|---:|---:|---:|---|---:|---|
| DCP1 | 4 Sparks · TP4/DCP1 | 1M | 16 | 8,192 | FP8 · 24 GiB/rank; SparkCache enabled | ~1.30M tokens | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| DCP2 | 4 Sparks · TP4/DCP2 | 1M | 16 | 8,192 | FP8 · 24 GiB/rank; SparkCache enabled | ~2.90M tokens | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| **DCP4 preferred** | **4 Sparks · TP4/DCP4** | **1M** | **16** | **8,192** | **FP8 · 24 GiB/rank; SparkCache enabled** | **~4.32M tokens** | **[Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)** |

All three default to 24 GiB of KV memory per rank. The reference capacities
were measured at 26/30/24 GiB for DCP1/2/4 respectively; vLLM reports the
actual model-wide capacity at startup. That capacity is shared across requests.
**DCP4 is the preferred DFlash2 profile** and the documented asynchronous
SparkCache configuration.

Set `SPARKCACHE_ENABLED=0` for vLLM's in-memory prefix cache alone, or `1` to
add persistent storage. DCP1/DCP2 use different publication settings from
DCP4; follow the matching section of the quickstart when enabling SparkCache.

SIRCL accelerates eligible collectives, with patched NCCL handling the other
paths. Startup checks require all four ranks to agree on the transport
configuration, and runtime checks detect transport failures. The
[public-image receipt](runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
records **qualified** four-rank DCP4 startup, persistent restoration, and
failure-containment checks.

The operator image is published at
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f`
and supports both caching methods. The quickstart includes installation,
readiness, and recovery instructions.

The external BF16 DFlash2 weights have a separate **CC BY-NC-ND 4.0** license
for research and evaluation. See the
[Inco AI model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license)
for its terms. The native-MTP3 profile below does not require these weights.

### GLM-5.3 Flash Spark with native MTP3 and mesh transport

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Start here |
|---|---|---:|---:|---:|---|---|
| NVFP4-Spark + native MTP3 + mesh · research-only | 4 Sparks · TP4/DCP4 | 1M | 16 | 8,192 | FP8 · 24 GiB/rank; SparkCache enabled | [Quickstart](docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md) |

This profile uses the NVFP4-Spark checkpoint's built-in three-token predictor;
no external draft checkpoint is required. It combines graph-native SIRCL for
selected decode shapes, dual-rail SIRCL for large prefill collectives, and
RoCEnante for selected small all-reduces. Target verification captures use
four-row increments through 64 rows.

The [profile package](runtime/glm53-spark-mtp3-mesh/README.md) provides the
public image, transport files, and temperature-one warmup. The
[mesh service](runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md) sets up and
monitors the hardware-forwarded paths between opposite Sparks. It checks
all-rank readiness before model startup and stops dependent serving when the
fabric is unhealthy. The quickstart documents coordinated recovery.

The [consolidated validation report](performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
collects installation and restart checks, cache restoration, three-pass
prefill measurements, decode matrices, Estonia accuracy, and long-context
retrieval results. This is an opt-in profile; the DFlash2/SIRCL recommendation
above is unchanged.

### Other model profiles

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Start here |
|---|---|---:|---:|---:|---|---|
| GLM-5.2 EXL3 3.5-bpw | 4 Sparks · TP4/DCP4 | 1M | 16 | 4,096 | NVFP4 DS-MLA · 9.25 GB/rank | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | 2 Sparks · TP2/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA · 16 GiB/rank | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | 4 Sparks · TP4/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA · 16 GiB/rank | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | 2 Sparks · TP2/DCP1 | 1M | 32 | 8,192 | FP8 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | 4 Sparks · TP4/DCP1 | 1M | 64 | 8,192 | FP8 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | 4 Sparks · TP4/DCP4 | 1M | 16 | 4,096 | NVFP4 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | 2 Sparks · TP2/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | 4 Sparks · TP4/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |

## Benchmark results

Throughput values below are tokens per second. Decode is sustained aggregate
output across active requests at temperature 1.0; C denotes concurrency.
The decode context is listed explicitly. Prefill entries identify their
context and mark integrated scouts. Coding peak uses a separate coding
prompt. Each linked record gives its sampling, cache settings, and repeat counts.

| Profile | Decode context | Prefill | C1 decode | C8 decode | Highest C at this context | Coding peak |
|---|---:|---:|---:|---:|---:|---:|
| [GLM-5.3 NVFP4-Spark · native MTP3 + mesh · 4 Sparks](performance/records/glm53-flash/spark-mtp3-mesh-20260905.md) | 8K | 2,703 (8K scout) | 48.2 | 168.8 | C16: 231.3 | — |
| [GLM-5.3 NVFP4-Spark · DFlash2 exact request-batch graphs · 4 Sparks](performance/records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md) | 16K | 2,717 (16K scout) | 43.05 | 134.3 | C16: 187.0 | — |
| [GLM-5.3 NVFP4 · DFlash2/B12X-KDA DCP4 · 4 Sparks](performance/records/glm53-flash/b12x-kda-dcp4-20260903.md) | 16K | 2,649 (16K scout) | 37.97 | — | C4: 90.36 | — |
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](performance/records/glm-3.5bpw/normalized-base-20260822.md) | 16K | 671 (16K) | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 16K | 1,926 (16K) | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 16K | 2,488 (16K) | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 16K | 1,367 (16K) | 29.50 | 142.20 | C16: 184.39 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 16K | 1,964 (16K) | 35.07 | 191.02 | C8: 191.02 | 48.46 |

The native-MTP3 mesh row uses one observation per cell with caching enabled.
Its [consolidated report](performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
also includes three-pass cold-prefix prefill results, the 32K/64K decode
matrices, Estonia **30/30** at C8, and **4/4** needle-hunt checks through
507,367 prompt tokens.

See [benchmark results and throughput tables](docs/RESULTS.md) for full
matrices, sample counts, exact settings, and limitations.

## Architecture

Two-Spark profiles use one direct 200 Gb/s cable and patched NCCL. Four-Spark
profiles use a `0-1-2-3-0` cable cycle. Their collective routing depends on
the profile: SIRCL accelerates supported operations, the native-MTP3 mesh
profile also uses RoCEnante, and patched NCCL supplies the remaining paths.

The virtual mesh uses the same four ring cables. ConnectX-7 hardware forwards
traffic between opposite nodes; no physical diagonal cables are added.

See [architecture](docs/ARCHITECTURE.md), [SIRCL](docs/SIRCL.md), and the
[deployment prerequisites](docs/PREREQUISITES.md).

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport and vLLM adapters |
| `runtime/` | Pinned runtime inputs and builders |
| `scripts/` | Site validation, preflight, launch, and evidence tooling |
| [`recipes/`](recipes/) | Machine-readable serving recipes and their operator-guide index |
| `performance/` | Benchmark methods, records, and sanitized receipts |
| `docs/` | Profile procedures, architecture, prerequisites, and evidence |

## Acknowledgements

SparkRing builds on vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache, ExLlamaV3,
and the work of the [local inference community](https://github.com/local-inference-lab/).

Luke and Local Inference Lab's [RoCEnante implementation](https://github.com/local-inference-lab/b12x/pull/295)
and [vLLM integration](https://github.com/local-inference-lab/vllm/pull/597)
provided the communication implementation and inspiration for the virtual-mesh
profile. SparkRing adapts that work to hardware-forwarded paths on the ring.

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

SparkRing code is licensed under Apache-2.0. See [`LICENSE`](LICENSE).
Model weights and bundled third-party components retain their own licenses;
review the selected model cards and third-party notices before deployment.
