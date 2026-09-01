# SparkRing

> Alpha software. SparkRing is evolving quickly. For repeatable deployments, use the immutable image digest and source revisions listed in each quickstart.

SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX 'Spark' systems
powered by the GB10 Grace Blackwell Superchip.

SparkRing supports GB10 pairs, four-node rings, and six-node rings.

Models run as tensor-parallel deployments over the direct fabric without an
external Ethernet or InfiniBand switch. [SIRCL](docs/SIRCL.md) provides custom
RDMA collectives where applicable, CUDA-graph command rings support repeated
decode work, and [patched NCCL](spark_transport/nccl/README.md) handles
communication outside SIRCL's supported paths.

The repository provides launch tooling, model profiles, test evidence, and
[performance data](performance/).

## Setup

Start with an SSH session on rank 0 and enough disk space for the intended
model weights. Use the [bootstrap guide](docs/BOOTSTRAP.md). The
model-independent `sparkring cluster init` workflow enrolls nodes, discovers
management and ConnectX-7 hardware, generates the cluster inventory, and
launches Ring Doctor before any model profile is selected.

## Resources

- [Supported models and profiles](#profiles)
- [Container image roles](#container-images)
- [Benchmark results](#benchmark-results)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below

## Container images

| Package | Purpose | When to use it |
|---|---|---|
| [`sparkring-glm53-sparkcache`](https://github.com/users/FujitsuPolycom/packages/container/package/sparkring-glm53-sparkcache) | Published Linux/ARM64 GLM-5.3 operator image with source-pinned vLLM, B12X GB10 kernels, DFlash2, and optional SparkCache. | Use the immutable digest from the [GLM-5.3 quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md). |
| [`sparkring-glm53-runtime`](https://github.com/users/FujitsuPolycom/packages/container/package/sparkring-glm53-runtime) | Source-pinned GLM-5.3 runtime bases used to construct later operator images. | Use only when a source-build procedure names an exact digest. |
| [`gb10-vllm-serving`](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) | Profile-specific GB10 serving images, including published DeepSeek inputs. | Use only when a model profile names an exact digest. |

Package tags are convenient labels, not reproducible identities. Deployment
guides use immutable digests.

## Profiles

### GLM-5.3 Flash

All three profiles use the GLM-5.3 Flash NVFP4 target, BF16 DFlash2 at depth
seven, FP8 KV, B12X GB10 kernels, and the same source-pinned ARM64 vLLM image.

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Approx. logical KV capacity | Start here |
|---|---|---:|---:|---:|---|---:|---|
| DCP1 | 4 Sparks · TP4/DCP1 | 1M | 16 | 8,192 | 26 GiB/rank; SparkCache enabled* | 1.30M tokens | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| DCP2 | 4 Sparks · TP4/DCP2 | 1M | 16 | 8,192 | 30 GiB/rank; SparkCache enabled* | 2.90M tokens | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| **DCP4 preferred**** | **4 Sparks · TP4/DCP4** | **1M** | **16** | **8,192** | **24 GiB/rank; SparkCache enabled*** | **4.32M tokens** | **[Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)** |

*Set
`SPARKCACHE_ENABLED=0` to use disable sparkcache, leaving only vLLM prefix caching by default. 1 = both. 

**DCP4 is preferred because it provides the best performance at high concurrency decode and the greatest available KVC space.

 The base image is published at
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:380283a506aeb8f9d486a3c64cd738e44268c3cc21590913ea9e4685869f256a` and supports both caching methods.

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

The public BF16 DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research
and evaluation; review its model card before use.

## Benchmark results

All values are tokens per second. Prefill uses a cold prompt with caching
disabled. Decode uses unique, cold prompt contexts at
temperature 1.0; decode values are aggregate throughput across active streams.

| Profile | Prefill | C1 decode | C8 decode | Highest tested decode | Coding peak |
|---|---:|---:|---:|---:|---:|
| [GLM-5.3 Flash DCP4 preferred default · 4 Sparks](performance/records/glm53-flash/dcp4-24g-default-20260901.md) | 2,513 | 40.20 | 116.73 | C16: 168.39 | 71.67 |
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](performance/records/glm-3.5bpw/normalized-base-20260822.md) | 671 | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 1,926 | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 2,488 | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 1,367 | 29.50 | 142.20 | C8: 142.20 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 1,964 | 35.07 | 191.02 | C8: 191.02 | 48.46 |

See [benchmark results and throughput tables](docs/RESULTS.md) for full
matrices, sample counts, exact settings, and limitations.

## Architecture

Two-Spark profiles use one direct 200 Gb/s cable and patched NCCL. Four-Spark
profiles use a switchless `0-1-2-3-0` cable cycle. SIRCL serves its tested
collective paths; patched NCCL handles the remaining paths.

See [architecture](docs/ARCHITECTURE.md), [SIRCL](docs/SIRCL.md), and the
[deployment prerequisites](docs/PREREQUISITES.md).

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport and vLLM adapters |
| `runtime/` | Pinned runtime inputs and builders |
| `scripts/` | Site validation, preflight, launch, and evidence tooling |
| `recipes/` | Machine-readable serving recipes |
| `performance/` | Benchmark methods, records, and sanitized receipts |
| `docs/` | Profile procedures, architecture, prerequisites, and evidence |

## Acknowledgements

SparkRing builds on work from vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache,
ExLlamaV3, and most importantly, the endless work by the 
[local inference community](https://github.com/local-inference-lab/).

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
