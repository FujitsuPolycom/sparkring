# SparkRing

SparkRing is a vLLM-based inference-serving stack with low-latency collective
communication for switchless clusters of NVIDIA GB10-based devices. 

SparkRing supports pairs, four-node rings, and six-node rings (in dev).

The collective communication stack combines [SIRCL](docs/SIRCL.md), [RoCEnante](third_party/b12x_roce/README.md), and [patched NCCL](spark_transport/nccl/README.md). The high-speed data fabric
needs no external Ethernet or InfiniBand switch; administration and vLLM api serving occur over a node/s 10Gbe NIC.

Four- and six-node rings use a virtual mesh built on custom RoCE RDMA routing and hardware forwarding in the ConnectX network ASICs. This creates paths between nodes that aren’t directly connected, carrying traffic over the existing ring cables without routing it through host CPUs. The result is mesh connectivity over a physical ring. 

The repository provides setup guides, launch tooling, model profiles,
reproducible benchmarks, and [test results](performance/).

> SparkRing is experimental. This repo is changing rapidly.

## Setup

1. Choose a [profile](#profiles) and check the [prerequisites](docs/PREREQUISITES.md).
2. For a four-node ring, follow the [bootstrap guide](docs/BOOTSTRAP.md).
   Two-node profiles include their own direct-link setup.
3. Follow the profile's quickstart, then run the
   [validation checks](docs/PROFILE_VALIDATION.md).

## Profiles

### Four Sparks

| Model / predictor | Serving stack | Transport | Layout | Context | Sequences | Batch | Guide |
|---|---|---|---|---:|---:|---:|---|
| **GLM-5.3 Flash NVFP4-Spark · MTP3 cache/checkpoint mesh** | [SparkRing vLLM/B12X image](runtime/glm53-spark-mtp3-mesh/performance/README.md) | [SIRCL + RoCEnante + NCCL](runtime/glm53-spark-mtp3-mesh/README.md) | TP4/DCP4 | 1M | 16 | 8,192 | [Quickstart](docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md) |
| GLM-5.3 Flash NVFP4 · BF16 DFlash2 | [SparkRing vLLM/B12X image](runtime/glm53-flash-jj-r8-gb10/README.md) | [SIRCL + NCCL](spark_transport/integrations/vllm/README.md) | TP4/DCP4; DCP1/2 | 1M | 16 | 8,192 | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw | [SparkRing vLLM/ExLlamaV3 build](runtime/exl3-r7/README.md) | [SIRCL + NCCL](docs/SIRCL.md) | TP4/DCP4 | 1M | 16 | 4,096 | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | [SparkRing vLLM/B12X image](runtime/deepseek0731-gb10/README.md) | [Patched NCCL](spark_transport/nccl/README.md) | TP4/DCP1 | 1M | 32 | 4,096 | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | [SparkRing vLLM/ExLlamaV3 build](runtime/qwen38/README.md) | [Patched NCCL](spark_transport/nccl/README.md) | TP4/DCP1 | 1M | 64 | 8,192 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| DeepSeek-V4-Flash-Vision-Exp with DSpark (research-only) | [Anemll image / MiaAI-Lab recipe](runtime/deepseek-vision-exp/profile.json) | [SparkRing patched NCCL](spark_transport/nccl/README.md) | TP4 | 1M | 48 | 12,288 | [Quickstart](docs/DEEPSEEK_V4_FLASH_VISION_EXP_TP4_QUICKSTART.md) |

The Vision-Exp [artifact contract](runtime/deepseek-vision-exp/profile.json)
identifies the Anemll image, MiaAI-Lab recipe, and SparkRing transport separately.
Contributor-reported results are linked from the guide; independent reproduction
of the selected artifacts is not claimed.

The four-Spark GLM-5.3 native-MTP3 profile uses hardware-forwarded mesh paths
and requires [managed-mesh setup](runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md).

DFlash2 profiles use a separate draft checkpoint with
[CC BY-NC-ND 4.0 terms](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license).

### Two Sparks

| Model / predictor | Serving stack | Transport | Layout | Context | Sequences | Batch | Guide |
|---|---|---|---|---:|---:|---:|---|
| **GLM-5.3 Flash NVFP4-Spark · native MTP3** | [SparkRing runtime image](runtime/sparkring/README.md) | [Patched NCCL](runtime/profiles/glm53-flash-spark-tp2/runtime.env.example) | TP2/DCP1 | 512K | 8 | 8,192 | [Quickstart](docs/GLM53_FLASH_SPARK_TP2_EXPERIMENTAL_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | [SparkRing vLLM/B12X image](runtime/deepseek0731-gb10/README.md) | [Patched NCCL](spark_transport/nccl/README.md) | TP2/DCP1 | 1M | 32 | 4,096 | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | [SparkRing vLLM/ExLlamaV3 build](runtime/qwen38/README.md) | [Patched NCCL](spark_transport/nccl/README.md) | TP2/DCP1 | 1M | 32 | 8,192 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |

The GLM-5.3 pair is **research-only**, uses 5 GiB KV per rank, and has a
[known video-color issue](https://github.com/FujitsuPolycom/sparkring/issues/229).

See the [profile index](docs/profiles/README.md) for evidence scopes and
[SparkCache compositions](recipes/sparkcache/README.md) for persistent-cache
support.

Qwen with SparkCache is unsupported; six-node profiles are research-only.

## Container images

| Package / runtime | Profile | Details |
|---|---|---|
| `sparkring` | GLM-5.3 Flash native-MTP3 pair | [Model-neutral package](runtime/sparkring/README.md) |
| `sparkring-glm53-sparkcache` | GLM-5.3 Flash DFlash2/SIRCL | [Operator image](runtime/glm53-flash-jj-r8-gb10/README.md) |
| `sparkring-glm53-sparkcache` | GLM-5.3 Flash native-MTP3 mesh | [Mesh image](runtime/glm53-spark-mtp3-mesh/performance/public-image.json) |
| `sparkring-glm53-runtime` | GLM source-build bases | [Runtime builder](runtime/glm53-flash/README.md) |
| `gb10-vllm-serving` | Profile-specific images, including DeepSeek | [Packages](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) |
| Anemll `dspark-vllm-gx10` | DeepSeek-V4-Flash-Vision-Exp with the MiaAI-Lab recipe | [Image, recipe, and transport provenance](runtime/deepseek-vision-exp/profile.json) |

Use the exact digest in the selected quickstart. Images sharing a package
name are not interchangeable; a model-neutral name does not qualify every profile. Images will be condensed and homogenized in future releases. 

## Benchmark results

Decode is sustained aggregate output at temperature 1.0.
Results attempt to reflect real world use-case numbers in all instances unless otherwise noted.
*structured data sweeps*,*temperature 0 and/or other out-of-spec configurations are not provided or recommended*

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

See [full results](docs/RESULTS.md) and the
[mesh validation report](performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for repeat counts, accuracy checks, settings, and limitations.

## Architecture

[Architecture](docs/ARCHITECTURE.md) · [SIRCL](docs/SIRCL.md) ·
[RoCEnante](third_party/b12x_roce/README.md) ·
[Mesh prerequisites](docs/PREREQUISITES.md#four-spark-managed-hardware-forwarded-mesh)

## Resources

[Validation](docs/PROFILE_VALIDATION.md) · [Deployment tooling](docs/DEPLOYMENT_SUITE.md) ·
[Contributing](CONTRIBUTING.md) · [Discussions](https://github.com/FujitsuPolycom/sparkring/discussions)

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Communication backends and vLLM adapters |
| `runtime/` | Pinned images, builders, and serving profiles |
| `scripts/` | Preflight, deployment, and validation tools |
| [`recipes/`](recipes/) | Machine-readable serving recipes |
| [`performance/`](performance/) | Measurement methods, evidence, and receipts |
| `docs/` | Operator guides and architecture |
| [`integrations/lil/`](integrations/lil/README.md) | Companion lifecycle and deployment integration |

## Acknowledgements

Built on vLLM, NVIDIA NCCL, B12X, ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).
Luke and Local Inference Lab's [RoCEnante implementation](https://github.com/local-inference-lab/b12x/pull/295)
and [vLLM integration](https://github.com/local-inference-lab/vllm/pull/597)
underpin the adapted mesh communication.
See [third-party notices](THIRD_PARTY_NOTICES.md).

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
retain their own terms; review the selected model cards and
[third-party notices](THIRD_PARTY_NOTICES.md) before deployment.
