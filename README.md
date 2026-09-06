# SparkRing

SparkRing serves large language models across NVIDIA DGX Spark computers
using vLLM, B12X kernels, and low-latency communication over direct links.

> SparkRing is experimental. Use the image digest and source revisions in
> your profile's quickstart. Qualification applies only to its documented tests.

## Setup

1. Choose a [profile](#profiles) and check the [prerequisites](docs/PREREQUISITES.md).
2. For a four-node ring, follow the [bootstrap guide](docs/BOOTSTRAP.md).
   Two-node profiles include their own direct-link setup.
3. Follow the profile's quickstart, then run the
   [validation checks](docs/PROFILE_VALIDATION.md).

## Profiles

Context is the per-request token limit; sequences are active requests; batch
is the scheduled-token budget per model step. KV memory also constrains which requests fit together.
512K means 524,288 tokens; 1M means 1,048,576.

### Two Sparks

| Model / predictor | Layout | Context | Sequences | Batch | Guide |
|---|---|---:|---:|---:|---|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 | TP2/DCP1 | 512K | 8 | 8,192 | [Quickstart](docs/GLM53_FLASH_SPARK_TP2_EXPERIMENTAL_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | 1M | 32 | 4,096 | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | TP2/DCP1 | 1M | 32 | 8,192 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |

The GLM pair is **research-only**, uses 5 GiB KV per rank, and has a
[known video-color issue](https://github.com/FujitsuPolycom/sparkring/issues/229).
Its configured context limit is not a full-context qualification.

### Four Sparks

| Model / predictor | Layout | Context | Sequences | Batch | Guide |
|---|---|---:|---:|---:|---|
| GLM-5.3 Flash NVFP4 · BF16 DFlash2 | TP4/DCP4 preferred; DCP1/2 available | 1M | 16 | 8,192 | [Quickstart](docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| GLM-5.3 Flash NVFP4-Spark · native MTP3 mesh | TP4/DCP4 | 1M | 16 | 8,192 | [Quickstart](docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw | TP4/DCP4 | 1M | 16 | 4,096 | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | 1M | 32 | 4,096 | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | TP4/DCP1 | 1M | 64 | 8,192 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |

The native-MTP3 mesh profile is **research-only** and requires
[managed-mesh setup](runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md).
DFlash2 remains the preferred four-node GLM profile; its external draft
weights have [separate CC BY-NC-ND 4.0 terms](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license).

See the [profile index](docs/profiles/README.md) for evidence scopes and
[SparkCache compositions](recipes/sparkcache/README.md) for persistent-cache
support. Qwen with SparkCache is unsupported; six-node profiles are research-only.

## Container images

| Package / runtime | Profile | Details |
|---|---|---|
| `sparkring` | GLM-5.3 Flash native-MTP3 pair | [Model-neutral package](runtime/sparkring/README.md) |
| `sparkring-glm53-sparkcache` | GLM-5.3 Flash DFlash2/SIRCL | [Operator image](runtime/glm53-flash-jj-r8-gb10/README.md) |
| `sparkring-glm53-sparkcache` | GLM-5.3 Flash native-MTP3 mesh | [Mesh image](runtime/glm53-spark-mtp3-mesh/public-image.json) |
| `sparkring-glm53-runtime` | GLM source-build bases | [Runtime builder](runtime/glm53-flash/README.md) |
| `gb10-vllm-serving` | Profile-specific images, including DeepSeek | [Packages](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) |

Use the exact digest in the selected quickstart. Images sharing a package
name are not interchangeable; a model-neutral name does not qualify every profile.

## Benchmark results

Recorded tokens per second; C1/C8 mean one/eight concurrent requests.
Decode is sustained aggregate output at temperature 1.0.
Decode context is shown separately from prefill context. These results
describe the linked workloads, not guaranteed performance.

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

The native-MTP3 mesh row is a single observation per cell with caching enabled.
Prefill scouts are integrated measurements, not isolated prefill benchmarks.
See [full results](docs/RESULTS.md) and the
[mesh validation report](performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for repeat counts, accuracy checks, settings, and limitations.

## Architecture

Pairs use a direct 200 Gb/s link. Four-node deployments use a cable ring;
the managed mesh adds hardware-forwarded paths without diagonal cables.
Profiles select patched NCCL, SIRCL, or RoCEnante for eligible operations.

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

Built on vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache, ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).
Luke and Local Inference Lab's [RoCEnante implementation](https://github.com/local-inference-lab/b12x/pull/295)
and [vLLM integration](https://github.com/local-inference-lab/vllm/pull/597)
underpin the adapted mesh communication.
See [third-party notices](THIRD_PARTY_NOTICES.md).

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
retain their own terms; review the selected model cards and
[third-party notices](THIRD_PARTY_NOTICES.md) before deployment.
