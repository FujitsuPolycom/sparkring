# SparkRing

SparkRing is a vLLM-based inference-serving stack with low-latency collective
communication for switchless clusters of NVIDIA GB10-based devices. 

SparkRing provides deployment profiles for pairs and four-node rings. Six-node work is research-only.

The collective communication stack combines [SIRCL](docs/SIRCL.md), [RoCEnante](third_party/b12x_roce/README.md), and [patched NCCL](spark_transport/nccl/README.md). The high-speed data fabric
needs no external Ethernet or InfiniBand switch; administration and vLLM api serving occur over a node/s 10Gbe NIC.

Four- and six-node rings use a virtual mesh built on custom RoCE RDMA routing and hardware forwarding in the ConnectX network ASICs. This creates paths between nodes that aren’t directly connected, carrying traffic over the existing ring cables without routing it through host CPUs. The result is mesh connectivity over a physical ring. 

The repository provides setup guides, launch tooling, model profiles,
reproducible benchmarks, and [test results](performance/).

> SparkRing is experimental. This repo is changing rapidly.

## Setup

1. Choose a [profile](#profiles) and check the [prerequisites](docs/PREREQUISITES.md).
2. For a shared-image GLM four-node ring, follow the [mesh host setup guide](docs/GLM53_SPARK_MESH_HOST_SETUP.md).
   Other four-node profiles use the [bootstrap guide](docs/BOOTSTRAP.md).
   Two-node profiles include their own direct-link setup.
3. Follow the profile's quickstart, then run the
   [validation checks](docs/PROFILE_VALIDATION.md).

## Profiles

<!-- BEGIN GENERATED PROFILES -->

Configured context is a per-request limit, not measured KV capacity or a completed long-context test.
Status describes the evidence scope; recommendation describes deployment navigation.

### Four Sparks

| Model / features | Layout | Configured context (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---|---|---|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache | TP4/DCP1 | 1048576 | qualified | recommended | [Guide](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | TP4/DCP1 | 1048576 | research-only | alternative | [Guide](profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | TP4/DCP1 | 430080 | implemented | alternative | [Guide](profiles/deepseek-v41-flash-cycle/README.md) |
| GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 | TP4/DCP4 | 1048576 | implemented | alternative | [Guide](profiles/glm52-exl3-r7-3.5bpw/README.md) |
| GLM-5.3 Flash NVFP4-Spark · native MTP3 | TP4/DCP1 | 1048576 | research-only | alternative | [Guide](profiles/glm53-flash-spark-tp4-dcp1/README.md) |
| GLM-5.3 Flash NVFP4-Spark · switched | TP4/DCP1 | 1048576 | research-only | alternative | [Guide](profiles/glm53-flash-spark-tp4-switched/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP4/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/qwen38-27b-exl3-k5k6/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | TP4/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |
| GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 + SparkCache | TP4/DCP4 | 1048576 | implemented | alternative | [Guide](profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) |

### Two Sparks

| Model / features | Layout | Configured context (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---|---|---|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache | TP2/DCP1 | 1048576 | qualified | recommended | [Guide](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/deepseek-v4-flash-0731-pair/README.md) |
| GLM-5.3 Flash NVFP4-Spark · native MTP3 | TP2/DCP1 | 1048576 | research-only | alternative | [Guide](profiles/glm53-flash-spark-tp2-dcp1/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP2/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/qwen38-27b-exl3-k5k6-pair/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | TP2/DCP1 | 1048576 | implemented | alternative | [Guide](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

### Retired GLM-5.3 profiles

| Model / features | Layout | Configured context (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---|---|---|
| GLM-5.3-Flash-NVFP4 | TP4/DCP4 | 1048576 | implemented | retired | [Guide](profiles/glm53-flash-nvfp4-dflash2-bf16-tp4/README.md) |
| GLM-5.3-Flash-NVFP4-Spark | TP4/DCP4 | 1048576 | research-only | retired | [Guide](profiles/glm53-mtp3-cache-checkpoints-tp4/README.md) |
| GLM-5.3-Flash-NVFP4-Spark | TP4/DCP4 | 1048576 | research-only | retired | [Guide](profiles/glm53-spark-mtp3-managed-mesh-tp4/README.md) |
| GLM-5.3-Flash-NVFP4 + SparkCache | TP4/DCP4 | 1048576 | qualified | retired | [Guide](profiles/sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/README.md) |

Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](docs/history/deployment-variants.md).

Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.
Switched deployments have no switched-hardware qualification. Qwen with SparkCache is unsupported;
six-node work remains research-only and is outside this deployment catalog.

<!-- END GENERATED PROFILES -->

## Container images

| Package / runtime | Profile | Details |
|---|---|---|
| `ghcr.io/fujitsupolycom/sparkring` | Generic R33 ARM64 image; exact profiles select TP2/TP4 topology and optional components | [Source build and profile verification](runtime/sparkring/jovian-r33/image/README.md) |
| `gb10-vllm-serving` | Profile-specific images, including DeepSeek | [Packages](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) |
| Anemll `dspark-vllm-gx10` | DeepSeek-V4-Flash-Vision-Exp with the MiaAI-Lab recipe | [Image, recipe, and transport provenance](runtime/deepseek-vision-exp/profile.json) |

Use the exact digest in the selected quickstart. Images sharing a package
name are not interchangeable; a model-neutral name does not qualify every profile.
Retired profiles retain their image references in their linked guides.
The [R33 publication record](runtime/sparkring/jovian-r33/publication.json)
contains the download digest and profile verification scope.

## Benchmark results

Each row links the exact measured configuration. These records include retired
profiles and are not benchmark results for the shared-image build.

Decode is aggregate output throughput. Each record specifies its sampling, workload and measurement conditions; results from different conditions are not matched comparisons.

| Profile | Decode context | Prefill | C1 decode | C8 decode | Highest C at this context | Coding peak |
|---|---:|---:|---:|---:|---:|---:|
| [GLM-5.3 NVFP4-Spark · native MTP3 + mesh · 4 Sparks](performance/records/glm53-flash/spark-mtp3-mesh-20260905.md) | 8K | 2,703 (8K scout) | 48.2 | 168.8 | C16: 231.3 | — |
| [GLM-5.3 NVFP4-Spark · DFlash2 exact request-batch graphs · 4 Sparks](performance/records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md) | 16K | 2,717 (16K scout) | 43.05 | 134.3 | C16: 187.0 | — |
| [GLM-5.3 NVFP4 · DFlash2/B12X-KDA DCP4 · 4 Sparks](performance/records/glm53-flash/b12x-kda-dcp4-20260903.md) | 16K | 2,649 (16K scout) | 37.97 | — | C4: 90.36 | — |
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](performance/records/glm-3.5bpw/normalized-base-20260822.md) | 16K | 671 (16K) | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 16K | 1,926 (16K) | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 16K | 2,488 (16K) | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [DeepSeek-V4.1-Flash · Engram on NVMe · DSpark k=5 · 4 Sparks](performance/records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md) | short prompts, temp 0 | 1,873 (16K) / 2,058 (64K) | 56.2 | — | C6: 159.9 | 77.3 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 16K | 1,367 (16K) | 29.50 | 142.20 | C16: 184.39 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 16K | 1,964 (16K) | 35.07 | 191.02 | C8: 191.02 | 48.46 |

See [full results](docs/RESULTS.md) and the
[mesh validation report](performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for repeat counts, accuracy checks, settings, and limitations.

### R33 throughput observations

Status: **research-only**. These reported values lack a complete public
harness/method record and must not be treated as matched comparisons with the
benchmarks above. Their linked records separately qualify bounded functional
checks and describe the missing measurement details.

| NVFP4-Spark MTP3 + SparkCache profile | Context | Prefill tok/s, median of 3 | C1 output tok/s | C4 aggregate output tok/s |
|---|---:|---:|---:|---:|
| [R33 four-Spark ring](performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md) | 8K | 3,274 | 51.4 | 132.5 |
| [R33 two-Spark pair](performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md) | 8K | 2,340 | 33.1 | 66.1 |

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
| `spark_transport/` | Communication backends and maintained fabric planning |
| `runtime/` | Shared configuration/launch behavior, image selection and release inputs |
| `scripts/` | Preflight, deployment, and validation tools |
| [`profiles/`](profiles/) | Authoritative deployment definitions and quickstarts |
| [`recipes/`](recipes/) | Generated legacy recipe exports |
| [`performance/`](performance/) | Measurement methods, evidence, and receipts |
| `docs/` | Architecture, operations, development and explicit history |
| [`integrations/vllm/`](integrations/vllm/README.md) | Framework adapters |
| [`integrations/lil/`](integrations/lil/README.md) | Companion lifecycle and deployment integration |

See the [layout and compatibility guide](docs/development/layout.md) before adding or relocating implementation.

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
