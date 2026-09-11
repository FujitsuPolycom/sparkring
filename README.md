# SparkRing

SparkRing is an inference-serving stack with low-latency collective communication
for switchless clusters of NVIDIA GB10-based devices. It supports two-node pairs
and four-node rings; six-node rings are experimental. Model profiles use vLLM
and [SGLang](runtime/deepseek-v41-sglang/README.md).

The collective communication stack combines SIRCL, RoCEnante, and patched NCCL.
The high-speed data fabric needs no external Ethernet or InfiniBand switch;
administration and inference API traffic use the management network, typically
through each node's 10GbE NIC.

Profiles that need communication between nonadjacent nodes can use a virtual
mesh over the four-node ring. Custom RoCE RDMA routing and hardware forwarding
in the ConnectX network ASICs create those paths over the existing ring cables,
without routing the traffic through host CPUs. This provides mesh connectivity
over a physical ring; the selected profile defines its transport requirements.

The repository provides setup guides, launch tooling, model profiles,
reproducible benchmarks, and test results. Validation applies to the exact
configurations and workloads recorded with each profile.

## Setup

1. Choose a deployment below and check the [prerequisites](docs/operations/prerequisites.md).
2. Follow its quickstart for host setup, image selection and launch commands.
3. Run the [validation checks](docs/operations/profile-validation.md).

## Profiles

Bold entries are recommended. Values describe the linked default profile.
Context is the per-request limit; counts are rounded. DCP and KV choices appear
in the same order. See the
[full catalog](profiles/README.md) for exact settings and alternatives.

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status | Quickstart |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | DCP1/DCP4 | 1M / ([2.3M](performance/capacity-references.md)/[8.4M](performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md)) | [Optional](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Validated | [Guide](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | DCP1 | 1M / [1M](performance/capacity-references.md) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development | [Guide](profiles/deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | DCP1 | 1M / — | No | Experimental | [Guide](profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | DCP1 | 1M / [2.2M](profiles/deepseek-v41-flash-cycle/recipe.json) | No | Development | [Guide](profiles/deepseek-v41-flash-cycle/README.md) |
| GLM-5.2 | [EXL3 3.5bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) | DCP4 | 1M / [1.2M](profiles/glm52-exl3-r7-3.5bpw/recipe.json) | [Optional](profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) | Development | [Guide](profiles/glm52-exl3-r7-3.5bpw/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | DCP1 | 1M / [8.7M](profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development | [Guide](profiles/qwen38-27b-exl3-k5k6/README.md) |

### Two Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status | Quickstart |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | DCP1 | 1M / [1.1M](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | [Optional](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Validated | [Guide](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | DCP1 | 1M / [1M](performance/capacity-references.md) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development | [Guide](profiles/deepseek-v4-flash-0731-pair/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | DCP1 | 1M / [4.1M](profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development | [Guide](profiles/qwen38-27b-exl3-k5k6-pair/README.md) |

<!-- END GENERATED PROFILES -->

\* KV capacity changes with configuration and enabled features, including
SparkCache. Linked sources provide the settings and basis for each figure.

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images](runtime/images/README.md#container-images)
- [Contributing](CONTRIBUTING.md) and [repository layout](docs/development/layout.md)
- [Community discussions](https://github.com/FujitsuPolycom/sparkring/discussions)

## Acknowledgements

Thanks to the contributors to vLLM, NVIDIA NCCL, B12X and ExLlamaV3, whose
serving, communication and kernel components are used by SparkRing profiles.

The RoCEnante integration adapts communication work by Luke (`lukealonso`)
and other [Local Inference Lab](https://github.com/local-inference-lab/) contributors.
See the [RoCEnante provenance](third_party/b12x_roce/README.md#attribution-and-design-origins)
and [third-party notices](THIRD_PARTY_NOTICES.md) for source origins, adaptations
and licensing.

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
retain their own terms; review the selected model cards and
[third-party notices](THIRD_PARTY_NOTICES.md) before deployment.
