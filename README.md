# SparkRing

SparkRing runs vLLM inference across pairs and four-node rings of NVIDIA
GB10 devices. Its communication stack combines SIRCL, RoCEnante and patched
NCCL to connect the high-speed data fabric without an external switch.
Administration and API traffic use the management network.

SparkRing is experimental. Six-node deployments are research-only.

## Setup

1. Choose a deployment below and check the [prerequisites](docs/operations/prerequisites.md).
2. Follow its quickstart for host setup, image selection and launch commands.
3. Run the [validation checks](docs/operations/profile-validation.md).

## Profiles

Bold entries are recommended. Values describe the linked default profile;
the SparkCache column links alternate configurations. Context is the per-request
limit; KV is the pool capacity. Counts are rounded; `~` marks an estimate.
See the [full catalog](profiles/README.md) for exact counts, testing scope and retired setups.

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model / features | Layout | Context (tokens) | KV (tokens) | SparkCache | Status | Quickstart |
|---|---|---:|---:|---|---|---|
| **GLM-5.3 Flash NVFP4-Spark · MTP3** | TP4/DCP1 | 1M | [~2.3M](performance/capacity-references.md) | Included · [off](profiles/glm53-flash-spark-tp4-dcp1/README.md) | Validated | [Guide](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | 1M | [~1M](performance/capacity-references.md) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development | [Guide](profiles/deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | TP4/DCP1 | 1M | — | No | Experimental | [Guide](profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | TP4/DCP1 | 430K | [2.2M](profiles/deepseek-v41-flash-cycle/recipe.json) | No | Development | [Guide](profiles/deepseek-v41-flash-cycle/README.md) |
| GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 | TP4/DCP4 | 1M | [1.2M](profiles/glm52-exl3-r7-3.5bpw/recipe.json) | [Optional](profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) | Development | [Guide](profiles/glm52-exl3-r7-3.5bpw/README.md) |
| GLM-5.3 Flash NVFP4-Spark · switched | TP4/DCP1 | 1M | [~2.3M](performance/capacity-references.md) | No | Experimental | [Guide](profiles/glm53-flash-spark-tp4-switched/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP4/DCP1 | 1M | [8.7M](profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development | [Guide](profiles/qwen38-27b-exl3-k5k6/README.md) |

### Two Sparks

| Model / features | Layout | Context (tokens) | KV (tokens) | SparkCache | Status | Quickstart |
|---|---|---:|---:|---|---|---|
| **GLM-5.3 Flash NVFP4-Spark · MTP3** | TP2/DCP1 | 1M | [1.1M](runtime/profiles/glm53-flash-spark-tp2/README.md) | Included · [off](profiles/glm53-flash-spark-tp2-dcp1/README.md) | Validated | [Guide](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | 1M | [~1M](performance/capacity-references.md) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development | [Guide](profiles/deepseek-v4-flash-0731-pair/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP2/DCP1 | 1M | [4.1M](profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development | [Guide](profiles/qwen38-27b-exl3-k5k6-pair/README.md) |

<!-- END GENERATED PROFILES -->

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images](runtime/images/README.md#container-images)
- [Contributing](CONTRIBUTING.md) and [repository layout](docs/development/layout.md)
- [Community discussions](https://github.com/FujitsuPolycom/sparkring/discussions)

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
