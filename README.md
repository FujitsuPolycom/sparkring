# SparkRing

SparkRing runs large language models across two or four NVIDIA DGX Sparks
cabled directly to each other, with no switch. Its collectives (SIRCL,
RoCEnante and patched NCCL) run over the ConnectX-7 cables, and on four-Spark
rings ConnectX hardware forwarding connects every Spark to every other over the
ring. Profiles use vLLM and [SGLang](runtime/deepseek-v41-sglang/README.md).

## Quick start

Cable the Sparks, check the [prerequisites](docs/operations/install.md), then
run one command on the Spark connected to your network.

Two Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

Four Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

It sets up every Spark, downloads the image and the model (or reuses a copy
already on the Sparks), and prints the API address when the model is ready. On
four Sparks it also applies the ring's
[ConnectX driver setting](docs/operations/install.md#four-spark-rings) and
repeats it at every boot. To run with Docker
Compose instead, see [Qwen on two Sparks with Compose](profiles/qwen38-flash-next-tp2/compose/README.md).

## Profiles

[Full profile catalog](profiles/README.md).

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[GLM-5.3-Flash](docs/operations/install.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 1 | 1M / — | No | Experimental |
| **[GLM-5.3-Flash](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 1/4 | 1M / ([2.3M](runtime/releases/shared-2026.09.3/correctness.json)/[8.4M](performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md)) | [Optional](profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-RL](docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 1 | 262K / — | No | Experimental |
| **[Qwen3.8-Flash-Next](profiles/qwen38-flash-next-qad-tp4/README.md)**<br>vLLM | [NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e) | 1 | 262K / [3.1M](runtime/releases/shared-2026.09.3/correctness.json) | No | Development |
| [DeepSeek-V4-Flash-0731](profiles/deepseek-v4-flash-0731/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 1 | 1M / [1M](performance/capacity-references.md) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development |
| [DeepSeek-V4-Flash-Vision-Exp](profiles/deepseek-v4-flash-vision-exp-tp4/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | 1 | 1M / — | No | Experimental |
| [DeepSeek-V4.1-Flash](profiles/deepseek-v41-flash-cycle/README.md)<br>vLLM | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 1 | 1M / [2.2M](profiles/deepseek-v41-flash-cycle/recipe.json) | No | Development |
| [DeepSeek-V4.1-Flash](profiles/deepseek-v41-flash-sglang-cycle/README.md)<br>SGLang | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | — | 262K / [1.5M](performance/records/deepseek-v41-flash/sglang-soak-20260912.md) | No | Development |
| [GLM-5.2](profiles/glm52-exl3-r7-3.5bpw/README.md)<br>vLLM | [EXL3 3.5bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) | 4 | 1M / [1.2M](profiles/glm52-exl3-r7-3.5bpw/recipe.json) | [Optional](profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) | Development |
| [Qwen3.8-27B](profiles/qwen38-27b-exl3-k5k6/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 1 | 1M / [8.7M](profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development |

### Two Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[GLM-5.3-Flash](docs/operations/install.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 1 | 262K / — | No | Experimental |
| **[GLM-5.3-Flash](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 1 | 1M / [1.1M](runtime/releases/shared-2026.09.3/correctness.json) | [Optional](profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-RL](docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 1 | 262K / — | No | Experimental |
| **[Qwen3.8-Flash-Next](profiles/qwen38-flash-next-tp2/README.md)**<br>vLLM | [NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e) | 1 | 262K / [2.9M](runtime/releases/shared-2026.09.3/correctness.json) | No | Development |
| [DeepSeek-V4-Flash-0731](profiles/deepseek-v4-flash-0731-pair/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 1 | 1M / [2.2M](performance/records/deepseek-v4-flash/image827a8e8c-tp2.json) | [Optional](profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development |
| [Qwen3.8-27B](profiles/qwen38-27b-exl3-k5k6-pair/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 1 | 1M / [4.1M](profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development |

<!-- END GENERATED PROFILES -->

\* KV capacity changes with configuration and enabled features, including
SparkCache. Linked sources provide the settings and basis for each figure.

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images and shared serving versions](runtime/images/README.md#container-images)
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
