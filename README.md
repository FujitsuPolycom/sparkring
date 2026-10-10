# SparkRing

SparkRing serves large language models on two to eight NVIDIA DGX Sparks
cabled directly to each other over ConnectX-7, with no switch. vLLM serves the
model; SparkRing provides the collectives (SIRCL), the tested profiles and
images, and a one-command installer.

## Quick start

1. Cable your Sparks as a pair, a ring of four or a ring of eight
   ([requirements](docs/operations/install.md#requirements)).
2. Pick a `--profile` from the [table below](#profiles), or build the commands
   in the [Install Builder](https://fujitsupolycom.github.io/sparkring/).
3. On the Spark connected to your network, run:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile PROFILE
```

The installer asks before it changes anything: `--plan` previews and `--yes`
approves, and running it again updates or switches models.

More: [documentation](docs/README.md) · [commands](docs/operations/commands.md) ·
[status dashboard](docs/operations/dashboard.md) ·
[Docker Compose files](docs/operations/compose-files.md) ·
[pinned install command](docs/operations/install-reference.md#get-the-package)

## SIRCL

SIRCL is SparkRing's collective layer for Sparks cabled without a switch, and
it replaces NCCL for vLLM's tensor-parallel collectives. It reaches Sparks that
aren't cabled to each other through the ConnectX-7 hardware relay. It picks its
schedule by message size from a tuning table measured on real rings, and
libsircl offers the same protocol behind NCCL's C API
([architecture](docs/architecture/sircl.md)).

## Profiles

| Model | Checkpoint | Sparks | `--profile` value | API port | Thinking |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 2 | `qwen38-flash-next-tp2` | 8000 | on · xhigh |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-qad-tp4` | 8015 | on · xhigh |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | always · max |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | always · max |
| GLM-5.3 | [NVFP4, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4) | 8 | `glm53-nvfp4-tp8` | 8015 | always · max |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 2 | `mimo-v26-flash-mopd-tp2` | 8020 | on |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 4 | `mimo-v26-flash-mopd-tp4` | 8020 | on |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | on · high |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 8000 | on · xhigh |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 8015 | on · xhigh |

`glm53-flash-tp2` and `glm53-flash-tp4` are aliases of the two GLM-5.3-Flash
profiles. The other eight-Spark profiles are Experimental
([profile catalog](profiles/README.md)).

## Speed

Decode at 32K context, output tok/s:

| Model | Checkpoint | Sparks | `--profile` | 1 / 4 / 8 streams | First token, 32K prompt |
|---|---|---|---|---|---|
| GLM-5.3 | NVFP4 | 8 | `glm53-nvfp4-tp8` | 47.9 / 99.8 / 143.7 | 24.4 s |
| GLM-5.3-Flash | CSF | 4 | `glm53-flash-nvfp4-spark-tp4` | 64.2 / 126.6 / 218.7 | 10.5 s |
| GLM-5.3-Flash | CSF | 2 | `glm53-flash-nvfp4-spark-tp2` | 39.5 / 87.1 / 123.2 | 15.3 s |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-qad-tp4` | 62.1 / 149.4 / 234.0 | 7.0 s |
| Qwen3.8-Flash-Next | QAD step 5500, MXFP8 attention | 2 | `qwen38-flash-next-tp2` | 51.1 / 117.1 / 179.0 | 8.4 s |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 63.8 / 127.5 / 182.8 | 7.6 s |

Each row's record is under `performance/records/`.

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images](runtime/images/README.md#container-images)
- [Contributing](CONTRIBUTING.md) and [repository layout](docs/development/layout.md)
- [Community discussions](https://github.com/FujitsuPolycom/sparkring/discussions)

## Acknowledgements

SparkRing builds on [Local Inference Lab](https://local-inference-lab.ai/)
([GitHub](https://github.com/local-inference-lab) ·
[Hugging Face](https://huggingface.co/local-inference-lab) ·
[Discord](https://discord.com/invite/localinferencelab)): the
[B12X](https://github.com/local-inference-lab/b12x) kernel library, the
[Qwen3.8-Flash-Next](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
and [GLM-5.3-Flash Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)
NVFP4 checkpoints, the RoCEnante RDMA transport by Luke (`lukealonso`) and
contributors ([provenance](third_party/b12x_roce/README.md#attribution-and-design-origins)),
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
and their [vLLM fork](https://github.com/local-inference-lab/vllm).

SparkRing's NCCL patches follow Joseph Rose's
[nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless).
Thanks to Eugene Rakhmatulin for the `eugr/spark-vllm-b12x` base image, and to
the vLLM, NVIDIA NCCL and ExLlamaV3 contributors. The
[third-party notices](THIRD_PARTY_NOTICES.md) give origins and licenses.

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
keep their own terms; check the model cards and
[third-party notices](THIRD_PARTY_NOTICES.md).
