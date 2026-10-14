# SparkRing

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-flash-tp2-dcp2`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp8`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`). https://local-inference-lab.ai/

SparkRing serves large language models on two or more NVIDIA DGX Sparks cabled
directly over ConnectX-7, with no switch, and installs them with one command.

## Quick start

1. Cable two to eight Sparks directly: a pair from p0 to p0, or a line or
   ring from each Spark's p0 to the next Spark's p1
   ([requirements](docs/operations/install.md#requirements)).
2. Pick a `--profile` from the [table below](#profiles), or build the commands
   in the [Install Builder](https://fujitsupolycom.github.io/sparkring/).
3. On the Spark connected to your network, run:

```bash
curl -fsSL https://fujitsupolycom.github.io/sparkring/install.sh | bash -s -- --profile PROFILE
```

This installs the release the [Install Builder](https://fujitsupolycom.github.io/sparkring/)
describes; the [pinned install command](docs/operations/install-reference.md#get-the-package)
installs `main` or another commit.

The installer asks before it changes anything: `--plan` previews and `--yes`
approves, and running it again updates or switches models.

More: [documentation](docs/README.md) · [commands](docs/operations/commands.md) ·
[status dashboard](docs/operations/dashboard.md) ·
[Docker Compose files](docs/operations/compose-files.md) ·
[pinned install command](docs/operations/install-reference.md#get-the-package)

## SIRCL

SIRCL is SparkRing's collective layer for switchless Sparks, including Sparks
reached through another Spark's ConnectX-7 relay
([architecture](docs/architecture/sircl.md)).
[net-sircl](https://github.com/FujitsuPolycom/net-sircl) brings it to any NCCL
application as a network plugin (research-only).

## Profiles

| Model | Checkpoint | Sparks | `--profile` value | Context | KV tokens | Thinking |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [CSF (NVFP4/MXFP8), Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD) | 2 | `qwen38-flash-next-tp2` | 262K | [2.5M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-kv-pools-20261012.md) | on · xhigh |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-tp4` | 262K | [2.8M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md) | on · xhigh |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 2 | `glm53-flash-tp2` | 1M | [1.5M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md) | always · high |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD), DCP2, research | 2 | `glm53-flash-tp2-dcp2` | 1M | [4.1M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md) | always · high |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 4 | `glm53-flash-tp4` | 1M | [5.7M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md) | always · high |
| GLM-5.3 | [NVFP4, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4) | 8 | `glm53-nvfp4-tp8` | 1M | [3.4M](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-kv-pools-20261012.md) | always · high |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 2 | `mimo-v26-flash-mopd-tp2` | 262K | [656K](performance/records/images/dev-20261004-kraken-matrix-20261004.md) | on |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 4 | `mimo-v26-flash-mopd-tp4` | 262K | [2.2M](performance/records/images/dev-20261004-kraken-matrix-20261004.md) | on |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 1M | [12.8M](performance/records/images/dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-kv-ring8-20261014.md) | on · high |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 262K | [2.9M](performance/records/images/dev-20261004-kraken-matrix-20261004.md) | on · xhigh |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 262K | [3.1M](performance/records/images/dev-20261004-kraken-matrix-20261004.md) | on · xhigh |

More profiles and options: [profile catalog](profiles/README.md).

## Speed

Output tok/s measured with llm-inference-bench 0.7.6 at temperature 1, one
start per profile with no install options; decode and prefill at 32K context
([benchmarks](performance/benchmarks.md)).

| Model | Checkpoint | Sparks | `--profile` | Decode, 1 / 4 / 8 streams | Prefill, 32K prompt | Coding peak | Layout |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 8 | `deepseek-v41-flash-tp8` | 73.7 / 174.2 / 258.9 | 4,851 | 133.8 | ring of eight |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 70.6 / 136.7 / 194.3 | 4,388 | 102.4 | ring of four |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 70.4 / 139.1 / 198.0 | 4,426 | 101.7 | path of four |
| GLM-5.3 | NVFP4 | 8 | `glm53-nvfp4-tp8` | 45.8 / 93.7 / 129.1 | 1,387 | 51.4 | ring of eight |
| GLM-5.3-Flash | CSF | 4 | `glm53-flash-tp4` | 68.1 / 135.3 / 203.5 | 3,165 | 79.7 | path of four |
| GLM-5.3-Flash | CSF | 2 | `glm53-flash-tp2` | 37.8 / 83.1 / 112.9 | 2,420 | 49.0 | pair |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-tp4` | 73.0 / 203.6 / 288.1 | 4,553 | 111.5 | ring of four |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-tp4` | 77.9 / 209.5 / 274.3 | 4,377 | 111.3 | path of four |
| Qwen3.8-Flash-Next | CSF | 2 | `qwen38-flash-next-tp2` | 68.1 / 149.7 / 219.0 | 3,716 | 90.7 | pair |

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md)
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
NVFP4 checkpoints,
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
and their [vLLM fork](https://github.com/local-inference-lab/vllm).

SparkRing includes software derived from RoCEnante (`b12x/comm/roce`, Local
Inference Lab), written by Jason (`original-el8`) in local-inference-lab/b12x
[PR 295](https://github.com/local-inference-lab/b12x/pull/295), with
dual-rail striping and graph-collective work by Luke Alonso (`lukealonso`),
licensed under Apache-2.0; b12x now lives in FlashInfer at
`flashinfer/experimental/b12x`
([provenance](third_party/b12x_roce/README.md#attribution-and-design-origins)).

SparkRing's NCCL patches follow Joseph Rose's
[nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless).
Thanks to Eugene Rakhmatulin for the `eugr/spark-vllm-b12x` base image, and to
the vLLM, NVIDIA NCCL and ExLlamaV3 contributors. The
[third-party notices](THIRD_PARTY_NOTICES.md) give origins and licenses.

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
keep their own terms; check the model cards and
[third-party notices](THIRD_PARTY_NOTICES.md).
