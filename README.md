# SparkRing

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp8`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`).

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

**[net-sircl](https://github.com/FujitsuPolycom/net-sircl)** brings SIRCL to
any NCCL application. It's an NCCL network plugin, published from this
repository's transport: PyTorch, vLLM, SGLang and other NCCL programs run
unmodified on directly cabled DGX Sparks, including peers reached through the
ConnectX-7 relays, with NCCL keeping its own algorithms and CUDA-graph support.
Version 0.1.0 is research-only and tested with NCCL 2.32.3.

## Profiles

| Model | Checkpoint | Sparks | `--profile` value | API port | Thinking |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [CSF (NVFP4/MXFP8), Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD) | 2 | `qwen38-flash-next-tp2` | 8000 | on · xhigh |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-tp4` | 8015 | on · xhigh |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 2 | `glm53-flash-tp2` | 8000 | always · max |
| GLM-5.3-Flash | [CSF (NVFP4/MXFP8)](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | 4 | `glm53-flash-tp4` | 8015 | always · max |
| GLM-5.3 | [NVFP4, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4) | 8 | `glm53-nvfp4-tp8` | 8015 | always · max |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 2 | `mimo-v26-flash-mopd-tp2` | 8020 | on |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 4 | `mimo-v26-flash-mopd-tp4` | 8020 | on |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | on · high |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 8000 | on · xhigh |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 8015 | on · xhigh |

`glm53-flash-tp2` and `glm53-flash-tp4` are short names of the profile IDs
`glm53-flash-nvfp4-spark-tp2` and `glm53-flash-nvfp4-spark-tp4`; both names
install the CSF checkpoint on this image. `qwen38-flash-next-tp2` installs the
Qwen CSF checkpoint on this image and serves it as
`Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2`; `--checkpoint qad-step5500-ple1000`
installs step 5500 instead. `qwen38-flash-next-tp4` is the short
name of the profile ID `qwen38-flash-next-qad-tp4`. The other eight-Spark
profiles are Experimental ([profile catalog](profiles/README.md)).

## Speed

Output tok/s of this release's profiles installed with no options, measured
with llm-inference-bench 0.7.6 at temperature 1 (`--concurrency 1,2,4,8
--contexts 8k,16k,32k,64k,128k --duration 20 --max-tokens 2048 --coding-peak`),
one start each, on release 2026.10.2's image `d52737a1`. Decode and prefill are
at 32K context; the coding peak is the median of five single-stream code
generations.

| Model | Checkpoint | Sparks | `--profile` | Decode, 1 / 4 / 8 streams | Prefill, 32K prompt | Coding peak | Layout |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 8 | `deepseek-v41-flash-tp8` | 73.7 / 174.2 / 258.9 | 4,851 | 133.8 | ring of eight |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 70.6 / 136.7 / 194.3 | 4,388 | 102.4 | ring of four |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 70.4 / 139.1 / 198.0 | 4,426 | 101.7 | path of four: four Sparks of a ring of eight |
| GLM-5.3 | NVFP4 | 8 | `glm53-nvfp4-tp8` | 45.8 / 93.7 / 129.1 | 1,387 | 51.4 | ring of eight |
| GLM-5.3-Flash | CSF | 4 | `glm53-flash-tp4` | 68.1 / 135.3 / 203.5 | 3,165 | 79.7 | path of four: four Sparks of a ring of eight |
| GLM-5.3-Flash | CSF | 2 | `glm53-flash-tp2` | 37.8 / 83.1 / 112.9 | 2,420 | 49.0 | pair |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-tp4` | 73.0 / 203.6 / 288.1 | 4,553 | 111.5 | ring of four |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-tp4` | 77.9 / 209.5 / 274.3 | 4,377 | 111.3 | path of four: four Sparks of a ring of eight |
| Qwen3.8-Flash-Next | CSF | 2 | `qwen38-flash-next-tp2` | 68.1 / 149.7 / 219.0 | 3,716 | 90.7 | pair |

Records: [ring of eight](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md), [rings of four and pairs](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring4-20261012.md),
[four-Spark GLM-5.3-Flash and Qwen](performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md). The DeepSeek path-of-four row ran
beside two other deployments measured on the rest of the ring at the same time;
the GLM-5.3-Flash and Qwen path-of-four rows ran on the two halves of one ring
of eight, their benchmarks overlapping for two minutes. Speed varies from start
to start: the same deployment measured twice differed by up to about 10 %, and
Qwen's four-Spark profile on a second ring of four measured 78.4 / 207.6 / 280.0
with a 4,569 tok/s prefill and a 110.4 coding peak.

Release 2026.10.3's [record](runtime/releases/2026.10.3/README.md#serving-defaults)
has these measurements' conditions. On four Sparks,
`--checkpoint qad-step5500-mxfp8-attention` decodes 6-8 % faster than the
default checkpoint in the median of two starts on the same ring.

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images](runtime/images/README.md#container-images)
- [net-sircl](https://github.com/FujitsuPolycom/net-sircl), SIRCL as an NCCL
  network plugin ([above](#sircl))
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
