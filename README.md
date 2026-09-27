# SparkRing

SparkRing runs large language models across two or four NVIDIA GB10-based
devices connected by direct-attach cables, without a network switch. SIRCL,
RoCEnante and patched NCCL provide collective transport and communication over
the ConnectX-7 links. On four-device rings, ConnectX hardware forwarding
provides connectivity between all devices over the existing ring cables. Model
profiles support vLLM and SGLang.

## Quick start

1. Check the [requirements](docs/operations/install.md#requirements).
2. Cable your GB10 devices as the requirements show.
3. On the device connected to your network, run the installer command with the
   profile for your model and device count. See also
   [SparkRing commands](docs/operations/commands.md).

Prefer Docker Compose? [Compose files for every profile](docs/operations/compose-files.md).

Two Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

Four Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

The command asks before it installs the SparkRing package on that device and
again before it changes any Spark; `--yes` answers both, and `--plan` installs
nothing. It sets up every Spark, downloads the image and the model (or reuses a
copy already on the Sparks), and prints the API address when the model is ready.
Each model also serves a live [status dashboard](docs/operations/dashboard.md).
On four Sparks it also applies the ring's
[ConnectX driver setting](docs/operations/install.md#four-spark-rings) and
repeats it at every boot. Running the same command again upgrades to the
branch's newest SparkRing, image and model settings and reuses files already
on the Sparks; to repeat an installation exactly, use the
[pinned command](docs/operations/install-reference.md#get-the-package). To run
with Docker Compose instead, see [Qwen on two Sparks with Compose](profiles/qwen38-flash-next-tp2/compose/README.md).

To install GLM, MiMo or DeepSeek instead, use a `--profile` value from [Profiles](#profiles).
All commands and flags: [SparkRing commands](docs/operations/commands.md).

## Profiles

`sparkring install --profile` accepts these profiles. Every profile runs image
`dev-20260927-h2dstaging-cuda1342-nccl2323-status031`. The four-Spark GLM and
DeepSeek figures were measured on it. The Qwen, MiMo and two-Spark GLM figures
were measured on its parent image,
`dev-20260925-qwendecode-cuda1342-nccl2323-status031`, which lacks vLLM's
host-to-device staging fix, and have not been measured on the selected image.

| Model | Checkpoint | Sparks | `--profile` value | API port | Decode (tok/s, one user) | Decode (tok/s, 8 / 16 users) | Prefill 16K (tok/s) |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 2 | `qwen38-flash-next-tp2` | 8000 | 49.3–84.8 | 198 / 284 | 4,077 |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-qad-tp4` | 8015 | 70.7–118.1 | 276 / 408 | 5,024 |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | 32.3–40.0 | 104 / — | 2,440 |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | 60.6–77.5 | 200 / 278 | 3,399 |
| MiMo-V2.6-Flash-RL | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 2 | `mimo-v26-flash-rl-tp2` | 8020 | 25.7–62.8 | — | 3,802 |
| MiMo-V2.6-Flash-RL | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 4 | `mimo-v26-flash-rl-tp4` | 8020 | 44.8–111.5 | — | 4,253 |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | 49.8–110.8 | 146 / 216 | 4,406 |

One-user decode ranges from prose to JSON prompts at temperature 0. Decode for
8 and 16 users is the aggregate llm-inference-bench rate at temperature 1.0
with no added context; the GLM pair profile serves at most 8 requests, and
MiMo is not yet measured this way. DeepSeek's figures were measured on the
selected image with its profile's settings: one-user decode and prefill on an
installed deployment, and 8 and 16 users in its settings search. Measurements:
[Qwen](performance/records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md),
[GLM](performance/records/images/dev-20260925-qwendecode-glm-prefill-20260926.md),
[MiMo](performance/records/images/dev-20260925-qwendecode-installer-profiles-20260926.md),
[DeepSeek](performance/records/images/dev-20260927-h2dstaging-deepseek-v41-tp4-20260927.md).

The [full profile catalog](profiles/README.md) lists every profile, including
SparkCache variants and models that `sparkring install` does not set up. Each
has its own setup guide.

## Documentation

- [Benchmarks and test results](performance/benchmarks.md)
- [Architecture](docs/architecture/overview.md) and [mesh host setup](docs/GLM53_SPARK_MESH_HOST_SETUP.md)
- [Container images and shared serving versions](runtime/images/README.md#container-images)
- [Contributing](CONTRIBUTING.md) and [repository layout](docs/development/layout.md)
- [Community discussions](https://github.com/FujitsuPolycom/sparkring/discussions)

## Acknowledgements

SparkRing is built on the work of [Local Inference Lab](https://local-inference-lab.ai/),
an independent nonprofit that develops open-source software and research for
running AI inference on personal hardware
([GitHub](https://github.com/local-inference-lab),
[Hugging Face](https://huggingface.co/local-inference-lab),
[Discord](https://discord.com/invite/localinferencelab)). SparkRing uses:

- [B12X](https://github.com/local-inference-lab/b12x), the kernel library for
  NVIDIA Blackwell whose kernels and loader run the Qwen, GLM and MiMo
  installer profiles on GB10.
- The [Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
  and [GLM-5.3-Flash NVFP4 Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)
  checkpoints that the Qwen and GLM profiles serve.
- RoCEnante, the RDMA transport for decode collectives, which originates with
  Luke (`lukealonso`) and other Local Inference Lab contributors; SparkRing
  adapts it ([provenance](third_party/b12x_roce/README.md#attribution-and-design-origins)).
- [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench),
  which measured the throughput matrices on the profile pages.
- Their [vLLM fork](https://github.com/local-inference-lab/vllm), which several
  catalog profiles build on.

Joseph Rose's [nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless)
ran NCCL over RoCE on four DGX Sparks without a switch, relaying traffic
between diagonal Sparks in two hops. SparkRing's NCCL compatibility patches
follow its approach to skipping NCCL's Tree and PAT connections.

Thanks also to Eugene Rakhmatulin for the `eugr/spark-vllm-b12x` base of the
installer image, and to the contributors to vLLM, NVIDIA NCCL and ExLlamaV3.
The [third-party notices](THIRD_PARTY_NOTICES.md) give source origins,
adaptations and licenses.

## License

SparkRing code is [Apache-2.0](LICENSE). Model weights and bundled components
retain their own terms; review the selected model cards and
[third-party notices](THIRD_PARTY_NOTICES.md) before deployment.
