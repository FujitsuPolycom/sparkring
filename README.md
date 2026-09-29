# SparkRing

SparkRing runs large language models across two or four NVIDIA GB10-based
devices connected by direct-attach cables, without a network switch. SIRCL,
RoCEnante and patched NCCL provide collective transport and communication over
the ConnectX-7 links. On four-device rings, ConnectX hardware forwarding
provides connectivity between all devices over the existing ring cables. Model
profiles support vLLM and SGLang.

## Quick start

1. Check the [requirements](docs/operations/install.md#requirements) and cable
   your Sparks as shown there.
2. Pick a `--profile` value from the [table below](#profiles) for your preferred model and
   number of Sparks.
3. On the Spark connected to your network, run:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile PROFILE
```

The installer requires approval for several operations. Use the `--yes` flag to auto-accept all prompts. Use `--plan` to display what the installer would do, without actually executing the install. The installer sets up every gb10 device, downloads the
image/models and distributes them if needed, and prints the API address when ready.
Run the same command again to update a profile or swap to another. Each model has a
[status dashboard](docs/operations/dashboard.md); on four Sparks the installer
also applies a [ConnectX driver setting](docs/operations/install.md#four-spark-rings)
at every boot.

More: [all commands](docs/operations/commands.md) ·
[Docker Compose files](docs/operations/compose-files.md) ·
[pinned install command](docs/operations/install-reference.md#get-the-package)

## Profiles

`sparkring install --profile` accepts these profiles:

| Model | Checkpoint | Sparks | `--profile` value | API port | Decode at 16K context, 1 / 4 / 8 / 16 users (tok/s) | Prefill 64K (tok/s) |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 2 | `qwen38-flash-next-tp2` | 8000 | 45.1 / 120 / 169 / 243 | 3,649 |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-qad-tp4` | 8015 | 64.9 / 167 / 235 / 341 | 4,524 |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | 38.0 / 78 / 106 / 85\* | 2,454 |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | 62.0 / 137 / 199 / 261 | 3,610 |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 2 | `mimo-v26-flash-mopd-tp2` | 8020 | 37.5 / 78 / 115 / 184 | 2,754 |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 4 | `mimo-v26-flash-mopd-tp4` | 8020 | 47.7 / 119 / 167 / 295 | 3,959 |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | 62.2 / 135 / 206 / 278 | 4,302 |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 8000 | — | — |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 8015 | — | — |

Decode is the total output rate with 1, 4, 8 and 16 users at once, each with
16K tokens of context, averaged over three runs of
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
0.6.2 at temperature 1.0. Prefill, from the same runs, is one cold 64K-token
prompt divided by its time to first token. \* The two-Spark GLM profile serves 8
requests at a time, so 16 users queue. Swift hasn't been measured this way yet.
Full results for each profile, from 8K to 128K context:
Qwen [two](performance/records/images/dev-20260928-plainstatus-qwen38-flash-next-tp2-context-20260929.md) and [four](performance/records/images/dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929.md) Sparks,
GLM [two](performance/records/images/dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929.md) and [four](performance/records/images/dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929.md),
MiMo [two](performance/records/images/dev-20260928-plainstatus-mimo-v26-flash-mopd-tp2-context-20260929.md) and [four](performance/records/images/dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929.md),
DeepSeek [four](performance/records/images/dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929.md).

Other setups, including SparkCache variants and models the installer doesn't
cover, each have their own guide in the [profile catalog](profiles/README.md).

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
