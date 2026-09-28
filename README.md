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
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

Four Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

The command asks before it installs the SparkRing package and before it
changes any Spark; `--yes` answers both, and `--plan` prints what the command
would do, also for an upgrade, and changes nothing. It sets
up every Spark, downloads the image and model unless the Sparks already hold
them, and prints the API address when the model is ready. Each model serves a
[status dashboard](docs/operations/dashboard.md), and on four Sparks the
installer also applies the ring's
[ConnectX driver setting](docs/operations/install.md#four-spark-rings) at every
boot. Running it again upgrades to the branch's newest release; the
[pinned command](docs/operations/install-reference.md#get-the-package) repeats
an installation exactly.

Other models: a `--profile` value from [Profiles](#profiles). Docker Compose:
[Qwen on two Sparks with Compose](profiles/qwen38-flash-next-tp2/compose/README.md).
All commands and flags: [SparkRing commands](docs/operations/commands.md).

## Profiles

`sparkring install --profile` accepts these profiles:

| Model | Checkpoint | Sparks | `--profile` value | API port | Decode, 1 / 8 / 16 users (tok/s) | Prefill 64K (tok/s) |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 2 | `qwen38-flash-next-tp2` | 8000 | 54.9 / 197 / 300 | 3,686 |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-qad-tp4` | 8015 | 78.4 / 261 / 410 | 4,534 |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 8000 | 57.6 / 181 / 258 | 3,628 |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 8015 | — | — |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | 38.0 / 108 / — | 2,419 |
| GLM-5.3-Flash | [NVFP4 Spark, Local Inference Lab](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | 65.9 / 208 / 267 | 3,631 |
| MiMo-V2.6-Flash-RL | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 2 | `mimo-v26-flash-rl-tp2` | 8020 | 31.0 / 123 / 193 | 2,473 |
| MiMo-V2.6-Flash-RL | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 4 | `mimo-v26-flash-rl-tp4` | 8020 | 47.2 / 197 / 340 | 4,002 |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | 63.5 / 195 / 277 | 4,271 |

Decode is the aggregate output rate from
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
0.6.2 at temperature 1.0 with no added context, 20 s per cell; prefill is one
cold 64K-token prompt divided by its time to first token. Every profile runs
image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`; the rows were
measured on its parent, `dev-20260928-toolchoice-cuda1342-nccl2323-status032`,
which differs only in the runtime-status dashboard, except the four-Spark Qwen
row, measured on `dev-20260927-b12xcache-cuda1342-nccl2323-status032`. The
tokens accepted per speculative step follow the sampled text at temperature
1.0, so one-stream decode varies between single runs: MiMo on two Sparks has
measured from 31.0 to 44.9 tok/s at the same step rate.
Measurements:
Qwen [two](performance/records/images/dev-20260928-toolchoice-qwen38-flash-next-tp2-20260928.md) and [four](performance/records/images/dev-20260927-b12xcache-qwen38-flash-next-tp4-20260927.md) Sparks,
Swift [two](performance/records/images/dev-20260928-toolchoice-swift15-qwen38-flash-next-tp2-20260928.md),
GLM [two](performance/records/images/dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp2-20260928.md) and [four](performance/records/images/dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp4-20260928.md),
MiMo [two](performance/records/images/dev-20260928-toolchoice-mimo-v26-flash-rl-tp2-20260928.md) and [four](performance/records/images/dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928.md),
DeepSeek [four](performance/records/images/dev-20260928-toolchoice-deepseek-v41-flash-tp4-20260928.md).

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
