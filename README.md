# SparkRing

SparkRing runs large language models distributed across two or four NVIDIA GB10-based devices
cabled directly to each other, with no switch. Its collectives (SIRCL,
RoCEnante and patched NCCL) run over the ConnectX-7 DACs, and on four-Spark
rings ConnectX hardware forwarding connects every Spark to every other over the
ring. Profiles use vLLM and SGLang.

## Quick start

Check the [requirements](docs/operations/install.md#requirements), cable the
Sparks as shown there, then run one command on the Spark connected to your
network.

Two Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

Four Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

The command sets up every Spark, downloads the image and the model (or reuses a
copy already on the Sparks), and prints the API address when the model is ready.
Each model also serves a live [status dashboard](docs/operations/dashboard.md).
On four Sparks it also applies the ring's
[ConnectX driver setting](docs/operations/install.md#four-spark-rings) and
repeats it at every boot. Run the same command again to update SparkRing and
the model; files already on the Sparks are reused. To run with Docker
Compose instead, see [Qwen on two Sparks with Compose](profiles/qwen38-flash-next-tp2/compose/README.md).

To install GLM or MiMo instead, use a `--profile` value from [Profiles](#profiles).
All commands and flags: [SparkRing commands](docs/operations/commands.md).

## Profiles

`sparkring install --profile` accepts these profiles:

| Model | Sparks | `--profile` value | API port | Decode (tok/s, one user) | Prefill 16K (tok/s) |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 2 | `qwen38-flash-next-tp2` | 8000 | 49.3–84.8 | 4,077 |
| Qwen3.8-Flash-Next | 4 | `qwen38-flash-next-qad-tp4` | 8015 | 70.7–118.1 | 5,024 |
| GLM-5.3-Flash | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | 32.3–40.0 | 2,440 |
| GLM-5.3-Flash | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | 59.0–73.2 | 2,900 |
| MiMo-V2.6-Flash-RL | 2 | `mimo-v26-flash-rl-tp2` | 8020 | 25.7–62.8 | 3,802 |
| MiMo-V2.6-Flash-RL | 4 | `mimo-v26-flash-rl-tp4` | 8020 | 44.8–111.5 | 4,253 |

Decode ranges from prose to JSON prompts. Measurements:
[Qwen](performance/records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md),
[GLM and MiMo](performance/records/images/dev-20260925-qwendecode-installer-profiles-20260926.md).

Older profiles, SparkCache variants and other models are in the
[full profile catalog](profiles/README.md). Each has its own setup guide.

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
