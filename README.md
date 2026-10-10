# SparkRing

SparkRing is a switchless serving stack for four NVIDIA GB10-based devices
cabled in a ring. vLLM and SGLang serve the model; SparkRing supplies the
collective transports over the ConnectX-7 links, the tested profiles and
images, and a one-command installer. Two-Spark profiles built from the same
work are included, and the installer can run two of them on one ring, one per
pair of Sparks, each with its own API endpoint.

## Quick start

1. Cable your Sparks as shown in the [requirements](docs/operations/install.md#requirements).
2. Pick a `--profile` from the [table below](#profiles), or build the commands
   in the [Install Builder](https://fujitsupolycom.github.io/sparkring/).
3. On the Spark connected to your network, run:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile PROFILE
```

The installer sets up every Spark, downloads and distributes the image and
model, and prints the API address when the model is ready. It asks before it
changes anything: `--yes` approves, `--plan` previews. Run it again to update
or switch models.

More: [documentation](docs/README.md) · [commands](docs/operations/commands.md) ·
[status dashboard](docs/operations/dashboard.md) ·
[Docker Compose files](docs/operations/compose-files.md) ·
[pinned install command](docs/operations/install-reference.md#get-the-package)

## Profiles

| Model | Checkpoint | Sparks | `--profile` value | API port | Thinking | Decode at 16K context, 1 / 4 / 8 / 16 users (tok/s) | Prefill 64K (tok/s) |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 2 | `qwen38-flash-next-tp2` | 8000 | on · xhigh | 46.5 / 119 / 165 / 230 | 3,590 |
| Qwen3.8-Flash-Next | [NVFP4 QAD, Local Inference Lab](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 4 | `qwen38-flash-next-qad-tp4` | 8015 | on · xhigh | 64.0 / 173 / 239 / 328 | 4,424 |
| GLM-5.3-Flash | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) where the image reads it, else [NVFP4 Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)†; Local Inference Lab | 2 | `glm53-flash-nvfp4-spark-tp2` | 8000 | always · max | 36.0 / 73 / 100 / 67\* | 2,460 |
| GLM-5.3-Flash | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) where the image reads it, else [NVFP4 Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)†; Local Inference Lab | 4 | `glm53-flash-nvfp4-spark-tp4` | 8015 | always · max | 62.1 / 133 / 200 / 231 | 3,500 |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 2 | `mimo-v26-flash-mopd-tp2` | 8020 | on | 30.2 / 76 / 120 / 182 | 2,806 |
| MiMo-V2.6-Flash-MOPD | [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) | 4 | `mimo-v26-flash-mopd-tp4` | 8020 | on | 62.8 / 124 / 181 / 317 | 4,094 |
| DeepSeek-V4.1-Flash | [FP8/MXFP4, DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | `deepseek-v41-flash-tp4` | 8015 | on · high | 57.0 / 135 / 201 / 275 | 4,396 |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 2 | `swift15-qwen38-flash-next-tp2` | 8000 | on · xhigh | 43.5 / 110 / 157 / 204 | 3,567 |
| Swift-1.5-Qwen3.8-Flash-Next | [NVFP4 experts/BF16, UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | 4 | `swift15-qwen38-flash-next-tp4` | 8015 | on · xhigh | 64.7 / 161 / 232 / 340 | 4,404 |

Decode is total output tokens/s with 1, 4, 8 and 16 users, each with 16K
tokens of context; prefill is one 64K-token prompt. Two runs at temperature 1.0
with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
0.7.6 on installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`
(release 2026.10.1) with the prepared transport, on two four-Spark rings
([all results, 16K–128K](performance/records/images/dev-20261004-kraken-matrix-20261004.md)).
Each row ran its profile's default checkpoint on that image.
\* The two-Spark GLM profile serves 8 requests at a time.
† The GLM profiles install the CSF checkpoint on an image whose vLLM reads it,
such as the release 2026.10.2 image, and NVFP4 Spark on every other image,
such as 2026.10.1's. Both GLM-5.3-Flash rows of this table were measured
with NVFP4 Spark (revision `a608241037e4`); the GLM-5.3-Flash rows of the
[SIRCL table](#results-on-sircl-ring-sessions) were measured with CSF (revision
`dec48abd33ef`)
([default checkpoint by profile](profiles/glm53-checkpoints.md#default-checkpoint-by-profile)).
`--profile glm53-flash-tp2` and `--profile glm53-flash-tp4` select the same
two profiles.

Five Experimental profiles serve on all eight Sparks of an eight-Spark ring
(`glm53-flash-csf-tp8`, `glm53-nvfp4-tp8`, `glm53-nvfp4-tp8-dcp1`,
`deepseek-v41-flash-tp8`, `qwen38-flash-next-qad-tp8`). They run only on
SIRCL ring sessions, with an image lock that lists them: the 2026.10.2
image's lists all but `qwen38-flash-next-qad-tp8`. Only `glm53-nvfp4-tp8` and
`glm53-nvfp4-tp8-dcp1` have measurements, none from an installation
([profile catalog](profiles/README.md)).

Thinking is the default for requests that don't set it: *on* (a request can
turn it off) or *always*, and its effort. `--reasoning-effort LEVEL` and
`--thinking off` change it per install ([details](docs/operations/install-reference.md#thinking)).

SparkCache variants, which keep the prefix KV cache on disk across restarts
through the external SparkCache connector, and models the installer doesn't
cover have their own
guides in the [profile catalog](profiles/README.md).

## Results on SIRCL ring sessions

Release 2026.10.2's image runs the collectives on SIRCL ring sessions with
NCCL off ([release record](runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)).
These results are Experimental: one eight-Spark ring, GPU clocks locked, the
[serving A/B runner](performance/harnesses/serving_ab/README.md) instead of
`sparkring install`, temperature 0, no prompt context, median of two starts.

| Profile | Model and checkpoint | Sparks | Output tok/s, 1 / 2 / 4 / 8 streams | First token, 32K prompt (s) | Record |
|---|---|---|---|---|---|
| `glm53-nvfp4-tp8` | GLM-5.3 NVFP4 `b472e4ee53f6` | 8 | 50.7 / 73.9 / 102.4 / 151.4 | 24.4 | [image `27e9f75c0d09`](performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009.md) |
| `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash CSF `dec48abd33ef` | 4 | 68.3 / 106.5 / 152.4 / 224.3 | 10.5 | [image `816c6d6a7e96`](performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) |
| `glm53-flash-nvfp4-spark-tp2` | GLM-5.3-Flash CSF `dec48abd33ef` | 2 | 42.0 / 61.4 / 88.9 / 133.5 | 15.3 | [image `816c6d6a7e96`](performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) |
| `qwen38-flash-next-qad-tp4` | Qwen3.8-Flash-Next QAD step 5500, MXFP8 attention | 4 | 91.0 / 143.6 / 218.5 / 310.6 | 6.8 | [image `816c6d6a7e96`](performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) |
| `qwen38-flash-next-tp2` | Qwen3.8-Flash-Next QAD step 5500, MXFP8 attention | 2 | 63.0 / 100.5 / 153.0 / 217.0 | 8.4 | [image `816c6d6a7e96`](performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) |
| `deepseek-v41-flash-tp4` | DeepSeek-V4.1-Flash FP8/MXFP4 `dba1be0a40aa` | 4 | 61.9 / 94.3 / 138.8 / 181.9 | 7.6 | [image `816c6d6a7e96`](performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) |

Two Sparks were a cabled pair and four Sparks four consecutive Sparks of the
ring of eight, whose ends reach each other through relays. Both images
precede the 2026.10.2 image, whose SIRCL is 0.3.2.

- **Pending:** installer qualification of the 2026.10.2 image on the
  eight-Spark ring.
- **Pending:** the four-Spark profiles installed and benchmarked on a ring of
  four, and the ring of four's measured SIRCL tuning row.

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
