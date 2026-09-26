# SparkRing

SparkRing is an inference-serving stack with low-latency collective communication
for switchless clusters of NVIDIA GB10-based devices. It supports two-node pairs
and four-node rings; six-node rings are experimental. Model profiles use vLLM
and [SGLang](runtime/deepseek-v41-sglang/README.md).

The collective communication stack combines SIRCL, RoCEnante, and patched NCCL.
The high-speed data fabric needs no external Ethernet or InfiniBand switch;
administration and inference API traffic use the management network, typically
through each node's 10GbE NIC.

Profiles that need communication between nonadjacent nodes can use a virtual
mesh over the four-node ring. Custom RoCE RDMA routing and hardware forwarding
in the ConnectX network ASICs create those paths over the existing ring cables,
without routing the traffic through host CPUs. This provides mesh connectivity
over a physical ring; the selected profile defines its transport requirements.

The repository provides setup guides, launch tooling, model profiles,
reproducible benchmarks, and test results. Validation applies to the exact
configurations and workloads recorded with each profile.

## Get started

**Before you start**, meet the host requirements in
[Install SparkRing](docs/operations/install.md):

- two Sparks cabled port p0 to port p0, or four in a ring with each Spark's p0
  cabled to the next Spark's p1;
- Ubuntu 24.04 ARM64 (DGX OS) with NVIDIA drivers, Docker, NVIDIA Container
  Toolkit and NetworkManager on every Spark;
- SSH enabled and a root login or a login with sudo on every Spark;
- one Spark, Node A, connected to your network. Workers need no network
  connection of their own.

**Two Sparks.** On Node A, run:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

**Four Sparks.** On Node A, from a console or a management connection that
does not use the ring cables, run:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

A ring whose ConnectX functions do not yet have the hairpin setting that
four-Spark forwarding needs stops at `Driver reload required`. Review and apply
the driver step, then install the model; repeat both after any Spark in the
ring reboots. The driver step has no hardware evidence
([Four-Spark rings](docs/operations/install.md#four-spark-rings)):

```bash
sudo sparkring setup --plan
sudo sparkring setup --allow-driver-reload
sudo sparkring install --profile qwen38-flash-next-qad-tp4
```

The [install script](install.sh) builds the SparkRing package from this
branch and installs it on Node A. `sparkring install` then lists what it will
change and asks `Proceed? [Y/n]`. It sets up every cabled Spark, downloads the
serving image and, unless a Spark already holds a verified copy, the model
checkpoint, and ends with the API address:

```text
Model ready: http://NODE_A_ADDRESS:8000/v1
```

Send a test request (four Sparks: port 8015, model
`Qwen3.8-Flash-Next-NVFP4-QAD-TP4`):

```bash
curl http://NODE_A_ADDRESS:8000/v1/chat/completions -H 'Content-Type: application/json'   -d '{"model": "Qwen3.8-Flash-Next-NVFP4-QAD-TP2", "messages": [{"role": "user", "content": "Hello"}]}'
```

The API has no key. Keep Node A on a trusted network
([Security and host exposure](docs/operations/install.md#security-and-host-exposure)).
From a second terminal, `sudo sparkring logs --follow` shows progress, which
is also saved to `/var/log/sparkring/install.log`. The
[installation guide](docs/operations/install.md) covers recovery, storage,
downloads, reuse of a checkpoint already on disk (`--model-path`) and
`--plan`/`--json` for scripted use.

| Sparks | Profile | API port | Status |
|---:|---|---:|---|
| 2 | `qwen38-flash-next-tp2` | 8000 | implemented |
| 4 | `qwen38-flash-next-qad-tp4` | 8015 | implemented |
| 2, 4 | `glm53-flash-nvfp4-spark-tp2`, `-tp4` | 8000, 8015 | research-only |
| 2, 4 | `mimo-v26-flash-rl-tp2`, `-tp4` | 8020 | research-only |

Both Qwen profiles serve
[Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e)
with three-token MTP drafting; their measured decode and prefill rates are in
the [installer tuning record](performance/records/qwen38-flash-next/installer-tuning-20260925.md).
Every installer profile runs on one shared serving image; `sparkring models`
lists them.

**Docker Compose.** Each installer profile's directory holds per-rank Compose
files for the same image, for example
[`profiles/qwen38-flash-next-tp2/compose`](profiles/qwen38-flash-next-tp2/compose).
[`standalone.yaml`](profiles/qwen38-flash-next-tp2/compose/standalone.yaml)
runs the Qwen TP2 profile on two Sparks from one file; it needs prepared
networking, the checkpoint on each Spark, and
[`runtime/common/loader-seccomp.json`](runtime/common/loader-seccomp.json) at
that path relative to the Compose file. [Compose deployments](docs/operations/compose.md)
describes both.

The [SparkRing image](docs/operations/images.md) contains the inference software.
Model weights, host drivers, Docker and network configuration are separate.
For existing deployments, verify prerequisites and reuse matching assets before
making changes. The [full validation runbook](docs/operations/profile-validation.md)
covers subsequent workload qualification and benchmarks.

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
