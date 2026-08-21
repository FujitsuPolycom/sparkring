# SparkRing

SparkRing serves two supported model profiles on four directly cabled NVIDIA DGX
Sparks. Four 200 Gb/s ConnectX-7 links form a switchless cycle; tensor-parallel
ranks use the cycle for inference traffic and the management network for SSH,
rendezvous, and the API.

## Profiles

| Profile | Model identity | Status | Start here |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | qualified on one four-Spark appliance; rebuilt images retain implemented status until promotion | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | implemented launch; SIRCL width 4096 is research-only | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |

The GLM profile is defined by
[`recipes/glm52-exl3-r7-3.5bpw.json`](recipes/glm52-exl3-r7-3.5bpw.json). The
DeepSeek profile is defined by
[`recipes/deepseek-v4-flash-0731.json`](recipes/deepseek-v4-flash-0731.json)
and uses the immutable published image pinned in
[`runtime/faststart-lock.json`](runtime/faststart-lock.json) and the tracked
per-rank environment template
[`scripts/config/deepseek-v4-flash-0731.env.example`](scripts/config/deepseek-v4-flash-0731.env.example).

## Architecture

```text
             management network
          |       |       |       |
         S0      S1      S2      S3

                  200 Gb/s
             S0 ========== S1
             ||             ||
    200 Gb/s ||             || 200 Gb/s
             ||             ||
             S3 ========== S2
                  200 Gb/s
```

SIRCL, the Switchless Inference RDMA Collective Layer, provides the qualified
collective path. Patched NCCL is the fallback for communication outside that
path. See [architecture](docs/ARCHITECTURE.md) and [SIRCL](docs/SIRCL.md).

## Prerequisites and evidence

Before deploying either profile, complete the four-Spark
[prerequisites](docs/PREREQUISITES.md). Measured results, conditions, and
limitations are in [results](docs/RESULTS.md). The two-profile registry is
[`docs/profiles/README.md`](docs/profiles/README.md).

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport and vLLM adapters |
| `runtime/` | Pinned runtime inputs and builders |
| `scripts/` | Site validation, preflight, launch, and evidence tooling |
| `recipes/` | Machine-readable serving recipes |
| `docs/` | Profile procedures, architecture, prerequisites, and evidence |

## License

Apache-2.0. See [`LICENSE`](LICENSE).
