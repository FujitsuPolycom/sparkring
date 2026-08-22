# SparkRing

> **This repository is changing rapidly.** Documentation, profiles, and
> branch history are being restructured as SparkRing's scope narrows to
> switchless multi-Spark collective transport and serving. Published
> branches may be rebased and documents may be moved, renamed, or
> replaced without a deprecation period. Pin a commit if you depend on a
> specific state of this tree.

SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX Spark systems
powered by the GB10 Grace Blackwell Superchip.

The 200 Gb/s ConnectX-7 links connect DGX Sparks directly, with no external
Ethernet or InfiniBand switch in the inference fabric. SparkRing supports
tensor-parallel deployments on two-node, four-node, and six-node (support in
progress) GB10 clusters. Model profiles cover setup, tested configurations,
limitations, and [performance results](docs/RESULTS.md).

The stack combines [SIRCL](docs/SIRCL.md) custom RDMA collectives,
CUDA-graph-replayable command rings, vLLM patches and adapters, and a
[patched NCCL fallback](#architecture) for communication not handled by the
custom path.

## Cluster sizes

- **2× DGX Spark — direct pair.** Models: DeepSeek-V4-Flash-0731; compatible
  with SparkCache.
- **4× DGX Spark — physical ring.** Models: GLM-5.2 EXL3 3.5-bpw;
  DeepSeek-V4-Flash-0731; compatible with SparkCache.
- **6× DGX Spark — physical ring (support in progress).** Target models: GLM |
  Kimi.

## Profiles

| Profile | Model identity | Topology | Status | Start here |
|---|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | qualified on one appliance; rebuilt images retain implemented status until promotion | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | two-Spark pair, TP2/DCP1 | implemented launch; SIRCL is unsupported | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | four-Spark cycle, TP4/DCP1 | implemented launch; SIRCL width 4096 is research-only | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | qualified durable prefix-state composition | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731` | two-Spark pair, TP2/DCP1 | qualified durable prefix-state composition | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731` | four-Spark cycle, TP4/DCP1 | qualified durable prefix-state composition | [SparkCache compositions](recipes/sparkcache/README.md) |

The GLM base profile is defined by
[`recipes/glm52-exl3-r7-3.5bpw.json`](recipes/glm52-exl3-r7-3.5bpw.json). The
two-Spark and four-Spark DeepSeek base profiles are defined by
[`recipes/deepseek-v4-flash-0731-pair.json`](recipes/deepseek-v4-flash-0731-pair.json)
and
[`recipes/deepseek-v4-flash-0731.json`](recipes/deepseek-v4-flash-0731.json).
The DeepSeek profiles use the immutable published image pinned in
[`runtime/faststart-lock.json`](runtime/faststart-lock.json) and the tracked
per-rank environment templates in [`scripts/config/`](scripts/config/).

Qualified durable prefix-state compositions for the two-Spark DeepSeek,
four-Spark DeepSeek, and four-Spark GLM profiles are in
[`recipes/sparkcache/`](recipes/sparkcache/README.md). They pin the exact
SparkCache wheel and runtime image used by each live store/restart/restore
gate. Those gates used a 4,096-token scheduler budget. Operators may choose
other values; 8,192 is known to work but is outside the published receipts.

## Architecture

```text
management LAN ─┬─────────────┬─────────────┬─────────────┐
            ┌───┴────┐    ┌───┴────┐    ┌───┴────┐    ┌───┴────┐
     API ──>│ rank 0 ╞════╡ rank 1 ╞════╡ rank 2 ╞════╡ rank 3 │
            └───╤────┘    └────────┘    └────────┘    └───╤────┘
                ╚═════════════════════════════════════════╝

  ═══  one 200 Gb/s ConnectX-7 DAC per edge (RoCEv2); the inference fabric
  ───  management LAN: SSH, rendezvous, rank-0 API; never a fabric edge
```

Four DGX Sparks serve one model as four tensor-parallel ranks, numbered 0
through 3. The inference fabric is a cycle of four direct cables - rank 0 to
rank 1, 1 to 2, 2 to 3, and 3 back to 0 - each one 200 Gb/s ConnectX-7 link
carrying RoCEv2. Every rank has exactly two fabric neighbors, and no switch
carries collective traffic. A separate management LAN reaches all four ranks
with SSH, rendezvous, and the client API that rank 0 serves; it is never a
fabric edge.

A rank reaches a non-adjacent fabric address only through a neighbor that
forwards the traffic. Routes to non-adjacent fabric subnets,
`net.ipv4.ip_forward=1`, and unrestricted `DOCKER-USER` ACCEPT rules between
the two fabric interfaces are therefore launch prerequisites on every rank,
checked by
[`scripts/ring_doctor.py`](scripts/ring_doctor.py).

Collectives do not take that relay. SIRCL, the Switchless Inference RDMA
Collective Layer, schedules a four-rank collective as the cycle's two perfect
matchings - ranks 0-1 with 2-3, then 1-2 with 3-0 - so every step is a neighbor
exchange and the two steps together use all four links. SIRCL holds
persistent RDMA sessions and device-published command rings that CUDA graph
replay resubmits without host work. Patched NCCL is the fallback for collective
shapes outside SIRCL's qualified families.

DeepSeek-V4-Flash-0731 has an implemented two-rank launch on a single cabled
pair using patched NCCL; SIRCL is unsupported on that topology. The GLM
profile requires the four-Spark cycle.

See [architecture](docs/ARCHITECTURE.md) and [SIRCL](docs/SIRCL.md).

## Prerequisites and evidence

Before deploying a profile, complete the applicable
[prerequisites](docs/PREREQUISITES.md). Measured results, conditions, and
limitations are in [results](docs/RESULTS.md). The six-profile registry is
[`docs/profiles/README.md`](docs/profiles/README.md).

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport and vLLM adapters |
| `runtime/` | Pinned runtime inputs and builders |
| `scripts/` | Site validation, preflight, launch, and evidence tooling |
| `recipes/` | Machine-readable serving recipes |
| `docs/` | Profile procedures, architecture, prerequisites, and evidence |

## Acknowledgements

SparkRing builds on work from vLLM, NVIDIA NCCL, B12X, ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
