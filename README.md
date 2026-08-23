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

SparkRing supports GB10 pairs, four-node rings, and Six-node rings (in dev).

Models run as tensor-parallel deployments over the direct fabric without an
external Ethernet or InfiniBand switch. [SIRCL](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/SIRCL.md) provides custom RDMA collectives
for qualified four-node paths, CUDA-graph command rings support repeated decode
work, and [patched NCCL](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/README.md) handles communication outside SIRCL's supported paths.

The repository provides launch tooling, model profiles, and qualification evidence, and [performance data](https://github.com/FujitsuPolycom/sparkring/tree/main/performance). Model-specific policy
belongs to profiles rather than the transport.

## Cluster sizes

- **2× DGX Spark — direct pair.** Models: DeepSeek-V4-Flash-0731 and
  Qwen3.8-27B EXL3 K5/K6. DeepSeek is compatible with SparkCache; the Qwen
  pair profile is cache-free and research-only.
- **4× DGX Spark — physical ring.** Models: GLM-5.2 EXL3 3.5-bpw,
  DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6. GLM and DeepSeek are
  compatible with SparkCache; Qwen SparkCache support is Pending.
- **6× DGX Spark — physical ring.** **In dev. Target models: GLM | KIMI.**

## Profiles

| Profile | Model identity | Topology | Status | Start here |
|---|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | normalized candidate at 1M context/16 sequences; historical 262K/eight-sequence qualification retained | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | two-Spark pair, TP2/DCP1 | normalized candidate live-benchmarked; SIRCL unsupported | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | four-Spark cycle, TP4/DCP1 | normalized candidate; SIRCL width 4096 research-only | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` | two-Spark pair, TP2/DCP1 | research-only live bring-up; 1M advertised through static YaRN; no performance result | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` | four-Spark cycle, TP4/DCP1 | **Experimental public builder**; candidate; historical runtime implemented; SIRCL unsupported | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | normalized candidate; historical durable-state receipt retained | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | two-Spark pair, TP2/DCP1 | normalized candidate; historical 131K/six-sequence receipt retained | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | four-Spark cycle, TP4/DCP1 | normalized candidate; historical 524K receipt retained | [SparkCache compositions](recipes/sparkcache/README.md) |

**Pending:** Qwen3.8-27B with SparkCache has no published composition recipe or
live cache evidence. It is not part of the eight base/composition profiles in
the table.

The GLM base profile is defined by
[`recipes/glm52-exl3-r7-3.5bpw.json`](recipes/glm52-exl3-r7-3.5bpw.json). The
two-Spark and four-Spark DeepSeek base profiles are defined by
[`recipes/deepseek-v4-flash-0731-pair.json`](recipes/deepseek-v4-flash-0731-pair.json)
and
[`recipes/deepseek-v4-flash-0731.json`](recipes/deepseek-v4-flash-0731.json).
The DeepSeek profiles use the immutable published image pinned in
[`runtime/faststart-lock.json`](runtime/faststart-lock.json) and the tracked
per-rank environment templates in [`scripts/config/`](scripts/config/).
The Qwen pair and cycle profiles are
[`recipes/qwen38-27b-exl3-k5k6-pair.json`](recipes/qwen38-27b-exl3-k5k6-pair.json)
and
[`recipes/qwen38-27b-exl3-k5k6.json`](recipes/qwen38-27b-exl3-k5k6.json).
They use the clean-checkout local image builder in
[`runtime/qwen38/`](runtime/qwen38/README.md), the public source pins from the
companion Qwen recipe, and the tracked pair/cycle environments in
`scripts/config/`.

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

DeepSeek-V4-Flash-0731 and Qwen3.8-27B have two-rank launches on a single
cabled pair using patched NCCL; SIRCL is unsupported on that topology. The
GLM profile requires the four-Spark cycle. Both Qwen topologies use patched
NCCL; their width-5,120 SIRCL path is unsupported.

See [architecture](docs/ARCHITECTURE.md) and [SIRCL](docs/SIRCL.md).

## Prerequisites and evidence

Before deploying a profile, complete the applicable
[prerequisites](docs/PREREQUISITES.md). Measured results, conditions, and
limitations are in [results](docs/RESULTS.md). The profile registry is
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

SparkRing builds on work from vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache,
ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
