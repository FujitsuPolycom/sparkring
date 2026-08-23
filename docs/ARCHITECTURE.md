# SparkRing architecture

SparkRing is a DGX Spark inference stack. It serves the GLM-5.2 EXL3 3.5-bpw,
DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6 profiles as four
tensor-parallel ranks on a switchless 200 Gb/s direct-cable cycle.
DeepSeek-V4-Flash-0731 and Qwen3.8-27B also have two-rank launches on a single
cabled pair using patched NCCL; SIRCL is unsupported on that topology.

## Topology

```text
management LAN ─┬─────────────┬─────────────┬─────────────┐
            ┌───┴────┐    ┌───┴────┐    ┌───┴────┐    ┌───┴────┐
     API ──>│ rank 0 ╞════╡ rank 1 ╞════╡ rank 2 ╞════╡ rank 3 │
            └───╤────┘    └────────┘    └────────┘    └───╤────┘
                ╚═════════════════════════════════════════╝

  ═══  one 200 Gb/s ConnectX-7 DAC per edge (RoCEv2); the inference fabric
  ───  management LAN: SSH, rendezvous, rank-0 API; never a fabric edge
```

Four ranks, numbered 0 through 3, are cabled as the cycle `0-1-2-3-0`. Each of
those four edges is one direct 200 Gb/s ConnectX-7 link carrying RoCEv2, so
every rank has exactly two fabric neighbors - rank 0 neighbors ranks 1 and 3,
rank 1 neighbors ranks 0 and 2, and so on around the cycle - and no switch
sits in the inference fabric. Ranks use both cycle neighbors for RDMA
communication.

The management network reaches all four ranks and carries SSH, rendezvous, and
rank 0's API traffic. It is not an inference-fabric edge.

A switchless fabric has no shared broadcast domain, so a rank reaches a
non-adjacent fabric address only through a neighbor that forwards the traffic.
Routes to each non-adjacent fabric subnet, `net.ipv4.ip_forward=1`, and an
unrestricted `DOCKER-USER` forward rule are launch prerequisites on every rank;
[prerequisites](PREREQUISITES.md) states the conditions and
[`scripts/ring_doctor.py`](../scripts/ring_doctor.py) verifies them.

The implemented two-Spark profiles hold ranks 0 and 1 only, joined by
one direct cable from cage 0 to cage 0, with rank 0 serving the API and no
relayed fabric hop. They use patched NCCL; SIRCL is unsupported on the pair.
The GLM profile requires the four-Spark cycle.

## Collective path

SIRCL (Switchless Inference RDMA Collective Layer) owns persistent RDMA
sessions and graph-replayable command rings for qualified tensor-parallel
collectives. On the four-rank cycle, a collective is scheduled as the cycle's
two perfect matchings - ranks 0-1 with 2-3, then 1-2 with 3-0 - so every step
is a neighbor exchange and no rank relays another rank's collective data. See
[SIRCL](SIRCL.md) for the transport boundary.

Patched NCCL is the fallback for collective shapes and phases outside SIRCL's
qualified path. The GLM profile uses SIRCL for qualified TP all-reduce and
vocabulary families; its DCP and indexer collectives remain stock. The
four-Spark DeepSeek profile uses
`scripts/config/deepseek-v4-flash-0731.env.example`; the two-Spark profile uses
`scripts/config/deepseek-v4-flash-0731-pair.env.example`. Both use patched
NCCL. Width-4096 SIRCL graph collectives are research-only on the four-Spark
cycle and unsupported on the pair.

The Qwen pair and four-Spark profiles use their topology-specific environments
in `scripts/config/` and patched NCCL. Their width-5,120 tensor-parallel path
is not admitted by SIRCL, so neither loads a custom SparkRing collective
adapter.

## Profile composition

The GLM deployment is generated from
`recipes/glm52-exl3-r7-3.5bpw.json` and its tracked runtime inputs. It combines
fixed MTP4, dynamic NVFP4 MLA key-value cache, bounded full-CKV gather, and the
exact-Q40 routing policy. The DeepSeek deployment uses the immutable published
runtime image in `runtime/faststart-lock.json`, its native DSpark speculation,
and `fp8_ds_mla` key-value cache geometry.

The Qwen deployments use the clean-checkout local ARM64 image builder in
`runtime/qwen38/` and the pair/cycle model recipes. The image is built once and distributed
with one content-addressed image ID. It combines EXL3 K5/K6 weights, Qwen MTP
depth 3, FP8 key-value cache, native prefix caching with recurrent-state
alignment, and full-decode CUDA graphs. External key-value caching is disabled
in the base profile.

The quickstarts own operational commands:
[GLM-5.2 EXL3 3.5-bpw](GLM52_35BPW_QUICKSTART.md) and
[DeepSeek-V4-Flash-0731](DEEPSEEK_V4_FLASH_QUICKSTART.md), and
[Qwen3.8-27B EXL3 K5/K6 pair](QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md), and
[Qwen3.8-27B EXL3 K5/K6 cycle](QWEN38_27B_EXL3_K5K6_QUICKSTART.md).
