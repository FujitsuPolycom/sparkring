# SparkRing architecture

SparkRing is a DGX Spark inference stack. It serves the GLM-5.2 EXL3 3.5-bpw
and DeepSeek-V4-Flash-0731 profiles as four tensor-parallel ranks on a
switchless 200 Gb/s direct-cable cycle; DeepSeek-V4-Flash-0731 also serves as
two ranks on a single cabled pair.

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
every rank has exactly two fabric neighbours - rank 0 neighbours ranks 1 and 3,
rank 1 neighbours ranks 0 and 2, and so on around the cycle - and no switch
sits in the inference fabric. Ranks use both cycle neighbours for RDMA
communication.

The management network reaches all four ranks and carries SSH, rendezvous, and
rank 0's API traffic. It is not an inference-fabric edge.

A switchless fabric has no shared broadcast domain, so a rank reaches the peer
opposite it on the cycle only through a neighbour that forwards the traffic.
Routes to each non-adjacent fabric subnet, `net.ipv4.ip_forward=1`, and an
unrestricted `DOCKER-USER` forward rule are launch prerequisites on every rank;
[prerequisites](PREREQUISITES.md) states the conditions and
[`scripts/ring_doctor.py`](../scripts/ring_doctor.py) verifies them.

The DeepSeek two-Spark pair holds ranks 0 and 1 only, joined by one direct
cable cage 0 to cage 0, with rank 0 serving the API and no relayed fabric hop.
The GLM profile requires the four-Spark cycle.

## Collective path

SIRCL (Switchless Inference RDMA Collective Layer) owns persistent RDMA
sessions and graph-replayable command rings for qualified tensor-parallel
collectives. On the four-rank cycle, a collective is scheduled as the cycle's
two perfect matchings - ranks 0-1 with 2-3, then 1-2 with 3-0 - so every step
is a neighbour exchange and no rank relays another rank's collective data. See
[SIRCL](SIRCL.md) for the transport boundary.

Patched NCCL is the fallback for collective shapes and phases outside SIRCL's
qualified path. The GLM profile uses SIRCL for qualified TP all-reduce and
vocabulary families; its DCP and indexer collectives remain stock. The
DeepSeek quickstart uses the patched NCCL configuration from
`scripts/config/deepseek-v4-flash-0731.env.example`; width-4096 SIRCL graph
collectives are research-only.

## Profile composition

The GLM deployment is generated from
`recipes/glm52-exl3-r7-3.5bpw.json` and its tracked runtime inputs. It combines
fixed MTP4, dynamic NVFP4 MLA key-value cache, bounded full-CKV gather, and the
exact-Q40 routing policy. The DeepSeek deployment uses the immutable published
runtime image in `runtime/faststart-lock.json`, its native DSpark speculation,
and `fp8_ds_mla` key-value cache geometry.

The two quickstarts own operational commands:
[GLM-5.2 EXL3 3.5-bpw](GLM52_35BPW_QUICKSTART.md) and
[DeepSeek-V4-Flash-0731](DEEPSEEK_V4_FLASH_QUICKSTART.md).
