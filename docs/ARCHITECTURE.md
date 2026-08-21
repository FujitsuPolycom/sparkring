# SparkRing architecture

SparkRing is a four-DGX-Spark inference stack. It serves the GLM-5.2 EXL3
3.5-bpw and DeepSeek-V4-Flash-0731 profiles as four tensor-parallel ranks on a
switchless 200 Gb/s direct-cable cycle.

## Topology

```text
             management network
          |       |       |       |
         S0      S1      S2      S3

             S0 ========== S1
             ||             ||
             ||             ||
             S3 ========== S2
```

Each edge is one direct 200 Gb/s ConnectX-7 link. The management network carries
SSH, rendezvous, and rank 0's API traffic; it is not an inference-fabric edge.
Ranks use both cycle neighbors for RDMA communication.

## Collective path

SIRCL (Switchless Inference RDMA Collective Layer) owns persistent RDMA
sessions and graph-replayable command rings for qualified tensor-parallel
collectives. On the four-rank cycle, a collective is scheduled as two perfect
matchings. See [SIRCL](SIRCL.md) for the transport boundary.

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
