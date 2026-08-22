# SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer**. It is SparkRing's
native collective transport for the directly cabled four-DGX-Spark cycle; it is
not a separate service or an NCCL fork.

## Implemented boundary

SIRCL maintains RDMA sessions, registered arenas, and device-published command
rings. CUDA graph replay submits pre-established work without Python or host
control work in the replay path.

A four-rank collective is decomposed into two perfect matchings of the physical
cycle. This scheduling is specific to the four-Spark topology documented in
[architecture](ARCHITECTURE.md). SIRCL does not claim a generic multi-node
collective interface or support beyond that topology.

## Profile use

The GLM-5.2 EXL3 3.5-bpw profile uses SIRCL for qualified tensor-parallel
all-reduce and vocabulary collective families. Patched NCCL handles operations
outside those families; DCP and indexer collectives use stock paths.

The DeepSeek-V4-Flash-0731 quickstart uses patched NCCL. Its width-4096 SIRCL
CUDA-graph configuration is research-only and excluded from functional profile
qualification. A bounded four-rank live validation established native replay,
API health, and zero overflow for the target and DSpark capture path; see the
[DeepSeek SIRCL evidence record](../performance/records/deepseek-v4-flash/sircl-width4096-live-validation-20260822.md).

The Qwen3.8-27B EXL3 K5/K6 profile uses patched NCCL. Its width-5,120
tensor-parallel shape is unsupported by SIRCL, so the profile does not load a
custom SparkRing collective adapter.

## Operational invariants

- All four ranks require the same topology, peer ordering, RDMA device mapping,
  and transport configuration.
- A collective shape not admitted to the native path must use the NCCL fallback.
- The management network is not an RDMA cycle edge.
- Transport evidence does not establish model correctness or performance unless
  the corresponding profile result states those conditions.

Deployment commands and profile limits are in the
[GLM quickstart](GLM52_35BPW_QUICKSTART.md) and
[DeepSeek quickstart](DEEPSEEK_V4_FLASH_QUICKSTART.md), and
[Qwen3.8-27B quickstart](QWEN38_27B_EXL3_K5K6_QUICKSTART.md).
