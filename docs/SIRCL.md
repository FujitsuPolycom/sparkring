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
qualification. A four-rank matched comparison established native replay,
API health, and zero overflow for the target and DSpark capture path; see the
[DeepSeek SIRCL evidence record](../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).

The GLM-5.3 Flash GB10 runtime can mount a source-built SIRCL bundle for its
implemented performance-testing lane. Its captured width-4096 path and eager
fused-prefill path have separate
admission gates. The fused path accepts contiguous TP4 BF16 `[Q, 4096]`
tensors from Q128 through Q8192 and uses two operation slots. Unsupported
signatures remain on NCCL. The GLM-5.3 profile captures Q8/Q16/Q32/Q64/Q128;
those captured collectives use graph-native SIRCL with direct doorbells. The
fused session uses four persistent QPs and two 67,109,888-byte operation
arenas. See the
[GLM-5.3 runtime guide](../runtime/glm53-flash-jj-r8-gb10/README.md) and the
[vLLM adapter contract](../spark_transport/integrations/vllm/README.md). The
[public SIRCL build receipt](../runtime/glm53-flash-jj-r8-gb10/sircl-public-build-receipt.json)
binds the native build and single-node test identity; it does not establish a
four-rank serving result.

Before native construction, the GLM-5.3 adapter exchanges a capability record
over the CPU process group. Shared protocol and artifact identities must match,
while each rank proves its own RDMA device and GID availability. Model output
is checked against every process-local native session after vLLM's existing
output synchronization. Fused kernels publish poison into mapped host control
state so this check can reject their output without adding CUDA synchronization.

The Qwen3.8-27B EXL3 K5/K6 pair and cycle profiles use patched NCCL. Their
width-5,120 tensor-parallel shape is unsupported by SIRCL, so neither loads a
custom SparkRing collective adapter.

## Operational invariants

- All four ranks require the same topology, peer ordering, RDMA device mapping,
  and transport configuration.
- A collective shape not admitted to the native path must use the NCCL fallback.
- The management network is not an RDMA cycle edge.
- Dual-rail prefill uses both RDMA device functions associated with each
  existing cabled cycle edge. It requires neither additional cables nor
  diagonal rank-to-rank links.
- Transport evidence does not establish model correctness or performance unless
  the corresponding profile result states those conditions.

Deployment commands and profile limits are in the
[GLM-5.2 quickstart](GLM52_35BPW_QUICKSTART.md),
[GLM-5.3 quickstart](GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md),
[DeepSeek quickstart](DEEPSEEK_V4_FLASH_QUICKSTART.md),
[Qwen3.8-27B pair quickstart](QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md), and
[Qwen3.8-27B cycle quickstart](QWEN38_27B_EXL3_K5K6_QUICKSTART.md).
