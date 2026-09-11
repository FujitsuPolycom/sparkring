# Spark Transport

`spark_transport` is SparkRing's communication layer for inference over directly
connected RDMA over Converged Ethernet (RoCE) links. It contains native
collectives, runtime adapters, patched NCCL integration, and tools for checking
the fabric and collective protocols.

The transport exchanges tensor buffers between ranks. Model profiles select
the runtime adapter, tensor signatures, topology, and verified artifacts needed
for their workload. GLM, DeepSeek, and Qwen profiles use different combinations
of these components; a profile's validation does not establish support for
every model or tensor shape.

Status: **implemented** components. **Qualified** results apply only to the
artifacts and conditions in the linked [profile records](../docs/profiles/README.md).
Hardware-forwarded mesh composition and six-node model profiles remain
**research-only**.

## Communication components

| Component | Purpose | Implementation and contract |
|---|---|---|
| **SIRCL — Switchless Inference RDMA Collective Layer** | Native four-rank collectives with persistent RDMA sessions, CUDA-graph submission, and eager prefill paths. | [SIRCL overview](../docs/SIRCL.md), [C/C++ interfaces](include/spark_transport/), and [native source](src/) |
| **RoCEnante integration** | Selected all-reduces over direct and hardware-forwarded opposite-peer paths in the four-rank mesh composition. | [Runtime overlay](../integrations/vllm/rocenante/README.md) and [adapted Local Inference Lab source and attribution](../third_party/b12x_roce/README.md) |
| **Patched NVIDIA NCCL** | Pair/cycle communication and fallback for collectives outside custom transport admission. Some model profiles use NCCL for all their collectives. | [Library patches, topology-specific environments, and invariants](nccl/README.md) |
| **Runtime adapters** | Select a collective implementation by process group, tensor geometry, execution mode, and enabled profile capabilities. | [vLLM adapter contract](../integrations/vllm/README.md) and [mesh composition](../runtime/glm53-spark-mtp3-mesh/README.md) |

SIRCL's native library is `libspark_transport_capi.so`. It provides BF16
all-reduce and specialized vocabulary all-gather interfaces. Its native
sessions require four participating ranks. The versioned all-reduce API
accepts tensor geometry, while fused prefill and vocabulary gathering retain
their own narrower shape contracts. Adapter admission can be narrower than
the native API.

The mesh overlay composes RoCEnante with SIRCL and NCCL rather than replacing
every collective. It handles selected tensor-parallel all-reduces and delegates
other calls to the saved backend. Decode-context-parallel and sparse-indexer
collectives retain their original NCCL dispatch in these maintained adapters.
The profile's bundle configuration determines the exact dispatch rules.

Patched NCCL has separate configurations for two-rank pairs and four- or
six-rank direct-cable cycles. That scope does not extend SIRCL's four-rank
native interfaces to other rank counts. Model support and six-rank research
limits are recorded by the deployment profiles.

## Physical links and hardware forwarding

The four-rank native cycle uses two direct neighbors per rank. Dual-rail
prefill uses both RDMA device functions associated with each cabled edge;
device, address, GID, and control-port assignments must agree across ranks.

The mesh composition adds communication with opposite ranks over the existing
physical ring. Packets cross two physical links through an intermediate
ConnectX-7 ASIC. The forwarding hop uses the NIC hardware; endpoints still
perform CPU posting and use GPU-mapped pinned host buffers. These paths share
the bandwidth of the physical cables.

See the [hardware-forwarding contract](fabric/cx7_hairpin_diagonal/README.md)
and [managed mesh operations](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md).
Fabric provisioning and model lifecycle are separate from a collective call.

## Dispatch and failure handling

- All ranks must agree on the selected library identity, protocol, group,
  tensor signature, and complementary peer configuration.
- A collective outside custom admission uses its configured NCCL backend.
  Shadow mode returns the reference result while checking the custom result;
  custom mode returns the native result only under its admission contract.
- All-reduce session-construction and enqueue failures terminate the worker.
  Vocabulary session construction permits fallback before enqueue. The
  selected adapter defines this boundary; it is not a general recovery rule.
- Failure after native work is enqueued terminates the worker. Retrying
  through NCCL in that process could reuse a CUDA stream with an unfulfilled
  wait or unfinished native operation.

The [adapter contract](../integrations/vllm/README.md) specifies supported modes,
tensor geometry, environment variables, and failure boundaries. Serving
instructions belong to the selected [profile quickstart](../docs/profiles/README.md).

## Directory map

| Path | Contents |
|---|---|
| [`include/spark_transport/`](include/spark_transport/) | Public C/C++ interfaces and protocol contracts |
| [`src/`](src/) | Sessions, verbs endpoints, CUDA operations, command rings, and topology checks |
| [`../integrations/vllm/`](../integrations/vllm/) | Maintained vLLM tensor admission, dispatch, and native-session checks; `spark_transport/integrations/vllm/` contains compatibility exports |
| [`nccl/`](nccl/) | Patched NCCL configuration and compatibility requirements |
| [`app/`](app/) and [`scripts/`](scripts/) | Collective probes and cable/rank qualification tools |
| [`tests/`](tests/) | Protocol, ABI, configuration, and source-contract checks |
| [`fabric/`](fabric/) | Maintained hardware-forwarding topology and planning |
| [`experiments/`](experiments/) | Transport variants and retained experimental interfaces |

Selected kernels under `experiments/tiled_prefill/` are linked into the native
library by [CMakeLists.txt](CMakeLists.txt). Directory placement alone does not
identify whether code participates in a serving artifact; use its build
targets and profile receipts.

## Build and validation

Run these commands from the repository root on an ARM64 CUDA environment with
CMake, a C++17 compiler, and libibverbs development headers:

```bash
cmake -S spark_transport -B build/spark-transport \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/spark-transport --target \
  spark_transport_capi \
  tp4_c_api_test \
  tp4_vocab_allgather_c_api_test \
  --parallel
ctest --test-dir build/spark-transport \
  -R "tp4_c_api_test|tp4_vocab_allgather_c_api_test" \
  --output-on-failure
```

This builds the native sources in this directory. A reproducible serving image
must use its profile's pinned sources, bundle configuration, and library hashes;
building the directory alone does not reconstruct every published image.

Run [cable qualification](CABLE_QUALIFICATION.md) and the relevant collective
probes in a stopped-model test window. Probes generate GPU/RDMA traffic.
Contract tests and cable checks do not establish model correctness or serving
performance; those require the profile's
[validation procedure](../docs/PROFILE_VALIDATION.md).
