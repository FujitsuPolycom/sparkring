# Spark Transport

## Status

`spark_transport` implements direct-cable, four-rank tensor-parallel
collectives for the supported GLM-5.2 EXL3 serving configuration and
implemented GLM-5.3 performance-testing paths:

- BF16 TP all-reduce; and
- BF16 vocabulary all-gather.

Each rank has two directly attached RoCE peers. The native protocol accepts
only the four-rank topology and fails closed on invalid rank, peer, device,
GID, tensor, session, or protocol state.

The exact-Q40 GLM-5.2 path has hidden width 6,144 and contiguous BF16 tensors
`[Q, 6144]`, with `Q` from 1 through 40. It is the only qualified custom
all-reduce geometry. The width-4,096 DeepSeek-V4-Flash-0731 path is
research-only: it may use the retained all-reduce admission surface, but has
no serving qualification.

The GLM-5.3 research path combines a captured width-4,096 transport with an
eager fused-prefill session. Fused prefill accepts contiguous BF16
`[Q, 4096]` tensors from Q128 through Q8192 and counter-rotates tensor halves
over both RDMA device functions on each existing cabled cycle edge. It requires
neither additional cables nor diagonal links. DCP, sparse-indexer, and every
unsupported signature remain on the patched NCCL fallback described in
[nccl/README.md](nccl/README.md).

## Build

Build the native library and its contract tests on an ARM64 CUDA environment
with CMake, a C++17 compiler, CUDA, and libibverbs development headers:

```bash
cmake -S spark_transport -B build/spark-transport \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/spark-transport --target \
  spark_transport_capi \
  spark_tp4_bidirectional_prefill_probe \
  spark_tp4_vocab_allgather_probe \
  tp4_c_api_test \
  tp4_vocab_allgather_c_api_test \
  --parallel
ctest --test-dir build/spark-transport \
  -R "tp4_c_api_test|tp4_vocab_allgather_c_api_test" \
  --output-on-failure
```

The serving artifact is `libspark_transport_capi.so`. All four ranks must
receive identical library bytes and identical transport configuration before
custom mode is selected.

## Deployment invariants

- Exactly four tensor-parallel ranks participate.
- Every rank uses its two direct RoCE peer addresses, devices, GIDs, and
  control ports consistently with the other ranks.
- Dual-rail prefill configures both RDMA device functions on each cabled ring
  edge. Its primary and secondary addresses, devices, GIDs, and control ports
  must be complete and non-overlapping.
- Inputs are CUDA-resident, contiguous BF16 tensors of an admitted shape.
- Native work begins only after session construction succeeds. A creation or
  admission failure uses the original vLLM/NCCL collective.
- A failure after native work is enqueued terminates the worker. In-process
  fallback is unsafe because the CUDA stream can contain an unfulfilled wait.

The vLLM adapter contract, including every consumed transport environment
variable, is specified in
[integrations/vllm/README.md](integrations/vllm/README.md).

## Cable qualification

Before serving, qualify each direct physical edge with
[CABLE_QUALIFICATION.md](CABLE_QUALIFICATION.md). Qualification proves
bidirectional payload integrity and reports latency under its stated
conditions; it does not qualify a model-serving result.

## NCCL fallback

Unsupported tensor signatures, DCP, and sparse-indexer collectives use vLLM's
NCCL dispatch. The patched NCCL configuration is required where the
switchless direct-cable topology is used; see
[nccl/README.md](nccl/README.md).
