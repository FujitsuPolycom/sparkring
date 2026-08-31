# Spark Transport

`spark_transport` is SparkRing's model-independent collective transport for
directly connected GPU systems. It owns topology validation, peer setup,
collective protocols, CUDA-stream ordering, and the boundary between custom
collectives and patched NCCL.

It does not define a model, checkpoint, quantization, context length, or
serving configuration. Model profiles select admitted tensor descriptors and
bind them to measured evidence outside this subsystem. See the
[profile registry](../docs/profiles/README.md).

## Implemented surface

The transport provides:

- BF16 row-oriented tensor-parallel all-reduce;
- BF16 tensor-parallel vocabulary all-gather;
- a C ABI for native callers;
- a vLLM adapter that admits only supported topology and tensor descriptors;
- direct-cable qualification tools; and
- patched NCCL dispatch for collectives outside the custom transport's
  admitted surface.

The custom protocol currently accepts a four-rank direct-cable cycle. Each
rank communicates with its two physical neighbors. Supported tensor geometry
is an explicit descriptor and qualification property, not a model identity.

The C ABI and vLLM adapter are **implemented**. Deployment qualification
belongs to the exact topology, tensor descriptor, runtime artifact, and model
profile named by an evidence record.

## Subsystem boundaries

| Concern | Location |
|---|---|
| Collective protocol and C ABI | `include/spark_transport/`, `src/` |
| Native probes and contract tests | `app/`, `tests/` |
| vLLM integration and descriptor admission | [`integrations/vllm/`](integrations/vllm/) |
| Patched switchless NCCL | [`nccl/`](nccl/) |
| Link and payload qualification | [`CABLE_QUALIFICATION.md`](CABLE_QUALIFICATION.md) |
| Model-specific settings and evidence | [`docs/profiles/`](../docs/profiles/) and [`recipes/`](../recipes/) |

## Build

Build the library and contract tests on an ARM64 CUDA environment with CMake,
a C++17 compiler, CUDA, and libibverbs development headers:

```bash
cmake -S spark_transport -B build/spark-transport \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/spark-transport --target \
  spark_transport_capi \
  spark_tp4_vocab_allgather_probe \
  tp4_c_api_test \
  tp4_vocab_allgather_c_api_test \
  --parallel
ctest --test-dir build/spark-transport \
  -R "tp4_c_api_test|tp4_vocab_allgather_c_api_test" \
  --output-on-failure
```

The serving artifact is `libspark_transport_capi.so`. Every participating
rank must use identical library bytes and transport configuration.

## Runtime invariants

- Every rank agrees on topology, peer addresses, devices, GIDs, ports, and
  collective sequence.
- Inputs are CUDA-resident contiguous tensors matching an admitted descriptor.
- Invalid topology, tensor, session, or protocol state is rejected before
  custom device work begins; the adapter may use its ordinary NCCL path.
- An error after custom CUDA work is enqueued terminates the worker. Continuing
  in-process is unsafe because the stream may contain an unfulfilled wait.
- Unsupported collective types use patched NCCL instead of being silently
  interpreted as a supported custom operation.

The complete adapter environment and fallback contract are documented in
[`integrations/vllm/README.md`](integrations/vllm/README.md).

## Link qualification

Run [`CABLE_QUALIFICATION.md`](CABLE_QUALIFICATION.md) before model serving.
The procedure verifies bidirectional payload integrity on each physical edge.
Its latency result applies only to the payload and conditions named by that
record; it does not qualify a model profile.

## Model profiles

Profiles map model-specific tensor shapes and serving settings onto the generic
transport interfaces. A profile may mark a descriptor **qualified**,
**research-only**, or **unsupported** without changing the transport API.

Start with the [profile registry](../docs/profiles/README.md) and the selected
profile's recipe and evidence record.
