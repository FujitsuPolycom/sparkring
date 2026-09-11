# GLM-5.2 TP4 vocabulary all-gather

## Status

Implemented native API and vLLM adapter for the GLM-5.2 tensor-parallel
vocabulary seam. The interface is limited to the exact four-rank contract:

```text
input per rank:  BF16 [Q, 38720], Q=1..configured query-row limit
output per rank: BF16 [Q, 154880]
group:           tp:0, world size 4
gather dimension: -1 or 1
```

`VLLM_SPARK_MAX_QUERY_ROWS` sets the adapter's row limit, defaults to 6, and
accepts values from 1 through 40. The native session supports rows 1 through 40.

The output is token-major:

```text
output[q] = [rank0[q], rank1[q], rank2[q], rank3[q]]
```

The dedicated worker performs this rank/query permutation. A rank-major raw
gather is not layout-compatible when `Q > 1`.

## Native interface and build targets

The vocabulary C API is declared in
`spark_transport/include/spark_transport/tp4_vocab_allgather_c_api.h` and provides
`spark_tp4_vocab_allgather_create`, `spark_tp4_vocab_allgather`, and
`spark_tp4_vocab_allgather_destroy`.

The supported build targets are:

```bash
cmake -S spark_transport -B build/spark-transport -DBUILD_TESTING=ON
cmake --build build/spark-transport --target \
  spark_transport_capi \
  spark_tp4_vocab_allgather_probe \
  tp4_vocab_allgather_c_api_test \
  --parallel
ctest --test-dir build/spark-transport \
  -R tp4_vocab_allgather \
  --output-on-failure
```

## Fail-closed operation

One native session supports all admitted `Q` values and requires a stable
caller CUDA stream. The adapter uses the candidate only for the exact
four-rank CUDA BF16 contract. Nonmatching signatures use the original
vLLM/NCCL collective. With the vocabulary graph option enabled in `custom`
mode, admitted captures use a prepared graph session; otherwise capture uses
the original collective. Session creation failure falls back before enqueue.

Shadow mode compares the final output byte-for-byte and returns the reference
result. `SPARK_TP4_VOCAB_SHADOW_PROMOTE=1` permits per-shape custom promotion
only after its configured shadow window passes. A native failure after enqueue
terminates the worker because its CUDA stream may contain an unfulfilled wait.

The environment contract is maintained in [README.md](../../../integrations/vllm/README.md).
