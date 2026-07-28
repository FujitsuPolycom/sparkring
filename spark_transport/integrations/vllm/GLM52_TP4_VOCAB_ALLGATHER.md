# GLM-5.2 TP4 vocabulary all-gather

## Scope

This path replaces only the steady-state tensor-parallel vocabulary
all-gather:

```text
input per rank:  BF16 [Q, 38720]
output per rank: BF16 [Q, 154880]
Q:               1 through 5
group:           tp:0, world size 4
gather axis:     final vocabulary axis
```

The output is token-major. For each query row, rank shards appear in rank
order:

```text
output[q] = [rank0[q], rank1[q], rank2[q], rank3[q]]
```

A raw rank-major four-rank gather instead produces
`[rank, query, vocab_shard]`. Reinterpreting that buffer as
`[query, full_vocab]` is correct only for `Q=1`. The dedicated CUDA worker
performs the required rank/query permutation while it writes the final
output.

## Native API

The vocabulary ABI is isolated in
`include/spark_transport/tp4_vocab_allgather_c_api.h`:

```c
spark_tp4_vocab_allgather_handle spark_tp4_vocab_allgather_create(...);

int spark_tp4_vocab_allgather(
    spark_tp4_vocab_allgather_handle handle,
    const void* input,
    void* output,
    uint32_t query_rows,
    void* cuda_stream,
    char* error,
    size_t error_bytes);

void spark_tp4_vocab_allgather_destroy(...);
```

One session accepts every supported `Q`. It allocates mapped transport
buffers for `Q=5`, sends only the active `Q * 77,440` input bytes, requires
one stable caller CUDA stream, and uses `SPARK_TP4_MAX_INFLIGHT` for bounded
submission.

## vLLM adapter

The vocabulary gather uses `GroupCoordinator._all_gather_out_place()`.
It does not pass through `PyNcclCommunicator.all_gather()`, so the generic
Spark all-gather hook cannot intercept it.

Enable the dedicated adapter in shadow mode first:

```bash
VLLM_SPARK_TP4_VOCAB_MODE=shadow
SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so
SPARK_TP4_VOCAB_CONTROL_PORT0=9990
SPARK_TP4_VOCAB_CONTROL_PORT1=9991
SPARK_TP4_VOCAB_SHADOW_COLLECTIVES=8
```

The admission gate requires all of:

- group `tp:0`;
- world size four;
- gather dimension `-1` or `1`;
- contiguous CUDA BF16 input;
- exact shape `[Q,38720]`, `Q=1..5`;
- no CUDA graph capture.

All near misses use the original vLLM path. Session creation failure also
falls back before native work is enqueued. A failure after native enqueue
terminates the worker because the CUDA stream can contain an unfulfillable
wait.

Shadow mode runs native and reference paths on the same stream and compares
the final token-major output byte-for-byte. Optional per-Q promotion is:

```bash
SPARK_TP4_VOCAB_SHADOW_PROMOTE=1
```

After deterministic and shadow validation, select the custom result with:

```bash
VLLM_SPARK_TP4_VOCAB_MODE=custom
```

## Deterministic four-rank probe

Build `spark_tp4_vocab_allgather_probe`, copy the binary to the same path on
all four Sparks, then run:

```powershell
.\scripts\run_tp4_vocab_allgather_probe.ps1 `
  -Binary /tmp/spark_tp4_vocab_allgather_probe `
  -Library /tmp/libspark_transport_capi.so `
  -Warmup 4 `
  -Iterations 100
```

The probe uses one dynamic-Q C API session, runs Q1 through Q5, and validates
every 16-bit element against a deterministic `(rank, query, column)` pattern.
Every rank must exit zero and report `mismatches=0` for all five widths.

This script creates temporary probe containers. It does not modify or restart
the GLM serving containers.

## Tests

GPU-free coverage:

```bash
cd integrations/vllm
python -m unittest test_spark_tp4_vocab_allgather_backend.py
```

Native build coverage:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target \
  tp4_vocab_allgather_layout_test \
  tp4_vocab_allgather_c_api_test \
  spark_tp4_vocab_allgather_probe
ctest --test-dir build -R tp4_vocab_allgather
```

The layout test covers Q1 through Q5, all four ranks, all 38,720 columns,
rank-major source offsets, token-major destination offsets, buffer bounds,
and invalid Q/coordinate rejection.
