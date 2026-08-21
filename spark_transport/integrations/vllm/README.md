# vLLM four-rank transport integration

## Status and scope

The adapter installs two custom candidates only:

- tensor-parallel BF16 all-reduce; and
- GLM-5.2 tensor-parallel vocabulary all-gather.

All other collectives retain vLLM's original NCCL dispatch. DCP and
sparse-indexer collectives require the patched-NCCL fallback in
[../../nccl/README.md](../../nccl/README.md).

## All-reduce admission

Custom all-reduce requires TP group `tp:0`, world size four, a contiguous
CUDA BF16 tensor, and an admitted two-dimensional `[Q, W]` shape.

- **Qualified GLM-5.2 path:** `W=6144`, `Q=1..40`; exact-Q40 serving uses
  this geometry.
- **Research-only DeepSeek-V4-Flash-0731 path:** `W=4096`. It has no serving
  qualification and must remain in shadow mode until a four-rank result is
  qualified.

Any non-matching collective calls the original vLLM/NCCL implementation.

`shadow` runs the candidate and reference operation, returns the reference
result, and checks numerical agreement. `custom` returns the native result
only after the selected signature has passed its required validation. Native
session creation failure falls back before enqueue. A failure after enqueue
terminates the worker; an in-process fallback could reuse a CUDA stream with
an unfulfilled native wait.

## Vocabulary all-gather admission

The dedicated vocabulary adapter intercepts
`GroupCoordinator._all_gather_out_place()` rather than the shared NCCL hook.
It requires group `tp:0`, world size four, gather dimension `-1` or `1`,
contiguous CUDA BF16 input, and exact input
`[Q, 38720]` for `Q=1..5`. It produces token-major BF16 `[Q, 154880]`:

```text
output[q] = [rank0[q], rank1[q], rank2[q], rank3[q]]
```

Shadow comparison is byte-exact. Session creation failure falls back before
enqueue; a native failure after enqueue terminates the worker.

The ABI, probe, and retained build targets are specified in
[GLM52_TP4_VOCAB_ALLGATHER.md](GLM52_TP4_VOCAB_ALLGATHER.md).

## Consumed environment variables

Set `PYTHONPATH` to this directory and mount
`libspark_transport_capi.so` read-only on every rank. The two adapters consume
the following variables:

| Variable | Purpose |
|---|---|
| `VLLM_SPARK_TP4_MODE` | All-reduce mode: `shadow`, `custom`, `disabled`, or unset. |
| `VLLM_SPARK_TP4_VOCAB_MODE` | Vocabulary mode: `shadow`, `custom`, or unset. |
| `SPARK_TP4_LIBRARY` | Required path to `libspark_transport_capi.so` when either custom candidate is enabled. |
| `SPARK_TP4_PEER0`, `SPARK_TP4_PEER1` | Required site-specific direct-peer addresses; do not use placeholder defaults for serving. |
| `SPARK_TP4_DEVICE0`, `SPARK_TP4_DEVICE1` | Local RoCE devices; defaults are `rocep1s0f0` and `rocep1s0f1`. |
| `SPARK_TP4_GID0`, `SPARK_TP4_GID1` | GID indices; default is `3` for each device. |
| `SPARK_TP4_CONTROL_PORT0`, `SPARK_TP4_CONTROL_PORT1` | All-reduce control-port base pair. |
| `SPARK_TP4_VOCAB_CONTROL_PORT0`, `SPARK_TP4_VOCAB_CONTROL_PORT1` | Vocabulary control-port pair. |
| `VLLM_SPARK_MAX_QUERY_ROWS` | Default-width all-reduce row limit. Set to `40` for the qualified GLM geometry. |
| `VLLM_SPARK_TP4_EAGER_WIDTHS` | Comma-separated all-reduce widths; unset admits only `6144`. Set `4096,6144` only for research shadow validation. |
| `SPARK_TP4_SHADOW_COLLECTIVES` | All-reduce shadow comparison window. |
| `SPARK_TP4_SHADOW_PROMOTE` | Promotes an all-reduce shape after its shadow window passes. |
| `SPARK_TP4_SHADOW_STRICT`, `SPARK_TP4_SHADOW_MAX_ULP` | All-reduce shadow comparison gates. |
| `SPARK_TP4_VOCAB_SHADOW_COLLECTIVES` | Vocabulary shadow comparison window. |
| `SPARK_TP4_VOCAB_SHADOW_PROMOTE` | Promotes a vocabulary shape after its byte-exact shadow window passes. |
| `SPARK_TP4_MAX_INFLIGHT` | Positive bound on native all-reduce and vocabulary submissions. |
| `SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS` | Positive timeout for vocabulary CUDA input staging before the native protocol begins. |

Every rank must use the same mode, admitted all-reduce widths, row limit,
library bytes, and non-overlapping control-port assignments. Invalid values or
conflicting port reservations fail installation instead of selecting a
different transport.

## Minimal qualified GLM configuration

```bash
PYTHONPATH=/opt/spark-vllm
VLLM_SPARK_TP4_MODE=shadow
VLLM_SPARK_TP4_VOCAB_MODE=shadow
SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so
VLLM_SPARK_MAX_QUERY_ROWS=40
SPARK_TP4_PEER0=<direct-peer-0>
SPARK_TP4_PEER1=<direct-peer-1>
```

Use `custom` only after deterministic four-rank native probes and the relevant
shadow windows pass. The patched NCCL runtime contract remains mandatory for
every operation not admitted by these two candidates.
