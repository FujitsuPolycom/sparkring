# vLLM four-rank transport integration

## Status and scope

The adapter installs custom candidates for two collective kinds:

- tensor-parallel BF16 all-reduce, including graph, eager, and dual-rail
  prefill variants; and
- GLM-5.2 tensor-parallel vocabulary all-gather.

All other collectives retain vLLM's original NCCL dispatch. DCP and
sparse-indexer collectives require the patched-NCCL fallback in
[../../spark_transport/nccl/README.md](../../spark_transport/nccl/README.md).

## All-reduce admission

Custom all-reduce requires TP group `tp:0`, world size four, a contiguous
CUDA BF16 tensor, and an admitted two-dimensional `[Q, W]` shape.

- **Qualified GLM-5.2 path:** `W=6144`, `Q=1..40`; exact-Q40 serving uses
  this geometry.
- **Research-only DeepSeek-V4-Flash-0731 path:** `W=4096`. It has no serving
  qualification and must remain in shadow mode until a four-rank result is
  qualified.
- **Implemented GLM-5.3 graph performance-testing path:** `W=4096`, `Q<=512`. It uses the
  captured-graph transport only when explicitly enabled by the runtime
  profile.
- **Implemented GLM-5.3 eager prefill performance-testing path:** `W=4096`, with a synchronous
  candidate at Q1024/Q2048/Q4096/Q8192 or a caller-stream fused candidate for
  every Q from 128 through 8192. The fused candidate requires strict dual-rail
  mode, which uses both RDMA device functions on each existing cabled ring
  edge. It does not require additional cables.

Any non-matching collective calls the original vLLM/NCCL implementation.

`shadow` runs the candidate and reference operation, returns the reference
result, and checks numerical agreement. `custom` returns the native result
only after the operator has completed the selected signature's required
validation. All-reduce native session construction and enqueue failures
terminate the worker. A failure after enqueue terminates the worker; an in-process fallback
could reuse a CUDA stream with an unfulfilled native wait.

Neither synchronous nor fused eager prefill runs during CUDA graph capture. Captured
width-4096 calls remain on the graph-native SIRCL path when that path is
configured, and otherwise fall through to NCCL. Eager tensors outside the
admitted row interval also fall through to NCCL on every rank.

## Vocabulary all-gather admission

The dedicated vocabulary adapter intercepts
`GroupCoordinator._all_gather_out_place()` rather than the shared NCCL hook.
It requires group `tp:0`, world size four, gather dimension `-1` or `1`,
contiguous CUDA BF16 input, and exact input
`[Q, 38720]` for `Q=1..VLLM_SPARK_MAX_QUERY_ROWS`. That limit defaults to `6`
and accepts values through `40`. It produces token-major BF16 `[Q, 154880]`:

```text
output[q] = [rank0[q], rank1[q], rank2[q], rank3[q]]
```

Shadow comparison is byte-exact. Session creation failure falls back before enqueue only in shadow mode.
Custom mode terminates on creation failure to prevent rank-split dispatch.
A native failure after enqueue terminates the worker.

The ABI, probe, and retained build targets are specified in
[GLM52_TP4_VOCAB_ALLGATHER.md](GLM52_TP4_VOCAB_ALLGATHER.md).

## Consumed environment variables

Set `PYTHONPATH` to this directory and mount
`libspark_transport_capi.so` read-only on every rank. The adapters consume
the following variables:

| Variable | Purpose |
|---|---|
| `VLLM_SPARK_TP4_MODE` | All-reduce mode: `shadow`, `custom`, `disabled`, or unset. |
| `VLLM_SPARK_TP4_VOCAB_MODE` | Vocabulary mode: `shadow`, `custom`, or unset. |
| `SPARK_TP4_LIBRARY` | Path to `libspark_transport_capi.so` for candidate execution in either `shadow` or `custom` mode. |
| `SPARK_TP4_PEER0`, `SPARK_TP4_PEER1` | Required site-specific direct-peer addresses; do not use placeholder defaults for serving. |
| `SPARK_TP4_DEVICE0`, `SPARK_TP4_DEVICE1` | Local RoCE devices; defaults are `rocep1s0f0` and `rocep1s0f1`. |
| `SPARK_TP4_GID0`, `SPARK_TP4_GID1` | GID indices; default is `3` for each device. |
| `SPARK_TP4_CONTROL_PORT0`, `SPARK_TP4_CONTROL_PORT1` | All-reduce control-port base pair. |
| `VLLM_SPARK_TP4_GRAPH_Q1` | Set to `1` to enable width-6144 all-reduce graph sessions and, with vocabulary mode `custom`, captured vocabulary all-gather. |
| `VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH` | Enables the implemented width-4096 captured-graph performance-testing session; mutually exclusive with the width-6144 graph paths. |
| `VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40` | Enables the exact-Q40 striped width-6144 graph session. |
| `VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL` | Graph payload-slot protocol: `serial_ack` or `two_slot_deferred_ack`. |
| `VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY` | Graph kernel strategy: `fused`, `split_64k`, or `tiered_64k`. |
| `VLLM_SPARK_SHARED_CAPTURE_STREAM` | Must be `1` for a graph-native TP4 session; all process-local TP ranks share the capture stream. |
| `SPARK_TP4_GRAPH_CONTROL_PORT0`, `SPARK_TP4_GRAPH_CONTROL_PORT1` | Graph all-reduce control-port pair. |
| `SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT0`, `SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT1` | Second graph control-port pair for exact-Q40 striping. |
| `SPARK_TP4_GRAPH_SUBMIT_CPU`, `SPARK_TP4_GRAPH_PROGRESS_CPU` | Distinct CPU indices for graph submission and RDMA progress. |
| `SPARK_TP4_GRAPH_DIRECT_DOORBELL` | Set to `1` to derive graph replay sequence on the device and use the round-zero payload doorbell for host notification. |
| `SPARK_TP4_GRAPH_STATUS_PATH` | Optional rank-local JSON status output for graph progress and collective audit data. |
| `SPARK_TP4_PERSISTENT_OUTPUT_SLOTS` | Bounded number of eager output buffers retained for pointer stability; `0` disables retention. |
| `SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS` | Positive value recorded by the rank capability vote; it does not configure the native control channel, which uses a fixed 10-second connect deadline. |
| `VLLM_SPARK_TP4_PREFILL_Q512` | Replaces the default all-reduce row set with Q1-Q512; does not extend vocabulary admission. Mutually exclusive with an external query-row provider. |
| `VLLM_SPARK_TP4_QUERY_ROW_PROVIDER` | Optional module exposing `provider_query_rows(environ)`; selects the default-width all-reduce row set, with rows in 1-512. |
| `SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0`, `SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1` | Captured vocabulary all-gather control-port pair. |
| `SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU` | CPU index reserved for captured vocabulary progress. |
| `VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM`, `VLLM_SPARK_TP4_DCP_GRAPH_SHADOW` | Reserve the configured DCP graph progress CPU for the corresponding research mode. |
| `SPARK_TP4_GRAPH_DCP_PROGRESS_CPU` | CPU index reserved for DCP graph progress. |
| `VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM` | Reserves the configured sparse-indexer graph progress CPU. |
| `SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU` | CPU index reserved for sparse-indexer graph progress. |
| `SPARK_TP4_DCP_COLLECTIVE_AUDIT` | Enables the DCP collective audit startup hook. |
| `VLLM_SPARK_TP4_DCP_MODE` | Identifies the DCP transport mode recorded by the DCP collective audit. |
| `SPARK_CUDAGRAPH_REPLAY_TIMING` | Enables the CUDA graph replay timing startup hook. |
| `SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES` | Positive number of FULL graph replays retained by the timing collector. |
| `SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH` | Rank-local file whose presence arms replay timing. |
| `SPARK_CUDAGRAPH_REPLAY_TIMING_STATUS_PATH` | Rank-local JSON output written when the timing window completes. |
| `SPARK_TP4_FLIGHT_RECORDER` | Reserved transport diagnostic selector; the GLM-5.3 launcher fixes it to `0`. |
| `VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL` | Set to `1` to enable the implemented eager width-4096 prefill performance-testing path. |
| `VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE` | `single` or `dual`; fused exposure requires `dual`. |
| `VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE` | `sync` for fixed admitted rows or `fused` for caller-stream Q128-Q8192. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0`, `SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1` | Primary prefill control-port base pair. Four admitted capacities consume this pair and the next six ports. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0`, `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1` | Required second-rail peer addresses in dual mode. All primary and secondary peer addresses must be distinct. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0`, `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1` | Required second-rail RoCE devices in dual mode. All primary and secondary device names must be distinct. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0`, `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1` | Second-rail GID indices in the inclusive range 0-255. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0`, `SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1` | Secondary prefill control-port base pair. |
| `SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS` | Positive setup and operation timeout for a prefill session. |
| `SPARK_TP4_VOCAB_CONTROL_PORT0`, `SPARK_TP4_VOCAB_CONTROL_PORT1` | Vocabulary control-port pair. |
| `VLLM_SPARK_MAX_QUERY_ROWS` | Shared vocabulary and default-width all-reduce row limit. Set to `40` for the qualified GLM geometry. |
| `VLLM_SPARK_TP4_EAGER_WIDTHS` | Generic eager all-reduce widths; unset admits only `6144`. Set `4096,6144` only for research shadow validation. The separately enabled bidirectional-prefill candidates admit width 4096 independently of this setting. |
| `SPARK_TP4_SHADOW_COLLECTIVES` | All-reduce shadow comparison window. |
| `SPARK_TP4_SHADOW_PROMOTE` | Promotes an all-reduce shape after its shadow window passes. |
| `SPARK_TP4_SHADOW_STRICT`, `SPARK_TP4_SHADOW_MAX_ULP` | A failed all-reduce comparison prevents promotion; strict mode `1` also raises an error. A positive maximum ULP adds a numerical gate; `0` leaves ULP diagnostic-only. |
| `SPARK_TP4_VOCAB_SHADOW_COLLECTIVES` | Vocabulary shadow comparison window. |
| `SPARK_TP4_VOCAB_SHADOW_PROMOTE` | Promotes a vocabulary shape after its byte-exact shadow window passes. |
| `SPARK_TP4_MAX_INFLIGHT` | Native all-reduce and vocabulary submission bound, 1-4096; default 64. |
| `SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS` | Vocabulary CUDA input-staging timeout before the native protocol begins, 5-3600 seconds; default 300. |

Every rank must use the same mode, admitted all-reduce widths, row limit,
library bytes, and non-overlapping control-port assignments. Invalid values or
conflicting port reservations fail installation instead of selecting a
different transport.

When enabled, the rank capability vote compares resolved eager widths and row
sets, graph admission flags and the selected graph kernel, as well as native
identities, prefill exposure, rail mode and protocol fields. Provider module names alone are
not compared: providers returning the same admitted rows are compatible.
Capability records use `sparkring-sircl-capability/v2`; mixed v1/v2 ranks fail
the ABI comparison. Update the adapter on every rank together. Device names,
GID mappings and local addresses remain rank-specific. The vote occurs when
an eligible collective first reaches it, so it cannot diagnose a rank that
never enters the vote; consistent launch configuration remains required.

## Minimal qualified GLM configuration

```bash
PYTHONPATH=/opt/spark-vllm
VLLM_SPARK_TP4_MODE=shadow
VLLM_SPARK_TP4_VOCAB_MODE=shadow
SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so
VLLM_SPARK_MAX_QUERY_ROWS=40
SPARK_TP4_PEER0='<direct-peer-0>'
SPARK_TP4_PEER1='<direct-peer-1>'
```

Use `custom` only after deterministic four-rank native probes and the relevant
shadow windows pass. The patched NCCL runtime contract remains mandatory for
every operation not admitted by these candidate paths.

## Implemented fused GLM-5.3 prefill performance-testing path

Every rank needs four local RoCE device names and four direct-peer addresses:
one primary and one secondary device-function/address pair for each of its two
cabled ring edges. The complete sanitized GLM-5.3
configuration is
[`runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example`](../../runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example).

The runtime validates the complete topology before creating a session. A
fused operation queues one native launch on vLLM's current CUDA stream and
uses two bounded operation slots internally. The native session owns its
registered buffers for the process lifetime; an operation timeout or protocol
error is process-fatal because safe in-process recovery is not established.
