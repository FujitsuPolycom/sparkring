# vLLM transport integration

The adapter offers SparkRing candidates for selected four-rank tensor-parallel
collectives. Every operation that does not match an admitted signature remains
on vLLM's NCCL path.

## Dispatch contract

The integration provides two candidate families:

- contiguous CUDA BF16 tensor-parallel all-reduce; and
- a dedicated tensor-parallel vocabulary all-gather.

Admission binds the process group, world size, tensor layout, dtype, shape,
runtime library, and profile-selected mode. Exact model geometries belong to
the profile that enables them.

Any near miss calls the original vLLM/NCCL implementation. Decode-context,
sparse-indexer, and other non-admitted collectives use the patched-NCCL
contract in [../../nccl/README.md](../../nccl/README.md).

## Modes

`shadow` runs the candidate and reference operation, returns the reference
result, and checks agreement. `custom` returns the native result only after the
selected signature passes its required validation. Disabled or unset modes use
the original vLLM path.

Native session creation failure returns to the reference path before enqueue.
A native failure after enqueue terminates the worker because the CUDA stream
may contain an unfulfilled wait; in-process fallback is unsafe at that point.

## Environment contract

Set `PYTHONPATH` to this integration directory and mount
`libspark_transport_capi.so` read-only on every rank.

| Variable | Purpose |
|---|---|
| `VLLM_SPARK_TP4_MODE` | All-reduce mode: `shadow`, `custom`, `disabled`, or unset |
| `VLLM_SPARK_TP4_VOCAB_MODE` | Vocabulary mode: `shadow`, `custom`, or unset |
| `SPARK_TP4_LIBRARY` | Path to the native library when a candidate is enabled |
| `SPARK_TP4_PEER0`, `SPARK_TP4_PEER1` | Site-specific direct-neighbour addresses |
| `SPARK_TP4_DEVICE0`, `SPARK_TP4_DEVICE1` | Local RoCE devices |
| `SPARK_TP4_GID0`, `SPARK_TP4_GID1` | RoCEv2 GID indices |
| `SPARK_TP4_CONTROL_PORT0`, `SPARK_TP4_CONTROL_PORT1` | All-reduce control-port bases |
| `SPARK_TP4_VOCAB_CONTROL_PORT0`, `SPARK_TP4_VOCAB_CONTROL_PORT1` | Vocabulary control ports |
| `VLLM_SPARK_MAX_QUERY_ROWS` | Profile-selected all-reduce row limit |
| `VLLM_SPARK_TP4_EAGER_WIDTHS` | Profile-selected all-reduce widths |
| `SPARK_TP4_SHADOW_COLLECTIVES` | All-reduce shadow comparison window |
| `SPARK_TP4_SHADOW_PROMOTE` | Permit per-signature promotion after the shadow window passes |
| `SPARK_TP4_SHADOW_STRICT`, `SPARK_TP4_SHADOW_MAX_ULP` | All-reduce comparison checks |
| `SPARK_TP4_VOCAB_SHADOW_COLLECTIVES` | Vocabulary shadow comparison window |
| `SPARK_TP4_VOCAB_SHADOW_PROMOTE` | Permit vocabulary promotion after its byte-exact window passes |
| `SPARK_TP4_MAX_INFLIGHT` | Bound native all-reduce and vocabulary submissions |
| `SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS` | Bound vocabulary CUDA-input staging |

Every rank must use identical modes, admitted widths, row limit, library bytes,
and non-overlapping port assignments. Invalid or conflicting values stop
adapter installation rather than selecting a different configuration.

## Profile-owned activation

A deployment profile must provide the exact admitted signatures, environment,
probe result, shadow policy, and patched-NCCL fallback. Do not derive those
values from this generic integration page or enable `custom` based only on
process health.

The model-specific vocabulary adapter contract is documented in
[GLM52_TP4_VOCAB_ALLGATHER.md](GLM52_TP4_VOCAB_ALLGATHER.md).
