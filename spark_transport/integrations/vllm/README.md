# vLLM integration probes

The first integration layer is observation-only. It patches
`CudaCommunicator.all_reduce()` in memory, records the input metadata, and
calls the original dispatch chain unchanged.

Mount this directory into the GLM container and put it first on
`PYTHONPATH`:

```bash
-v /path/to/spark_transport/integrations/vllm:/opt/spark-vllm:ro
-e PYTHONPATH=/opt/spark-vllm
-e VLLM_SPARK_TRACE_ALLREDUCE=1
-e VLLM_SPARK_TRACE_PATH=/cache/spark-allreduce-rank.jsonl
```

Use a rank-specific output path in the real four-node launch. Records are
written on the first occurrence of a signature and at power-of-two counts, so
the tracer does not log every layer invocation.

Each JSON line contains:

- TP group name;
- shape and stride;
- dtype and element size;
- exact byte count;
- contiguity;
- cumulative call count.

The tracer does not synchronize CUDA, time kernels, allocate GPU memory, or
change the chosen all-reduce backend.

## Direct-cable TP4 backend

Build `libspark_transport_capi.so`, mount it and this integration directory
into all four containers, and set:

```bash
-e PYTHONPATH=/opt/spark-vllm
-e VLLM_SPARK_TP4_MODE=shadow
-e SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so
-e SPARK_TP4_SHADOW_COLLECTIVES=10000
```

The adapter intercepts only TP world-size four, contiguous CUDA BF16
`[Q, W]` inputs under the admitted row and width policy — by default
`[Q, 6144]` for rows `1..VLLM_SPARK_MAX_QUERY_ROWS` (default 6), with rows
governed by the query-row policy and widths by `VLLM_SPARK_TP4_EAGER_WIDTHS`
as described below. Every other collective uses the original vLLM/NCCL
dispatch unchanged.

`shadow` executes both paths, returns the NCCL result, and accumulates exact
and numerical comparison statistics on the GPU. Its summary includes bitwise
mismatches, non-finite disagreements, maximum absolute and BF16 ULP error,
and counts beyond one, two, and four ULPs. Optional strict validation uses:

```bash
-e SPARK_TP4_SHADOW_STRICT=1
-e SPARK_TP4_SHADOW_MAX_ULP=0
```

Zero leaves BF16 ULP distance diagnostic-only while retaining relative,
absolute, and non-finite correctness gates. Near-zero BF16 results can be
many representable values apart while remaining numerically insignificant.

Any native exception terminates that worker because an already-enqueued CUDA
event wait cannot safely fall back in-process. Restart all four ranks after a
native error. Switch to `custom` only after the shadow distribution and
deterministic model-output tests pass.

### Opt-in persistent eager outputs

`SPARK_TP4_PERSISTENT_OUTPUT_SLOTS=N` replaces the recurring
`torch.empty_like()` in each eager native all-reduce session with one
preallocated `[N, Q, 6144]` backing tensor and prebuilt slot views. The default
is `0` (disabled), and values above 4096 are rejected.

This is an inference-only performance experiment. A session admits one exact
BF16 CUDA signature, and all producers, consumers, and later slot reuse must
remain CUDA-stream ordered. Use at least 256 slots for the measured GLM-5.2
MTP4 inventory so a same-shape slot is not reused inside one target round.
The first live A/B must keep eager execution, DCP1, MTP4, and all transport
settings fixed, and must reject any output or acceptance change.

`SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS` controls only how long the
eager vocabulary progress thread may wait for an already-enqueued CUDA stream
to reach its input-staging kernel. Its default is 300 seconds. Once the GPU
publishes that staging doorbell, all RDMA protocol and peer-consumption waits
retain their five-second watchdog. This distinction prevents long prefill or
JIT work ahead of a vocabulary collective from being misclassified as a
fabric hang.

### Opt-in graph-native Q1--Q5

After the eager Q1--Q5 path is validated, graph-native mixed-Q capture can be
enabled only with a frozen custom policy. The environment flag retains its
original `GRAPH_Q1` name for deployment compatibility:

```bash
-e VLLM_SPARK_TP4_MODE=custom
-e VLLM_SPARK_TP4_GRAPH_Q1=1
-e SPARK_TP4_GRAPH_CONTROL_PORT0=9970
-e SPARK_TP4_GRAPH_CONTROL_PORT1=9971
-e SPARK_TP4_GRAPH_SUBMIT_CPU=10
-e SPARK_TP4_GRAPH_PROGRESS_CPU=11
-e VLLM_SPARK_SHARED_CAPTURE_STREAM=1
```

The first eligible eager custom Q1--Q5 warmup creates one distinct graph-only
verbs session with Q5 (61,440-byte) capacity. Capture itself never opens
sockets, allocates mapped memory, or creates a progress thread. That one ready
session records exact contiguous CUDA BF16 `[Q,6144]` nodes for every
`Q in [1,5]`; each captured node passes its fixed Q to the native generic
capture ABI. An unprepared eligible capture records the original collective.
Every other adapter and unsupported tensor signature still uses the original
path during capture.

The graph session refuses to start unless the server command line contains a
positive `--kv-cache-memory-bytes`, the two CPU IDs are distinct, and the
source-composed shared capture-stream patch is explicitly enabled. The patch
gives target, speculative-prefill, speculative-decode, piecewise, and full
capture one process-lifetime stream per device and rejects overlapping capture.
The native session pins and verifies only the calling/submission thread on the
submission CPU, then pins and verifies its persistent verbs progress thread on
the progress CPU. This preserves the measured two-core topology without
pinning the whole vLLM process. The container/process cpuset must allow both
CPUs.

After capture and after a real request, query replay state inside each worker:

```python
from spark_tp4_backend import graph_q1_status_snapshot

print(graph_q1_status_snapshot())
```

The result is RPC-serializable. A production gate requires `captured_nodes > 0`,
an increase in `completed_sequence` across the request,
`published_sequence == consumed_sequence == completed_sequence`, zero
`overflow_sequence`, and both affinity-verification flags. A capture-node
count without advancing native sequences proves only graph definition.

Do not combine graph Q1 with `shadow`: a captured graph cannot change policy
after validation. See
[GRAPH_NATIVE_TP4_Q1.md](../../GRAPH_NATIVE_TP4_Q1.md) for the native
multi-node soak and failure contract.

### Separate PIECEWISE prefill buckets

The first prefill milestone keeps the existing C8 MTP4 decode capture plan
unchanged and appends eight PIECEWISE-only prefill buckets:

```text
decode padding:  1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40
FULL decode:     5,10,15,20,25,30,35,40
prefill:         48,72,144,224,288,352,432,512
```

The prefill policy is based on the observed query rows below. It uses eight
captures and bounds padding for those shapes to at most 26 rows:

| Observed | PIECEWISE bucket |
|---:|---:|
| 48 | 48 |
| 69, 72 | 72 |
| 143 | 144 |
| 210 | 224 |
| 279 | 288 |
| 348 | 352 |
| 417 | 432 |
| 486 | 512 |

Launch and serve independently attest
`VLLM_SPARK_DECODE_CAPTURE_SIZES`,
`VLLM_SPARK_FULL_DECODE_CAPTURE_SIZES`,
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES`, and their ordered union in
`VLLM_SPARK_GRAPH_CAPTURE_SIZES`. The initial prefill set may later be
replaced by a broader policy only if it remains strictly increasing above
Q40, has at most 16 buckets, terminates at Q512, and pads every observed shape
by no more than 32 rows. Missing, partial, malformed, or inconsistent
attestations fail before vLLM starts.

The compilation patch may synthesize exact uniform speculative widths only
through Q40, and the FULL dispatcher additionally enforces the Q40 cap and
whole-Q5 requests. Thus Q48--Q512 are PIECEWISE candidates only. With
`VLLM_SPARK_TP4_PREFILL_Q512=1`, the native contiguous BF16 TP all-reduce
session may admit those PIECEWISE widths through Q512 when its arena was
created with matching capacity. Query, vocabulary, and DCP collectives remain
bounded at Q40.

The four-rank serving experiment found no uncached-prefill improvement and a
19.8% C2 decode regression, so this wider plan is retained as default-off
protocol groundwork rather than a promoted serving configuration. The proven
default remains the Q40 decode plan.

### Opt-in eager admitted hidden widths

Status: offline-tested groundwork, default-off. No serving evidence.

The eager TP4 all-reduce native session moves a raw byte payload and does not
depend on the model's hidden width; the historical width restriction is
adapter-side admission. `VLLM_SPARK_TP4_EAGER_WIDTHS` replaces the admitted
eager hidden-width list. Unset or empty, admission is exactly the historical
contiguous CUDA BF16 `[Q, 6144]`. When set (comma-separated element widths,
for example `4096,6144`), eager all-reduce admits contiguous CUDA BF16
`[Q, W]` for each listed `W` under the same query-row bounds as before. Every
other admission term is unchanged: four ranks, `tp:0`, BF16, CUDA,
contiguous.

- Eager only. Graph capture and replay, dual-port striped, vocabulary, DCP,
  and all-gather admission are untouched and remain width-bound. A non-6144
  tensor observed during an active graph capture is routed to the stock path
  and recorded as `graph_width_ineligible` rather than captured.
- Eager all-reduce control ports are derived from the ordered set of
  admissible payload sizes. With the variable unset, the mapping is
  bit-identical to the historical per-Q scheme. Setting the variable remaps
  those ports, so all four ranks must share the same value;
  `validate_active_port_namespace` fails installation on any overlap or
  malformed value.
- An identical payload byte count is the same native operation: widths whose
  `Q * W * 2` coincide share one session and one port pair.
- Recommended rollout is shadow mode first (`VLLM_SPARK_TP4_MODE=shadow`) so
  each new signature accumulates its numerical comparison window before any
  promotion. No performance or maturity claim is attached: prior admission
  widenings (Q512 prefill, isolated dual-port Q40) regressed serving until
  qualified, so this remains admission groundwork pending a matched
  four-Spark comparison.

`VLLM_SPARK_TP4_QUERY_ROW_PROVIDER` replaces the default-width query-row
policy. Unset, the admitted rows are the contiguous range
`1..VLLM_SPARK_MAX_QUERY_ROWS`. When set to an importable module name, that
module owns the row set: it is imported lazily at installation, must expose
`provider_query_rows(environ) -> Iterable[int]`, and its rows are validated
in the generic core (integers, unique, positive, at most 512). A configured
provider that is missing, lacks the interface, or returns invalid rows fails
installation with a diagnostic naming the variable and module. The variable
is mutually exclusive with `VLLM_SPARK_TP4_PREFILL_Q512` and, like the width
list, is launch identity: all four ranks must configure the same value. Row
providers constrain the default width only; non-default extension widths
always admit the contiguous range.
[`spark_tp4_query_row_provider.py`](spark_tp4_query_row_provider.py) is the
single resolver both the admission gate and the port namespace consult.

## Direct-cable TP4 all-gather

The all-gather adapter is separately opt-in:

```bash
-e VLLM_SPARK_TP4_ALLGATHER_MODE=shadow
-e SPARK_TP4_ALLGATHER_SHADOW_COLLECTIVES=8
-e SPARK_TP4_ALLGATHER_BASE_PORT=9490
```

It patches `PyNcclCommunicator.all_gather()` but admits only exact GLM-5.2
signatures measured in the trace:

- `[Q, 2, 2048]` INT32 for Q1--Q5: sparse-indexer candidate merge;
- `[1, 38720]` BF16: the legacy Q1 vocabulary gather seam;
- `[753664]` UINT8: sparse CKV gather;
- `[23552]` UINT8: bounded prefill-CKV gather.

Every signature also requires world size four, contiguous CUDA input/output, matching
dtypes, and an output containing exactly four rank segments. Everything else
continues through NCCL. `shadow` runs custom into a persistent candidate
buffer, runs NCCL into the real output on the same CUDA stream, and compares
the results byte-for-byte. Any mismatch is fatal because all-gather has no
floating-point reduction-order tolerance.

The two CKV signatures additionally require
`SPARK_TP4_ALLGATHER_ENABLE_CKV=1`. They stay on the original collective by
default even after a short byte-exact shadow: the 753,664-byte path still
needs a sustained long-prefill sequence/credit soak. Fixed-K4 DCP decode does
not use that bulk CKV signature, so the fence does not reduce its intended
steady-state custom coverage.

Set `SPARK_TP4_ALLGATHER_SHADOW_PROMOTE=1` to promote each exact signature
independently after its configured shadow sample completes with zero byte
mismatches. Promotion starts on that signature's next call and does not
require a model reload. Signatures that have not completed validation remain
in shadow mode.

## Direct-cable TP4 vocabulary all-gather

The ordinary GLM-5.2 vocabulary gather is dispatched through
`GroupCoordinator._all_gather_out_place()`, so it does not pass through the
generic `PyNcclCommunicator.all_gather()` adapter above. Enable its dedicated
adapter separately:

```bash
-e VLLM_SPARK_TP4_VOCAB_MODE=shadow
-e SPARK_TP4_VOCAB_SHADOW_COLLECTIVES=8
-e SPARK_TP4_VOCAB_SHADOW_PROMOTE=0
-e SPARK_TP4_VOCAB_CONTROL_PORT0=9990
-e SPARK_TP4_VOCAB_CONTROL_PORT1=9991
```

It admits only TP world size four, `dim=-1`, and contiguous CUDA BF16 inputs
of shape `[Q, 38720]` for Q1 through Q5. The native result has the final
token-major `[Q, 154880]` layout. Every non-exact signature continues through
the original vLLM path.

With custom vocabulary mode, the existing
`VLLM_SPARK_TP4_GRAPH_Q1=1` graph flag also opts the exact Q1--Q5 vocabulary
path into graph capture. The first eligible eager warmup creates one separate
graph-only vocabulary session on dedicated ports:

```bash
-e SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0=10110
-e SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1=10111
```

The graph session shares the configured submission CPU and current CUDA capture
stream with TP all-reduce, but uses a distinct vocabulary progress CPU
(`SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU`, default 12; TP progress is normally
11). Capture never creates a session, opens sockets, or runs preflight: if
eager warmup has not produced a ready graph session, the call records the stock
vocabulary collective and increments `cold_fallbacks`. A ready session captures
through:

```c
spark_tp4_vocab_allgather_handle spark_tp4_vocab_graph_create(
    const spark_tp4_vocab_graph_config* config,
    char* error, size_t error_bytes);

int spark_tp4_vocab_capture_allgather(
    spark_tp4_vocab_allgather_handle handle,
    const void* input, void* output, uint32_t query_rows,
    void* cuda_stream, char* error, size_t error_bytes);

int spark_tp4_vocab_get_graph_status(
    spark_tp4_vocab_allgather_handle handle,
    spark_tp4_vocab_graph_status* status, size_t status_bytes,
    char* error, size_t error_bytes);
```

The original eager config and `spark_tp4_vocab_allgather_create` ABI remain
unchanged. The separate graph config adds unsigned
`graph_submit_cpu_plus_one` and `graph_progress_cpu_plus_one`; the graph
session encodes CPU 10/12 as 11/13 and is created only through
`spark_tp4_vocab_graph_create`. `vocab_graph_status_snapshot()` exposes
captured and replay sequence counters for worker RPC. Captured-node and
cold-fallback counts are also kept on the group coordinator. Any native capture
failure terminates the worker; no stock fallback is attempted after a graph
node may have been partially defined.

`shadow` returns the original vLLM result while comparing it byte-for-byte
with the direct-cable candidate. Set
`SPARK_TP4_VOCAB_SHADOW_PROMOTE=1` only after validation; each Q promotes
independently after its shadow window completes without a mismatch. Native
session creation failure falls back before enqueue. Any failure after a
native call starts terminates that worker because its CUDA stream cannot be
safely reused for in-process fallback.

See [GLM52_TP4_VOCAB_ALLGATHER.md](GLM52_TP4_VOCAB_ALLGATHER.md) for the
layout contract, build targets, and four-node probe procedure.

## Direct-cable DCP query and attention combine

The DCP adapter is a separate opt-in path:

```bash
-e VLLM_SPARK_TP4_DCP_MODE=shadow
-e SPARK_TP4_DCP_SHADOW_COLLECTIVES=8
-e SPARK_TP4_DCP_SHADOW_PROMOTE=0
-e SPARK_TP4_DCP_CONTROL_PORT0=9890
-e SPARK_TP4_DCP_CONTROL_PORT1=9891
```

It patches the two vLLM calls used by the active GLM-5.2 DCP path:

- `GroupCoordinator._all_gather_out_place()` admits only group `dcp:0`, world
  size four, `dim=1`, and contiguous CUDA BF16 `[Q, 16, 576]` query input. It
  writes contiguous `[Q, 64, 576]`.
- `vllm.v1.attention.ops.common.cp_lse_ag_out_rs()` admits only the same
  group, CUDA BF16 attention output with logical shape `[Q, 64, D]`, where
  `D` is 256 or the production latent width 512. It accepts either exact
  token-major strides `(64*D, D, 1)` or live head-major strides
  `(D, Q*D, 1)`, CUDA FP32 `[Q, 64]` LSE on the same device,
  `is_lse_base_on_e=True`, and either `return_lse=True` or the production
  output-only contract. Native code reads both supported layouts directly,
  without a Python layout copy. The LSE input may be strided and is made
  contiguous before the native call. It returns contiguous token-major BF16
  `[Q, 16, D]` output and, when requested, FP32 `[Q, 16]` LSE.

Both operations accept only `Q` 1 through 5 and share one generic native DCP
handle and its ordered queue per group coordinator. CUDA graph capture and
every non-exact signature use the original vLLM path.

`shadow` runs the native operation and then the stock operation with the same
vLLM arguments, returning the stock result. Query output is compared
byte-for-byte. Combine output and LSE use independent numeric tolerances:

```bash
-e SPARK_TP4_DCP_COMBINE_OUTPUT_RTOL=0.01
-e SPARK_TP4_DCP_COMBINE_OUTPUT_ATOL=0.0625
-e SPARK_TP4_DCP_COMBINE_LSE_RTOL=0.000002
-e SPARK_TP4_DCP_COMBINE_LSE_ATOL=0.00002
```

Validation windows and promotion are independent for every Q and for query
versus combine. With `SPARK_TP4_DCP_SHADOW_PROMOTE=1`, a signature promotes
on its next call only after its own window passes.

Native session creation failure falls back before enqueue and disables the
adapter for that coordinator. Once a native combine call begins, a native
failure, stock-path failure, or tolerance failure terminates the worker; the
ordered CUDA stream cannot safely fall back in-process. The native query and
fused combine probes have both been validated across Q1 through Q5 on the
four-rank direct-cable cluster.

### Semantics-preserving stock timing

`SPARK_TP4_STOCK_TIMING=1` is a measurement-only eager-mode gate. With the
vocabulary and DCP adapters installed, it bypasses their native candidates
and calls the original vLLM operations unchanged. Stream-ordered CUDA events
bracket exactly one fixed-K4 inventory:

- 79 Q5 plus 3 Q1 query calls;
- 79 Q5 plus 3 Q1 stock combine blocks, representing 164 logical
  collectives;
- 1 Q5 plus 4 Q1 vocabulary calls.

The recorder first observes the Q3 startup/profile signature but does not arm.
After startup and the prefix-cache priming request complete, the operator must
write one unique run ID to
`/cache/jit/spark-stock-timing-rank${RANK}.arm` on every rank. An arm marker
that exists before Q3, a changed or removed run ID, Q2/Q3/Q4 after arming, a
CUDA-stream change, a bucket overflow, or an excessive host span invalidates
the measurement. After 169 wrapper calls representing 251 logical
collectives, it reports rank/run ID, device time by family and Q, their sum,
the covered device and host timelines, event calibration, overflow count, and
an explicit `valid=true|false`. It then becomes a no-op wrapper around the
original operations. This is intentionally not a transport stub: activations,
expert routing, reduction order, and outputs remain stock.

After the unarmed 32K prefix request has completed, use
`scripts/arm-glm52-stock-timing.ps1 -PrefixPrimed`. The helper generates one
run ID, verifies that every container is running and has observed its startup
Q3 signature, and writes the rank-specific markers. Then send one short
request using the identical cached prefix and require four matching
`valid=true` reports before using any timing value.
