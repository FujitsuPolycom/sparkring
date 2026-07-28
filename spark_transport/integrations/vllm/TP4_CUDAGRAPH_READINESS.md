# TP4 CUDA-graph readiness audit

Date: 2026-07-25

Scope: current vLLM TP4 all-reduce, all-gather, and DCP-query adapters plus
their native sessions and CUDA workers. This is a static/read-only audit; no
container was restarted or deployed.

## July 26 implementation update

The Q1 all-reduce blocker described below now has an opt-in adapter path.
`VLLM_SPARK_TP4_GRAPH_Q1=1` with `VLLM_SPARK_TP4_MODE=custom` prepares a
separate graph-only verbs session during the first eligible eager Q1--Q5
warmup. Capture never
creates that session: an unprepared capture records the original collective.
A prepared session can define repeated Q1 nodes on one stable stream, using
dedicated control ports and the device-published command ring on replay.

GPU-free adapter tests pass, and one native executable graph passed a
128-node/four-rank soak. Changing-input replay, multiple executable graphs
sharing one session, and four-rank vLLM capture/replay remain required before
this becomes a measured production path. The all-gather, DCP, vocabulary, and
non-Q1 blockers in the original audit still apply.

### Live replay observability and preflight

The native session now exposes a nonblocking, read-only
`spark_tp4_get_graph_status()` C ABI. Its Python representation reports:

- captured Q1 node definitions;
- published, consumed, and completed replay sequences;
- command-ring overflow;
- capture/polling/host-native-atomic readiness;
- verified submission/progress affinity and their CPU IDs.

`spark_tp4_backend.graph_q1_status_snapshot()` returns only plain Python
dictionaries, so it can be invoked through the existing worker RPC mechanism.
Capture is not evidence of execution. A live gate must take a snapshot before
and after a real replay and require:

```text
captured_nodes > 0
completed_sequence_after > completed_sequence_before
published_sequence == consumed_sequence == completed_sequence
overflow_sequence == 0
replay_advanced == true
replay_caught_up == true
submit_affinity_verified == progress_affinity_verified == true
```

An overflow is marked fatal in the status ABI and remains process-fatal in the
progress thread. Other asynchronous CUDA/verbs failures abort the worker
immediately and cannot be polled afterward. The query does not add a fallback,
wait for completion, or change the post-enqueue failure policy.

Graph-session creation also fails closed unless all of these deployment
preconditions hold:

1. the server command line contains exactly one positive
   `--kv-cache-memory-bytes` value;
2. `SPARK_TP4_GRAPH_SUBMIT_CPU` and
   `SPARK_TP4_GRAPH_PROGRESS_CPU` are present, nonnegative, and distinct;
3. `VLLM_SPARK_SHARED_CAPTURE_STREAM=1` enables the fail-closed source patch
   that unifies target, speculative, piecewise, and full capture;
4. the native session can pin the creating/submission thread exclusively to
   the configured submission CPU;
5. the native progress thread can pin itself exclusively to the configured
   progress CPU.

The fixed KV-cache byte requirement is intentionally specific to the current
vLLM launch. In this vLLM version it makes `determine_available_memory()` return
before `profile_run()`/`profile_cudagraph_memory()`, avoiding the otherwise
unproven throwaway profiling-capture stream. If that vLLM control flow changes,
the prerequisite must be re-audited rather than silently relaxed.

## Verdict

Custom TP4 graph replay is **not ready in the production adapters**. A native
Q1 all-reduce vertical slice now has a replay-safe device publisher; the
all-gather and DCP findings below remain unchanged.

- DCP-query is capture-safe only because it explicitly captures the original
  collective instead of TP4.
- TP4 all-gather and non-Q1 all-reduce currently enter or return to their
  original capture paths. Only an explicitly prewarmed exact Q1 all-reduce
  may enter the graph-native path.
- All three native protocols pass a host-generated sequence as a by-value
  kernel argument. CUDA graph replay reuses that captured value and does not
  rerun the Python/C++ submission path, so repeated native replay cannot make
  protocol progress correctly.

The Q1 vertical slice avoids replay-time CPU callbacks entirely. Its captured
kernel claims and publishes a sequence through a `cudaHostAllocMapped` ring,
and a persistent CPU thread polls that ring and performs verbs progress.

## Exact blockers

### 1. Capture-time host submission waits for a kernel that is only recorded

All-reduce calls native unconditionally for eligible tensors:
`spark_tp4_backend.py:334-344`. All-gather does the same at
`spark_tp4_allgather_backend.py:318-327`.

The native calls immediately:

1. increment a host sequence;
2. push it into a host submission deque;
3. record/launch a kernel with that sequence;
4. wake the progress thread.

See `tp4_session.cpp:170-224`, `tp4_allgather_session.cpp:170-228`, and
`tp4_dcp_session.cpp:176-239`.

During stream capture the CUDA launch becomes a graph node and does not execute.
The progress thread nevertheless receives the submission and waits up to five
seconds for `producer_sequence`, which the unexecuted kernel cannot publish.
The timeout reaches `fatal_async_failure` and aborts the worker.

vLLM does execute a warmup before full capture, so sessions are usually already
constructed. That avoids cold-session allocation in the common path but does
not solve the capture submission deadlock.

### 2. Replay reuses a captured scalar sequence and skips host progress

`GpuTp4TensorWorker::enqueue`, `GpuTp4AllgatherWorker::enqueue`, and
`GpuTp4DcpQueryWorker::enqueue` pass `sequence` by value to their kernels.
The value is therefore fixed in the captured kernel node.

vLLM replay calls `CUDAGraph.replay()` directly. Python adapter code, the C API,
the native host sequence increment, deque insertion, and progress-thread
notification do not run again.

Even if capture-time progress were deferred until first replay, later replays
would reuse the same sequence. Doorbell comparisons would see already-satisfied
values, no new RDMA submission would exist, and stale data/races would result.

### 3. DCP intentionally excludes native TP4 from graphs

`spark_tp4_dcp_backend.py:257-258` delegates to the original collective whenever
the current stream is capturing. This is safe, and its GPU-free test locks in
that behavior, but it means every replay permanently uses the original captured
collective. Promotion after capture cannot change the recorded graph.

### 4. Cold-path work is not capture-safe

Every adapter lazily creates backends/sessions. Native construction opens
control channels, connects verbs endpoints, allocates CUDA-mapped host memory,
barriers with peers, and starts a progress thread. Python also mutates caches
and allocates error buffers.

These actions must be completed before capture. Warmup currently makes that
likely, not contractual. A new signature, different capture order, or capture
without the vLLM warmup would run blocking network setup and `cudaHostAlloc`
inside capture.

### 5. Shadow state and promotion are CPU-time decisions

Shadow modes allocate candidates, mutate Python counters, run extra comparison
kernels, and eventually call `.item()`. Those actions occur while defining the
graph and are not replayed. A graph also cannot switch from shadow/original to
custom after capture because its nodes are already fixed.

Shadow and flight-recorder counts collected during graph definition do not
describe replay traffic. Graph capture must use a frozen policy:

- shadow/validation: capture the original collective only;
- custom: capture TP4 only after explicit readiness succeeds.

### 6. Stream identity is an unproven invariant

Each native session binds to the first `cudaStream_t` and rejects any other
stream (`tp4_session.cpp:186-189`, `tp4_allgather_session.cpp:186-189`,
`tp4_dcp_session.cpp:194-197`).

vLLM warmup and full capture currently run in the same capture-manager context,
but the adapter has no explicit assertion that warmup, capture, and replay use
the supported stream topology. Piecewise graphs and future capture-manager
changes can violate this assumption. Native graph mode must be keyed/gated on
one proven stream per session.

For the exact current launch, fixed `--kv-cache-memory-bytes` removes vLLM's
memory-profiling capture stream before graph-session preparation. The adapter
now enforces that narrow prerequisite. Multiple production executable graphs
and their replay stream identity remain to be proven.

### 7. Pointer lifetime is accidental rather than owned

The mapped RDMA buffers and native workers are stable for the native session,
which is good for captured kernel arguments. The Python wrappers never call
the exposed destroy functions, so handles currently live until process exit.
That leak incidentally keeps native pointers valid but is not an explicit graph
lifetime contract.

Input/output tensor pointers are also captured:

- all-gather receives caller-owned output;
- DCP custom allocates output per invocation;
- all-reduce uses `torch.empty_like`.

PyTorch's graph pool can provide stable captured allocation addresses, but the
native session must outlive every graph that references its mapped buffers.
Adding a normal Python destructor without graph ownership would create a
use-after-free risk. The first graph implementation should make sessions
intentionally process-lifetime; graph-aware teardown can follow later.

### 8. Replay errors cannot fall back

The current fatal-after-enqueue policy is correct: once a TP4 kernel waits on
doorbells, falling through to the original collective can deadlock ranks.
During replay the C API is not called, so no synchronous return code is
available. Network/progress failures must remain process-fatal, and readiness
failures must be resolved before graph capture.

## Smallest implementation sequence

### Step 0: make CUDA graphs safe today using the original collectives

Add the existing DCP capture predicate to all-reduce and all-gather. During
capture:

- do not create a backend or session;
- do not enter shadow accounting;
- delegate directly to the original collective.

Add GPU-free tests for both modes proving capture creates no native state. This
is the smallest safe change and enables vLLM CUDA graphs, but TP4 is not used
on replay.

### Step 1: introduce an explicit native capture-readiness contract

Before capture:

- create every exact-signature session during warmup;
- finish mapped-memory allocation, verbs connection, peer barriers, and worker
  creation;
- validate the capture stream identity;
- expose a C API readiness query;
- freeze policy to `custom` for that graph (never shadow/promote).

If any condition fails, capture the original collective. Do not attempt lazy
initialization in capture.

Keep capture-ready native sessions in an explicit process-lifetime registry
until graph teardown ownership exists.

### Step 2: make one submission execute on every replay

The implemented minimal native design is:

1. Allocate a fixed 64-slot command ring with `cudaHostAllocMapped` before
   capture and require `cudaDevAttrHostNativeAtomicSupported`.
2. In the captured Q1 kernel, system-scope CAS reserves capacity and claims
   one unique sequence.
3. The same kernel writes its descriptor, executes
   `__threadfence_system()`, publishes in exact sequence order, and retains
   the claimed value in shared memory for its transport doorbells.
4. When graph mode is configured, the persistent progress thread switches
   from condition-variable waits to adaptive busy polling. Replay performs no
   host callback, CUDA API call, allocation, or lock acquisition.
5. The CPU completes slots in order. A publisher waits when 64 sequences are
   outstanding, so no live descriptor can be overwritten.

The Q1 implementation is in `gpu_tp4_tensor.cu`, `tp4_session.cpp`, and
`tp4_graph_command.*`. DCP extensions must carry fixed `q` metadata in their
device-published descriptors; graph capture already specializes per `q`.

### Step 3: prove one signature end to end

Implement graph-native replay first for TP all-reduce `[1, 6144]` BF16 only.
Keep every other signature on capture fallback. On four nodes:

- warm once, capture once, replay at least 10,000 times;
- change static input contents before every replay;
- assert outputs and doorbell sequences change monotonically;
- verify exactly one host submission and two RDMA rounds per replay;
- verify a second stream is rejected before capture;
- inject a progress timeout and confirm fatal termination rather than fallback.

### Step 4: extend the proven primitive

Once the sequencing primitive is stable:

1. add remaining TP all-reduce row counts;
2. add fixed-size all-gather signatures;
3. add DCP `q=1..5`, carrying `q` in the replay descriptor.

Only after all signatures pass replay-soak and pointer-lifetime tests should
the adapters permit native custom capture by default.
