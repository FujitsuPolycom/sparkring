# Graph-native TP4 Q1 vertical slice

## Scope

This vertical slice makes contiguous BF16 `[Q, 6144]`, Q1 through Q5, TP4
all-reduces replayable from CUDA Graphs. The opt-in vLLM adapter still routes
only its audited signatures; the native mixed-Q path and probe do not by
themselves claim production integration readiness.

The public prototype entrypoint is
`Tp4AllreduceSession::capture_all_reduce(input, output, q, stream)`, with
`capture_q1_all_reduce()` retained as the Q1 convenience entrypoint. It may be
called repeatedly
while its stable caller stream is actively capturing so one session can
serve every Q1 all-reduce node in the model graph. Every call must precede
both eager submission and graph replay. The session and every captured input
and output allocation must outlive replay, and executable graphs must replay
on the stream used for capture.

The Python all-reduce adapter now has an opt-in
`VLLM_SPARK_TP4_GRAPH_Q1=1` policy. A dedicated graph-only session is
connected during the first eligible eager custom Q1--Q5 warmup, on ports
separate from the eager sessions. During capture any exact contiguous BF16
`[Q, 6144]`, Q1--Q5 signature may use the native entrypoint only when that
session is already ready; cold capture falls back without opening sockets or
allocating mapped memory. Generic all-gather, vocabulary, DCP query/combine,
and unsupported all-reduce signatures still capture their original vLLM
paths.

## Q1-to-Q5 integration audit

The Q1 primitive is not the hot production shape under the current fixed-MTP4
launch. vLLM rounds the configured graph sizes to multiples of five, so target
decode, speculative prefill, and even single-request draft decode normally
capture `[5, 6144]`, not `[1, 6144]`. The current Q1 adapter may therefore
replace none of the important steady-state all-reduces even when its native
probe passes.

The production design is one graph-only session sized for the Q5 maximum
(61,440 bytes) per rank, with each replay descriptor carrying its active
`q` and byte count. It is not five per-Q sessions: five sessions would add
five busy-poll progress threads, ten connections, separate sequence domains,
and more ports while preserving no useful ordering advantage. A max-sized
mapped layout can safely operate on shorter prefixes, and the verbs endpoint
already accepts a dynamic byte count.

One prerequisite above the transport layer is now implemented behind an
explicit opt-in. Target, speculative prefill, speculative decode, piecewise,
and full graph capture share one strong-reference, process-lifetime CUDA
stream per device. An explicit lock protects stream construction and the
active-capture guard across threads; overlapping capture fails closed.
Relaxing the same-stream assertion would permit concurrent staging-buffer
overwrite and remains unacceptable.

The mixed-Q rollout proceeds in this order:

1. add validated `q` and `payload_bytes` fields to the 64-byte command
   (implemented);
2. make a Q5-sized worker operate on the descriptor's shorter byte count
   (implemented);
3. encode Q in the cross-rank doorbell and fail on a rank-local mismatch
   (implemented);
4. soak alternating Q1--Q5 executable graphs with changing input
   (probe implemented; live gate pending);
5. unify all vLLM capture managers on one per-device stream (implemented,
   offline lifecycle gate passed; live stream-ID proof pending); and
6. route every exact contiguous BF16 `[Q, 6144]`, Q1--Q5, through one
   process-lifetime graph session.

For the current DCP1/SP0/fixed-MTP4 single-request lane, the exact
steady-state census is 160 Q5 and nine Q1 TP all-reduces, followed by one Q5
and four Q1 vocabulary all-gathers. No indexer, CKV, DCP, EP, PP, or DP
collectives occur in steady decode. The graph entry point therefore captures
only sizes 1 and 5; this avoids creating unused Q6/Q8 graphs whose unsupported
all-reduces would remain on NCCL. Once the mixed-Q all-reduce passes live, a
graph-native Q1/Q5 vocabulary gather is the only remaining transport primitive
needed for a socket-free steady target round.

## Repeated-node four-rank gate

The 128-node graph soak captured 128 distinct Q1 all-reduce nodes and launched
that graph twice for warmup plus 100 measured iterations. Every rank completed
exactly 13,056 published, consumed, and completed sequences:

| Rank | Host submit/call | Device/call | Overflow | Mismatches |
|---:|---:|---:|---:|---:|
| 0 | 2.414 us | 36.696 us | 0 | 0 |
| 1 | 2.416 us | 36.696 us | 0 | 0 |
| 2 | 2.436 us | 36.696 us | 0 | 0 |
| 3 | 2.446 us | 36.696 us | 0 | 0 |

All four ranks used the identical staged probe SHA-256
`107694738f947dbeb01147ce2d898579bf2ff1c66b4740869e03dc6831e37aaf`.
This proves repeated-node sequence ownership at the transport layer. It does
not yet prove that vLLM captures the intended nodes or that its full graph
replays successfully.

The probe now changes its stable input allocation before every graph launch.
Adjacent launches alternate an exactly representable BF16 offset, and the
captured validator reads a stream-ordered replay marker to calculate the
corresponding new four-rank reduction. This catches stale input or output
data without using a floating-point tolerance. The input update is outside
the graph on the same stream, matching the production stable-pointer contract.

An opt-in multi-graph mode captures graph A with 3 Q1 nodes and graph B with
128 Q1 nodes on the same session and stream before any replay. It alternates
A/B launches, synchronizes each launch, and requires published, consumed, and
completed counters to equal the exact cumulative node count after every
launch. Before replay one it also requires the captured-node inventory to
equal exactly 131 while every replay counter remains zero. Immediately after
replay one it attempts a third capture and requires
the documented
`graph TP4 capture cannot add nodes after the first replay` rejection. It then
continues alternating the two existing executable graphs, proving that the
rejected capture did not corrupt them.

This expanded probe has compiled and its host command-ring model has passed,
but the new changing-input/multi-graph mode has not yet passed the four-rank
live gate. Do not treat it as production evidence until that run is recorded.

`-MixedQValidation` strengthens that lifecycle gate. It requires
`-MultiGraphValidation`, constructs one Q5-capacity session, and captures:

- graph A cycling the deterministic pattern Q1, Q3, Q5; and
- graph B with exactly 128 nodes cycling Q1, Q2, Q3, Q4, Q5.

With the default A3/B128 shapes, the captured-node histogram is
Q1=27, Q2=26, Q3=27, Q4=25, Q5=26. That is 390 active rows and exactly
4,792,320 active bytes per A/B launch cycle. Before every graph replay, the
probe changes all five input rows. A captured validator checks every active
BF16 output element for that node; inactive rows are not counted as validated.
The existing sequence, affinity, rejected-late-capture, overflow, and mismatch
gates remain mandatory.

## Why the eager path cannot simply be captured

`Tp4AllreduceSession::all_reduce()` currently performs two actions on the
calling CPU thread:

1. append a `Submission` containing the next sequence to a host deque; and
2. launch `tp4_tensor_all_reduce` with that sequence as a kernel argument.

Stream capture records the kernel launch, but the deque append happens only
during capture. Replaying that graph would therefore reuse the captured
sequence without publishing another verbs submission. The first replay
might consume the one capture-time descriptor; later replays would not
advance the transport.

## Replay state machine

The vertical slice preallocates a 64-slot `Tp4GraphCommandRing` with
`cudaHostAllocMapped` when the TP4 session is created. Capture is rejected
unless `cudaDevAttrHostNativeAtomicSupported` is true.

```text
captured Q1--Q5 GPU kernel
  -> atomically reserve capacity and claim the next replay sequence
  -> write {sequence, trace, q, payload_bytes} into its mapped ring slot
  -> system fence, then monotonically publish that exact sequence
  -> stage round 0 and publish the existing producer doorbell
  -> wait while the progress thread submits and completes verbs round 0
  -> reduce/stage round 1 and wait for verbs round 1
  -> write the fixed output allocation

persistent progress thread
  -> busy-poll the producer cache line with short adaptive spin/yield
  -> consume the next ring descriptor
  -> run the existing two-round VerbsEndpoint submission/completion path
  -> advance the completed sequence on a separate consumer cache line
```

There is no host node or callback in the captured graph. Replay calls only
`cudaGraphLaunch`; the captured kernel performs the replay publication.
The progress thread is notified once when graph mode is configured and then
polls without a mutex or condition-variable wakeup on each replay. All verbs
work remains on that persistent thread.

The ring separates GPU-owned claim/publication counters from CPU-owned
consume/completion counters by cache line. The publishing thread uses
system-scope compare-and-swap plus `__threadfence_system()` before exposing a
descriptor. The consumer uses acquire loads and release stores. NVIDIA
documents host-native atomics on coherent Grace Hopper/Grace Blackwell
systems and the host visibility guarantee of system fences:

- <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html>

All CUDA allocation and graph construction happen before replay. All verbs
work remains on the session's persistent progress thread.

## Safety properties

- The mixed-Q prototype accepts only exact contiguous BF16 `[Q, 6144]`,
  Q1--Q5 descriptors. Invalid Q/byte pairs fail closed.
- One Q5-capacity session is bound to one stable CUDA stream and may define
  multiple Q1--Q5 nodes before the first replay.
- Eager and graph submissions cannot be mixed on the same session.
- A device CAS claims each sequence exactly once. A replay keeps its claimed
  sequence in block-shared memory; it never reads a global "latest" value.
- A claim is permitted only while
  `claimed_sequence - completed_sequence < 64`. A full ring backpressures the
  GPU publisher without overwriting a live slot.
- Descriptors publish strictly in sequence order. Sequence exhaustion,
  impossible counter ordering, or publication regression sets
  `overflow_sequence`; the progress thread then terminates the process.
- Same-stream replay is mandatory. Graph clones or concurrent replay on a
  second stream are outside the contract because transport and tensor
  buffers are session-owned and fixed.
- Session destruction first synchronizes the caller stream while the verbs
  progress thread is still alive. It then stops the thread and releases
  mapped state.
- No fallback is safe after a replay publishes its descriptor. A process
  must terminate on transport failure, as with the eager native path.

## Native validation

Build on a Spark node with the CUDA and verbs development packages:

```bash
cmake -S spark_transport -B spark_transport/build \
  -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build spark_transport/build -j --target \
  spark_transport_capi tp4_graph_command_test spark_tp4_graph_q1_probe
ctest --test-dir spark_transport/build -R tp4_graph_command_test \
  --output-on-failure
```

Stage the same `spark_tp4_graph_q1_probe` binary on all four nodes, then
launch the ranks concurrently with the peer mapping already used by
`run_tp4_tensor_probe.ps1`:

```text
rank 0: peer0=192.0.2.2 peer1=192.0.2.4
rank 1: peer0=192.0.2.1 peer1=192.0.2.3
rank 2: peer0=192.0.2.4 peer1=192.0.2.2
rank 3: peer0=192.0.2.3 peer1=192.0.2.1
```

The checked-in four-rank runner performs the executable SHA-256 preflight, concurrent
launch, timeout handling, exact sequence checks, log collection, and cleanup:

```powershell
.\spark_transport\scripts\run_tp4_graph_q1_probe.ps1 `
  -Warmup 10 -Iterations 100 `
  -ControlPort0 9960 -ControlPort1 9961 `
  -CpuSet 10,11 `
  -MaxGraphSubmitUs 25 -MaxDeviceUs 75
```

The default remains the original single-graph performance mode. Its tensor
contents now change before every replay, but command-line defaults, expected
sequence calculation, and performance thresholds remain compatible.

Verify identical SHA-256 hashes for the rebuilt probe and
`libspark_transport_capi.so` on every rank. Leave `SPARK_TRANSPORT_TRACE`
unset for performance gates.

Each invocation uses:

```bash
spark_tp4_graph_q1_probe \
  --rank RANK --peer0 PEER0 --peer1 PEER1 \
  --device0 rocep1s0f0 --device1 rocep1s0f1 \
  --gid0 3 --gid1 3 \
  --control-port0 9470 --control-port1 9471 \
  --warmup 10 --iterations 100 \
  --max-graph-submit-us 25 --max-device-us 75
```

The gate passes only if every rank exits zero and reports:

- `publisher=device`;
- `correct=true`;
- `passed=true`;
- `submit_gate=pass` and `device_gate=pass`;
- `mismatched_elements=0`;
- `published=consumed=completed=110`;
- `overflow=0`.
- `graph_launches=input_updates=110`;
- `monotonic_sequences=true`.

Repeat with 1,000 measured replays after the 100-replay smoke test; the
expected sequence is then 1,010. Finally run 10,000 measured replays without
performance thresholds as a ring/order soak and require sequence 10,010,
zero overflow, and zero mismatches on all ranks.

```powershell
.\spark_transport\scripts\run_tp4_graph_q1_probe.ps1 `
  -Warmup 10 -Iterations 10000 `
  -ControlPort0 9968 -ControlPort1 9969 `
  -CpuSet 10,11 -DisablePerformanceGates
```

Before vLLM integration, also capture many Q1 nodes in one executable graph.
This catches the production pattern that a single model graph contains many
collective nodes sharing one transport session:

```powershell
.\spark_transport\scripts\run_tp4_graph_q1_probe.ps1 `
  -Warmup 2 -Iterations 100 -OperationsPerGraph 128 `
  -ControlPort0 9976 -ControlPort1 9977 `
  -CpuSet 10,11 -DisablePerformanceGates
```

The expected published, consumed, and completed sequence is
`(2 + 100) * 128 = 13,056`. Require zero overflow and zero mismatches. The
probe uses a distinct output allocation for every captured Q1 node, matching
the graph allocator's stable-pointer contract more closely than replaying one
node repeatedly.

Before vLLM integration, run the stronger two-executable-graph lifecycle
gate. Performance gates are intentionally forbidden in this mode because it
synchronizes and verifies every launch:

```powershell
.\spark_transport\scripts\run_tp4_graph_q1_probe.ps1 `
  -Warmup 2 -Iterations 100 `
  -MultiGraphValidation `
  -GraphAOperations 3 -GraphBOperations 128 `
  -ControlPort0 9978 -ControlPort1 9979 `
  -CpuSet 10,11 -DisablePerformanceGates
```

The exact terminal state is:

```text
graph launches = (2 + 100) * 2 = 204
input updates  = 204
sequence       = (2 + 100) * (3 + 128) = 13,362
```

Every rank must additionally report:

- `mode=multi`;
- `graph_a_operations=3` and `graph_b_operations=128`;
- `graph_launches=204` and `input_updates=204`;
- `captured_nodes=131` and `pre_replay_capture_valid=true`;
- `published=consumed=completed=13362`;
- `monotonic_sequences=true`;
- `post_replay_capture_rejected=true`;
- `mismatched_elements=0`, `overflow=0`, and `passed=true`.

Multi-graph mode fails closed if graph A exceeds 16 nodes, graph B is not
exactly 128 nodes, or a performance threshold is supplied. A sequence mismatch
after any launch terminates immediately rather than relying only on the final
counters.

Run the mixed-Q gate only after rebuilding and staging one identical probe
binary on all four ranks:

```powershell
.\spark_transport\scripts\run_tp4_graph_q1_probe.ps1 `
  -Warmup 2 -Iterations 100 `
  -MultiGraphValidation -MixedQValidation `
  -GraphAOperations 3 -GraphBOperations 128 `
  -ControlPort0 9980 -ControlPort1 9981 `
  -CpuSet 10,11 -SubmitCpu 10 -ProgressCpu 11 `
  -DisablePerformanceGates
```

In addition to the A3/B128 lifecycle gates above, every rank must report:

```text
mixed_q=true
session_capacity_bytes=61440
q1_nodes=27 q2_nodes=26 q3_nodes=27 q4_nodes=25 q5_nodes=26
active_bytes_per_graph_cycle=4792320
validated_active_bytes_total=488816640
```

The final byte count is `(2 + 100) * 4,792,320`. The runner computes these
values from its requested graph sizes and fails if any rank reports a
different histogram or byte count. This mode has been built but intentionally
has not been run live while the model is available.

The 25-us submission and 75-us device thresholds are regression gates for the
observed cluster, whose paired eager Q1 device mean was about 42 us. Also
retain the relative gate `graph device <= 1.5 * paired eager device` on every
rank if the eager baseline moves. The superseded host-node implementation
measured roughly 780 us host submission and 989 us device time per call; any
result in that range means a callback or progress-wakeup serialization
remains.

Before staging, rebuild both `spark_transport_capi` and
`spark_tp4_graph_q1_probe`; the mapped command-ring layout changed from a
64-byte to a 128-byte header. The generic
`spark_tp4_capture_all_reduce(..., q, ...)` symbol was added while the Q1
entrypoint remains as a compatibility wrapper. Mixing an old library with a
new probe or Python adapter is unsupported.

## Remaining production work

1. Pass the changing-input mixed-Q A3/B128 alternating four-rank lifecycle
   gate, then validate the opt-in adapter in a four-rank vLLM capture/replay.
2. Prove behavior under graph destruction, worker teardown, and CUDA
   asynchronous errors. Keep graph cloning and cross-stream replay rejected.
3. Replace process-fatal ring diagnostics with the repository's shared
   asynchronous error reporting without adding blocking to the publication
   hot path.
4. Prove the source-composed shared capture stream and serialized replay on
   all four live workers; the pristine-image lifecycle probe has passed.
5. Require equal per-Q capture counts, zero cold-capture fallback, advancing
   native completion sequences, and no captured NCCL for Q1--Q5.
6. Extend graph-native replay to remaining steady-state all-gather/vocabulary
   signatures before claiming a socket-free captured decode round.
7. Run four-node bounded model-output and pinned performance gates before
   promotion.
