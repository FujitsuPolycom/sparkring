# SIRCL baseline for optimization agents

This packet is the mandatory starting point for work on SIRCL correctness,
benchmarking, or overlap. It describes the code in this checkout, not an
intended future design. Read the cited seams before proposing a change.

SIRCL is the direct-cable collective layer under `spark_transport/`; SparkRing
is the surrounding runtime, launcher, adapter, validation, and fallback stack.
SparkCache is a separate persistent-context subsystem. The public boundary and
current scope are stated in [`docs/SIRCL.md`](../SIRCL.md#implemented-data-path).

## Evidence vocabulary

Every technical claim in an agent report, design, test, or benchmark must be
tagged mentally—and explicitly when ambiguity is possible—as one of:

- **Observed:** read directly from current source, an attested runtime state, or
  a captured artifact. Cite the source seam or artifact and its identity.
- **Measured:** produced by an executed, identity-bound experiment. State the
  hardware, topology, payload, mode, warmup, repetitions, statistic, gates, and
  artifact hash. A measured simulator result is still only simulator evidence.
- **Modeled:** produced by a CPU/GPU simulator, reference function, analytic
  model, or mocked test. It may test an invariant but does not prove native
  scheduling, RDMA progress, CUDA ordering, numerical equivalence, or speed.
- **Proposed:** a design not present in the current path. State which observed
  invariant it changes and which native evidence would promote it.
- **Unknown:** not proven by the available source or evidence. Preserve the
  unknown; do not convert it into an assumption.

Repository maturity words (`planned`, `candidate`, `offline-validated`,
`live-validated`, `accepted`) have the meanings in [`AGENTS.md`](../../AGENTS.md#terminology).
Reference-lane measurements do not transfer to the public-functional lane or a
different runtime/checkpoint. The required claim label and sealed-cell rules
are in [`docs/RESULTS.md`](../RESULTS.md#4-methodology-the-claim-discipline).

## Current TP4 all-reduce state machine

### Topology and round order

For rank `r`, the schedule is exactly:

```text
round 0: peer = r XOR 1, device 0   (0<->1, 2<->3)
round 1: peer = r XOR 3, device 1   (0<->3, 1<->2)
```

The code rejects any round other than 0 or 1 and binds the device index to the
round in [`spark_transport/src/tp4_schedule.cpp`](../../spark_transport/src/tp4_schedule.cpp#L7)
(lines 7-22). `Tp4AllreduceSession` constructs one channel, mapped arena, and RC
endpoint for each round, barriers both channels, and only then starts progress
([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L283), lines
283-333).

The production TP4 all-reduce does **not** currently run two independent round
workers. A single call enqueues one CUDA kernel on the caller stream
([`gpu_tp4_tensor.cu`](../../spark_transport/src/gpu_tp4_tensor.cu#L269), lines
269-314). That kernel spans both rounds. The session owns one progress thread
([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L531), lines
531-561), and its `progress()` function calls `exchange_round()` for round 0,
waits for it to finish, then calls round 1
([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L645), lines
645-678). Therefore, overlapping the existing round-0 and round-1 calls is a
**proposed protocol change**, not a scheduler switch.

### One round, precisely

For either round and one doorbell token:

1. The GPU stages the round's send data, applies a system-scope fence, and
   publishes `producer_sequence`. For round 0 this is the original input
   ([`gpu_tp4_tensor.cu`](../../spark_transport/src/gpu_tp4_tensor.cu#L195),
   lines 195-202). For round 1 it is the BF16 partial sum generated after round
   0 (lines 203-220).
2. The sole progress thread waits for the local producer token, posts the RC
   payload write followed by the remote doorbell write, and waits for local
   send completion ([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L124),
   lines 124-151).
3. The GPU waits for `remote_sequence`, consumes the received buffer, and
   publishes `consumer_sequence`. Round 0 computes the pairwise BF16 partial
   sum; round 1 adds the two partial sums into the output
   ([`gpu_tp4_tensor.cu`](../../spark_transport/src/gpu_tp4_tensor.cu#L201),
   lines 201-241).
4. The progress thread writes the consumer token to the peer's
   `acknowledgement_sequence` and waits for the peer acknowledgement before the
   exchange returns ([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L149),
   lines 149-169).

This acknowledgement is the current buffer-reuse barrier. Removing or moving
it requires a new ownership proof; a throughput-only test is insufficient.
The four control words are defined in
[`gpu_doorbell.hpp`](../../spark_transport/include/spark_transport/gpu_doorbell.hpp#L8)
(lines 8-14).

TP2's standalone probe is a one-edge, one-round primitive with the same
producer/remote/consumer/ack vocabulary
([`app/tp2_probe.cpp`](../../spark_transport/app/tp2_probe.cpp#L173), lines
173-216). It is useful evidence for a single exchange, but it is not proof that
a proposed TP4 two-round or multi-slot schedule is correct.

## CPU workers, GPU work, and streams

- Each `Tp4AllreduceSession` owns one `progress_thread_`; it is not one worker
  per edge or round. Eager work is queued under `submission_mutex_`, then the
  progress thread waits for the caller-stream input gate and executes both
  rounds ([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L563),
  lines 563-602).
- Eager `all_reduce()` enqueues the GPU kernel asynchronously on the supplied
  CUDA stream. When the caller changes streams, `CudaStreamHandoff` orders the
  old and new streams before submission
  ([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L360), lines
  360-426).
- Graph capture requires an active and stable caller stream. A device kernel
  claims and publishes a descriptor in a 64-slot mapped command ring; the same
  session progress thread is the sole CPU consumer
  ([`tp4_graph_command.hpp`](../../spark_transport/include/spark_transport/tp4_graph_command.hpp#L89),
  lines 89-122; [`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L605),
  lines 605-637).
- Graph producer and consumer counters live on separate cache lines. A claim is
  blocked when 64 descriptors are outstanding, and publication is ordered by
  system-scope atomics/fences
  ([`gpu_tp4_tensor.cu`](../../spark_transport/src/gpu_tp4_tensor.cu#L57), lines
  57-120). `completed_sequence`, not merely `consumed_sequence`, releases ring
  capacity.
- Session destruction first drains the retained caller stream while keeping
  verbs progress alive, then stops and joins the progress thread
  ([`tp4_session.cpp`](../../spark_transport/src/tp4_session.cpp#L336), lines
  336-357).

Do not call the existing 64 command descriptors "64 transport ingress
buffers." The descriptor ring can queue replay commands, but the current TP4
session still has one mapped send/receive arena per round and serializes both
round exchanges for each consumed descriptor.

## Ownership, barriers, and failure boundary

Current ownership is sequence-based:

- GPU producer: claim descriptor (graph only), fill payload/descriptor, then
  release-publish the sequence.
- CPU progress thread: sole descriptor consumer and verbs submitter/completion
  reaper for its session.
- Remote GPU: consume only after the remote doorbell matches the expected
  token; publish consumer completion only after the reduction/copy.
- CPU progress thread: return from a round only after both local GPU completion
  and peer consumption acknowledgement.
- Graph slot: reusable only after CPU publishes `completed_sequence`.

Exact graph mode accepts only equality. An observed value below the expected
sequence keeps waiting; an observed value above it publishes overflow and
enters the fatal wait. Non-graph mode may accept a later sequence. Ring
overflow or invalid graph ordering is therefore fatal
([`gpu_tp4_tensor.cu`](../../spark_transport/src/gpu_tp4_tensor.cu#L123), lines
123-152). Native failure after the CUDA stream has acquired an unfulfillable
wait cannot safely fall back in-process: the vLLM adapter terminates the worker
([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py#L815),
lines 815-847 and 868-888). Any proposed overlap must specify fail-closed
behavior for every partially published slot and every edge failure.

## Admission, fallback, and accounting

The vLLM all-reduce adapter admits only its exact world-size, group, shape,
dtype, CUDA, and contiguity contract; an ineligible signature calls the
original implementation and records the reason
([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py#L190),
lines 190-235; lines 799-847). In shadow mode, the native candidate is compared
with the original result, but the original result remains authoritative until
the signature passes and optional promotion is enabled (lines 860-945).

Accounting has distinct and incomplete surfaces:

- `spark_collective_audit.py` counts calls through wrapped **stock** seams by
  eager/capture phase, family, reason, and a bounded set of pointer-free
  signatures. It is enabled only when `SPARK_TP4_GRAPH_STATUS_PATH` is set
  ([`spark_collective_audit.py`](../../spark_transport/integrations/vllm/spark_collective_audit.py#L214),
  lines 214-245 and 263-309).
- Graph adapters expose custom captured-node events and native
  published/consumed/completed/overflow sequences
  ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py#L535),
  lines 535-588 and 768-780).
- `spark_graph_status_reporter.py` periodically combines all-reduce, DCP,
  indexer, vocabulary, and stock snapshots into a schema-v3 atomic status file
  ([`spark_graph_status_reporter.py`](../../spark_transport/integrations/vllm/spark_graph_status_reporter.py#L18),
  lines 18-56 and 79-106).

A zero stock delta proves only that the instrumented stock seams did not grow
inside the bounded window. It does not by itself prove how many eager custom
calls ran, that every possible fallback seam is wrapped, that RDMA carried the
payload, or that all ranks observed the same window. A valid gate must capture
before/after snapshots on all four ranks, require positive rank-synchronous
native progress on every rank where the workload expects native execution,
require zero relevant stock delta and zero
dropped-signature ambiguity, and inspect transport/runtime logs.

## Existing numerical and benchmark entrypoints

These are the exact public entrypoints in this checkout:

| Purpose | Entrypoint | What it proves—and does not prove |
|---|---|---|
| FP32 arithmetic audit | [`spark_transport/scripts/run_tp4_numerical_audit.ps1`](../../spark_transport/scripts/run_tp4_numerical_audit.ps1), invoking [`tp4_numerical_audit.py`](../../spark_transport/integrations/vllm/tp4_numerical_audit.py) | Generates deterministic BF16 cancellation/scale cases; compares custom TP4 and `dist.all_reduce` with FP32 truth; reports MAE/RMSE/max/exact/bit mismatches. The runner sets the reference process group to NCCL **Socket** (`run_tp4_numerical_audit.ps1`:102-105), does not time either arm, and prints the aggregate only on rank 0 (`tp4_numerical_audit.py`:120-153). It is not a same-stack SIRCL-vs-patched-NCCL-IB performance benchmark. |
| Eager custom TP4 probe | [`spark_transport/scripts/run_tp4_tensor_probe.ps1`](../../spark_transport/scripts/run_tp4_tensor_probe.ps1), invoking `spark_tp4_tensor_probe` from [`app/tp4_tensor_probe.cu`](../../spark_transport/app/tp4_tensor_probe.cu) | Exercises the native two-round session, validates results, supports alternating streams/delayed submission, and emits `TP4_TENSOR`. Its per-call timer surrounds `session.all_reduce()` submission (`tp4_tensor_probe.cu`:253-264); completion is synchronized after the loop (`tp4_tensor_probe.cu`:267-273). Interpret its metric according to that timing boundary, not automatically as end-to-end device latency. |
| Graph Q1 probe | [`spark_transport/scripts/run_tp4_graph_q1_probe.ps1`](../../spark_transport/scripts/run_tp4_graph_q1_probe.ps1), invoking `spark_tp4_graph_q1_probe` | Exercises graph capture/replay and verifies command sequences, overflow, output, submission and device timing. Its documented evidence scope is [`spark_transport/GRAPH_NATIVE_TP4_Q1.md`](../../spark_transport/GRAPH_NATIVE_TP4_Q1.md). |
| Other native families | The `run_tp4_{allgather,dcp_graph,indexer_graph,vocab_graph}*` scripts under [`spark_transport/scripts/`](../../spark_transport/scripts/) and matching targets in [`spark_transport/CMakeLists.txt`](../../spark_transport/CMakeLists.txt#L56) | Family-specific standalone probes only. Passing one family does not promote another family or end-to-end serving. |

There is no existing public entrypoint here that by itself supplies an
identity-sealed, timed, same-stack SIRCL-versus-patched-NCCL comparison with
before/after fallback counters. That is a required new harness/evidence cell,
not something to infer from the numerical audit or historical result tables.

## Identity and evidence contract

Before comparing two cells, record and verify at minimum:

1. repository commit and dirty-state description;
2. runtime lock ID and lock-file SHA-256; serving image repository digest and
   local image ID; in-image runtime manifest self-hash;
3. vLLM, Torch, CUDA, NCCL, SIRCL library, adapter bundle, and benchmark
   executable/script identities (commit/version plus SHA-256 where applicable);
4. host OS/kernel, NVIDIA driver and firmware, GPU/NIC identities, relevant
   clocks/power mode, and container runtime;
5. all four rank roles, exact direct-cable edge mapping, HCA/GID/MTU/link rate,
   management interface, and pre/post link/error counters;
6. model/checkpoint identity if a model is in the loop; TP/DCP/MTP, cache,
   graph, allocator, environment, and launch arguments;
7. payload shape/dtype/layout, collective family, eager/graph mode, stream
   policy, warmup, measured repetitions, ordering/randomization, timeout, and
   statistic;
8. synchronized per-rank before/after custom status, stock/fallback counters,
   overflow/fatal counters, request/probe correctness, and complete failure
   disposition;
9. raw artifact SHA-256, UTC window, sanitization method, and an explicit lane,
   maturity, hardware, and evidence-scope label.

The public runtime requires identical attested artifacts on all ranks
([`docs/PUBLIC_FUNCTIONAL_TARGET.md`](../PUBLIC_FUNCTIONAL_TARGET.md#5-the-acceptance-gate),
acceptance stage 1). Use [`runtime/runtime-lock.json`](../../runtime/runtime-lock.json) as
the pin source, not prose copied into a goal. Public evidence must be processed
through [`scripts/collect_evidence.py`](../../scripts/collect_evidence.py),
whose residue check fails before writing if sensitive identifiers survive
([`collect_evidence.py`](../../scripts/collect_evidence.py#L598), lines 598-665).

For an A/B transport benchmark, hold every identity and setting constant
except the declared transport selection. Do not compare SIRCL in one image with
stock or differently patched NCCL in another and call the delta transport-only.
Both cells must prove which path actually executed; a requested mode string is
not execution evidence.

## Forbidden inference patterns

Agents must not:

- describe a simulator, mocked queue, CPU BF16 loop, or descriptor-layout test
  as the native CUDA/RDMA state machine;
- call a sequential BF16 sum "NCCL ring order" without proving the exact NCCL
  algorithm, protocol, channels, chunk schedule, and reduction order for that
  run;
- infer two-round overlap from two HCAs, two endpoints, two mapped arenas, or a
  64-slot command ring;
- infer safe buffer reuse from eventual correctness; prove the producer,
  consumer, acknowledgement, completion, and failure transitions;
- treat finiteness of only the last iteration, a permissive relative-to-NCCL
  error multiplier, or agreement with one BF16 implementation as an FP32
  numerical gate;
- treat `correct=true`, process exit zero, zero stock delta, or positive custom
  count as a standalone correctness proof;
- conflate comparability validation with a performance pass, or absence of a
  regression with an optimization win;
- combine eager submission time, graph host-launch time, GPU elapsed time,
  transport completion time, and model throughput under one "latency" label;
- transfer measurements between lanes, commits, binaries, drivers, firmware,
  checkpoints, topologies, graph modes, or cache states;
- report planned multi-slot ingress as implemented, or an offline model as
  live-validated;
- silently weaken a gate because the required counter or identity is absent.
  Missing proof makes the result indeterminate.

## Staged optimization preflight

These checks are staged so ordinary source exploration and contributions are
not blocked on access to a live cluster. Complete only the stage reached by the
claim or action.

### Before changing protocol code

- [ ] I read this file, `AGENTS.md`, the targeted adapter, the native session,
  the GPU kernel, and the exact probe/harness I will modify.
- [ ] I drew the current producer/consumer/ack/completion state machine from
  source and identified every ownership boundary the proposal changes.
- [ ] I stated whether the work is observed, modeled, proposed, or measured and
  did not use a model as native proof.
- [ ] I named one narrow hypothesis, a falsifier, the single intended variable,
  and all controlled confounders.
- [ ] I specified per-iteration correctness against FP32 truth or an exact
  reference as appropriate, including non-finite handling and tolerances chosen
  independently of the candidate result.

### Before native execution

- [ ] I identified the exact custom and stock execution counters and the gaps in
  those counters; missing evidence fails closed.
- [ ] I specified native stress for wraparound/backpressure, alternating
  streams, delayed ranks/edges, timeout, partial publication, and teardown.
- [ ] I prepared a same-stack identity manifest and per-rank before/after
  evidence schema before collecting a performance number.
- [ ] I will inspect the generated remote plan before any read-only command and
  obtain explicit authorization before host mutation or stopping serving, per
  `AGENTS.md`.

### Before a performance or acceptance claim

- [ ] I separated host submission, GPU elapsed, transport completion, and
  end-to-end model metrics.
- [ ] I will publish only sanitized artifacts and will label lane, maturity,
  hardware, and evidence scope without promotion beyond the passed gate.

An unchecked box blocks only its stage. The correct output is then a bounded
source, evidence, or harness task rather than a native or performance claim.
