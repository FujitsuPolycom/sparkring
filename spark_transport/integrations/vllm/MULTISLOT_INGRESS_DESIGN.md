# Multi-Slot Ingress Design for GLM Decode Overlap

## Status

**Design contract only. No native code changes. No performance model.**
This document defines the unimplemented changes required before
multi-slot overlap can occur. The previous CPU timing model was
removed because it modeled an overlap pattern that the actual native
execution mechanics cannot produce.

## Observed current behavior (cited from source)

### Single buffer per round

`tp4_session.cpp` line 285:
```cpp
layout_ = make_tp2_buffer_layout(options_.payload_bytes);
```
`gpu_tp2.cu` lines 145-159 (`make_tp2_buffer_layout`): allocates exactly
one `send_offset`, one `receive_offset`, and one `DoorbellControl`
per buffer. The session creates two such buffers (`buffer0_`, `buffer1_`)
at `tp4_session.cpp` lines 289-301.

### Sequential eager progress

`tp4_session.cpp` lines 563-602 (`progress_loop`): the progress thread
pops one `Submission` from `submissions_` deque, calls
`eager_input_gates_->wait()` (CUDA event), then calls `progress()` which
calls `exchange_round()` for round 0, then round 1, before returning to
pop the next submission. There is no overlap between consecutive
collectives.

### Sequential graph progress

`tp4_session.cpp` lines 605-637 (`progress_graph_commands`): the
verbs progress thread consumes one `Tp4GraphCommand` descriptor from
the 64-slot ring (`kTp4GraphCommandCapacity = 64`,
`tp4_graph_command.hpp` line 10), then calls `progress()` for both
rounds, then calls `tp4_graph_command_complete()` before consuming the
next command.

### GPU kernel performs both rounds in one launch

`gpu_tp4_tensor.cu` lines 155-243 (`tp4_tensor_all_reduce`): a
single kernel on the caller stream performs round 0 (staging, RDMA
doorbell, wait, hadd) and round 1 (RDMA doorbell, wait, hadd) in one
launch. There is no stream split or kernel split that would allow the
GPU to overlap rounds of different collectives.

### One QP, one CQ, one MR per edge

`verbs_endpoint.hpp` lines 57-62: `VerbsEndpoint` has one
`ibv_qp* queue_pair_`, one `ibv_cq* completion_queue_`, one
`ibv_mr* memory_region_`. `verbs_endpoint.cpp` lines 195-228
(`write`): posts one RDMA Write WR per call.
`verbs_endpoint.cpp` lines 231-254 (`wait_for_send`): polls the CQ for
exactly one completion with a matching `work_id`, with a 5-second
timeout. This is one-sided RDMA Write — no receive WR is posted by the
receiver; the NIC writes directly into the registered MR.

### Doorbell protocol

`gpu_doorbell.hpp` lines 8-17: `DoorbellControl` has
`command_sequence`, `producer_sequence`, `remote_sequence`,
`consumer_sequence`, `acknowledgement_sequence`, `observed_sequence`,
`mismatch_count`, and `reserved` fields, each 64-bit, cache-line
aligned (`alignas(64)`, `sizeof == 64`). The protocol in
`tp4_session.cpp` lines 124-170 (`exchange_round`):
1. Store `producer_sequence` (GPU publishes)
2. Post RDMA write of payload + doorbell
3. Wait for send completion
4. Wait for `consumer_sequence` (GPU consumes remote data)
5. Post RDMA write of `consumer_sequence` as ack
6. Wait for `acknowledgement_sequence` (peer consumed)

## Why buffer slotting alone cannot produce overlap

The previous design assumed that allocating N independent buffer slots
per round would allow round 0 of collective N+1 to overlap round 1 of
collective N. This is false given the current execution mechanics:

1. **One progress thread serializes both rounds.** `progress()`
   (`tp4_session.cpp:640-673`) calls `exchange_round()` for round 0,
   then `exchange_round()` for round 1, synchronously. The thread
   cannot start round 0 of N+1 until round 1 of N returns.

2. **One GPU kernel does both rounds.** `tp4_tensor_all_reduce`
   (`gpu_tp4_tensor.cu:155-243`) is a single kernel launch on the
   caller stream. The GPU cannot overlap round 0 of N+1 with round 1
   of N because both rounds of one collective are in one kernel.

3. **Buffer slotting only prevents memory contention.** Different
   slots prevent `send0` overwrite, but the verbs thread and GPU
   kernel are still serialized by the single-thread, single-kernel
   design. Slotting is necessary but not sufficient for overlap.

## Required changes (all unimplemented)

### 1. Interleavable round progress

Round 0 of collective N+1 must be able to progress while round 1
of collective N is still in flight. The current single progress
thread (`tp4_session.cpp:640-673`) serializes both rounds per
collective.

**Possible mechanisms (not mandatory):**
- Split progress into per-round functions with separate verbs workers
- Event-driven single-thread with non-blocking round dispatch
- Persistent multi-collective kernel that processes multiple
  collectives' rounds in one launch

### 2. Safe per-slot ownership

Buffer ownership must prevent the GPU kernel of collective N+1
from overwriting `send0` while the verbs thread is still reading
it for collective N (`gpu_tp4_tensor.cu:195-198`). The current
per-buffer `DoorbellControl` (`gpu_doorbell.hpp:8-17`) must be
extended to per-slot or a new ownership protocol must be designed.

### 3. GPU execution concurrency

The GPU must be able to execute round 0 of collective N+1
concurrently with or ordered relative to round 1 of collective N.
The current single kernel on the caller stream
(`gpu_tp4_tensor.cu:155-243`) prevents this.

**Possible mechanisms (not mandatory):**
- Split into separate kernels on separate CUDA streams
- Persistent kernel that processes an internal queue of round
  operations
- Multi-collective kernel that batches rounds from different
  collectives

## Unresolved correctness risks

1. **QP ordering**: RC QPs guarantee ordering for writes to the same
   QP, but concurrent round 0 and round 1 exchanges on the same QP
   may interleave doorbell sequences in ways the current protocol
   does not handle.

2. **CQ capacity**: the current `wait_for_send` (`verbs_endpoint.cpp:231-254`)
   polls for exactly one completion. Concurrent exchanges need either
   non-blocking posting or a CQ with multiple outstanding WRs.

3. **CUDA event ordering**: the current `CudaEventGatePool`
   (`cuda_event_gate.hpp`) supports per-sequence events, but the
   single-kernel, single-stream design prevents overlap. Splitting
   kernels across streams requires new event dependencies.

4. **Teardown**: the destructor (`tp4_session.cpp:336-358`)
   synchronizes the caller stream and drains submissions. With
   multi-slot, it must verify all N slots' acknowledgements are
   complete before destroying the verbs thread.

5. **Fail-closed invariant**: there is no fallback after native
   enqueue (`tp4_session.cpp:172`, `fatal_async_failure`). Multi-slot
   must preserve this — a slot timeout must remain process-fatal.

## Claims deliberately not made

- This design does not claim SIRCL is a generic `torch.distributed`
  ProcessGroup.
- This design does not claim a measured or modeled speedup.
- This design does not change the fail-closed invariant.
- This design does not affect NCCL fallback for prefill or
  non-matching tensors.
- This design does not prove slot safety; it identifies the
  unimplemented requirements.
