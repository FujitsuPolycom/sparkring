# TP4 tiled bidirectional-ring bulk lane

Status: design proposal, research-only. Nothing in this document is
implemented, and native work begins only if the measurement sequence in
section 6 passes its gate. Measured inputs are cited to their records;
projections are labeled as fits, not measurements. An earlier revision
of this proposal framed the lane around dual-port striping and gated it
against the transport's own sequential path; both framings are
superseded here. Striping describes the physical port utilization; the
algorithmic content of this proposal is a bidirectional-ring
reduce-scatter/all-gather schedule executing under the transport's
existing tiled-capacity contract, gated against NCCL.

## 1. Problem

The transport's collective paths are engineered for decode-class
payloads, which measurement shows are overhead-bound: kernel-strategy
fixes, not bandwidth, removed the dominant cost at every measured
decode shape (docs/DUAL_PORT_STRIPING_PROBE_20260818.md).

Bulk payloads invert the proportions, and the serving stack produces
one today: chunked-prefill all-reduces of BF16 [2048, 4096] (16 MiB),
which take the stock NCCL path because they exceed every admission
bound. KV-cache migration and weight streaming sit in the same class.

Linear fits over the probe record's measured points expose the
problem quantitatively. The transport's best sequential kernels fit
approximately 85 µs + 1.81e-4 µs/byte; the graph-captured NCCL
control fits approximately 910 µs + 1.10e-4 µs/byte. NCCL's higher
floor buys a shallower slope, so the fits cross near 11.1 MiB:

| 16 MiB all-reduce | Projected from fits |
|---|---:|
| Sequential custom path | ~3.12 ms |
| NCCL | ~2.75 ms |

Above roughly 11 MiB the existing custom path is projected to lose to
NCCL. A bulk lane that merely extends the sequential schedule upward
therefore has no serving value; the incumbent for the 16 MiB prefill
collective is NCCL, and NCCL is the baseline this proposal gates
against.

## 2. Why a bidirectional ring

On four ranks, recursive doubling moves 2N bytes per rank per
all-reduce and performs 2N element additions per rank. A ring
reduce-scatter plus all-gather moves 1.5N bytes per rank and performs
0.75N additions, and on this topology its two transfer directions map
onto the two NIC ports transmitting simultaneously, for an effective
2 x 200 Gb/s per rank.

The algorithmic counts favor the ring by construction; what they do
not establish is cost. Element-addition count is not GPU reduction
cost: memory traffic, staging copies, and per-tile bookkeeping may
dominate, and the fits above show the non-wire intercept is the
quantity that decides the outcome. The measurement sequence exists to
attribute those components before any implementation.

At 16 MiB the ring's projected range is wide precisely because of
that attribution uncertainty: roughly 1.22 ms if reduction and
staging scale with the reduced element count, to roughly 2.28 ms if
they do not scale at all. The bottom of that range beats NCCL by 2.3x;
the top loses. This is a measurement question, not a design argument.

Reference wire arithmetic, stated exactly: one 16 MiB (16,777,216
byte) traversal at 200 Gb/s (25 decimal GB/s) is approximately
671 µs.

## 3. Design

A bidirectional-ring reduction schedule and native executor for the
transport's existing tiled-capacity contract. This is deliberately
not a new session architecture: `tp4_tiled_session.hpp` and the
prefill capacity-pool contract
(integrations/vllm/TP4_PREFILL_CAPACITY_POOL.md) already define
tiles, active bytes, tickets, credits, ragged tails, and bounded
arenas. Those are reusable contracts, not an implemented transport
engine — their native dispatch is explicitly unsupported today — and
this proposal supplies the missing executor rather than a parallel
abstraction.

- **Fixed capacity, per-operation active bytes.** The setup handshake
  pins, once, in an exact versioned record: maximum active bytes,
  tile bytes, world size and topology, schedule and protocol
  versions, datatype and alignment. Ragged operations then carry
  `active_bytes` in their per-operation descriptors, following the
  capacity-pool contract. No per-operation payload-size handshake
  exists.
- **Eager, not graph-captured.** Prefill runs eager in the serving
  profiles, and transfer callers are host-driven. A captured bulk
  collective would be a separate proposal.
- **Tiled pipeline.** Tile-granular staging overlaps reduction with
  wire transfer; tile size is a measured parameter, not a constant
  chosen here.
- **Two operations.** `all_reduce(active_bytes)` as the
  bidirectional-ring reduce-scatter/all-gather, and
  `write(active_bytes)` as a dual-direction raw copy to one peer for
  KV migration and weight streaming (no reduction).
- **Fail-closed semantics, stated precisely.** Before any native
  enqueue, falling back to the stock path is safe and permitted.
  After a native enqueue or RDMA publication, no fallback is
  attempted under any condition, including operations whose output
  was never copied out: peer arena and queue-pair state may already
  have advanced, so the worker terminates, exactly as the existing
  eager collective paths behave.
- **Arena reuse.** Registered mapped arenas, per-port RC queue pairs,
  doorbell and credit machinery, and the bounded in-flight window are
  composed from the existing primitives.

### Serving integration

A research-only admission input routes all-reduces above the bulk
threshold (found by measurement, expected near the fitted 11 MiB
crossover) and matching the signature gate to the bulk executor.
Everything below the threshold is untouched. The audit records the
routing; fallback remains counted, never assumed.

### Out of scope

- Decode collectives, the graph sessions, and the graph-only striped
  schedule: unchanged.
- Two-rank literal striping: the qualified four-node ring cables each
  port to a different neighbor, so a two-port two-rank pair requires
  a different physical cable plan. That is a separate proposal and
  does not block this lane.

## 4. Measurement sequence

1. Single-edge two-rank probe at 1/4/8/16/32 MiB with
   producer/exchange/reduction/acknowledgement decomposition — bounds
   the per-component costs (with the caveat that two-rank
   decomposition bounds components but cannot fully compose the
   four-rank result: topology, step count, and reduction order
   differ).
2. Four-rank write-only dual-direction probe — actual per-port
   bandwidth with both directions active, no reduction, port byte
   counters bracketing.
3. Local reduce-only kernel over the same buffers, sizes, and
   mapped-memory layout — isolates reduction and memory-traffic cost.
4. Sequential versus bidirectional-ring all-reduce using the same
   reducer and tiled staging — isolates the schedule.
5. NCCL graph and eager controls extended through 16 and 32 MiB,
   dual-rail, with both port counters and the loaded-version banner
   retained.
6. Gate: native serving work is authorized only if the measured
   bidirectional ring beats the measured NCCL result at 16 MiB.
   Beating the transport's own sequential path authorizes nothing.

Every quoted result distinguishes wire utilization (bytes per port
against link rate) from algorithm bandwidth (payload over time), and
any bound coverage (sizes skipped, single-session statistics) is
stated rather than implied.

## 5. Adoption

Stage 6's gate feeds a research-only serving admission evaluated on
measured time-to-first-token at matched contexts. Adoption into any
qualified profile requires its own qualification evidence. If the
gate fails, this document records the negative result and the lane
ends there.
