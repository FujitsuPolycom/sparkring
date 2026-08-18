# Bulk striped transport: a bandwidth lane for multi-MiB payloads

Status: design proposal, research-only. Nothing in this document is
implemented, and adopting any stage requires the measurement gates in
section 6. The problem statement is grounded in measured collective
latencies; the wire-time figures are arithmetic from link rate, not
measurements.

## 1. Problem

The transport's collective paths are engineered for decode-class
payloads, which measurement shows are overhead-bound, not
wire-bound: a BF16 [256, 4096] all-reduce (2 MiB) completes in
436 µs on the four-rank ring while its per-traversal wire time at
200 Gb/s is roughly 84 µs — about one fifth of the total
(docs/DUAL_PORT_STRIPING_PROBE_20260818.md). Kernel-strategy fixes
(split_64k/tiered_64k), not extra bandwidth, are what removed the
dominant cost there, and the dual-port striped schedule showed no
median advantage at any measured decode shape.

The proportions invert for bulk payloads. The serving stack already
produces one today: chunked-prefill all-reduces of BF16 [2048, 4096]
(16 MiB), which currently take the stock NCCL path because they
exceed every admission bound. At 200 Gb/s a 16 MiB traversal is
roughly 640 µs of wire time — the dominant term at that size, and
the term that doubling effective bandwidth would halve. The same
applies to any future multi-MiB transfer: KV-cache migration between
tenants or hosts, weight streaming, snapshot shipping.

No existing path serves this class:

- Eager all-reduce sessions admit at most 512 rows.
- The graph sessions are decode-shaped (Q512 capacity ceilings) and
  graph-only.
- The existing `dual_port_striped` wire schedule is graph-only,
  fused-kernel-only, and fixed to two exact decode capacities; it is
  a decode experiment, not a bandwidth lane.

## 2. Why striping is the right tool here and not for decode

A single collective's rounds are data-dependent (round 1 consumes
round 0's partial sums), so the two NIC ports cannot be overlapped
across rounds of one recursive-doubling collective. The only way one
transfer uses both ports is to split its payload across them —
striping. For decode payloads that attacks the minority wire slice;
for bulk payloads it attacks the majority slice. The lane is
therefore scoped to bulk payloads only, with the boundary set by
measurement (section 6), expected in the 4–16 MiB region where wire
time crosses half of total time.

Topology changes what striping means:

- **Four-rank ring.** Each port faces a different neighbor, so
  striping one peer-to-peer transfer across ports is impossible.
  The bandwidth-optimal bulk collective is instead a bidirectional
  ring reduce-scatter + all-gather: every rank transmits on both
  ports simultaneously in opposite ring directions, for an
  effective 2 x 200 Gb/s per rank.
- **Two-rank pair.** With both QSFP cages cabled to the same peer,
  striping is literal: half the byte range per port, one reduction
  (or raw copy) at the receiver. This is the only configuration in
  which a second port helps a two-rank deployment at all.

## 3. Design

One new session family, `BulkStripedSession`, alongside the existing
collective sessions:

- **Byte-range contract, no row geometry.** Bulk payloads are
  contiguous byte ranges with an alignment requirement (proposed:
  256-byte). The row-geometry machinery (query-row contracts,
  provider seams, graph command descriptors) does not apply and is
  not reused. The setup handshake exchanges an exact geometry record
  {payload_bytes, chunk_bytes, schedule, topology}, following the
  exact-record pattern the graph sessions use — no hashed identity.
- **Eager, not graph-captured.** Prefill runs eager in the serving
  profiles, and bulk transfers (migration, streaming) are host-driven.
  Graph capture is out of scope for the lane; a captured bulk
  collective would be a separate proposal.
- **Chunked pipeline.** Payloads are processed in fixed-size chunks
  (proposed initial size: 1 MiB, tuned by measurement) so reduction
  compute overlaps wire transfer and per-chunk protocol overhead
  amortizes to noise at bulk sizes.
- **Two operations.**
  - `all_reduce(bytes)`: four-rank bidirectional ring
    reduce-scatter + all-gather, or two-rank striped
    exchange-and-add.
  - `write(bytes)`: striped raw copy to one peer (the KV-migration /
    weight-streaming primitive; no reduction, so it generalizes to
    any registered destination arena).
- **Arena reuse.** Registered `cudaHostAllocMapped` arenas, per-port
  RC QPs, doorbell/credit machinery, and the bounded in-flight window
  are the existing primitives; the session composes them rather than
  introducing new transport mechanics.

### Integration: serving prefill

The vLLM adapter gains a bulk admission class behind its own
research-only input, mutually independent of the decode inputs:
an all-reduce whose payload exceeds the bulk threshold and matches
the signature gate (world size, group, dtype, contiguity) routes to
the bulk session instead of stock. Everything below the threshold is
untouched. The audit records the routing so fallback remains counted,
never assumed.

### Integration: transfers

`write(bytes)` is exposed through the C API for host-driven callers
(SparkCache migration, weight staging). No vLLM involvement.

## 4. What this explicitly does not change

- Decode collectives: the split-kernel sequential paths remain the
  decode default; no schedule change is proposed for them.
- The existing graph-only `dual_port_striped` schedule: unchanged,
  still a decode-shape experiment.
- NCCL remains the path for every collective family the lane does
  not admit.

## 5. Build stages

1. **Probe first.** Extend the four-rank probe (and revive the
   two-rank probe harness) with a bulk mode: sequential versus
   bidirectional-ring versus NCCL at 4/8/16/32 MiB, port byte
   counters bracketing every leg, wire-utilization derivation
   (bytes moved per port per second against link rate), correctness
   on every rank. This stage alone answers whether the lane is worth
   building and locates the admission threshold.
2. **Native session.** `BulkStripedSession` with the two operations,
   both topologies, exact geometry handshake, chunk pipeline.
3. **vLLM bulk admission.** Research-gated prefill routing with
   audit coverage; serving A/B on time-to-first-token at 8K/64K/128K
   contexts with decode-regression guard.
4. **Transfer callers.** SparkCache/KV-migration adoption of
   `write(bytes)`.

## 6. Adoption gates

- Stage 1 proceeds to stage 2 only if measured bidirectional-ring
  (or two-rank striped) bandwidth at 16 MiB exceeds the best
  sequential result by at least 1.5x with correct results on all
  ranks.
- Stage 3 ships as research-only and is evaluated on measured
  time-to-first-token, not projected; adoption into any qualified
  profile requires its own qualification evidence.
- Every quoted number distinguishes wire utilization (bytes per port
  against link rate) from algorithm bandwidth (payload over time);
  the striping probe record documents why conflating them misleads.
