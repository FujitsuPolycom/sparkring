# TP4 tiled bidirectional-ring bulk lane

A design proposal for a four-rank collective path serving multi-MiB
payloads. The design reduces and gathers around the cabled ring in
both directions at once, so that each rank transmits on both NIC cages
simultaneously, and streams the payload through a bounded pool of
fixed-size tiles rather than through a payload-sized arena. No
implementation of it exists; section 1 bounds what that means.

## 1. Status and evidence boundary

Status: research-only design proposal. Nothing specified in this
document is implemented. There is no native executor, no CUDA kernel,
no C ABI entry point, no control-port reservation, and no vLLM
dispatch path for this lane. Section 5 describes separate contracts
this design reuses, some of which do have implemented parts; that
status belongs to those contracts and not to this lane. Adoption of
any stage requires the gates in section 9.

Statements here fall into four classes, and none may be promoted to
another without the measurement that would justify it:

- **Measured** — values recorded in
  [`docs/DUAL_PORT_STRIPING_PROBE_20260818.md`](../docs/DUAL_PORT_STRIPING_PROBE_20260818.md)
  under that record's stated conditions and limitations.
- **Fitted** — linear fits over those measurements, and any projection
  evaluated outside the measured payload range. The two paths are
  fitted differently: the transport fit uses the only two payloads
  measured for a single kernel, while the primary NCCL fit is a
  three-point least-squares fit. Section 2 states each construction.
- **Structural** — consequences of the cabling and of the collective
  algorithms, computed from the nameplate link rate.
- **Assumed** — premises adopted without measurement here, listed
  under "Assumed" in section 2. Each names the leg of section 8 that
  would replace it with a measurement.

Three limitations apply to everything below. The measurements were
taken with isolated single-replay device timing, which does not
represent serving, where collectives pipeline with compute. No payload
above 6,291,456 bytes was measured on any path. No large-message
bandwidth measurement exists for this fabric at all, so every wire
figure derives from a nameplate 200 Gb/s rather than from an observed
rate.

## 2. Measured facts versus fitted projections

### Measured

Four-rank all-reduce, rank-0 p50, isolated single-replay device
timing. The wire schedule named `sequential` is recursive doubling
with one NIC cage active per round; `dual_port_striped` is recursive
doubling with the payload halved across both cages per round.

| Payload bytes | Schedule | Kernel | p50 µs |
|---:|---|---|---:|
| 491,520 | sequential | split_64k | 174.0 |
| 491,520 | sequential | fused | 618.5–622.8 |
| 491,520 | dual_port_striped | fused | 190.5–190.7 |
| 6,291,456 | sequential | split_64k | 1,223.7 |
| 6,291,456 | sequential | fused | 7,582–7,700 |
| 6,291,456 | dual_port_striped | fused | 1,228.5–1,234.6 |

Stock NCCL, under a comparable graph-captured device-timed control on
the same ring, at all six measured payloads: 796.2 µs at 12,288 bytes,
820.5 µs at 65,536 bytes, 907.6 µs at 491,520 bytes, 1,140.1 µs at
2,097,152 bytes, 1,149.9 µs at 2,359,296 bytes, and 1,600.8 µs at
6,291,456 bytes.

Serving shapes, from the capture-inventory trace of the
DeepSeek-V4-Flash-0731 profile at width 4096: eager prefill chunks
reach [2048, 4096] = 16,777,216 bytes, the capture inventory holds
approximately 87 captured width-4096 all-reduce nodes per full-forward
graph, and every width-4096 collective in that profile takes the stock
NCCL path. The inventory records which nodes exist, not how often each
is replayed; replay frequency under sustained traffic is unmeasured,
so no per-second or per-step collective rate follows from it.

Admission ceiling: eager all-reduce is width-generic behind
`VLLM_SPARK_TP4_EAGER_WIDTHS`, but admits at most 512 query rows, and
only when `VLLM_SPARK_TP4_PREFILL_Q512=1`. At width 4096 that ceiling
is 4,194,304 bytes. A 16,777,216-byte prefill chunk is four times the
largest payload the eager path can accept, so the gap this lane
addresses is payload size, not width.

### Fitted

Linear fits of the form `fixed + marginal x bytes`. The two paths have
different amounts of data behind them, and the fits are constructed
differently as a result.

The transport fit is a two-point fit over 491,520 bytes (174.0 µs) and
6,291,456 bytes (1,223.7 µs). Those are the only two payloads measured
for any single transport kernel, so no other construction is
available.

The NCCL fit is a least-squares fit over the three largest measured
NCCL payloads — 2,097,152 bytes (1,140.1 µs), 2,359,296 bytes
(1,149.9 µs), and 6,291,456 bytes (1,600.8 µs) — giving
895.9 µs + 1.1194e-4 µs/byte. The three smaller measured NCCL payloads
are excluded because they sit inside its approximately 800 µs fixed
floor, where the marginal term is a small fraction of the total and
including them biases the slope toward the floor rather than toward
the payload range the lane targets.

The construction does not drive the result. As a robustness check
only, and not as the projection this document uses, a two-point fit
over the two largest measured NCCL payloads alone (2,359,296 and
6,291,456) gives 879.4 µs + 1.1467e-4 µs/byte and moves the projected
crossing from 11.2 MiB to 11.4 MiB.

| Path | fixed µs | marginal µs/byte | projected at 16,777,216 B |
|---|---:|---:|---:|
| Transport, sequential split_64k | 85 | 1.810e-4 | ~3,120 µs, projected |
| Stock NCCL | 896 | 1.119e-4 | ~2,770 µs, projected |

These two lines are projected to cross at approximately 11.2 MiB, a
payload at which neither was measured. Below the projected crossing
the transport wins on NCCL's fixed cost; above it NCCL wins on
marginal cost, which is 1.62 times cheaper per byte. The
16,777,216-byte prefill chunk lies above the crossing. Both
evaluations extrapolate well beyond the measured range and rest on two
points each; section 8 measures the crossing rather than inferring it.

Separating each fitted marginal into the structural wire term of
section 3 and an unattributed remainder. The striped row is a third
fit, over the same two payloads as the transport row (491,520 and
6,291,456 bytes) and using the midpoint of each reported range,
190.6 µs and 1,231.5 µs:

| Path | marginal | wire term | remainder |
|---|---:|---:|---:|
| Transport sequential, 2N on one cage at a time | 1.810e-4 | 8.0e-5 | 1.010e-4 |
| Transport striped, 2N across two cages | 1.795e-4 | 4.0e-5 | 1.395e-4 |
| NCCL, 1.5N on one cage | 1.119e-4 | 6.0e-5 | 5.2e-5 |

Two consequences follow. First, the striped fit's total marginal sits
within 0.84% of the sequential split_64k fit while its wire term is
half, leaving a combined fused-kernel and striped-protocol remainder of
about 3.85e-5 µs/byte over split_64k — within 4% of the 4.0e-5 µs/byte
the schedule saves. The kernel and the protocol changed together, so
that remainder is not attributable to either alone. The probe binary requires the `fused` kernel for
`dual_port_striped`, so the recorded measurements cannot separate the two
effects. Second, the transport's off-wire cost per byte is about twice
NCCL's, and a wire-only improvement does not close that.

### Assumed

- 200 Gb/s is treated as 25 GB/s of payload throughput per cage per
  direction.
- NCCL is assumed to run a unidirectional ring, moving 1.5N bytes per
  rank. Section 3 states the counter evidence and its limits.
- Reduction cost is assumed to track element-add count. This is the
  assumption the design turns on, and leg 3 of section 8 measures it
  rather than relying on it.

## 3. Directional rail model

Four Sparks form a 4-cycle whose edge set decomposes into two perfect
matchings, each carried by its own NIC cage. Every rank therefore
faces one ring neighbour through `rocep1s0f0` and the other through
`rocep1s0f1`. The links are full duplex, so a rank can in principle
transmit on both cages and receive on both cages at the same time.
That simultaneity is a property of the cabling and the link standard;
it has not been demonstrated under load on this fabric, and leg 2 of
section 8 exists to demonstrate it.

Per-rank cost of one N-byte four-rank all-reduce:

| Schedule | transmitted | per cage | cages transmitting | wire wall time | element-adds |
|---|---:|---:|---:|---:|---:|
| Recursive doubling, sequential | 2N | N | one per round | 2N/B | 2N |
| Recursive doubling, striped | 2N | N | two | N/B | 2N |
| Unidirectional ring | 1.5N | 1.5N | one | 1.5N/B | 0.75N |
| Bidirectional ring | 1.5N | 0.75N | two | 0.75N/B | 0.75N |

At 16,777,216 bytes and 25 GB/s those wire wall times are 1,342 µs,
671 µs, 1,007 µs and 503 µs. A single traversal of 16 MiB is 671 µs;
640 µs corresponds to 16 decimal MB and is not the figure this design
uses.

The bidirectional ring splits the payload in two, runs an independent
ring reduce-scatter and all-gather over each half in opposite
directions, and gives each direction the cage facing that direction's
successor. It moves the same 1.5N bytes per rank as a unidirectional
ring but spreads transmission across both cages, so its structural
upper bound against a single-cage ring is a factor of two in wire wall
time. It also performs 0.75N element-adds where recursive doubling
performs 2N.

Counter evidence for the NCCL assumption: on the graph-captured NCCL
control, rank 0 recorded about 4 GB on `rocep1s0f0` and about 20 MB on
`rocep1s0f1`. `port_xmit_data` counts transmitted octets only. A
unidirectional ring on this cabling transmits to its successor through
one cage and receives from its predecessor through the other, which
produces exactly this transmit asymmetry — but the counter does not
establish the complementary receive traffic, and no other pattern has
been excluded. Leg 5 of section 8 captures `port_rcv_data` alongside
`port_xmit_data`. Until it does, the factor of two is a structural
upper bound on the opportunity, not a measured headroom result.

## 4. Two independent deliverables

Raw transfer and reduction carry different risk and must not share a
gate or a maturity label.

### 4a. Raw striped write

The cabling constrains what a striped write can mean. Each rank has a
direct link to exactly two peers, its ring neighbours, and each of
those is reachable through exactly one cage. A byte-range copy to one
directly attached peer therefore uses one cage and one cage only;
there is no way to put a second cage behind it, because the second
cage does not reach that peer.

Three forms are available, and the deliverable is the two-destination
write:

- **Two-destination write.** One logical operation carrying a distinct
  byte range to each ring neighbour, both cages transmitting at the
  same time. This is the only single-hop form in which both cages
  carry payload for one operation, and it is the form leg 2 of section
  8 measures. It serves callers that place or replicate data around
  the ring rather than callers that must reach one named peer.

  Byte semantics, so the gate is reproducible without further
  interpretation. The quoted 16,777,216 bytes is the **total** the
  operation carries, not a per-destination figure. It is partitioned
  into two equal contiguous ranges of 8,388,608 bytes, the first to
  the ring successor and the second to the ring predecessor, each
  leaving through the one cage that reaches its destination. The timed
  interval opens when the source rank submits the operation and closes
  when completions for both destinations have been signalled at the
  source. Throughput is the total 16,777,216 bytes divided by that
  interval, so the figure is aggregate across both cages. The
  single-cage baseline it is compared against carries the same total
  of 16,777,216 bytes as one range to one neighbour, timed and divided
  the same way; the comparison therefore holds bytes constant and
  varies only how many cages carry them.
- **Two-path write to the rank two hops away.** That rank sits at the
  far side of the 4-cycle, so two edge-disjoint two-hop paths reach it,
  one through each neighbour. Half the range can travel each path with
  the neighbour relaying, which puts both local cages to work at the
  cost of a store-and-forward hop and of consuming relay bandwidth on
  ranks that are not party to the transfer. Specified here as
  available; not part of the two-destination-write deliverable, and no
  gate covers it.
- **Single-destination two-cage write.** Requires both cages of a rank
  to terminate on the same peer, which the qualified cabling does not
  provide. It is excluded with two-rank striping in section 10 and is
  not proposed here.

The performance question for the two-destination write is whether both
cages transmit simultaneously at close to twice single-cage
throughput. Leg 2 answers exactly that, because the operation it
measures and the operation being specified are the same shape. There
is no reduction remainder to attribute.

Performance is not the whole of its maturity. Before any caller
adopts it, this lane must define destination arena ownership and
registration lifetime, completion signalling to the caller, payload
validation sufficient for a caller to detect a truncated or corrupted
transfer, and the ordering each caller requires between a transfer and
its own state changes. Those are caller-contract questions, not
wire-schedule questions, and a bandwidth measurement answers none of
them.

### 4b. Ring all-reduce

The bidirectional ring described in section 3. It carries the entire
unattributed remainder from section 2. Projecting it at 16,777,216
bytes under the two bounding attributions of that remainder:

Every value in this table is projected. The lane has not been built,
so none of its costs has been measured at any payload.

| Attribution of the remainder | projected marginal | projected total | versus NCCL projection | NCCL projected to regain advantage at |
|---|---:|---:|---:|---:|
| Entirely reduction, scaling as 0.75N/2N | 6.79e-5 | ~1,220 µs | 2.27x | no crossover under these linear fits |
| None of it scales with the schedule | 1.310e-4 | ~2,280 µs | 1.22x | ~41 MiB |

Both bounds beat the NCCL projection at 16,777,216 bytes, on NCCL's
fixed cost. Only the first keeps that advantage as payloads grow. Under the
second, the lane serves 16 MiB prefill but loses on the larger
transfers a streaming or migration caller would present. The
attribution therefore decides not only whether the lane is worth
building but how far it generalises.

## 5. Reused tiled contracts versus missing native implementation

A tile-pool contract exists in this repository, and this design reuses it rather
than defining a second one. Its status is uneven and must not be
overstated:

| Component | Status |
|---|---|
| Capacity selector, `integrations/vllm/spark_tp4_prefill_capacity_pool.py`, with unit test | Implemented, offline-validated |
| Source-level tile contract, `include/spark_transport/tp4_tiled_session.hpp` | Unsupported: header definitions exist and no translation unit consumes them. Coverage is partial — Python tests cross-check the header's capacity maxima, while its native layout, ticket, and credit helpers remain uncompiled and untested |
| Native executor, CUDA kernels, C ABI, control-port reservation, vLLM dispatch | Unsupported; none exists |

Reused from the tile contract: the pool layout with its slot and lane
geometry, per-operation active-byte descriptors, generation-tagged
tickets, the cumulative consumed-through credit window, ragged final
tiles, and a registered arena whose size does not depend on payload
size.

Unsupported, and required by this lane: the native side in full,
together with the layout and ticket tests that the other session
families have and this contract lacks.

The contract is also width-specific, and reusing it for this lane
requires extending it. Both halves fix the row width at 6144 BF16
elements — `kTp4TiledBf16ElementsPerQueryRow` in the header, and
`TARGET_WIDTH` in the capacity selector — so capacity classes, tile
counts, and active-byte totals are all derived from a row size the
DeepSeek-V4-Flash-0731 profile does not use. That profile is width
4096.

Two extensions would resolve it, and one must be chosen before the
contract can carry this lane:

- Pin a row width per session, exchanged in the setup handshake of
  section 6, and derive capacity classes from it.
- Drop rows from the contract entirely and express capacity, tiles,
  and descriptors in bytes, leaving row geometry to the caller that
  has it.

The byte-only expression matches this lane's operations more directly and
removes a concept the lane does not otherwise need. Either way the
extension is native work carrying its own qualification: the capacity
selector, the tile descriptors, and their tests are validated at
width 6144 and at no other width, and no evidence covers width 4096.

The arena argument for reuse is quantitative. The exact-payload
striped layout allocates about 4N per endpoint arena and one arena per
cage, so about 8N per rank — roughly 128 MiB at a 16,777,216-byte
payload, and growing with it. The tile pool is 8,389,632 bytes per
edge and constant.

## 6. Capacity, ragged-block, and descriptor design

The setup handshake pins, once per session: maximum active bytes, tile
bytes, world size and topology, schedule version, protocol version,
datatype, and alignment. It does not pin the size of any individual
operation. Each operation carries its own active-byte count, validated
against the pinned maximum. Chunked prefill emits a ragged final chunk
per sequence, so a handshake that fixed an exact payload size would
force either one session per distinct chunk size or padding every
chunk up to capacity.

Session capacity derives from launch configuration — the batched-token
budget multiplied by model width and datatype size — and is validated
at construction. Section 7 removes any mid-flight escape, so a shape
above capacity must be rejected at admission and routed to NCCL, never
discovered at enqueue.

The ring adds a block decomposition above the tile decomposition. A
payload splits eight ways: four ranks by two directions. Both levels
can be ragged, and a block size is not generally a whole number of
tiles. The tile descriptor's operation-offset and active-byte fields
express this, but the mapping between ring blocks and pool tiles is
unimplemented, and it is the principal descriptor cost of the design.

Tile size and slots per edge become schedule-coupled parameters rather
than free ones. At a 16,777,216-byte payload with 512 KiB tiles, a
block is 2 MiB, or four tiles, and each direction runs six dependent
steps; an eight-slot pool then holds two steps in flight. That sizing
must be derived for the schedule, not inherited from the decode
defaults.

Pipeline fill and drain set the lower payload bound. Six dependent
steps per direction, at a per-step protocol latency on the order of
the 4.5 µs p50 measured for a 16 KB write into a registered mapped
arena, plus one tile of wire time, puts fill and drain near 60–80 µs
by arithmetic rather than by measurement. Against 503 µs of ring wire
time at 16,777,216 bytes that is 12–16%; at 2,097,152 bytes it exceeds
the transfer itself. The lane is therefore expected to pay off
somewhere between 4 and 8 MiB, with the boundary set by measurement.

## 7. Fail-closed semantics

Admission is a predicate evaluated before any enqueue: world size,
communicator group, datatype, contiguity, and active bytes within
capacity. Operations that fail it route to NCCL and are counted in the
collective audit, so fallback stays recorded rather than assumed.

After the first enqueue or RDMA publication there is no fallback. Peer
queue-pair and credit state may have advanced by that point, and a
unilateral retreat by one rank desynchronises the others. A failure
past that point terminates the worker. Unexpected generations, invalid
active-byte counts, and poisoned slots remain process-fatal, as the
tile credit contract specifies.

One consequence is a simplification: because no rollback path has to
be preserved, output placement is chosen for speed rather than for
recoverability, and the lane need not stage results to keep a fallback
viable.

## 8. Ordered measurement plan

Leg 1 runs on the two-rank probe binary this repository builds. Leg 5
runs the patched deployment NCCL fallback with counter capture and
needs no transport prototype.
Legs 2, 3 and 4 each require a prototype that has no implementation,
so the plan and the gates distinguish two scopes of implementation.
**Probe scope** is a research prototype living in the probe harness:
sufficient to run a leg and produce a number, not reusable, not
reachable from the C ABI or from any serving path, and disposable if
the leg fails. **Product scope** is a reusable session with a C ABI
entry point, control-port reservation, and vLLM dispatch.

Section 9 orders its authorisations so that no leg is conditioned on
its own result: leg 1 stands on this document alone, its outcome
authorises the probe-scope prototypes for legs 2 through 4, and those
legs together with leg 5 authorise product scope.

Each leg can stop the sequence.

1. **Single-edge component bounds.** The two-rank probe
   (`spark_tp2_probe`, built from `app/tp2_probe.cpp`) run with
   `--bytes` at 1, 4, 8, 16 and 32 MiB, reporting its producer,
   exchange, and reduce-and-ack phases separately. Establishes an achievable large-message rate for
   one cage and a reduction cost at bulk sizes. Two nodes, one cable,
   no code changes. This bounds components only: the four-rank path
   differs in topology, step count, and reduction order, so a two-rank
   decomposition cannot explain the four-rank remainder by itself.
2. **Four-rank two-destination write.** The operation of section 4a:
   each cage carrying a distinct byte range to its own ring neighbour,
   both transmitting at once, no reduction. `port_xmit_data` and
   `port_rcv_data` bracket every leg. Establishes whether both cages
   transmit simultaneously and at what fraction of twice single-cage
   throughput, and gates the raw write deliverable on its own. The
   measured operation and the specified primitive are the same shape,
   so the result transfers directly.
3. **Local reduce-only kernel.** Same arenas, mapped layout, and
   datatype, at the ring's block geometry — 2 MiB blocks at a
   16,777,216-byte payload — not at full payload width. This
   attributes the remainder of section 2 and decides section 4b.
4. **Sequential versus bidirectional ring**, with the same reducer and
   the same tiled staging on both arms. Breaks the kernel-and-schedule
   coupling that leaves the 6,291,456-byte schedule comparison in
   the record at docs/DUAL_PORT_STRIPING_PROBE_20260818.md uninterpretable.
5. **Extended NCCL control** through 16 and 32 MiB with both cage
   counters, and with intermediate points at 8 and 12 MiB so that the
   projected crossing near 11.2 MiB is measured rather than
   extrapolated. Curvature is not itself a stop condition, and its
   direction matters: if measured NCCL rises faster than the linear
   fit past 6,291,456 bytes, the case for this lane strengthens,
   because NCCL is worse at the target payload than section 2
   projects. If it rises more slowly, the case weakens. The stop
   condition is margin, not shape — the sequence stops when measured
   NCCL at 16,777,216 bytes sits close enough to the ring's candidate
   range that the p50 margin required in section 9 is out of reach.
6. **Serving evaluation**, only if the gates in section 9 pass.

## 9. Adoption gates

Four authorisations, in order. Each names what it permits and what must
hold before it takes effect.

**1. Leg 1.** Carries no precondition beyond this document and needs
no implementation. It measures achievable single-cage large-message
throughput and a two-rank reduction cost at bulk sizes.

**2. Probe-scope prototypes for legs 2 through 4.** Permits the
two-destination write, the reduce-only kernel at block geometry, and a
bidirectional ring, each inside the probe harness. Nothing built under
this authorisation is reusable, reachable from the C ABI, or reachable
from a serving path, and none of it survives a failed leg.

Its precondition is a recomputation, not a further measurement:
substitute leg 1's measured per-cage rate for the 25 GB/s nameplate
rate throughout section 3, and leg 1's measured reduction cost for the
optimistic attribution bound in section 4b. If the recomputed ring
projection at 16,777,216 bytes does not clear the p50 margin required
in authorisation 4 against the NCCL fit, the premise is eliminated and
no prototype is authorised. Leg 1 measures two ranks reducing two
contributions over a full payload, while the ring reduces four
contributions over eighth-payload blocks, so this recomputation is a
screen and not the attribution; leg 3 performs the attribution.

**3a. Product-scope implementation, two-destination raw write.** A
reusable session, its C ABI entry point, control-port reservation, and
the destination-side ownership and completion machinery of section 4a
are authorised by leg 2 and the single-cage baseline of leg 1 alone,
against the numerical raw-write gates stated below. This deliverable
carries no reduction and no ring schedule, so it neither requires nor
inherits the ring's comparison against the patched deployment NCCL
fallback; a ring result cannot authorise it and its own result cannot
authorise the ring.

**3b. Product-scope implementation, ring all-reduce.** A reusable
session, its C ABI entry point, control-port reservation, the
tile-contract width extension of section 5, and vLLM dispatch require
legs 2 through 5 to
show the ring beating the patched deployment NCCL fallback, not merely
beating the transport's own sequential schedule. Unpatched upstream
NCCL cannot initialise on this switchless topology, so the incumbent
is the patched build the deployment actually loads (NCCL 2.30.7 by its
own banner). The capture-inventory trace of the
DeepSeek-V4-Flash-0731 profile, recorded 2026-08-18, gives that
fallback as the path every width-4096 collective in that profile
dispatches to, so it is the incumbent for this payload class and an
internal improvement slower than it has no serving value.

**4. Serving admission.** Requires all of:

- Exact integer-valued oracle equality on all four ranks, plus the
  declared random-BF16 association gate. Ring reduce-scatter changes
  summation order relative to both recursive doubling and NCCL, so
  bitwise agreement with either is not the criterion and must not be
  substituted for one.
- Ring p50 at least 10% below the patched deployment NCCL fallback at
  16,777,216 bytes, over bracketed repeated arms rather than a single
  session.
- Ring p95 no worse than that fallback at the same payload.
- Both transmit rails showing balanced positive counter deltas, with
  receive counters captured alongside.
- Zero overflow, poison, credit, and timeout failures on every rank.

The raw write deliverable of section 4a carries its own admission
gates, and they are numerical rather than a reference to leg 2's
result:

- Aggregate two-destination p50 throughput at 16,777,216 bytes at
  least 1.5 times the single-cage p50 throughput leg 1 measures at the
  same payload, over bracketed repeated arms rather than a single
  session. A result at or below the single-cage baseline is a
  regression and fails.
- Aggregate p95 no worse than the single-cage baseline.
- Balanced positive transmit deltas on both cages, with receive
  counters captured alongside.
- Destination arena ownership and registration lifetime, completion
  signalling, payload validation, and caller ordering implemented and
  covered by tests, not specified only.
- Zero credit and timeout failures on every rank.

It does not inherit the ring's gates, and the ring does not inherit
its result.

Anything admitted under these gates enters as research-only.
Promotion into a qualified profile requires that profile's own
qualification evidence.

## 10. Exclusions

- **Decode collectives.** The split-kernel sequential paths are the
  decode default, and no schedule change in this design applies to
  them.
- **The `dual_port_striped` schedule.** Graph-only, fused-kernel-only,
  and restricted to two exact decode capacities. This design changes
  none of that.
- **Graph capture.** This lane is eager. Prefill runs eager in the
  serving profiles, and host-driven transfers are not captured. A
  captured bulk collective would be a separate proposal.
- **Two-rank striping, and the single-destination two-cage write that
  depends on it.** Excluded, and not a prerequisite for the four-rank
  lane. The qualified cabling connects each cage to a different
  neighbour, so striping one transfer to one named peer across both
  cages requires a different cable plan and its own qualification.
  Section 4a specifies what the qualified cabling does support.
- **Migration and streaming callers.** No caller adopts the raw write
  primitive until it passes the maturity items in section 4a.
- **NCCL.** Stock NCCL is the path for every collective family this lane
  does not admit.
