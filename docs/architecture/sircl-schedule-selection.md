# SIRCL placement-aware schedule estimates

**Experimental, research-only.** The offline selector estimates schedules and
per-lane windows for two to eight Sparks. It prints conditional queue certificates,
paths, rejected candidates and predictions. It does not configure a session,
contact a Spark, install routes, or replace any runtime tuning decision.

From the repository root, with `spark_transport/sircl` on `PYTHONPATH`:

```bash
python -m sparkring_sircl.schedule_selection --fabric ring8 --placement 0-5 --co-resident 6-7
python -m sparkring_sircl.schedule_selection --fabric ring8 --placement 0-3 --co-resident 4-7 --collective all_gather
python -m sparkring_sircl.schedule_selection --fabric-document fabric.json --placement 0-7 --sizes 4096,65536,4194304,33554432
```

The implementation is [schedule_selection.py](../../spark_transport/sircl/sparkring_sircl/schedule_selection.py).
The separate [subset policy](../../spark_transport/sircl/SUBSET_RING.md) remains
direct chain for large flat all-gather and pieces for large all-reduce.
The selector's estimates are hypotheses for measurement, including when a
candidate uses an operation or reservation policy that runtime does not support.

## Inputs and identity

The inputs are the fabric, ordered rank placement, disjoint co-resident placements,
collective, per-rank input bytes, eager/graph mode and flat/non-flat gather shape.
All-reduce bytes mean the full input; gather bytes mean one rank's shard;
reduce-scatter bytes mean the full input before scattering. Sizes are positive
multiples of 16. The CLI accepts inclusive ranges and explicit rank permutations.
Pair means two cables between two Sparks; `path2` means one cable.

For a setup document, the selector uses the existing `sparkring-fabric/v1`
validator. It preserves the cabling, fabric identity, complete document digest
and installed relay-plan reference. It validates uniform function names. The
document records a relay-plan digest, not its forwarding rules: path estimates
assume the universal same-function shortest-path table that SIRCL derives from
those cables. A relayed candidate is refused when the document has no SIRCL
relay plan. Verification of the live table against that reference is outside
this offline program. Synthetic `ringN`/`pathN` inputs model that table without
asserting that any fabric has been configured.

The output key includes physical cables, the document/reference when supplied,
ordered placement, every co-resident placement, collective, mode, gather shape,
model revision, native and kernel source hashes, and SIRCL version/ABI. Positions
are not reduced to a shape label: 0-5 beside 6-7 is distinct from an isolated
six-Spark cycle. Overlapping deployments, duplicate ranks, invalid cables and
out-of-range positions are rejected. Unlisted positions are assumed idle.
Multiple independent sessions on the same positions are outside this model;
their QPs need explicit additional reservations before admission.

## Candidate paths and schedules

Each candidate chooses one or two lanes. Paths come from `routes.derive_lanes`:
the shortest physical route, or the existing primary/secondary split over tied
shortest routes. It never invents a forwarding route absent from the modeled
table. The physical direct-chain order is derived independently of rank order.
The first implementation does not search longer alternate paths, all Hamiltonian
orders, kernel grids, pieces, or channel counts. Those require additional
measurements and, for alternate routes, a matching installed relay plan.

| Candidate | Modeled bytes per directed peer/edge | Admission |
|---|---|---|
| One-shot all-reduce | S to each other rank | S <= 128 KiB |
| Two-shot all-reduce | 2S/N to each other rank | S <= 2 MiB |
| Pieces all-reduce | 2S/N to each other rank | Any modeled size |
| Pieces gather / scatter | S / S/N to each other rank | Any modeled size |
| Ring reduce / gather / scatter | 2(N-1)S/N / (N-1)S / (N-1)S/N per successor edge | A direct chain order exists; closing route bounded; at most one ring lane per queue |
| Chain all-reduce | S in each direction on each chain edge | Every chain edge direct |
| Chain gather / scatter | At cut i, iS and (N-i)S in opposite directions, divided by N for scatter | Direct chain; gather must be flat |

The size ceilings are model capacities, not a claim about an installed session's
allocation. A route with more than the supported three relays is refused. On a
path of eight, for example, a direct chain remains modelable while a ring's
closing path is refused. Some sparse placements have no admitted candidate;
the JSON records the reasons and the CLI exits 1. An unsafe fallback is never
substituted. One-shot and two-shot are distinct candidates for all-reduce only;
the gather/scatter pieces models include their existing small-operation paths.

## Cost model and calibration

For each candidate, the selector accumulates traffic on each directed cable and
on each endpoint NIC's send and receive resources. The two PCIe functions share
their cable's bound. Cable directions and NIC sends allow 24 GB/s; NIC receives
allow 26.8 GB/s. Pure hardware forwarding is charged to cable resources and
hairpin queues, not to endpoint DMA. Rates use decimal GB/s.

The predicted period is a launch/stage term plus the maximum of resource time
and each relayed lane's `lane_volume * RTT(window, relays) / window`. The launch
term is 5 us plus 1.5 us per dependency stage. Pieces adds 1.5 us per additional
2 MiB piece. These constants are assumptions, not measured kernel timings.
Eager and graph have separate measurement keys but share this provisional
latency estimate. Memory/reduction throughput, overlapping kernels and receiver
backpressure are not fitted. Ties prefer fewer lanes; this is an explicit model
tie-break, not evidence that one lane outperforms two on hardware.

Co-resident groups reserve all-pairs traffic at two lanes per peer. The model
divides a shared resource's bandwidth equally among groups using it. Placement
alone cannot determine the other deployment's offered load. This equal-share
scenario is neither a bandwidth guarantee nor a DCP2 performance prediction.
Queue safety is independent of that fairness assumption.

Calibration uses the retained `ring8-hw-20261013` records:

| Record | Observation | Use |
|---|---|---|
| `sircl-subset-ring/RESULTS.md`, cb519293 | TP6 ring, 65,536 B/lane, about 9.4 GB/s; pieces about 11.5 GB/s | Effective two-lane RTT inferred as 13.94 us |
| `net-sircl-0.1.1-profile2/RESULTS.md`, SIRCL section | Path4 closing edge, 393,216 B/lane, 21.11 GB/s at 256 MiB eager | Effective RTT inferred as 37.25 us |
| `net-sircl-0.1.1-profile2/RESULTS.md`, net-sircl C | 163,840-byte allowance, measured post-to-CQE 40.3 us | Profiled completion-latency reference |
| `net-sircl-0.1.1-route/RESULTS.md`, DP | 245,760-byte allowance, 46.4 us | RTT grows with flight size |
| `net-sircl-0.1.1-smallsizes/RESULTS.md`, DP | 57.0 us, changed posting policy/load | Congestion sensitivity, not an independent byte-size anchor |
| `net-sircl-0.1.1-round9/RESULTS.md` | Cap 8 beat 16/24/64 on the relayed path | Avoid assuming more channels create bandwidth; channel search is deferred |

For two relays, `RTT_us = 13.94 + (W-65536)*(37.25-13.94)/327680`.
The code retains the unrounded anchor values. For another relay count h it
adds `4*(h-2)` us and floors at 4 us. Hop scaling is an extrapolation.
The congested estimate multiplies RTT by `57/46.4`. Neither estimate is a
confidence interval. The raw net-sircl profile is perturbed by instrumentation
and a different transport; using 40 us as SIRCL's RTT would predict only
3.28 GB/s at 64 KiB, contradicting the measured 9.4 GB/s. The inferred SIRCL
calibration reconciles that discrepancy explicitly; it is not a direct RTT
measurement. Two fitted anchors are not independent validation.

## Conditional queue proof

A queue is `(relay position, PCIe function class, egress port)`. With minimum
MTU 256 B and maximum framing/queue-accounting allowance 128 B per packet,
each byte of full-packet data consumes at most 1.5 bytes in its data direction
and generates at most 0.5 bytes of ACK traffic in the reverse direction.
Short tails, flags, software credits and their ACKs reserve
`C(q) = 16384 * (number of QPs through q + number through reverse(q))`.
The 16,384 B comes from 64 native credits, at most two WRs per credit and
128 B of control/tail allowance per WR, following the subset helper model.

For summed windows D(q), every candidate must satisfy:

```text
1.5 * D(q) + 0.5 * D(reverse(q)) + C(q) <= 524288
```

ACK-only queues are included. `packet_windows` divides the residual capacity
among the forward and reverse lanes, rounds down to chunks, then constrains
each lane by every data and ACK queue it traverses. It recomputes each final
inequality and refuses zero-chunk relayed lanes. Windows are also limited to
60 chunks. Ring candidates use 32 KiB chunks; others use 4 KiB chunks. The
output includes reservations for co-resident groups as well as the target.
These are joint requirements, not reservations that existing deployments have
accepted. A measurement never bypasses the queue proof.

The proof assumes completion-only release (`SIRCL_FORWARD_PROOF=0`), no retained
retransmission copies, the stated packet-accounting bound, and no unknown
relayed sessions. Hardware qualification must check those assumptions. It is
not an unconditional statement about physical NIC occupancy. It also assumes
all arena/link payload and ACKs are drained before switching candidate paths
or windows. The runtime does not implement this placement-wide transition
protocol. Consequently **no selector output is an executable session policy**.
In particular, copying a selected ring window into an existing session is unsafe.

The active subset runtime preserves OFF arena pacing and delivery-proof release.
Its simultaneous arena/ring bound is different from a drained offline candidate.
[The subset bound](../../spark_transport/sircl/SUBSET_RING.md#larger-window-analysis)
accounts for that overlap: at TP6 on 0-5 the maximum common 32 KiB-aligned ring
window in the conservative fixed-arena shared-QP envelope is 196,608 B, not 393,216 B.

## Measurements override the model

`--measurements FILE` accepts `sircl-placement-measurements/v1` with the exact
`key` printed by the selector and a `rows` array. Each row names `bytes`, the
candidate's full `id`, positive finite `period_us`, `exact: true`, `drops: 0`,
`counters_valid: true`, `isolation_passed: true`, and a nonempty `evidence`
reference. A candidate ID includes a hash of its paths and joint windows.
The fastest admitted measured candidate wins over all model predictions at
that exact size. Unmeasured sizes use the model; no measured interval is
extrapolated. Wrong keys and failed gate records are rejected. Unknown candidate
IDs cannot win. This small research schema is separate from runtime v2 tables;
existing measured defaults are comparison evidence, not silently imported as
placement-qualified measurements.

Measured selection also requires `--conditions FILE`, a JSON object recording
`image`, `drivers`, `dtype`, `shape`, `rotate_buffers` and `co_workload`. These
values become part of the exact key. Use explicit `none` descriptions for
absent co-resident load; do not leave measurement conditions implicit. Model-only
queries need no conditions file. A row/padded shape description must include
its row geometry, not merely say `non-flat`.

### Comparison with retained choices

These are model outputs at graph-mode input sizes, not timing measurements.
Runtime rows come from `runtime/common/sircl-tuning-defaults.json`, the cycle-4
v2 table it references, and SIRCL's built-in cycle-8 decisions.

| Fabric / placement | Collective and bytes | Model schedule | Retained choice/evidence | Agreement |
|---|---|---|---|---|
| Pair | Reduce 4/32 MiB | Chain, one lane | Pair row: ring | Disagrees; model does not price chain kernel overhead |
| Path4 | Reduce 4/32 MiB | Ring, two lanes | Path-4 row: ring | Schedule agrees; modeled packet window differs |
| Cycle4 | Reduce 64 KiB / 4 MiB | Pieces / pieces | Measured v2: one-shot / ring | Disagrees at both sizes |
| Cycle4 | Reduce 32 MiB | Ring | Measured v2: ring | Schedule agrees |
| Cycle8 | Reduce 4/32 MiB | Ring, one lane | Built-in decisions: ring | Schedule agrees; lane tie-break is unmeasured |
| Ring8 0-5 beside 6-7 | Reduce 32 MiB | Chain, one lane | Runtime subset: pieces; measured 64 KiB ring window lost to pieces | Unvalidated chain hypothesis, not a runtime recommendation |
| Ring8 0-5 beside 6-7 | Flat gather 4 MiB / 16 MiB | Pieces / chain | Chain measured 1.20 / 1.32 times pieces | Misses the 4 MiB winner; agrees at 16 MiB |
| Cycle7 | Reduce 32 MiB | Ring, one lane | Generic rules, no measured row | Requires measurement |

The 4 MiB gather candidates are close in this model (about 1,233 us pieces
versus 1,236 us chain). A narrow synthetic tie must not override the observed
chain gain. At 16 MiB, predicted chain bus bandwidth is 17.10 GB/s versus
16.33-16.34 measured. Predicted TP6 pieces at 64 MiB is 17.00 GB/s versus
11.45-11.46 measured: the missing kernel/transport costs materially matter.
This disagreement is a reason to measure placements, not to publish predictions
as defaults. The fitted path4 endpoint is 21.11 GB/s at 384 KiB, while the
selector's stricter 320 KiB packet envelope predicts about 20.09 GB/s at
256 MiB; the hardware record at 384 KiB measured 21.11-21.13 GB/s.

`sparkring fabric tune` currently measures configured group shapes and stores
SIRCL v2 timing tables through the ring harness. The existing promotion process
is specified in [releases](../development/releases.md). It does not yet produce
the placement schema or coordinate joint reservations. A placement measurement
requires a harness run with explicit `--groups`, universal fabric scope, every
co-resident session and its workload recorded, exact eager/graph results,
per-queue counters and the candidate's actual lane/window receipt. A converter
must join those receipts with the exact selector key and candidate ID; this
converter and the reservation/transition runtime are not implemented here.

An eventual install-time decision can enumerate paths, prove joint queue
reservations, prepare kernels, measure candidate schedules and bind a table
to fabric/build/co-residency. Runtime can then look up a size/collective/mode
decision within that admitted reservation. A topology, group or window change
requires drain and re-admission at install/session setup time. Runtime must
never independently enlarge a reservation based on a fast timing row. Today,
runtime retains its existing explicit overrides, matching tables and generic
rules; it does not import this selector.

## Hardware validation plans

### TP6 with DCP2, maximum 25-minute outage

The owner frees positions 0-5 for the harness and reserves DCP2 on 6-7 for a
single fixed decode probe. No other DCP2 client is allowed. Keep the existing
NIC settings and universal relay table. Pre-stage source, image, commands,
receipt collection and rollback without changing the deployment.

The retained hardware run required about 12 minutes to restore TP6. Reserve
13 minutes for restoration and 2 minutes for stop/start and idle/A/A baselines.
That leaves 10 minutes for three ABBA blocks (12 arms): **at most 50 seconds per
arm, including session setup and counters**. The prior 124-case arms took
3-4 minutes each and cannot fit. Use a reduced matrix: small all-reduce at
16 B/64 KiB, all-reduce at 8/64 MiB, flat gather at 4/16 MiB, padded/row gather
at 4 MiB, eager and graph, exactness before timing and eight rotated buffers.
Pilot the arm duration within the budget; if 50 seconds is infeasible, abort
and restore rather than calling a partial block a completed 25-minute test.

A is pieces with the original pacing; B is the opt-in direct-chain policy.
There is no larger-window all-reduce B executable in this revision. A future
ring B requires a proved transition/arena policy and matching emulation first.
Do not substitute the drained advisory ring window. Each block is A-B-B-A;
pair adjacent A/B arms and compare each case, not only an aggregate mean.

Snapshot all eight Sparks' drops, `rx_out_of_buffer`, relay-rule drops, sequence
errors, ACK timeouts and retries before/after each arm. Watch those counters
during execution. A reset or missing counter invalidates the arm. Require exact
results, zero counter increases, every case at least 0.97 times pieces
throughput (inverse latency ratio for small cases), and DCP2 decode within an
A/A noise band declared before B. If A/A is too noisy to assess a 3% change,
the test is inconclusive. Abort on a safety/error gate or at minute 12; restore
TP6 and verify both APIs by minute 25. The retained run's external DCP2 client
made its isolation conclusion inconclusive; it is not an isolation baseline.

### Placement validation for fabric tune

These are separate owner-approved reservations. No hardware action is part of
the offline implementation. Estimates include setup, exactness, candidate
timing, repeated winner/control arms and restoration. They assume a reduced
candidate set and warm build caches; expand the reservation before a full sweep.

| Placement on ring8c | Owner must free or reserve | Estimated reservation |
|---|---|---|
| 0-5 plus 6-7 | Free TP6; DCP2 exclusive fixed workload, no external clients | 25 min reduced ABBA above; 60-75 min for a full placement sweep |
| 0-3 plus 4-7 | Free all eight; two controlled four-rank groups, idle and concurrent arms | 60-90 min |
| 0-7 | Free all eight; no serving co-residents | 45-60 min |
| Pairs | Free tested pair and every relay used by a nonadjacent pair; controlled complements | 20-30 min per distinct pair/overlap class; 60-90 min for adjacent, distance-2 and opposite classes |

At each placement, measure 4 KiB, 28/64/128 KiB, 1/4/16/64 MiB for
all-reduce, flat gather and reduce-scatter; add padded gather controls and
boundary-adjacent sizes. Measure the model winner, pieces, the matching runtime
row and any competing candidate within 10% of predicted time. Record one/two
lanes separately. Use eager and graph, eight rotated buffers and at least
100 timed calls for large cases, with an A/A noise sample and three ABBA
blocks for each proposed promoted setting. First admit only settings executable
under the existing runtime proof. Unsupported advisory candidates are deferred,
not forced into `fabric tune` by an override.

Save the setup document, relay-plan digest, ordered groups, workload descriptions,
image/driver/build receipts, actual windows, per-arm counters, exactness and
per-rank periods. Require the same safety, 0.97 per-case and co-resident noise
gates as above. Report a model miss even when another candidate passes. Do not
promote subset evidence to `path-4` or `cycle-8`: a placement-qualified table must
retain its fabric and co-resident key. Default behavior changes only through a
separate measured promotion after runtime admission exists.
