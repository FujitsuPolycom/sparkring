# Collective critical-path attribution: measurement design

Status: research-only design. Nothing in this document has been executed.
Every quantity it defines is a quantity to be collected, not a result. The
only numbers here that were observed are the ones explicitly labelled
**measured**, each with the document that states the conditions producing it.
Numbers labelled **derived** are arithmetic over other numbers; numbers
labelled **assumed** are inputs this design accepts without evidence in this
checkout and, where possible, states how to replace with evidence.

Scope: four directly cabled DGX Sparks (GB10, sm_121, aarch64) serving one
tensor-parallel vLLM deployment across ranks 0-3, whose data-plane collectives
run either on the SparkRing RDMA transport under `spark_transport/` or on the
patched ring-safe NCCL fallback. [SIRCL transport](../../docs/SIRCL.md) defines the
transport; [Architecture](../../docs/ARCHITECTURE.md) defines the topology.

## The interval this exists to narrow

One serving request of 5.687 s wall time has been analysed into a bound on
its collective transport's contribution of at least 0.674 s and at most
4.881 s. Those three numbers are **assumed** inputs to this design: the
analysis that produced them is not an artifact in this checkout, so nothing
here reproduces or verifies them. Their construction is what matters, and it
is stated:

- The lower end divides the bytes the collectives must move by the link's
  line rate. It is a wire-time floor.
- The upper end is the longest-rank residency: the total wall time any one
  rank spent inside collective calls.

Both ends are computable from an inventory and a timer. Neither is the
quantity a reader wants, which is how much shorter the request would be if
the collectives cost nothing. Three mechanisms separate residency from that
quantity:

1. **Arrival skew.** A collective call's elapsed time on one rank includes
   the time that rank waited for its slowest peer. That wait is produced by
   load imbalance elsewhere in the step, yet it accrues inside the timed
   region and is charged to the collective.
2. **Overlap.** Communication issued on a stream other than the one carrying
   dependent compute can proceed concurrently with it. Where that happens,
   removing the collective saves less than its residency, and possibly
   nothing.
3. **Per-rank disagreement.** Ranks arrive at a collective at different
   times, so their residencies differ. The maximum over ranks is the largest
   of four numbers, not the shared cost of one operation.

This document specifies the arms that measure each mechanism, the validity
gates that reject a capture, the detection threshold below which no effect
may be claimed, and — in [What remains unmeasured](#what-remains-unmeasured) —
the parts of the interval that survive the whole campaign.

## Quantities and their symbols

For one collective instance `i` on rank `r`:

| Symbol | Name | How it is obtained |
|---|---|---|
| `R(i,r)` | naked residency | timed region containing only the collective |
| `G(i,r)` | gated residency | timed region containing only the collective, entered immediately after a synchronizing gate |
| `B1(i,r)` | first-gate residency | timed region containing the first gate |
| `B2(i,r)` | second-gate residency | timed region containing the second gate |
| `F(i)` | wire-time floor | payload bytes times the algorithm's per-rank wire multiplier, divided by the link rate |
| `E` | exposure coefficient | slope of end-to-end wall time against delay injected into the collective's stream region |

The model this design uses, **derived** and not an identity:

```text
R(i,r) = skew(i,r) + transport(i) + local(i,r)
```

It is exact only for a collective that cannot begin moving bytes until every
rank has entered it. A pipelined ring algorithm begins its first chunk as
soon as one neighbour pair is ready, so part of `transport` overlaps part of
`skew` and the split over-attributes to skew by at most the time to move one
chunk. State that bound with any skew number produced by this design; do not
present the split as exact.

## Instruments this reuses

Four modules in this repository already record the facts the arms need. This
design adds no second instrumentation for anything they cover.

| Module | What it already provides | Use here |
|---|---|---|
| `spark_transport/integrations/vllm/spark_collective_audit.py` | pointer-free per-call signatures (shape, dtype, CUDA residence, contiguity, world size, communicator unique name) and `classify_stock_family`, which maps a seam plus signature to a semantic family such as `vocabulary_all_gather` or `dcp_output_reduce_scatter` | Supplies the family and payload half of an instance key, and the inventory that says which collectives a profile issues at all |
| `spark_transport/integrations/vllm/spark_dcp_collective_audit.py` | source-pinned, fail-closed wrappers at the DCP combine and reduce-scatter seams that record without changing execution | The seam pattern each timed wrapper follows: wrap, record, delegate unchanged |
| `performance/harnesses/q2r_phase_timing/phase_timing.py` | a fixed-capacity CUDA-event collector with a `COLLECTIVE` phase kind, preallocated event pairs, an injected `Event` factory (so it is testable with no GPU), NVTX ranges on the same descriptor, a non-blocking `drain()`, and explicit `dropped`/`errors` counters | The recorder for every timed region in the gated and naked arms. Its contract — never query, never synchronize, never allocate while armed — is exactly what a serving-path timer must satisfy |
| `spark_transport/nccl/probe_dcp4_collectives.py` | a model-free four-rank probe that builds the same TP and DCP communicator scopes vLLM builds, validates values rather than only completion, times with CUDA events, and reports p50 and p99 over fixed iteration counts | The model-free harness the device-layer arms extend |

Two instruments are deliberately not reused.

- `performance/harnesses/vllm/stock_timing.py` times a fixed
  inventory of 169 wrapper calls belonging to one GLM-5.2 MTP4 round and
  invalidates itself on any other shape, family, or stream. It demonstrates
  the event-pool and calibration-event technique this design adopts, but its
  expected-call table is specific to that round and cannot express a general
  inventory.
- `performance/harnesses/q2r_phase_timing/live_installer.py` pins the
  exact source SHA-256 of every seam it patches in one deployed vLLM build.
  That fail-closed discipline is required here, but the pinned hash set
  belongs to the image being measured and must be recollected against
  whichever image a campaign runs, not copied.

## Arm: gated device timing

**Purpose.** Separate transport time from arrival skew, per collective
instance and per rank.

**What is timed.** Three consecutive regions on the same CUDA stream and the
same communicator, each bracketed by its own preallocated event pair:

```text
[e0] gate_1 [e1] gate_2 [e2] collective under test [e3]
B1 = e0->e1     B2 = e1->e2     G = e2->e3
```

The gate is a synchronizing collective of minimal payload on the same
communicator as the collective under test — not a host barrier followed by a
stream synchronization. A host-side barrier plus `torch.cuda.synchronize()`
also removes skew, but it drains the stream and therefore destroys the
overlap the [overlap arm](#arm-overlap-determination) exists to measure; it
is admissible only as a cross-check, never as the primary gate.

**How the gate's own cost is subtracted.** `B1` contains arrival skew plus
the gate's own cost. `B2` is entered by every rank within the exit spread of
`gate_1`, so it carries almost no incoming skew and its residency is the
gate's own cost. Therefore:

```text
skew(i,r) >= median(B1(i,r)) - median(B2(i,r))
```

This is a lower bound, not an estimate: `B2` still absorbs whatever exit skew
`gate_1` produced, so it overstates the gate's cost and understates the
difference. `G` is measured after `gate_2` and is unaffected by that
overstatement, which is why `G` — not `B1 - B2` — carries the transport
claim.

**Why two gates rather than one gate and a separate naked run.** A single
gate would force the skew estimate to be a difference across two collections,
which drift separates. Both gates sit inside one timed sequence, so the
subtraction happens within one iteration on one rank.

**Validity gates.** A gated capture is rejected unless all hold:

- the gate's own cost (`median(B2)`, maximised over ranks) is at most 10% of
  the gated collective it precedes. A gate comparable to the collective has
  changed what it was placed there to isolate. `performance/harnesses/bench/collective_attribution.py`
  enforces this constant.
- per-rank `median(G)` values agree across ranks within 25% peak-to-peak of
  their median. Ranks that entered together should leave together;
  disagreement means the gate did not equalise arrival, so `G` is not a
  transport time.
- every rank recorded the same number of samples for `B1`, `B2`, and `G`, and
  the recorder reports zero capacity drops, zero unregistered-descriptor
  drops, and zero record or drain errors.
- the stream identity recorded at each timed region is constant for the whole
  epoch. A stream change mid-epoch makes two event pairs incomparable.

## Arm: naked timing

**Purpose.** Reproduce the residency ceiling on the same inventory, in the
same units, so the amount gating removes is a difference between two
measurements rather than between a measurement and a quoted number.

**What is timed.** One region per collective instance, containing only the
collective, with nothing placed before it. Identical descriptors, identical
inventory, identical iteration counts to the gated arm, collected in a
separate session.

`R - G` per rank is a second, independent estimate of arrival skew. It is
contaminated by session-to-session drift where `B1 - B2` is not, so the two
estimates are reported side by side and their disagreement is evidence about
drift rather than something to average away.

## Arm: measured link rate

**Purpose.** Replace an assumed line rate in the wire-time floor with a rate
this fabric was observed to reach.

The floor is `payload_bytes * wire_multiplier * 8 / rate`. Two of its three
inputs are **assumed** by this design:

- The multiplier depends on the algorithm. A ring all-reduce over four ranks
  moves `2(N-1)/N = 1.5` payloads per rank. The two-perfect-matching
  decomposition [SIRCL transport](../../docs/SIRCL.md) specifies for the four-Spark
  cycle sends a full payload in each of two rounds, so its multiplier is 2.0.
  A floor quoted without naming the algorithm is not a floor.
  `collective_attribution.py` refuses a capture whose instances do not state
  both the multiplier and the basis for it.
- The rate. [Dual-port striping and graph-kernel probe record,
  2026-08-18](../records/transport/dual-port-striping-20260818.md) states, as an open item,
  that no measurement in that record establishes large-message bandwidth on a
  single link and that every wire-time derivation from the 200 Gb/s nameplate
  rate presently assumes it.

**What to collect.** Single-link large-message achieved bandwidth: one
directly cabled pair, payloads spanning the serving inventory's largest
messages, `port_xmit_data` and `port_rcv_data` bracketing every leg on both
ports, values validated rather than only completion. The result raises the
floor — an achieved rate is below nameplate — which narrows the interval from
below without any change to the ceiling. A capture document records
`rate_basis` as `nameplate` or `measured`, and the report states which it
used.

## Arm: overlap determination

**Purpose.** Decide whether a collective can overlap compute in this
deployment at all, because that decision changes the interpretation of every
other number here.

Three determinations, cheapest first.

1. **Stream census (decisive for the negative case).** Record, for every
   timed collective, the CUDA stream it is enqueued on, and for the compute
   that consumes its output, the stream that compute is enqueued on. If they
   are the same stream, that collective cannot overlap that compute, by
   stream ordering, and its residency is fully exposed to the step's device
   timeline. `performance/harnesses/vllm/stock_timing.py` already records
   `int(stream.cuda_stream)` per call and invalidates on change; the same
   read supplies this census. A same-stream result requires no further
   overlap work for that collective.
2. **Envelope reconciliation.** The phase collector registers a
   `step_envelope` descriptor around the model-execution call and nested
   `collective` descriptors inside it. Nested durations are attribution and
   are not disjoint, so they must not be summed as if they were. Their sum
   exceeding the envelope is positive evidence of concurrency; their sum
   falling below it bounds how much of the envelope is unattributed, which is
   not the same as proving serialisation.
3. **Injected-delay sweep.** See the next section. A slope near zero is the
   only positive evidence that a region is genuinely overlapped, because it
   observes wall time not moving when the region is made longer.

**How overlap changes interpretation.** If the exposure coefficient for a
collective family is `E`, then that family's contribution to wall time is
`E` times its gated transport time, not its gated transport time. Report both
and never silently substitute one for the other. `E = 1` means the reported
transport time is the contribution; `E = 0` means the family costs the
request nothing however long it takes, and its whole residency belongs to
some other cause.

## Arm: counterfactual estimation

The quantity of interest is a counterfactual — the request's duration in a
world where the collective is free — and no timer reads it. Three
approximations attack it from different directions, and each establishes a
different thing.

### Injected-delay sweep

**What it does.** Insert a calibrated device-side delay of `D` microseconds
into the collective's stream region, sweeping `D` over several nonnegative
values, and record end-to-end wall time at each. Fit wall time against `D`.
The slope is the fraction of a marginal collective microsecond that reaches
wall time.

**What it establishes.** The exposure of the region at its current operating
point, in the direction of making the collective slower. It is the only arm
in this design that observes wall time responding to a change in the
collective, so it is the only one that measures exposure rather than
assuming it.

**What it does not establish.** The saving from removing the collective.
Reading the slope down to zero collective cost extrapolates outside every
delay observed and assumes exposure stays linear there — a plausible
assumption near the operating point and a false one if some other rank or
stream becomes the binding constraint first. `collective_attribution.py`
prints this qualification with any exposed-seconds figure it reports, and
refuses to report a slope at all when the swept delay span is smaller than
the layer's detection threshold applied to the undelayed wall time.

### Comparison against the patched ring-safe NCCL fallback

**What it does.** Runs identical shapes, identical inventory, and identical
prompts on the SparkRing transport and on the patched ring-safe NCCL
fallback, alternating arms.

**What it establishes.** The difference between two transports at those
shapes. That is a real and decision-relevant number: it bounds what changing
transport recovers.

**What it does not establish.** The cost of communication. Both arms move the
same bytes over the same cables; their difference omits everything both share.
The already-collected device-timed comparison in
[Dual-port striping and graph-kernel probe record,
2026-08-18](../records/transport/dual-port-striping-20260818.md) illustrates the size of the
gap between the two questions: under isolated single-replay device timing the
patched fallback showed an approximately 800 µs floor at every payload it
measured while the transport's best configurations measured 174 µs at
491,520 bytes and 1,223 µs at 6,291,456 bytes (**measured**; conditions,
including that the serving stack was torn down first and that the timing is
explicitly not representative of pipelined serving, are stated in that
record).

### Null-collective substitution

**What it does.** Replaces the collective with an operation that preserves
control flow, stream ordering, tensor shapes, and output allocation but moves
no bytes between ranks.

Three constructions, with different confounds:

| Construction | Preserves numerics | Confound |
|---|---|---|
| Skip the exchange and use the rank-local value | no | Divergent numerics change the sampled token stream, which changes the workload. Usable for timing structure over a fixed token count; never for an end-to-end latency comparison on the same prompt. |
| Exchange with self through a device-local copy | no | Same numerics problem, plus the copy's own cost remains in the timed region. |
| Replay a cached correct output recorded from a prior identical run | yes, for a deterministic replay | Requires fixed prompt, fixed seed, temperature zero, and `ignore_eos`, and is valid only for a replay of exactly that request. The device copy of the cached output remains in the timed region. |

**What it establishes.** An upper bound on what removing communication could
save, and only after the null's own cost — kernel launch, the copy, the
stream-ordering dependency that still exists — has been measured with the
same instrument and subtracted. An unsubtracted null substitution
systematically understates the saving.

**What it does not establish.** The floor. Removing the bytes does not remove
the requirement to move them; the wire-time floor is unaffected by any null
arm and remains the interval's lower end.

The numerics confound is not a detail. A campaign that compares end-to-end
latency between an arm that produces the correct token stream and an arm that
does not is comparing two workloads, and every downstream number is void.
Only the cached-output construction avoids it.

## Instance identity and granularity

An aggregate over all collectives is not actionable. Every measurement in
this design is keyed per instance and per rank.

An instance key is:

```text
family | communicator unique name | world size | payload shape |
element width | step ordinal | call ordinal within step
```

`classify_stock_family` in `spark_collective_audit.py` supplies the family
from a seam and a pointer-free signature; the signature supplies shape,
element width, world size, and communicator name. The two ordinals come from
the phase collector's reservation sequence, which preserves call order within
an epoch.

**Cross-rank joining requires the ordinals to agree across ranks.** If rank 2
issues a collective rank 3 does not, the sequences desynchronise and every
subsequent join is wrong. Cross-rank ordinal agreement is therefore a
validity gate on the capture, not a convenience: reject the capture rather
than joining by nearest match.

Per instance the campaign reports, per rank, the median and interquartile
range of `G`, `R`, `B1`, `B2`, and the derived skew, with sample counts; and
per instance the wire-time floor, the slowest-rank transport median, and the
cross-rank spread. `occurrences` records how many times an instance's
signature occurs in the attributed request, so per-request totals are a sum
over the inventory rather than an extrapolation from one call.

## Statistical treatment

### Reported form

No quantity is reported as a single number. Every timing is a median with its
interquartile range, its minimum, its maximum, and its sample count. Medians
rather than means, because a serving path's occasional long sample is a real
event that should not be allowed to move the central estimate.

### Repetition counts

Within a session, at the device layer: 200 timed iterations per instance per
arm after at least 20 warmup iterations, matching the convention already used
by `probe_dcp4_collectives.py`. The first call to a shape pays allocation and
algorithm selection and is not a steady state.

Across sessions: at least three independent sessions per arm at the device
layer, because cross-session spread exceeds within-session spread.
[Dual-port striping and graph-kernel probe record,
2026-08-18](../records/transport/dual-port-striping-20260818.md) records approximately 2%
spread in p50 across three same-configuration 200-iteration runs, and
follower-rank medians agreeing with rank 0 within 2.3% (**measured**;
conditions in that record).

At the end-to-end layer: alternating arms, at least 12 paired windows. The
count is **derived** from the repository's own observed serving dispersion,
below.

### Detection threshold

This repository has published serving comparisons that its own analysis later
declined to treat as effects. The threshold below is set so that this design
cannot repeat that.

Two observations fix the scale (**measured**; conditions in [EXL3 performance
campaign, 2026-08-02](EXL3_AB_CAMPAIGN_20260802.md), which states the
harness version, salts, artifact hashes, and window settings):

- Same-configuration C8 decode aggregates over three repeats of one arm gave
  ranges of 16.20, 24.28, and 16.32 tok/s against medians of 73.04, 60.48,
  and 61.04 tok/s — 22%, 40%, and 27% of median.
- One arm measured twice within a session, opening and closing, drifted
  -5.34% at C1, +7.45% at C2, +0.59% at C4, and -0.77% at C8.

The three relative ranges are 22.2%, 26.7%, and 40.1% of median; taking the
middle one, and using the expectation of about 1.69σ for the range of three
normal samples, gives σ ≈ 15.8% of median (**derived**, and a weak estimate
from three samples). Requiring a two-sided 95% interval half-width of 10% of
median gives `n >= (1.96 * 15.8 / 10)^2 = 9.6`, hence at least 10 paired
windows; this design rounds to 12.

**This design commits to two thresholds, and to refusing any claim below
them:**

| Layer | Threshold | Basis |
|---|---|---|
| Device-timed collectives | 5% of the compared median | Above both the ~2% cross-session p50 spread and the 2.3% cross-rank median agreement observed for isolated device-timed collective probes |
| End-to-end serving quantities | 10% of the compared median | Policy floor for the C1 attribution design; a campaign must measure its own same-configuration dispersion and raise this threshold when that dispersion is larger |

A difference below its layer's threshold is reported as `indeterminate` — a
result stating that this instrument at this repetition count cannot separate
the difference from run-to-run variation. It is never reported as a small
effect. `collective_attribution.py` fixes both floors in code and refuses a
`--detect-percent` that would lower either.

**Concurrency restriction.** No attribution claim is made at C8 or under any
competing traffic. The dispersion above exceeds every effect this campaign
could plausibly resolve, and the same record shows the dispersion is driven
by per-lane imbalance rather than by anything the transport controls. Claims
are restricted to C1, which is also the concurrency at which the motivating
request's interval was constructed.

**Threshold calibration before any comparison.** The first collection of any
campaign is a null comparison: the identical configuration collected twice
under the alternating schedule, analysed by the same tool as if it were an
A/B. Its paired-difference distribution is the campaign's own dispersion
estimate, and `collective_attribution.py --plan` converts that estimate into
the repetition count the campaign then owes. If the null comparison returns a
dispersion tighter than the figures above, the thresholds above still hold:
a short null comparison can miss structure that appears over longer windows,
and the repository has observed exactly that.

## Validity gates

A capture is rejected, not adjusted, when any of these fail:

1. Any record error, drain error, capacity drop, or unregistered-descriptor
   drop reported by the phase collector.
2. Cross-rank disagreement in the per-instance call sequence.
3. A gate cost above 10% of the gated collective it precedes.
4. Cross-rank spread of gated medians above 25% of their median.
5. A stream identity change within one epoch.
6. Any difference between the gated and naked arms other than the gate: a
   different inventory, a different layer, a different link rate, or the same
   session label on both arms.
7. Value mismatch in any collective the model-free probe validates. Timing a
   collective that computed the wrong answer measures nothing.

Gates 3, 4, and 6 are enforced by `collective_attribution.py`; gates 1, 2, 5,
and 7 belong to the recorder and the probe.

## Analysis tool

`performance/harnesses/bench/collective_attribution.py` (safety class OFFLINE) reads a
gated capture, a naked capture, and an optional injected-delay sweep, and
emits the narrowed interval with the qualifications that bound it. It takes
no measurement of its own; every number it prints is arithmetic over numbers
some other instrument recorded.

```bash
# The document shapes the probe must produce.
python performance/harnesses/bench/collective_attribution.py --print-schema

# Narrow the interval from two arms, optionally scaled by measured exposure.
python performance/harnesses/bench/collective_attribution.py \
    --gated gated.json --naked naked.json --exposure sweep.json \
    --json report.json

# Size a campaign from a dispersion the null comparison measured.
python performance/harnesses/bench/collective_attribution.py \
    --plan --dispersion-percent 15.8 --detect-percent 10
```

Capture documents declare schema `sparkring-collective-attribution/v1`; a
sweep declares `sparkring-collective-exposure/v1`; the report declares
`sparkring-collective-attribution-report/v1`.

| Exit code | Meaning |
|---:|---|
| 0 | The arms were comparable and a report was emitted |
| 2 | Two valid documents that do not form a pair, or a refused threshold |
| 3 | An input file is missing or unreadable |
| 4 | A document is not valid JSON or violates its declared schema |

The tool refuses several things by design: an instance without a stated wire
multiplier and a stated basis for it, a link rate that declares neither
`nameplate` nor `measured`, a naked arm carrying gate timings, a repeated
instance key, both arms from one session, and any `--detect-percent` below
its layer's floor.

## What this closes

- **The residency ceiling.** The gated arm replaces the sum of per-rank
  residencies with a sum of per-rank gated transport medians, and reports the
  difference explicitly as skew that belongs to some other cause. Whether
  that difference is large is exactly what the campaign finds out; it cannot
  be predicted from this document.
- **The floor's rate assumption.** A measured single-link large-message rate
  replaces the nameplate rate, raising the floor.
- **Whether overlap exists.** The stream census settles the same-stream case
  outright, and the delay sweep settles the rest with positive evidence.
- **Which collectives matter.** Per-instance attribution over the audited
  inventory says which family and which payload size carries the time, and
  therefore where any change would have to act.
- **Whether a reported difference is an effect.** The thresholds refuse the
  claim rather than reporting a small number.

## What remains unmeasured

Everything below survives a complete, successful campaign. Any report built
on this design must restate it.

- **The true counterfactual.** The delay sweep measures exposure only for
  nonnegative delays; the null arms approach from the other side but never
  remove stream ordering, kernel launch, or the buffer's existence. The
  residual between them is bounded by the null's own measured cost, but the
  assumption that exposure stays linear between the two anchors is an
  assumption and stays one.
- **The critical path itself.** CUDA events have no common time base across
  devices, so this design produces per-rank attributions and a bound formed
  by summing per-instance slowest-rank medians. That bounds the transport's
  contribution; it does not identify a path through the four ranks'
  dependency graph. Identifying one requires a calibrated cross-rank clock
  offset, which nothing here implements.
- **Why the skew exists.** The gated arm measures how much a rank waited. It
  says nothing about what made its peer late. Attributing skew to a cause —
  routing imbalance, per-rank expert counts, scheduling — is separate work.
- **The pipelining correction.** The skew/transport split over-attributes to
  skew by at most one chunk's transfer time for a pipelined algorithm. That
  bound is stated, not measured; measuring it requires per-chunk visibility
  the transport publishes no per-chunk timing seam.
- **Any concurrent or C8 behaviour.** Excluded by the dispersion restriction
  above, not by absence of interest.
- **Whether the motivating 5.687 s request is representative.** One request
  is one sample. This design measures a workload's collective inventory under
  controlled conditions. Re-deriving that specific request's interval from
  workload statistics requires its own inventory, which the audit can supply,
  and a deterministic replay of it, which requires fixed prompt, fixed seed,
  temperature zero, and `ignore_eos`.
- **Tail behaviour.** Three sessions support a median. They do not support a
  p95 or p99 claim, and this design makes none.
- **Anything about the transport's absolute quality.** Every number here is
  an attribution within one deployment on one four-Spark appliance. It is not
  a benchmark of either transport against any other system.
