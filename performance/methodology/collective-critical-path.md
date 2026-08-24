# GLM TP4/DCP4 collective-attribution design

Status: research-only. The offline report analyzer is implemented and tested,
but this repository contains no live gated/naked capture, delay sweep, or
attribution report. This document is a design for one future collection, not
performance evidence.

## Scope

This design applies only to the GLM-5.2 EXL3 3.5-bpw four-Spark TP4/DCP4
profile. It compares its SIRCL transport with the patched NCCL fallback on the
same directly cabled four-rank cycle. It does not describe the Qwen profiles,
where SIRCL does not support the width-5120 all-reduce, or the DeepSeek
profiles.

The aim is to bound how much collective transport contributes to one
controlled decode workload. CUDA events provide per-rank intervals, not a
clock-synchronised distributed critical path.

## What to collect

For every collective instance `i` and rank `r`, record:

| Value | Meaning |
|---|---|
| `R(i,r)` | Naked residency: only the collective is timed. |
| `G(i,r)` | Gated residency: the collective is timed immediately after two same-communicator synchronising gates. |
| `B1(i,r)`, `B2(i,r)` | Residencies of the first and second gates. Their difference is a lower bound on arrival skew. |
| `F(i)` | Wire-time floor: payload bytes × stated per-rank wire multiplier / measured link rate. |
| `E` | Exposure: slope of request wall time against calibrated delay added to the collective stream region. |

Use the existing collective audit to identify the family, communicator, shape,
element width, and call order. Include the step and call ordinals in the
instance key, and reject a capture if the ordinal sequence differs across
ranks.

## Measurement arms

1. **Gated device timing.** On the collective's stream and communicator,
   time `gate_1`, `gate_2`, then the collective with preallocated CUDA events.
   `G` is the transport-residency estimate; `median(B1) - median(B2)` is a
   conservative skew lower bound. Do not use a host barrier plus device
   synchronisation as the primary gate: it changes stream-overlap behaviour.

2. **Naked device timing.** Time the same instance inventory with no preceding
   gate, in a distinct session. `R - G` is a drift-sensitive second view of
   arrival skew, not a value to average with the gate estimate.

3. **Link-rate measurement.** Measure single-link large-message achieved
   bandwidth across the relevant payload range, validate collective values,
   and bracket each leg with port counters. Name the collective algorithm and
   its wire multiplier. A nameplate rate is an assumption, not a measurement.

4. **Overlap and exposure.** First record the collective and consuming-compute
   streams. A same-stream dependency cannot overlap. For other cases, run a
   multi-point device-delay sweep and regress request wall time on added delay.
   `E` measures marginal exposure at the operating point; applying it down to
   zero transport cost is an extrapolation.

## Capture gates

Reject, rather than adjust, a capture when any of these conditions fails:

- a recorder reports an error, capacity drop, or unregistered descriptor;
- ranks disagree on the collective sequence;
- the second gate costs more than 10% of the collective it precedes;
- gated medians differ by more than 25% peak-to-peak across ranks;
- a timed descriptor changes CUDA stream within the epoch;
- the paired arms differ in inventory, layer, link rate, or session; or
- the value-validating probe reports a mismatch.

Collect enough independent sessions to measure the workload's own dispersion
before choosing a detection threshold. Do not reuse thresholds from removed
or unmatched benchmark records. No claim about concurrent serving, C8, tails,
or another model profile follows from a C1 collection.

## Reporting and limits

Report per instance and rank: median, interquartile range, min/max, sample
count, `R`, `G`, `B1`, `B2`, wire-time floor, and cross-rank spread. Sum
slowest-rank instance medians only as a bound, not as a reconstructed critical
path. State the link-rate basis and whether exposure was measured.

The existing offline analyzer,
`performance/harnesses/bench/collective_attribution.py`, accepts gated and
naked capture documents and an optional exposure sweep. Its schemas are
`sparkring-collective-attribution/v1`,
`sparkring-collective-exposure/v1`, and
`sparkring-collective-attribution-report/v1`.

Even a successful campaign does not identify a cross-rank dependency path,
explain why a rank arrived late, establish tail behaviour, or prove the exact
counterfactual saving from removing communication. A null substitution also
retains launch, copy, allocation, and stream-ordering costs; it is not a free
collective.
