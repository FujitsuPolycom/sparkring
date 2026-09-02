# GLM-5.3 burst-prefill scheduler cadence

## Result

The GLM-5.3 TP4/DCP4 general-serving profile uses prefill scheduler interval
two. On eight simultaneous requests with approximately 6K unique input tokens,
interval two improved median aggregate output by 1.6% and reduced median
maximum time to first token by 3.9% relative to interval eight.

| Scheduler interval | Median aggregate tok/s | Median average TTFT | Median maximum TTFT | Median p90 request latency |
|---:|---:|---:|---:|---:|
| 2 | 57.07 | 14.64 s | 20.82 s | 41.75 s |
| 8 | 56.16 | 15.16 s | 21.66 s | 42.63 s |

Every retained sample completed all eight requests without an error. Four
interval-two samples and three interval-eight samples were retained. A warmup
sample that triggered monitored Triton and KDA compilation was excluded before
measurement.

Issue 164 records a separate campaign where interval two improved C8 output by
23% and reduced maximum TTFT from 34 to 24 seconds. The repeated registry-image
comparison found a smaller benefit in the same direction and no measured
tradeoff on the tested burst.

## Scope

The immutable Linux/ARM64 artifact was
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762`.
It ran on four GB10 systems at TP4/DCP4 with 24 GiB of FP8 KV per rank,
an 8,192-token scheduler budget, DFlash2 depth seven, and SparkCache enabled.

The machine-readable sample set is
[`summary.json`](../../receipts/glm53-flash/scheduler-cadence-20260902/summary.json).

Interval eight remains available as an operator override when protecting
established decode streams is more important than admitting a simultaneous
prefill burst.
