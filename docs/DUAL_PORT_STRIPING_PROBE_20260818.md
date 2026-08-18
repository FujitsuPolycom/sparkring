# Dual-port striping and graph-kernel probe record, 2026-08-18

Status: research-only. Model-free transport measurements on the four
directly cabled DGX Sparks; no serving claim, no promotion. The main
conclusions are stated first; everything else here is conditions,
data, and limitations. A same-day follow-up session completed the
originally missing control legs, the counter instrumentation, and a
comparable NCCL control; those results are folded in below.

## Conclusion

The graph all-reduce's `fused` kernel is pathological at large
payloads on the sequential wire schedule: 7,582-7,700 µs p50 for a
6 MiB collective, an order of magnitude above the alternatives. Two
mechanisms independently eliminate the pathology:

- `split_64k` or `tiered_64k` graph kernels on the sequential
  schedule: 174 µs p50 at 0.5 MiB, 1,223 µs p50 at 6 MiB.
- The `dual_port_striped` wire schedule (which requires the `fused`
  kernel): 191 µs p50 at 0.5 MiB, 1,229-1,235 µs p50 at 6 MiB.

With the control legs complete, the split kernels are the better fix
at both payloads: at 0.5 MiB they beat striping by about 9% on both
median and tail (p95 182-184 µs versus 197 µs), and at 6 MiB they
match striping's median (0.99x) while striping keeps a modestly
tighter tail (p95 1,259 µs versus 1,382-1,559 µs). Striping never
beats the best sequential kernel; its earlier apparent wins were
measured against the pathological fused baseline.

A graph-captured, device-timed NCCL control on the same ring (see
below) shows an approximately 800 µs floor for NCCL all-reduce at
every payload measured, versus 174-1,223 µs for the transport's best
configurations at the matched payloads.

## Instrument and conditions

`spark_tp4_graph_q1_probe` (the binary staged as
`/var/tmp/spark_tp4_graph_q1_probe_dual_port_v1` on each rank,
2026-08-12 build lineage), four synchronized ranks over the direct
links, RDMA devices `rocep1s0f0/f1` with odd ranks inverted, GID 3,
control ports 9470/9471. Common configuration for every quoted
measurement: `--allreduce-protocol two_slot_deferred_ack`,
`--graph-submit-cpu 14 --graph-progress-cpu 15`, 20 warmup, 200 timed
iterations, `--timing-mode isolated`
(`timing_scope=device_output_ready_single_replay`: device-side
output-ready time per single graph replay, excluding graph submit,
about 2 µs per call, and all host synchronization). Every quoted leg
reported `mismatched_elements=0 correct=true passed=true` on all four
ranks, and follower-rank medians agree with rank 0 within 2.3%.

The serving stack was torn down before each probe session. The
original session did not record port byte counters; the follow-up
session bracketed every leg with rank-0
`/sys/class/infiniband/rocep1s0f*/ports/1/counters/port_xmit_data`
snapshots. Every follow-up transport leg moved an essentially
identical byte total on each of the two ports (about 110 MB per port
per 220-iteration leg) regardless of wire schedule, consistent with
equal-per-port transmission and no competing traffic.

## Measurements

Fixed-Q payloads at the default width: Q40 = 491,520 bytes; Q512 =
6,291,456 bytes. Values are rank-0 p50/p95 in µs.

| Payload | Schedule | Kernel | p50 | p95 |
|---|---|---|---:|---:|
| Q40 | sequential | fused | 618.5-622.8 | 688.1-700.4 |
| Q40 | sequential | split_64k | 174.0 | 184.3 |
| Q40 | sequential | tiered_64k | 174.1 | 182.4 |
| Q40 | dual_port_striped | fused | 190.5-190.7 | 194.6-196.6 |
| Q512 | sequential | fused | 7,582-7,700 | 8,026-8,255 |
| Q512 | sequential | split_64k | 1,223.7 | 1,559.2 |
| Q512 | sequential | tiered_64k | 1,223.4 | 1,382.1 |
| Q512 | dual_port_striped | fused | 1,228.5-1,234.6 | 1,259.2-1,265.4 |

The Q512 sequential-fused range spans three same-configuration runs
(~2% spread); the Q40 fused and striped ranges span the original and
follow-up sessions. All bandwidth derivations from these numbers are
algorithm bandwidth (payload divided by time), not wire utilization.

## NCCL control

The follow-up session ran a comparable NCCL control: four ranks in
throwaway containers using the deployment image and per-rank serving
environment, `torch.distributed.all_reduce` captured into a CUDA
graph per payload, individual replays timed with device events
(synchronized single-replay samples, 20 warmup, 200 iterations),
result correctness asserted. The loaded runtime identified itself as
NCCL 2.30.7+cuda13.0 (`NCCL_DEBUG=VERSION` banner; PyTorch's
compiled-against tuple reports 2.29.7); logs from all four ranks were
preserved before container removal.

| Bytes | p50 | p95 |
|---:|---:|---:|
| 12,288 | 796.2 | 1,229.8 |
| 65,536 | 820.5 | 1,511.1 |
| 491,520 | 907.6 | 1,301.4 |
| 2,097,152 | 1,140.1 | 1,583.7 |
| 2,359,296 | 1,149.9 | 1,678.1 |
| 6,291,456 | 1,600.8 | 2,156.5 |

Under this scope NCCL shows an approximately 800 µs floor at every
payload. At the matched points the transport's best configurations
measure 174 µs (491,520 bytes) and 1,223 µs (6,291,456 bytes) under
the same isolated single-replay device timing. Two topology
observations from the same control: rank 0's byte counters show NCCL
moved about 4 GB on `rocep1s0f0` and only about 20 MB on
`rocep1s0f1` (highly asymmetric rail use, unlike the transport's
equal-per-port totals), and a single-port variant
(`NCCL_IB_HCA=rocep1s0f0`) fails deterministically (two runs,
identical `ibv_modify_qp` timeout to the far-rail GID) because each
rank reaches one ring neighbor only through its second port: on this
switchless topology both rails are required for connectivity, so no
single-port NCCL configuration exists to measure.

Probe-enforced constraints (from the binary's own validation):
`dual_port_striped` requires graph-only execution,
`two_slot_deferred_ack`, the `fused` kernel, and fixed Q40 or Q512.
The striped-with-split-kernel combination therefore does not exist to
measure, and the schedule comparison is only available on the
artifact-prone fused kernel: the kernel-pathology conclusion is
consistent with the data but is an inference across two changed
variables, not a controlled isolation.

## Limitations

- Isolated single-replay timing does not represent serving, where
  collectives pipeline with compute; neither the absolute values nor
  the schedule ratios are established under pipelined conditions.
  This applies equally to the NCCL control and its comparison.
- Limited-session statistics: one or two 200-iteration sessions per
  configuration (three for sequential-fused Q512); p95 values are
  weakly supported; raw samples were not retained.
- Any projection to serving payloads is unvalidated: no decode-step
  shape trace exists, and speculative-verification batch shapes are
  assumptions, not measurements.
- The NCCL control used the stock configuration reachable through the
  deployment environment; no NCCL tuning (algorithm, protocol,
  channel, or chunk-size overrides) was attempted, so the control
  bounds the deployment's NCCL path, not NCCL's best case.
- An earlier same-day eager host-timed NCCL run is superseded by the
  graph-captured control above and is not quoted.

## Decode-step shape trace, DeepSeek-V4-Flash-0731 serving profile

A same-day trace armed the collective audit
(`SPARK_TP4_GRAPH_STATUS_PATH`) on the DeepSeek serving profile
(width 4096, 32 maximum sequences, DSpark speculation with seven
draft tokens, CUDA graphs enabled) and collected the stock-collective
signature inventory from all four ranks after graph capture plus a
small request burst. One instrumentation fact this required: the
status-reporter thread previously started only when the width-6144
graph session was prepared, so on any other profile the audit
recorded signatures in memory that were never written; the trace ran
with a backend patch that starts the reporter whenever the audit is
armed. Findings, capture-inventory scope only (signature counts do
not measure how often a captured graph bucket is replayed):

- Every `[Q, 4096]` collective in the profile currently takes the
  stock NCCL path (`ineligible_signature` / capture-phase
  `graph_transport_disabled`); the transport carries none of them.
- The captured graph buckets are Q = 48 to 512 in steps of 8 (eight
  rows per sequence: seven draft tokens plus one target), so the
  32-sequence verification batch is Q256 = exactly 2 MiB. Q288 is
  simply the 36-sequence bucket, not an extra per-sequence row.
- About 87 width-4096 all-reduces occur per forward pass
  (approximately two per layer), the drafter adds width-256
  collectives of at most 16 KiB, and eager prefill chunks reach
  [2048, 4096] = 16 MiB, above the 512-row eager admission bound.
- A single maximum-capacity [Q <= 512, 4096] graph session would
  cover every traced verification shape.

## Follow-up measurements this record motivates

1. Graph-bucket replay frequency under sustained traffic (the
   inventory above shows which buckets exist, not how often each is
   selected).
2. Repeated sessions for tail statistics if any default-configuration
   decision comes to rest on p95 differences.
