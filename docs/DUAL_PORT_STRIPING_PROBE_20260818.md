# Dual-port striping and graph-kernel probe record, 2026-08-18

Status: research-only. Model-free transport measurements on the four
directly cabled DGX Sparks; no serving claim, no promotion, no NCCL
comparison established. The one defensible conclusion is stated first;
everything else here is conditions, data, and limitations.

## Conclusion

The graph all-reduce's `fused` kernel is pathological at large
payloads on the sequential wire schedule: 7,582-7,700 µs p50 for a
6 MiB collective, an order of magnitude above the alternatives. Two
mechanisms independently eliminate the pathology and converge:

- `split_64k` or `tiered_64k` graph kernels on the sequential
  schedule: 1,223 µs p50 at 6 MiB.
- The `dual_port_striped` wire schedule (which requires the `fused`
  kernel): 1,229-1,235 µs p50 at 6 MiB, with a tighter tail
  (p95 1,259 µs versus 1,382-1,559 µs for the split kernels).

At 6 MiB, striping therefore offers no median advantage over the best
sequential kernel; its demonstrated benefit is tail variance. Whether
striping helps at 0.5 MiB against the best sequential kernel is
unmeasured (the control leg is missing; see limitations).

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

The serving stack was torn down before each probe session. Link
idleness was not instrumented (no port byte counters recorded).

## Measurements

Fixed-Q payloads at the default width: Q40 = 491,520 bytes; Q512 =
6,291,456 bytes. Values are rank-0 p50/p95 in µs.

| Payload | Schedule | Kernel | p50 | p95 |
|---|---|---|---:|---:|
| Q40 | sequential | fused | 618.5 | 700.4 |
| Q40 | dual_port_striped | fused | 190.5 | 194.6 |
| Q512 | sequential | fused | 7,582-7,700 | 8,026-8,255 |
| Q512 | sequential | split_64k | 1,223.7 | 1,559.2 |
| Q512 | sequential | tiered_64k | 1,223.4 | 1,382.1 |
| Q512 | dual_port_striped | fused | 1,228.5-1,234.6 | 1,259.2-1,265.4 |

The Q512 sequential-fused range spans three same-configuration runs
(~2% spread). All bandwidth derivations from these numbers are
algorithm bandwidth (payload divided by time), not wire utilization.

Probe-enforced constraints (from the binary's own validation):
`dual_port_striped` requires graph-only execution,
`two_slot_deferred_ack`, the `fused` kernel, and fixed Q40 or Q512.
The striped-with-split-kernel combination therefore does not exist to
measure, and the schedule comparison is only available on the
artifact-prone fused kernel: the kernel-pathology conclusion is
consistent with the data but is an inference across two changed
variables, not a controlled isolation.

## Limitations

- Missing control: Q40 sequential with `split_64k`/`tiered_64k` was
  not run, so no striping-versus-best-sequential ratio exists at
  0.5 MiB.
- A same-day NCCL `all_reduce` timing exists but is not comparable
  and is not quoted here: it ran eager (not graph-captured), was
  host-timed including launch and synchronization overhead, used
  both RoCE ports (`NCCL_IB_HCA` lists both), used 524,288 bytes
  rather than 491,520 at the small point, and its container logs were
  deleted before the loaded `libnccl` identity could be verified.
- Isolated single-replay timing does not represent serving, where
  collectives pipeline with compute; neither the absolute values nor
  the schedule ratios are established under pipelined conditions.
- Single-session statistics: one 200-iteration session per
  configuration (three for sequential-fused Q512); p95 values are
  weakly supported; raw samples were not retained.
- Any projection to serving payloads is unvalidated: no decode-step
  shape trace exists, and speculative-verification batch shapes are
  assumptions, not measurements.

## Follow-up measurements this record motivates

1. Q40 sequential `split_64k`/`tiered_64k` (the missing control).
2. A graph-captured, device-timed NCCL control at matched payloads,
   single-port and dual-rail variants, with `NCCL_DEBUG=VERSION`
   output and logs retained.
3. Port byte counters (`port_xmit_data`) bracketing every leg.
4. A decode-step collective shape trace from a serving run, before
   any tokens-per-second projection.
