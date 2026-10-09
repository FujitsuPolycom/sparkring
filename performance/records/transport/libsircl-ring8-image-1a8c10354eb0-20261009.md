# libsircl 0.6.0 of the release image on the cycle of eight Sparks

Lane: **public-functional**. Status: **implemented**. Evidence scope:
**live-validated collectives of the libsircl library that installer image
`1a8c10354eb0` carries, with nccl-tests and libsircl's own checks; no serving
measurement**. Hardware: eight NVIDIA DGX Sparks (GB10) cabled as one ring of
eight with ConnectX-7 RoCE, one rank per Spark on GPU 0.

This record is the gate `gate-ring8-image-1a8c10354eb0-20261009T185859Z` of
the libsircl library in the image of release 2026.10.2
([release record](../../../runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)).
Its report and comparison are beside it
([report.md](libsircl-ring8-image-1a8c10354eb0-20261009/report.md),
[compare.md](libsircl-ring8-image-1a8c10354eb0-20261009/compare.md)); host
identities are position labels.

## Conditions

- **Library under test:** `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0` of
  image `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`,
  built from `spark_transport/libsircl` git tree
  `dbf3607484dd47df4cf6c8238eb5b3272466effb` with nvcc 13.4 and gcc 13.3.0; its
  layer receipt SHA-256 `4da7173b…`, as the release lock records. Every rank
  ran these bytes (8 of 8 ranks checked, one library SHA-256).
- **Routing:** settings for the cycle of eight, logged for every rank at its
  own position: peer routes, chain order 0-7 and forward windows
  (`LIBSIRCL_FORWARD_WINDOWS`), the forms `tools/site_routes.py` prints.
- **Harnesses:** libsircl's bit-exact check (`library_rank.py`) with
  point-to-point channels off and on; nccl-tests v2.21.1, bfloat16, 8 B to
  256 MiB doubling, `-n 20 -w 5`, in six arms; libsircl's timing sweep
  (`perf_rank.py`, bfloat16 all-reduce, 120 size checks per sweep).
- **Arms:** `off-coll` (all-reduce, all-gather, reduce-scatter, broadcast,
  channels off), `ring-coll` (the first three under
  `SIRCL_LARGE_SCHEDULE`, `SIRCL_GATHER_SCHEDULE` and `SIRCL_SCATTER_SCHEDULE`
  set to `ring`), `on-coll` and `on-p2p` (channels on, no session forward
  windows; all-to-all, send-receive and hypercube in `on-p2p`), and two arms
  that must be refused: `off-p2p` (point-to-point without channels) and
  `on-budget` (hypercube with channels under the session's window budget).
- **Time:** 2026-10-09, 19:02 to 19:11 UTC. No NVIDIA NCCL arm ran; the NCCL
  figures below are from the gate of snapshot `a3477af2` on the same ring
  ([libsircl status](../../../spark_transport/libsircl/STATUS.md#hardware-the-path-of-four-at-positions-4-7-and-the-cycle-of-eight-snapshot-a3477af2)).

## Result

Verdicts, every one passed:

- Bit-exact check: exit 0 on 8 of 8 ranks with channels off and with channels
  on.
- nccl-tests `off-coll`, `ring-coll` and `on-coll`: every job exited 0 on every
  rank, `#wrong 0` on every row (104, 78 and 104 rows), every receipt forwarded
  0, healthy, with no refusals. `on-p2p`: every job exited 0, `#wrong 0` on 78
  rows; the 52 in-place rows of all-to-all and send-receive have no in-place
  result.
- `off-p2p` and `on-budget` were refused on every rank (exit status 3), as
  expected.
- Bytes: every communicator receipt names the layer receipt's four kernel
  packs (272 receipts).
- Cycle plan: every eight-rank communicator without a schedule setting took
  it (96 communicators; 16 bit-exact runs with
  `LIBRARY_RANK_EXPECT_CYCLE_PLAN=1`), and from 8 MiB the default all-reduce
  ran within 1.15 times the ring schedules' time (24 size comparisons).

All-reduce, slowest rank's mean eager time in the timing sweep, µs (bus
bandwidth GB/s):

| Size | Default schedule | Ring schedules | Snapshot `a3477af2`, default |
|---|---|---|---|
| 4 KiB | 18.23 | 18.45 | 18.2 |
| 1 MiB | 154.8 (11.85) | 155.1 (11.83) | 154.4 |
| 4 MiB | 540.9 (13.57) | 612.7 (11.98) | 537.0 |
| 8 MiB | 773.8 (18.97) | 774.5 (18.95) | 1,106.8 |
| 64 MiB | 4,909.3 (23.92) | 4,910.4 (23.92) | 9,200.9 |
| 256 MiB | 19,288.4 (24.35) | 19,289.5 (24.35) | 36,794.1 |

nccl-tests, rank 0's out-of-place time per iteration, µs (bus bandwidth
GB/s), `off-coll` arm against NVIDIA NCCL 2.32.3 in the `a3477af2` gate:

| Test | Size | libsircl 0.6.0, default | NVIDIA NCCL 2.32.3 |
|---|---|---|---|
| all-reduce | 4 KiB | 29.12 | 120.3 |
| all-reduce | 8 MiB | 804.60 (18.25) | 782.6 |
| all-reduce | 256 MiB | 19,313.8 (24.32) | 19,940 (23.56) |
| all-gather | 256 MiB | 9,737.60 (24.12) | 10,127 (23.19) |
| reduce-scatter | 256 MiB | 9,813.25 (23.94) | 10,320 (22.76) |
| broadcast | 1 MiB | 554.42 (1.89) | 104.8 |
| broadcast | 256 MiB | 138,648 (1.94) | 11,082 (24.22) |

The full tables of every arm are in [report.md](libsircl-ring8-image-1a8c10354eb0-20261009/report.md).

## Conclusion

On the cycle of eight Sparks, the libsircl of the release image is correct
in every collective and point-to-point line it runs, refuses the two layouts
it must refuse, and runs the cycle plan by default. From 8 MiB its default
all-reduce, all-gather and reduce-scatter run within 1.15 times the ring
schedules' time, which snapshot `a3477af2` reached only with the schedules
set explicitly. At 256 MiB its
all-reduce reaches 24.3 GB/s bus bandwidth against NVIDIA NCCL 2.32.3's
23.6 GB/s on the same ring in the earlier gate. Broadcast stays at 1.9 GB/s
against NCCL's 24.2 GB/s: libsircl has no ring broadcast (unsupported).

## Limitations

- The NCCL comparison spans two gates on the same ring: no NCCL arm ran in
  this one. Both images carry the same NVIDIA NCCL 2.32.3, the release image
  through its parent `aba309e4610c`.
- nccl-tests and the timing sweep time collectives, not serving; no serving
  measurement uses libsircl (`--transport libsircl` is research-only).
- One run per arm.
