# libsircl 0.6.0 of image 1a8c10354eb0 on the cycle of eight Sparks

Lane: **public-functional**. Status: **implemented**. Evidence scope:
**live-validated collectives of the libsircl library that installer image
`1a8c10354eb0` carries, with nccl-tests and libsircl's own checks; no serving
measurement**. This record is gate
`gate-ring8-image-1a8c10354eb0-20261009T185859Z`, evidence of release 2026.10.2
([release record](../../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)). Release 2026.10.2's image, `d52737a109e0`, carries this library with one
change, the current-device fix of source tree `030419b8`.

## Conditions

- **Hardware:** eight NVIDIA DGX Sparks (GB10) cabled as one ring of eight
  with ConnectX-7 RoCE, one rank per Spark on GPU 0; 2026-10-09, 19:02 to
  19:11 UTC.
- **Library:** `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0` of image
  `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`,
  built from `spark_transport/libsircl` tree
  `dbf3607484dd47df4cf6c8238eb5b3272466effb`, as the release lock records.
- **Routing:** the route planner's settings for the cycle of eight on every
  rank: peer routes, chain order 0-7 and forward windows.
- **Harnesses:** libsircl's bit-exact check (`library_rank.py`) with
  point-to-point channels off and on; nccl-tests v2.21.1, bfloat16, 8 B to
  256 MiB doubling, `-n 20 -w 5`; libsircl's all-reduce timing sweep
  (`perf_rank.py`).
- **Arms:** `off-coll` (all-reduce, all-gather, reduce-scatter and broadcast,
  channels off), `ring-coll` (the first three with the ring schedules set),
  `on-coll` and `on-p2p` (channels on), and two arms that must be refused:
  `off-p2p` (point-to-point without channels) and `on-budget` (hypercube
  above the session's window budget).
- **Baseline:** no NVIDIA NCCL arm ran in this gate. The NCCL column is
  NVIDIA NCCL 2.32.3 in the gate of libsircl snapshot `a3477af2` on the same
  ring
  ([libsircl status](../../../spark_transport/libsircl/STATUS.md#hardware-the-path-of-four-at-positions-4-7-and-the-cycle-of-eight-snapshot-a3477af2)).

## Result

nccl-tests, rank 0's out-of-place time per iteration, µs (bus bandwidth
GB/s):

| Collective | Size | libsircl 0.6.0, default | libsircl 0.6.0, ring schedules set | NVIDIA NCCL 2.32.3 |
|---|---|---|---|---|
| all-reduce | 4 KiB | 29.1 | 29.8 | 120.3 |
| all-reduce | 8 MiB | 804.6 (18.25) | 787.1 (18.65) | 782.6 |
| all-reduce | 256 MiB | 19,313.8 (24.32) | 19,294.2 (24.35) | 19,940 (23.56) |
| all-gather | 256 MiB | 9,737.6 (24.12) | 9,736.3 (24.12) | 10,127 (23.19) |
| reduce-scatter | 256 MiB | 9,813.3 (23.94) | 9,834.8 (23.88) | 10,320 (22.76) |
| broadcast | 256 MiB | 138,648 (1.94) | — | 11,082 (24.22) |

## Verdict

Every check passed:

- **Bytes:** all 8 ranks ran one library SHA-256, the lock's, and every
  communicator receipt names the layer receipt's four kernel packs (272
  receipts).
- **Correctness:** the bit-exact check exited 0 on 8 of 8 ranks with channels
  off and on. In `off-coll`, `ring-coll`, `on-coll` and `on-p2p` every
  nccl-tests job exited 0 on every rank with `#wrong 0` on every row (104,
  78, 104 and 78 rows). `off-p2p` and `on-budget` were refused on every rank
  (exit status 3), as required.
- **Cycle plan:** every eight-rank communicator without a schedule setting
  took it (96 communicators, 16 bit-exact runs expecting it). From 8 MiB the
  default all-reduce ran within 1.15 times the ring schedules' time in the
  timing sweep (24 size comparisons; 256 MiB: 19,288 against 19,290 µs eager;
  4 KiB: 18.2 µs).

Conclusion: image `1a8c10354eb0`'s libsircl is correct on the cycle of eight
and runs the cycle plan by default; its 256 MiB all-reduce reaches 24.3 GB/s
bus bandwidth against NVIDIA NCCL 2.32.3's 23.6 GB/s in the `a3477af2` gate.
Broadcast stays at 1.9 GB/s against NCCL's 24.2 GB/s: libsircl has no ring
broadcast (unsupported). Limitations: the NCCL comparison spans two gates on
the same ring; one run per arm; collectives only, no serving measurement
(`--transport libsircl` is research-only).
