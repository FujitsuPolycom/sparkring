# libsircl 0.6.0 of image 1a8c10354eb0 on two rings of four Sparks

Lane: **public-functional**. Status: **implemented**. Evidence scope:
**live-validated collectives of the libsircl library that installer image
`1a8c10354eb0` carries, on two four-Spark rings at once, with nccl-tests and
libsircl's own checks; no serving measurement**. This record is gate
`gate-ring4-image-1a8c10354eb0-20261010T005453Z`, evidence of release 2026.10.2
([release record](../../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)),
the four-Spark counterpart of the
[gate on the cycle of eight](libsircl-ring8-image-1a8c10354eb0-20261009.md).

## Conditions

- **Hardware:** eight NVIDIA DGX Sparks (GB10) cabled as two independent
  rings of four (ring A and ring B) with ConnectX-7 RoCE, each recorded by
  `sudo sparkring setup` as a `cycle-4` fabric; one rank per Spark on GPU 0;
  both rings at once, 2026-10-10 from 00:55 UTC. Ring B's positions 0 and 1
  ran GPU driver 580.178.04 with kernel 7.0.0-1019, its positions 2 and 3
  driver 580.173.02 with kernel 6.17.0-1029.
- **Library:** `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0` of image
  `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`,
  source tree `dbf3607484dd47df4cf6c8238eb5b3272466effb` (lock
  `installer-image-c7c35fe0.json`).
- **Routing:** per ring, the route planner's `ring:4` layout: two lanes, at
  most one relay, chain order 0-3.
- **Harnesses and arms:** those of the cycle of eight: the bit-exact check
  with point-to-point channels off and on; nccl-tests v2.21.1, bfloat16, 8 B
  to 256 MiB doubling, `-n 20 -w 5`, in the arms `off-coll`, `ring-coll`,
  `on-coll`, `on-p2p`, `on-budget` and `off-p2p`; the all-reduce timing sweep.
  On a ring of four every relayed pair's window fits the session's budget, so
  the planner expects `on-budget` to pass; `off-p2p` must be refused. No
  NVIDIA NCCL arm ran.

## Result

nccl-tests `off-coll` arm, rank 0's out-of-place time per iteration, µs (bus
bandwidth GB/s):

| Collective | Size | Ring A | Ring B |
|---|---|---|---|
| all-reduce | 4 KiB | 25.6 | 18.5 |
| all-reduce | 8 MiB | 630.2 (19.97) | 625.9 (20.10) |
| all-reduce | 256 MiB | 16,566.1 (24.31) | 16,604.2 (24.25) |
| all-gather | 256 MiB | 8,320.8 (24.20) | 8,345.6 (24.12) |
| reduce-scatter | 256 MiB | 8,645.5 (23.29) | 8,675.5 (23.21) |
| broadcast | 256 MiB | 50,804.7 (5.28) | 50,055.4 (5.36) |

## Verdict

Every check passed on both rings:

- **Bytes:** every rank ran the lock's library SHA-256 (4 of 4 per ring), and
  every communicator receipt names the four kernel packs (136 per ring).
- **Correctness:** the bit-exact check exited 0 on 4 of 4 ranks with channels
  off and on. In `off-coll`, `on-coll` (104 rows each), `ring-coll` and
  `on-p2p` (78 rows each) every job exited 0 on every rank with `#wrong 0`.
  `on-budget` passed and `off-p2p` was refused (exit status 3) on every rank,
  as the planner expects on a ring of four.
- **Cycle plan:** every four-rank communicator without a schedule setting
  took it (48 per ring), and from 8 MiB the default all-reduce ran within
  1.15 times the ring schedules' time in the timing sweep on 24 of 24 size
  comparisons per ring (256 MiB: 16,551 µs on ring A and 16,562 µs on ring B,
  24.3 GB/s).

Conclusion: image `1a8c10354eb0`'s libsircl is correct on both rings of four
and runs the cycle plan by default; the two rings agree within 1 % at
256 MiB. Broadcast reaches 5.3 GB/s: libsircl has no ring broadcast
(unsupported at bandwidth). Limitations: no NCCL comparison; one run per
arm; ring A's default 4 KiB all-reduce in the timing sweep (17.8 µs) was
slower than its ring-schedule run and ring B's (about 11.5 µs), which one run
cannot separate from spread; ring B mixed two driver and kernel versions;
collectives only, no serving measurement.
