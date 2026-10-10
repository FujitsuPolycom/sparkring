# libsircl 0.6.0 of the release image on two rings of four Sparks

Lane: **public-functional**. Status: **implemented**. Evidence scope:
**live-validated collectives of the libsircl library that installer image
`1a8c10354eb0` carries, on two four-Spark rings at once, with nccl-tests and
libsircl's own checks; no serving measurement**. Hardware: eight NVIDIA DGX
Sparks (GB10) cabled as two independent rings of four (ring A and ring B)
with ConnectX-7 RoCE, each recorded by `sudo sparkring setup` as a
`cycle-4` fabric; one rank per Spark on GPU 0.

This record is the gate `gate-ring4-image-1a8c10354eb0-20261010T005453Z` of
the libsircl library in the image of release 2026.10.2
([release record](../../../runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)),
the four-Spark counterpart of the
[gate on the cycle of eight](libsircl-ring8-image-1a8c10354eb0-20261009.md).
Each ring's report and comparison are beside it
([ring A](libsircl-ring4-image-1a8c10354eb0-20261010/ring-a/report.md),
[ring B](libsircl-ring4-image-1a8c10354eb0-20261010/ring-b/report.md)); host
identities are position labels 0 to 3 within each ring.

## Conditions

- **Image and lock:** image
  `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  lock `installer-image-c7c35fe0.json` (built from commit `c7c35fe0`). Library
  `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0`, SHA-256
  `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`, source tree
  `dbf3607484dd47df4cf6c8238eb5b3272466effb`; every rank of both rings ran these
  bytes (4 of 4 ranks per ring, one library SHA-256).
- **Routing:** per ring, the route planner's `ring:4` layout
  (`spark_transport/libsircl/tools/site_routes.py`): two lanes, at most one
  relay, chain order 0-3, on that ring's recorded fabric document.
- **Drivers:** ring B's positions 0 and 1 ran GPU driver 580.178.04 with
  kernel 7.0.0-1019, and its positions 2 and 3 driver 580.173.02 with kernel
  6.17.0-1029.
- **Harnesses and arms:** as for the cycle of eight: libsircl's bit-exact
  check with point-to-point channels off and on; nccl-tests v2.21.1, bfloat16,
  8 B to 256 MiB doubling, `-n 20 -w 5`, in the arms `off-coll`, `ring-coll`
  (ring schedules set), `on-coll`, `on-p2p`, `off-p2p` (must be refused) and
  `on-budget`; libsircl's timing sweep (`perf_rank.py`, 60 size checks per
  sweep). On a ring of four every relayed pair has a window within the
  session's budget, so the planner expects `on-budget` to pass, where on the
  cycle of eight it must be refused.
- **Time:** both rings at once, 2026-10-10 from 00:55 UTC. No NVIDIA NCCL arm
  ran.

## Result

Every verdict passed on both rings:

- Bytes: every rank's library and kernel packs match the lock and the layer
  receipt; every communicator receipt names the four packs (136 receipts per
  ring).
- Bit-exact check: exit 0 on 4 of 4 ranks with channels off and on (8 runs
  per ring).
- nccl-tests: `off-coll` and `on-coll` (104 rows each), `ring-coll` and
  `on-p2p` (78 rows each): every job exited 0 on every rank, `#wrong 0`,
  every receipt forwarded 0, healthy, no refusals. `on-budget` passed on every
  rank, as the planner expects. `off-p2p` was refused on every rank (exit
  status 3), as required.
- Cycle plan: every four-rank communicator without a schedule setting took it
  (48 communicators; 8 bit-exact runs expecting it), and from 8 MiB the
  default all-reduce ran within 1.15 times the ring schedules' time on 24 of
  24 size comparisons per ring.

All-reduce, slowest rank's mean eager time in the timing sweep, µs (bus
bandwidth GB/s):

| Size | Ring A, default | Ring A, ring schedules | Ring B, default | Ring B, ring schedules |
|---|---|---|---|---|
| 4 KiB | 17.76 | 11.61 | 11.55 | 11.38 |
| 8 MiB | 619.0 (20.33) | 611.5 (20.58) | 612.6 (20.54) | 613.9 (20.50) |
| 256 MiB | 16,551.4 (24.33) | 16,542.9 (24.34) | 16,561.7 (24.31) | 16,569.3 (24.30) |

nccl-tests, rank 0's out-of-place time per iteration at 256 MiB, `off-coll`
arm, µs (bus bandwidth GB/s):

| Test | Ring A | Ring B |
|---|---|---|
| all-reduce | 16,566.1 (24.31) | 16,604.2 (24.25) |
| all-gather | 8,320.76 (24.20) | 8,345.64 (24.12) |
| reduce-scatter | 8,645.51 (23.29) | 8,675.51 (23.21) |
| broadcast | 50,804.7 (5.28) | 50,055.4 (5.36) |

## Conclusion

On both rings of four, the libsircl of the release image is correct in every
collective and point-to-point line, passes the window budget the planner
expects for a ring of four, refuses point-to-point without channels, and runs
the cycle plan by default. Its 256 MiB all-reduce takes 16.55 ms, 24.3 GB/s
bus bandwidth, on both rings, which agree within 1 % in every 256 MiB
figure above. Broadcast reaches 5.3 GB/s on a ring of four: libsircl has no
ring broadcast (unsupported at bandwidth).

## Limitations

- No NVIDIA NCCL arm ran on the rings of four, so this record has no NCCL
  comparison.
- Ring A's default 4 KiB time (17.76 µs) is higher than its ring-schedule
  time and ring B's (about 11.5 µs); one run per arm cannot say whether that
  is spread.
- Ring B mixed two driver and kernel versions.
- nccl-tests and the timing sweep time collectives, not serving.
