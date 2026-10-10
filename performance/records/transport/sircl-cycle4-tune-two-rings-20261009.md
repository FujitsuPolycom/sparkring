# SIRCL tune of the cycle of four on two separate rings, 2026-10-09

Status: **measured; promoted into the default tuning table's `cycle-4` row. One quick ring-harness tune per
ring, GPU clocks not locked, one ring on mixed GPU drivers and kernels; no serving comparison with SIRCL's
rules on a cycle of four.**

The default SIRCL tuning table ([sircl-tuning-defaults.json](../../../runtime/common/sircl-tuning-defaults.json))
served a deployment on every Spark of a four-Spark cycle with the `cycle` row: SIRCL's own rules. This record
holds the measurement behind its `cycle-4` row and the SIRCL table that row names,
[cycle-4.json](../../../runtime/common/sircl-tuning/cycle-4.json) (hash `df333d97acb1a4b2`).

## Conditions

- **Hardware:** eight DGX Sparks (GB10, one ConnectX-7 each, RoCE) cabled as two separate cycles of four.
  `sudo sparkring setup` recorded each cycle's fabric: fabric `097062777e17` (ring A) and fabric
  `f1cb76795938` (ring B). Each cycle has 2 lanes per peer and at most 1 relay on a lane.
- **Package and image:** SparkRing package revision `e2590e79` on every Spark. Image configuration ID
  `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952` with the lock
  [installer-image-c7c35fe0.json](../images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json):
  SIRCL 0.3.2, native ABI 9, native source hash `09db871538819191`, kernel source hash `00f8d50c2cc5d98c`.
- **Harness:** SIRCL's ring harness (`python -m sparkring_sircl.ring tune`) from the image's build commit
  `c7c35fe0`. It staged those SIRCL sources on each Spark and built their native library inside the image,
  and the sources' tuning key equals the image's. Each ring ran its whole cycle as one group.
- **Sweep:** `--quick`. Per collective (all-reduce, all-gather, reduce-scatter, all-to-all), mode (eager and
  CUDA graph replay) and per-rank size (8 sizes: 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB and
  64 MiB), the harness timed the following candidates:
  - one-shot and two-shot at launch grids 4, 8, 16 and 32;
  - from 256 KiB, two-shot pieces, tiles and scatter ops;
  - the chain at pieces of 256 KiB, 512 KiB and 1 MiB;
  - the ring at each of those pieces and at staggers 0 and 1;
  - each chain and ring candidate at 1, 2 and 4 blocks per role.

  Every case cycled through 8 input and output windows. NCCL was not measured. Each ring's sessions ran
  with 8 link slots of 1 MiB.
- **Drivers:**
  - Ring A: positions 0-3 on GPU driver 580.173.02, kernel 6.17.0-1029-nvidia.
  - Ring B: positions 0-1 on 580.178.04, kernel 7.0.0-1019-nvidia; positions 2-3 on 580.173.02,
    kernel 6.17.0-1029-nvidia.

  The eight-Spark runs recorded earlier the same day ran on the same mix.
- **Other load:** none on ring A. Two Sparks of ring B each ran an idle container (`ubuntu:24.04`,
  `sleep infinity`, no GPU), so ring B's preflight and tune ran with the harness's `--force`.
- **Clocks:** not locked; the Sparks read 2,411 MHz during the runs.

## Result

| Ring | Fabric | Harness run | Measurement | Tune cases (measured, exact) | Ring's own table |
|---|---|---|---|---|---|
| A | `097062777e17` | `20261010-001824-tune4-a` | 30 min | 1,102, 1,102 | `bff6cee56c7d342c` |
| B | `f1cb76795938` | `20261010-002309-tune4-b` | 29 min | 1,098, 1,098 | `9f4c93ab51a28faf` |

Both runs passed. The rings ran at the same time, each on its own four Sparks. The rings' own tables
(digests `bff6cee56c7d342c` and `9f4c93ab51a28faf`) are held in the maintainer's run archive, not in the
repository; the merged table, [cycle-4.json](../../../runtime/common/sircl-tuning/cycle-4.json), is.

`scripts/promote_sircl_tuning.py --row cycle-4 --ring A --ring B` merged the two tables:

- **Merge rule:** a measurement (collective, mode, per-rank size, choice) exact on both rings keeps the
  slower ring's median. `sparkring_sircl.tuning.build_document` then decides each size from those times.
- **Counts:** 1,256 measurements were exact on both rings. 8 were exact on one ring only and were dropped.
- **Where the rings disagree:** the rings' own tables chose differently at 29 of the 64 measured points.
  24 of them differ only in the launch grid of a one-shot, two-shot, pieces or all-to-all op. The other 5 differ
  in a ring or chain piece, at 256 KiB to 64 MiB.
- **Cost:** at every measured point, the merged choice took at most 4.3 % longer on ring A than ring A's own
  fastest candidate there, and at most 1.1 % longer on ring B. At half or more of the points it was each
  ring's own fastest.

The merged table records `SIRCL_LINK_SLOTS=8` and `SIRCL_LINK_SLOT_BYTES=1048576`, which a session applies
where its environment leaves them unset. It decides, per collective, by mode:

- **All-reduce:**
  - one-shot up to about 90 KiB per rank;
  - two-shot up to 1.2 MiB;
  - above that, the ring at stagger 1: 512 KiB pieces from 1.2 MiB (to 1.4 MiB eager, 2.4 MiB graph),
    then 256 KiB pieces to 14 MiB, and 512 KiB pieces above.
- **All-gather:**
  - pieces up to 0.6 MiB per rank shard (in graph mode with one ring interval at 0.42-0.59 MiB);
  - above that, the ring at stagger 1: 256 KiB and 512 KiB pieces to 2 MiB, 512 KiB pieces above, and in
    graph mode 1 MiB pieces from 45 MiB.
- **Reduce-scatter:**
  - scatter ops up to 2.4 MiB (graph) or 2.8 MiB (eager), with one chain interval at 256-362 KiB in eager
    mode;
  - above that, the ring at stagger 1 with 512 KiB pieces, and 1 MiB pieces from 38 MiB in eager mode.
- **All-to-all:** launch grids only.

## Conclusion

The `cycle-4` row of the default table names this table. Sessions of SIRCL 0.3.2 whose group is a whole
cycle of four take its choices and settings. Sessions of the compatible SIRCL 0.3.1 take the row, which sets
nothing, and keep SIRCL's rules. The row's `evidence` names the fabrics, runs, image, SIRCL build and each
Spark's driver and kernel.

Limitations:
- **Sizes:** the quick sweep measured 8 sizes. Between them the table interpolates, and above 64 MiB its
  last choice applies.
- **Runs:** each ring ran once, with clocks not locked.
- **Drivers:** ring B ran on two driver and kernel versions.
- **No serving comparison:** no serving run compared the row with SIRCL's rules on a cycle of four.
