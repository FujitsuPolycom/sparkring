# Setup and SIRCL tuning of two rings of four Sparks

Lane: **public-functional**. Status: **implemented**. Evidence scope:
**verdicts of `sudo sparkring setup --re-form` and of SIRCL's ring-harness
tune on two four-Spark rings; no serving measurement**. Hardware: the eight
NVIDIA DGX Sparks (GB10) of the eight-Spark ring, recabled as two independent
rings of four (ring A and ring B) with ConnectX-7 RoCE. Positions are 0 to 3
within each ring.

This record holds the verdicts of the setup runs, whose outputs are not in
the repository, and of the tunes, whose runs and merge the
[SIRCL tune record](sircl-cycle4-tune-two-rings-20261009.md) holds. It belongs to release 2026.10.2
([release record](../../../runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)).

## Setup

- **Conditions:** `sudo sparkring setup --re-form` on each ring's Node A, with
  a SparkRing package that holds commit `e2590e79` (a re-form keeps the
  worker's preparation and removes the moved fabric's relay table). It
  renumbers the fabric addresses of the recabled Sparks
  ([change the layout](../../../docs/operations/install-reference.md#change-the-layout)).
- **Result:** both re-forms finished with exit status 0 and printed
  `Fabric verified: 4 cables on 4 Sparks (cycle-4)`. The cable speed check
  measured 212.7 to 213.4 Gb/s on every cable of both rings.
- **Use:** the libsircl gate on both rings ran on the fabric documents these
  setups recorded
  ([record](libsircl-ring4-image-1a8c10354eb0-20261010.md)).

## SIRCL tune

- **Conditions:** SIRCL's ring harness `tune` with the quick size set (every
  fourth size from 4 KiB to 64 MiB) on the `cycle-4` group shape, on both
  rings. Ring B's positions 0 and 1 ran GPU driver 580.178.04 with kernel
  7.0.0-1019, its positions 2 and 3 driver 580.173.02 with kernel 6.17.0-1029.
- **Result:** both tunes passed: 1,102 and 1,098 cases on the two rings,
  every output exact.
- **Use:** the measured `cycle-4` row of the default tuning table
  ([sircl-tuning-defaults.json](../../../runtime/common/sircl-tuning-defaults.json))
  merges these tunes; the [SIRCL tune record](sircl-cycle4-tune-two-rings-20261009.md)
  holds the runs, both rings' table digests and the merge.

## Limitations

- Verdicts only for the setups: their outputs are not committed.
- One run per ring. Ring B mixed two driver and kernel versions; the row's
  evidence records each Spark's driver and kernel.
