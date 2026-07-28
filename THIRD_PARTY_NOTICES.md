# Third-Party Notices

SparkRing is licensed under the Apache License, Version 2.0. The full license
text is in the `LICENSE` file at the root of this repository.

Copyright 2026 SparkRing contributors.

This document identifies third-party material contained in this repository and
third-party projects that this repository references, patches, or interoperates
with. Except as stated below, all files in this repository are original
SparkRing work licensed under Apache-2.0.

## 1. NVIDIA NCCL (portions included)

The following patch files under
`spark_transport/experiments/nccl_switchless_ring/` contain portions of NVIDIA
NCCL:

- `nccl-2.29.7-skip-tree-pat.patch` — against NCCL v2.29.7,
  `src/transport/generic.cc`
- `nccl-2.30.7-skip-tree-pat.patch` — against NCCL v2.30.7-1,
  `src/transport/generic.cc`
- `nccl-2.30.7-advertise-all-listener-gids.patch` — against NCCL v2.30.7-1,
  `src/transport/net_ib/connect.cc`

The context and removed lines in these unified diffs are verbatim NVIDIA NCCL
source code:

> Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The NCCL files these patches modify are licensed under the Apache License,
Version 2.0, per NCCL's `LICENSE.txt`. NCCL itself is not distributed in this
repository. Applying these patches and building or distributing a patched
`libnccl` binary requires compliance with NCCL's complete `LICENSE.txt`,
including preservation of its copyright notices and license text.

The lines added by `nccl-2.30.7-advertise-all-listener-gids.patch` are original
SparkRing work. For the lines added by the two skip-tree patches, see Section 2.

## 2. josephdrose/nccl-spark-switchless (approach credit)

The switchless skip-Tree/skip-PAT approach reproduced by
`nccl-2.29.7-skip-tree-pat.patch` and `nccl-2.30.7-skip-tree-pat.patch`
originates from Joseph Rose's `josephdrose/nccl-spark-switchless`. That
repository states no license. The guard reproduced in these patches is a
minimal (approximately four-line) environment-variable gate — the
`NCCL_SKIP_TREE_CONNECT` check and its log message. No other code from that
project is included in this repository. This entry records credit for the
approach.

## 3. vLLM (referenced; not distributed; no code included)

No vLLM source code is distributed in this repository. The files under
`spark_transport/integrations/vllm/` and the vLLM-facing files under
`spark_transport/experiments/` are original SparkRing adapters: they
monkey-patch a running vLLM installation at runtime and verify the exact
upstream source by SHA-256 hash before installing any modification, declining
to install (falling back to stock behavior) if the hash does not match.

vLLM is licensed under the Apache License, Version 2.0, Copyright the vLLM team
and contributors. Obtaining and running vLLM is subject to its own license and
notices. SparkRing is not a fork of vLLM.

## 4. B12X / Eldritch vLLM fork (not included)

The deployed runtime these adapters were validated against was built from a
private vLLM-derivative fork ("B12X" / "Eldritch"), pinned in this repository
by the version string
`0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626`.
That fork is not included in, and not published from, this repository.

## 5. eugr/spark-vllm-docker (acknowledgment; no code included)

Mod-packaging and build-cache concepts were studied from
`eugr/spark-vllm-docker` (MIT License, Copyright 2026 Eugene Rakhmatulin). No
code from that project was copied into this repository.

## 6. RTL8127 kernel experiment (withheld)

A SparkRing-authored RTL8127 kernel-handoff prototype, licensed GPL-2.0-only by
design for linkage into Realtek's GPL `r8127` driver, exists but is withheld
from this snapshot. The approach it describes requires Realtek's GPL `r8127`
(11.014.00) driver source, which must be obtained separately under its own
license. No GPL-licensed code is included in this snapshot.

## 7. References are not inclusion

Documentation and version strings in this repository refer to third-party
projects, including NVIDIA NCCL, vLLM, FlashInfer, the B12X/Eldritch vLLM
fork, `josephdrose/nccl-spark-switchless`, `josephdrose/joe-spark-patches`,
and `eugr/spark-vllm-docker`. A reference to an upstream project or design is
not a claim that its source code is included here. Building any of the optional
patched dependencies described in this repository requires obtaining each
upstream project under, and complying with, that project's own license and
notices.

## 8. License headers

After the exclusions described above, no file in this snapshot carries a
third-party SPDX identifier or copyright header. Any file that bears such a
header must retain it verbatim.
