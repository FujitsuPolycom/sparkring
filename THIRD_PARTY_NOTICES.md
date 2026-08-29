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
`spark_transport/nccl/` contain portions of NVIDIA
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

## 3. vLLM (referenced and patched)

The unified diffs under `runtime/deepseek0731-gb10/patches/` contain context
and removed lines from vLLM, pinned to the source revision recorded by that
runtime contract. The added lines port upstream vLLM fixes and SparkRing's
DeepSeek GB10 integration. The files under `spark_transport/integrations/vllm/`
and the vLLM-facing files under `spark_transport/experiments/` remain original
SparkRing adapters: they verify exact upstream source before installing any
runtime modification and decline to install when the source differs.

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

## 9. EXL3 R7 runtime builder components

The `runtime/exl3-r7/` builder package assembles an ARM64/SM121 container image
from the following upstream components. Each is identified by an exact Git
commit in `runtime/exl3-r7/pins.json` or `runtime/exl3-r7/prepare_build_deps.py`
and embedded as an OCI label in the built image. None of these components'
source code is distributed in this repository; the builder fetches them at
build time from their public repositories.

### 9a. local-inference-lab vLLM fork

- Repository: `https://github.com/local-inference-lab/vllm.git`
- Exact commit: `e2666d9a65f41fc376607531453cbd57c4c71016`
- License: Apache License 2.0 (inherited from `vllm-project/vllm`)
- Role: Built into the image as the serving engine; patched with a receipt-gated
  integration patch (SHA-256 pinned in `pins.json`).

### 9b. B12X (local-inference-lab/b12x)

- Repository: `https://github.com/local-inference-lab/b12x.git`
- Exact commit: `7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b`
- License: Apache License 2.0
- Role: SM120/SM121 CuTe DSL kernel library for mixed-Trellis MoE, MLA
  attention, and fused GEMM. Built into the image as a pip package from source.

### 9c. ExLlamaV3 (brandonmmusic-max/exllamav3)

- Repository: `https://github.com/brandonmmusic-max/exllamav3.git`
- Exact commit: `704aefd743b390af4bd0fb429d1906f9b964c7d8`
- License: MIT License
- Role: EXL3 quantization encoder and extension. Cloned at build time, patched
  with the inherited ARM64 external-collectives patch (SHA-256 pinned in the
  Containerfile), and built in place.

### 9d. InstantTensor (voipmonitor/InstantTensor)

- Repository: `https://github.com/voipmonitor/InstantTensor.git`
- Exact commit: `49b4010afc1cae0441e71fe0b0bffc24fa05e932`
- License: Apache License 2.0
- Role: Ultra-fast distributed Safetensors weight loader. Cloned and installed
  as a pip package from source.

### 9e. NVIDIA CUTLASS

- Repository: `https://github.com/NVIDIA/cutlass.git`
- Exact commit: `da5e086dab31d63815acafdac9a9c5893b1c69e2`
- License: BSD-3-Clause
- Role: CUDA Templates for high-performance linear algebra; consumed as a source
  dependency by the vLLM build. Staged locally via
  `prepare_build_deps.py` with a receipt-gated inventory.

### 9f. Triton kernels (triton-lang/triton)

- Repository: `https://github.com/triton-lang/triton.git`
- Exact commit: `0add68262ab0a2e33b84524346cb27cbb2787356`
- Subdirectory: `python/triton_kernels/triton_kernels`
- License: MIT License
- Role: Triton kernel sources consumed by the vLLM build. Staged locally via
  `prepare_build_deps.py` with a receipt-gated inventory.

### 9g. QuACK kernels

- Package: `quack-kernels` 0.5.0 from the Python Package Index
- Wheel SHA-256: `08821ebfb8e638cc20308d5c59410c6dbb3b637ccc7b07bd57c7a9261a06af74`
- License: Apache License 2.0; the wheel contains its `LICENSE` file
- Role: CuTe DSL kernels used by the R7 runtime. The builder applies six
  hash-bound Python 3.12 annotation compatibility edits to `layout_utils.py`
  and `copy_utils.py`, then verifies the resulting file hashes. The edits are
  SparkRing-authored; the remaining files retain their upstream license.

### 9h. Apache TVM FFI

- Repository: `https://github.com/apache/tvm-ffi`
- Package: `apache-tvm-ffi` 0.1.10
- ARM64 wheel SHA-256: `3829216a8500c2f61062e48c627f6db6c3fa49416b3ffa85bc04243ae5d759f7`
- License: Apache License 2.0
- Notice: Copyright 2024-present The Apache Software Foundation; the wheel
  contains the upstream `LICENSE` and `NOTICE` files.
- Role: ARM64 FFI runtime installed under `/opt/sparkring-r7-tvm-ffi` and
  selected by the profile's `PYTHONPATH`.

### 9i. Inherited base image

The R7 image is built from an operator-supplied ARM64 parent image. The parent
is identified by an immutable sha256 image ID (`BASE_IMAGE_ID`), never a
mutable tag alone, and the build uses the resolved image ID. This repository
does not establish one universal parent-image license: the builder therefore
requires the audited SPDX expression for the exact parent as
`BASE_IMAGE_LICENSES`. The value is recorded as
`org.sparkring.parent.licenses` and included in the image's combined OCI
license expression. A registry publisher must audit the parent image's source,
license, and redistribution terms before publishing a derived image.

### 9j. OCI provenance labels

The built image carries these OCI labels for every component: exact upstream
revision (`org.sparkring.*.commit`), source repository
(`org.opencontainers.image.source`), license
(`org.opencontainers.image.licenses`), and the parent image's immutable ID
(`org.sparkring.parent.image-id`). Building or distributing the image requires
compliance with each component's license, including preservation of copyright
notices and license texts.

## 10. GLM-5.3 Flash profile artifacts (referenced; not included)

The GLM-5.3 Flash TP4/DCP1 recipes reference model and runtime artifacts that
are not distributed in this repository:

- `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `520de24eabf507659eaef7c70f14fd584527facc`, MIT, used as the ModelOpt
  mixed-precision target checkpoint;
- `incoai/GLM-5.3-Flash-DFlash2` revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, CC BY-NC-ND 4.0, used as
  the BF16 speculative drafter; its model card limits use to research and
  evaluation and directs commercial licensing inquiries to Inco AI;
- `local-inference-lab/vllm` commit
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`, Apache-2.0, used as the
  serving runtime; and
- `local-inference-lab/b12x` commit
  `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`, used as the GB10 kernel
  implementation under Apache-2.0 and its repository notices.

SparkRing records these identities and validates compatible image content; it
does not redistribute the model weights, vLLM source, or B12X source. Operators
must obtain each artifact under its own terms. The complete profile provenance
and known lineage limitations are in `runtime/glm53-flash/pins.json`.
