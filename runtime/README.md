# SparkRing runtime

`runtime/` contains reviewable runtime inputs for the supported GLM-5.2 EXL3
3.5-bpw, GLM-5.3 Flash, DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6 serving
configurations. It does not contain model weights, site configuration, registry
credentials, or a live-deployment result.

## Components

| Path | Role |
|---|---|
| [`exl3-r7/`](exl3-r7/README.md) | GLM-5.2 EXL3 3.5-bpw R7 ARM64 image builder and its verification tests |
| [`glm53-flash/`](glm53-flash/README.md) | GLM-5.3 Flash target, BF16 DFlash2, vLLM, B12X, patched NCCL, and SparkCache identity and attestation contract |
| [`glm53-flash-jj-r8-gb10/`](glm53-flash-jj-r8-gb10/README.md) | One GLM-5.3 Flash R8 ARM64 image, source builder, and adjustable TP4/DCP1/DCP2/DCP4 launcher with SparkCache |
| [`glm53-flash-e10536a/`](glm53-flash-e10536a/README.md) | Implemented source builder for vLLM e10536a with internal MTP5 and opt-in adaptive depth; live serving unqualified |
| [`glm53-flash-b12x-kda-adaptive-mtp/`](glm53-flash-b12x-kda-adaptive-mtp/README.md) | Implemented source builder for adaptive MTP and live-tensor B12X KDA at vLLM `0b67266a`; live serving unqualified |
| [`deepseek0731-gb10/`](deepseek0731-gb10/README.md) | DeepSeek-V4-Flash-0731 GB10 parser, K5 sparse-row, and native PR431 image layer |
| [`qwen38/`](qwen38/README.md) | Public-source ARM64 image builder for the Qwen3.8-27B EXL3 K5/K6 pair and cycle profiles |
| [`faststart-lock.json`](faststart-lock.json) | Immutable ARM64 base-image and model-identity pins |
| [`build-public-overlay.py`](build-public-overlay.py) | Builds the reviewed Python overlay bundle |
| [`public-overlay-files.json`](public-overlay-files.json) | Explicit source-file allowlist for the public overlay |
| [`test_public_overlay.py`](test_public_overlay.py) | Offline contract coverage for allowlisting and manifest generation |

## GLM-5.2 EXL3 R7 builder

[`runtime/exl3-r7/`](exl3-r7/README.md) is the supported builder for the
GLM-5.2 EXL3 3.5-bpw serving image. It consumes pinned inputs, verifies the
runtime chain, and produces a locally tagged ARM64 image. It is a build
surface, not evidence that a four-rank appliance has started or served a
request.

Read the builder README before changing its image inputs, patches, or
verification code. A changed pin, patch preimage, or build artifact requires a
test that fails on the corresponding drift.

## Faststart lock

`faststart-lock.json` has schema `sparkring-faststart-lock/v1`. It records:

- the immutable manifest and configuration digests of the ARM64 base image;
- the GLM-5.2 model repository, immutable revision, and `config.json` hash;
- the supported serving-image digest and platform.

The builder must use immutable digests rather than mutable image tags. A lock
entry is a contract: do not replace a failed verification value with a hash
calculated from an arbitrary local checkout or image.

## Qwen3.8-27B builder

[`runtime/qwen38/`](qwen38/README.md) builds the Qwen runtime from public,
immutable source inputs over a pinned CUDA ARM64 parent. It produces a local
image and requires no published Qwen image or maintainer-held runtime archive.
The image contains no checkpoint or site configuration. Build it once, record
its image ID, and distribute the same saved OCI archive to every rank before
running the applicable Qwen pair or cycle quickstart.

## Public overlay

Run the overlay builder from the repository root:

```bash
python runtime/build-public-overlay.py \
  --output build/public-overlay
```

The builder accepts only files named in `public-overlay-files.json`, copies
them into the fixed runtime layout, and writes
`sparkring-overlay-manifest.json` with SHA-256 entries for every admitted
file. It rejects an unsupported source layout, a missing allowlisted source,
and unsafe relative paths. Add an overlay member only by updating the
allowlist and validating the generated manifest through the builder.

The builder test suites cover the GLM, DeepSeek, and Qwen runtime contracts:

```bash
python -m pytest runtime/exl3-r7 runtime/glm53-flash runtime/deepseek0731-gb10 runtime/qwen38 -q
```

[`runtime/glm53-flash/`](glm53-flash/README.md) builds the source-pinned
GLM-5.3 ARM64 parent runtime. It verifies the vLLM, B12X, patched NCCL,
InstantTensor, source-receipt, image-label, and SBOM inputs recorded in
`runtime/glm53-flash/pins.json`. SparkCache owns the derived
`deploy/glm53_flash/Containerfile`; SparkRing validates the published derived
image through exact labels and an in-container source contract. Run the
GLM-5.3 builder, publisher, profile, and launcher contracts with:

```bash
python -m pytest runtime/glm53-flash runtime/glm53-flash-e10536a \
  runtime/glm53-flash-jj-r8-gb10 \
  runtime/glm53-flash-b12x-kda-adaptive-mtp \
  scripts/test_glm53_flash_profile.py \
  scripts/test_prepare_glm53_e105_profile.py \
  scripts/test_prepare_glm53_b12x_kda_adaptive_mtp_profile.py \
  scripts/test_sparkring_generic_launcher.py -q
```

## DeepSeek-V4-Flash-0731

[`runtime/deepseek0731-gb10/`](deepseek0731-gb10/README.md) builds the
DeepSeek-specific layer over the generic GB10 serving image. The lock records
both the hardened image digest and the unchanged generic rollback digest. The
per-rank container environments are defined by topology:
[`scripts/config/deepseek-v4-flash-0731-pair.env.example`](../scripts/config/deepseek-v4-flash-0731-pair.env.example)
for a two-rank pair and
[`scripts/config/deepseek-v4-flash-0731.env.example`](../scripts/config/deepseek-v4-flash-0731.env.example)
for a four-rank cycle. The selected environment template and immutable image
digest must agree. Neither offline validation nor a successful launch on one
topology qualifies the other topology.

## Scope and safety

Overlay generation and its tests are **OFFLINE**. An image build changes the
local container store. Pulling, distributing, starting, or replacing images
on configured hosts is **MUTATES HOST** or **STOPS SERVING** and requires
explicit authorization for the named hosts and action.
