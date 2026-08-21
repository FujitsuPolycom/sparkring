# SparkRing runtime

`runtime/` contains the reviewable runtime inputs shared by the supported
GLM-5.2 EXL3 3.5-bpw and DeepSeek-V4-Flash-0731 serving configurations. It
does not contain model weights, site configuration, registry credentials, or a
deployment launcher.

## Components

| Path | Role |
|---|---|
| [`exl3-r7/`](exl3-r7/README.md) | GLM-5.2 EXL3 3.5-bpw R7 ARM64 image builder and its verification tests |
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

The R7 test suite covers the surviving runtime build contracts:

```bash
python -m pytest runtime/exl3-r7 -q
```

## DeepSeek-V4-Flash-0731

The lock records the published ARM64 serving image that registers the
DeepSeek-V4 model surface. Its per-rank container environment is defined in
[`scripts/config/deepseek-v4-flash-0731.env.example`](../scripts/config/deepseek-v4-flash-0731.env.example).
The environment template and immutable image digest must agree; neither
substitutes for a four-rank serving qualification.

## Scope and safety

Overlay generation and its tests are **OFFLINE**. An image build changes the
local container store. Pulling, distributing, starting, or replacing images
on configured hosts is **MUTATES HOST** or **STOPS SERVING** and requires
explicit authorization for the named hosts and action.
