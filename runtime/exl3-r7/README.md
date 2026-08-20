# EXL3 R7 3.5-bpw fixed-MTP4 ARM64 builder

This directory is a **reviewable public builder package** for the EXL3 R7
candidate runtime documented in
[`docs/EXL3_R7_FIXED_MTP4_PROFILE.md`](../../docs/EXL3_R7_FIXED_MTP4_PROFILE.md).
It builds an ARM64/SM121 container image from pinned public source trees with
fail-closed receipt verification.

**Maturity:** public-functional-lane, **offline-validated builder candidate**.
The operator configuration assembled from the same pinned component lineage is
live-validated, but an image built from this clean-checkout builder has not yet
passed that live gate. It is not the repository default, not an accepted
public-functional matrix, and not registry-available.

The builder compiles `spark_transport_capi` from the same SparkRing revision
and installs the manifest-bounded public vLLM adapter overlay. It therefore
does not depend on an inherited, operator-held SIRCL binary. It installs the
pinned public QuACK and ARM64 TVM FFI wheels and applies the hash-bound
compatibility edits used by the accepted operator profile. The exact-Q40 EXL3
and model-runner overlays remain a separate, published composition layer
because they require their own source-input hashes, local image identity, and
pre-graph numerical parity receipt. See
[`docs/EXL3_R7_OPERATOR_REPRODUCTION.md`](../../docs/EXL3_R7_OPERATOR_REPRODUCTION.md).

## Platform

| Property | Value |
|---|---|
| Architecture | `linux/arm64` |
| GPU target | SM121 (NVIDIA GB10 / DGX Spark, CUDA 13.2) |
| Python | 3.12 |
| Parent image | Supplied by the operator through `BASE_IMAGE` and `BASE_IMAGE_ID`; see "Choosing the parent image" |

## Inputs

- `pins.json` — authoritative pin set: release repository/commit, component
  base commits, patch paths, patch SHA-256 hashes, and result Git tree hashes.
- `prepare_build_deps.py` — CUTLASS and Triton kernel source pins with
  receipt-gated inventory verification.
- `Containerfile` — multi-stage build that clones ExLlamaV3 and InstantTensor
  at pinned commits, applies the ARM64 external-collectives patch, builds
  vLLM and B12X from the prepared source bundle, compiles SIRCL for SM121,
  installs the public adapter overlay, and runs `verify_runtime.py`.
- `PREPARED_SOURCES` environment variable (optional) — a pre-verified source
  directory whose receipt is checked by `prepare_context.py --verify`.

## Outputs

- A local container image tagged `${IMAGE:-sparkring-r7:arm64-sm121}`.
- The image's immutable ID is printed by `build-image.sh` on success.
- OCI labels embedded in the image record every component's upstream commit,
  the parent image ID, the source receipt SHA-256, and the license.

## Choosing the parent image

`BASE_IMAGE` and `BASE_IMAGE_ID` are operator inputs, not pins in this
repository, and the builder verifies whichever image is supplied against the ID
given alongside it. Two images in the build chain are usable as that parent, and
both are produced from this repository rather than obtained from a registry:

| Candidate | What it is | Built by |
|---|---|---|
| `sparkring/gb10-vllm-base` | The GB10 runtime layer: SM121 kernels, aarch64 cu132 wheels, the pinned vLLM, B12X, SparkInfer, NCCL, FlashInfer, and DeepGEMM | `scripts/build-gb10-vllm-base-image.sh`, which `scripts/bootstrap_nf3.py` and `scripts/bootstrap_exl3.py` invoke |
| `sparkring/glm52-exl3-tr3-3.25bpw` | The 3.25-bpw public-functional serving image, itself derived from the runtime base | `scripts/bootstrap_exl3.py` |

An image built by this builder records its parent in the
`org.sparkring.parent.image-id` label, which is how a deployment identifies
which of the two it descends from. That label on a built 3.5-bpw image and this
document have disagreed; the label is the record of what was built, and this
table describes what the builder accepts rather than asserting one lineage.

A published image can serve as the parent rather than one built locally:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:35b29616dc05677b98f647282e81a99fbca1969791ccbfca711c11a44285385e
```

`scripts/pull_pinned_images.py` retrieves it along with every other image the
locks pin. To use the 3.25-bpw serving image as the parent instead, run its
bootstrap. Either way, read the ID the builder must be given:

```bash
docker image inspect <parent-image-ref> --format '{{.Id}}'
```

## Build command

```bash
BASE_IMAGE=<parent-image-tag> \
BASE_IMAGE_ID=<parent-image-sha256-id> \
BASE_IMAGE_LICENSES=<parent-image-spdx-expression> \
  ./runtime/exl3-r7/build-image.sh
```

`BASE_IMAGE`, `BASE_IMAGE_ID`, and `BASE_IMAGE_LICENSES` are **required**.
The script resolves the parent image to its immutable ID, builds from that
resolved ID rather than the mutable name, and fails closed if the observed ID
does not match `BASE_IMAGE_ID`. The license value must be the audited SPDX
expression for the exact parent image; it is combined with the licenses of the
components added by this builder in the standard OCI license label.

To use pre-prepared sources instead of fetching at build time:

```bash
PREPARED_SOURCES=/path/to/prepared/sources \
BASE_IMAGE=... BASE_IMAGE_ID=... BASE_IMAGE_LICENSES=... \
  ./runtime/exl3-r7/build-image.sh
```

## Immutable identity contract

1. **Parent image:** identified by `BASE_IMAGE_ID` (sha256 image ID), never a
   mutable tag alone. The script compares the engine-resolved ID against the
   required value and exits 78 on drift.
2. **Source trees:** `prepare_context.py` fetches each component at its
   pinned commit, applies the pinned patch, and verifies the Git `write-tree`
   hash matches `result_tree` in `pins.json`. A `receipt.json` is written
   and self-verified before returning.
3. **Prepared sources:** when `PREPARED_SOURCES` is set, the receipt is
   verified by `prepare_context.py --verify` — not just checked for
   existence. The verifier re-checks schema, release commit, every component's
   `base_commit`, `patch_sha256`, and `result_tree` against `pins.json`,
   re-computes the actual Git tree of each checked-out component, and rejects
   unstaged or untracked content not represented by that tree.
4. **Build dependencies:** `prepare_build_deps.py` stages CUTLASS and
   Triton kernel sources with a per-file SHA-256 inventory receipt. The
   `--verify` flag re-checks the inventory against the receipt.
5. **Python runtime wheels:** QuACK 0.5.0 and Apache TVM FFI 0.1.10 are fetched
   from hash-bound public wheel URLs. The build verifies the resulting runtime
   file hashes.
6. **OCI labels:** the built image carries
   `org.sparkring.parent.image-id`, `org.opencontainers.image.source`,
   `org.opencontainers.image.revision`, `org.opencontainers.image.licenses`,
   and per-component commit labels for vLLM, B12X, ExLlamaV3, InstantTensor,
   CUTLASS, and Triton kernels.

## What is excluded

- No model weights, checkpoints, or safetensors files are included.
- No site addresses, SSH keys, credentials, or private host paths.
- No registry push step. The image is local only.
- The SM121 diagnostic probes (`sm121_shared_dense_probe.py`,
  `sm121_routed_expert_probe.py`) and qualification harnesses
  (`mtp*_qualification.py`, `endpoint_benchmark.py`) are GPU-only tools
  for live validation; they are not part of the build.
- The nonfinite-trace and shared-capture overlay builders are vLLM source
  patch generators for live runtime overlays; they are not built into the image.

## Maturity

This builder is a **public-functional-lane, offline-validated candidate**. It
is:

- derived from the component lineage used by the four-Spark operator result in
  `docs/EXL3_R7_FIXED_MTP4_PROFILE.md`, without claiming that a clean-checkout
  image built here has passed that live gate;
- **not accepted** as a public-functional matrix;
- **not the default** advertised configuration;
- **not registry-available** — the image is built and used locally;
- **not claimed** to be correct for hardware other than the four-Spark
  appliance it was validated on.

See [`docs/STATUS.json`](../../docs/STATUS.json) for the machine-readable
maturity record and [`AGENTS.md`](../../AGENTS.md) for the lane and maturity
terminology.
