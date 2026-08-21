# GLM-5.2 EXL3 3.5-bpw ARM64 builder

This directory builds the ARM64/SM121 runtime used by the GLM-5.2 EXL3
3.5-bpw profile. Filesystem and environment contracts retain the literal
`exl3-r7` identifier where operators must type it.

Status: **implemented**. The builder and its receipt checks pass offline without
a live cluster. A derived image is not qualified until the four-Spark promotion
checklist passes against that image's immutable ID.

## Inputs

- `pins.json` identifies every source tree and resulting Git tree.
- `prepare_context.py` materializes and verifies those trees.
- `prepare_build_deps.py` stages CUTLASS and Triton kernel sources with a
  per-file receipt.
- `exllamav3-arm64-external-collectives.patch` disables x86-only CPU
  collective sources on ARM64.
- `Containerfile` builds vLLM, B12X, ExLlamaV3, InstantTensor, SIRCL, and the
  supported-profile Python overlay.
- `runtime/faststart-lock.json` pins the published parent image.

The builder accepts an optional preverified source directory through
`PREPARED_SOURCES`. Its receipt and Git trees are rechecked before use.

## Parent image

Pull the immutable parent on the ARM64 build host:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
docker image inspect \
  ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028 \
  --format '{{.Id}}'
```

`BASE_IMAGE_ID` is the local content-addressed image ID reported by Docker,
not the registry manifest digest. The builder verifies the observed ID before
creating a build context.

## Build

```bash
BASE_IMAGE=ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028 \
BASE_IMAGE_ID=<docker-image-id> \
BASE_IMAGE_LICENSES=<audited-spdx-expression> \
  ./runtime/exl3-r7/build-image.sh
```

The default output tag is `sparkring-r7:arm64-sm121`. Set `IMAGE` to choose
another local tag. The script prints the immutable output image ID.

## Verification invariants

The build fails closed unless:

1. the parent image resolves to `BASE_IMAGE_ID`;
2. every source commit, patch hash, and resulting Git tree matches
   `pins.json`;
3. the prepared dependency inventory matches its receipt;
4. QuACK and Apache TVM FFI wheels match their pinned hashes;
5. the SIRCL library and supported-profile overlay build from this checkout;
6. installed runtime files match the hashes enforced by
   `verify_runtime.py`.

The image records the parent ID, source receipt, component revisions, and SPDX
license expression in OCI labels.

## Scope

The image contains no model weights, site addresses, credentials, registry
push action, or live-cluster acceptance claim. Diagnostic probes and trace
overlay generators remain operator-invoked qualification tools and are not
enabled by the serving entrypoint.

Continue with the
[GLM quickstart](../../docs/GLM52_35BPW_QUICKSTART.md) and
[promotion checklist](../../docs/GLM52_35BPW_PROMOTION_CHECKLIST.md).
