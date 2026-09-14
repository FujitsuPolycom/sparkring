# R37 shared feature image

Status: **Development**. This composition layers optional Qwen collective and
prefill features over the published R37 cache64 image. It preserves the
complete inherited vLLM, B12X, NCCL, SIRCL and SparkCache payload. Feature hooks
are baked into the image; model weights and site configuration are external.

The [descriptor](descriptor.json) pins the parent image/receipt and every
addition. The [local build record](local-build.json) identifies the built image
and verification scope. No publication or profile-default promotion is implied.

## Included capabilities

| Capability | Selection and scope |
|---|---|
| SparkCache 64-group Qwen hybrid capture/restore fixes | Inherited from [cache64](../lil-r37-cache64/README.md); enabled through the existing cache profile |
| Patched dual-domain NCCL, GLM HC/coalescing and SIRCL | Inherited unchanged; existing profiles own their activation |
| Qwen collective policy | `SPARKRING_FEATURES=qwen-collectives`; 20 KiB all-reduce cutoff by default, applicable to TP2/TP4 |
| Qwen HC fusion and MTP prefill GEMMs | `SPARKRING_FEATURES=qwen-prefill`; bounded QAD TP4 evidence, not admitted for TP2 |
| Both Qwen features | `SPARKRING_FEATURES=qwen-collectives,qwen-prefill` |

`SPARKRING_FEATURES` is empty by default. Unknown or duplicate selections stop
startup. The collective policy still requires an enabled RoCEnante transport
and the correct profile/site device map. The image includes the adaptive
transport-selection hook, which is inactive without an explicit
`SPARKRING_TRANSPORT_PROFILE` selection.

The prefill feature derives a source-specific compiler-cache directory beneath
the configured cache roots. Repeated Python startup preserves that namespace
rather than nesting another directory. Profiles retain their model, TP/DCP,
context, sequence, batch and KV settings.

## Build locally

Run from the repository root on a Linux ARM64 builder with the parent cached
or accessible. The build needs no model weights or GPU workload interruption.

```bash
PARENT_REF=ghcr.io/fujitsupolycom/sparkring@sha256:de885a8a3f687d1966b918f913ab95b0da33a84422313ed4c10ba5477c66f523
docker pull --platform linux/arm64 "$PARENT_REF"
docker tag "$PARENT_REF" sparkring-build-parent:r37-cache64-384bc56e4a1f
python3 runtime/images/feature_extension.py prepare \
  --descriptor runtime/images/compositions/lil-r37-shared/descriptor.json \
  --repository . --output /tmp/sparkring-shared-build
FEATURE_DESCRIPTOR_SHA=$(sha256sum /tmp/sparkring-shared-build/descriptor.json | cut -d' ' -f1)
FEATURE_IMAGE=sparkring-local:r37-arm64-${FEATURE_DESCRIPTOR_SHA:0:12}
docker build --pull=false --network=none \
  --build-arg BASE_IMAGE=sparkring-build-parent:r37-cache64-384bc56e4a1f \
  --build-arg DESCRIPTOR_SHA256="$FEATURE_DESCRIPTOR_SHA" \
  --tag "$FEATURE_IMAGE" /tmp/sparkring-shared-build
docker run --rm --network none --entrypoint /opt/venv/bin/python "$FEATURE_IMAGE" \
  /opt/sparkring/bin/feature-extension.py verify
```

The context must not exist. Preparation reconstructs the descriptor's exact
LF/CRLF bytes and rejects source-content changes. Installation verifies the
parent, refuses replacement of inherited files, records additions and preserves
the parent receipt for comparison. Verification checks the complete resulting
inventory, not just the feature directory. Build metadata/image IDs can differ
between independent builds; retain each build's actual descriptor and receipt.

Inspect image capabilities without starting a model:

```bash
docker run --rm --network none --entrypoint /opt/venv/bin/python "$FEATURE_IMAGE" \
  -c 'import json,sparkring_features; print(json.dumps(sparkring_features.description(), indent=2))'
```

Existing published profiles remain pinned to their existing releases. Adopting
this image requires profile/release admission and serving checks for the exact
selected combination; do not bypass an existing launcher's image validation.

## Checkpoint scheduling limitation

The existing Qwen SparkCache profile selects
`--recurrent-checkpoint-policy aligned`. This image does not add the proposed
request-boundary checkpoint connector support or change that policy. Different
checkpoint policies can change prefill chunking even with the same batch-token
limit. Cache-disabled prefill results must not be presented as SparkCache-enabled
results. Request-boundary SparkCache support remains separate implementation
and qualification work.
