# R37 shared image with Qwen prefill kernels

Status: **Development**. This composition installs Qwen HC row sharding,
recurrent-checkpoint coalescing, PLE checkpoint export and paired sparse-attention
representative scoring into the shared ARM64 SparkRing image. Profiles select
HC sharding and coalescing; the paired scorer selects eligible shapes at runtime.
Model weights and site configuration remain external.

The [source provenance](provenance.json) identifies the upstream contributions
and the R37 compatibility adaptations. The [patch](runtime.patch) reconstructs
14 Python files from exact installed preimages. The [descriptor](descriptor.json)
pins the parent image and receipt, patch, resulting files and installer.
The inherited dual-domain NCCL, SIRCL, RoCEnante, GLM integrations, Qwen HC fusion,
MTP GEMMs and SparkCache binaries are unchanged.

The [lease contract](vllm-connector-jobs-r37-qwen-prefill.json) binds the scheduler
and checkpoint source files used by asynchronous SparkCache capture. It preserves
the required block-lifetime semantics. Source agreement does not establish
external-cache restore correctness; serving evidence is recorded separately.

## Build locally

Use a Linux ARM64 Docker host with the published shared parent cached. Its
immutable registry reference is
`ghcr.io/fujitsupolycom/sparkring@sha256:aef597a5ee70f7b4e0807901e43456b6ac8d2234247ab4df6a6cfe031e5169c6`.
Run from the repository root:

```bash
PARENT_ID=sha256:2540686d726a28eb07784f9d2db5dc1f795404c7874fc1d6c11f018cd789adc2
docker image inspect "$PARENT_ID"
docker tag "$PARENT_ID" sparkring-build-parent:r37-shared-2540686d726a
python3 runtime/images/source_extension.py prepare \
  --descriptor runtime/images/compositions/lil-r37-qwen-prefill/descriptor.json \
  --repository . --output /tmp/sparkring-qwen-prefill-build
DESCRIPTOR_SHA=$(sha256sum /tmp/sparkring-qwen-prefill-build/descriptor.json | cut -d' ' -f1)
IMAGE=sparkring-local:lil-r37-qwen-prefill-${DESCRIPTOR_SHA:0:12}
docker build --pull=false --network=none \
  --build-arg PARENT_IMAGE=sparkring-build-parent:r37-shared-2540686d726a \
  --tag "$IMAGE" /tmp/sparkring-qwen-prefill-build
docker run --rm --network=none "$IMAGE" verify
docker image inspect --format '{{.Id}}' "$IMAGE"
```

Preparation refuses an existing output directory. Installation checks the entire
parent inventory, applies the patch in a temporary tree, checks every result,
and installs a distinct receipt and verified serving entry point. It performs
no dependency resolution or native-library rebuild. Python-defined GPU kernels
compile during runtime preparation and require GPU tests.

Independent builds can have different Docker layer identities while producing
identical installed-source receipts. Distribute one selected image to every
rank and verify the exact image ID before rendering a deployment.

## Run through the profile

Use the explicit [local source-image options](../../../../docs/operations/compose.md)
with the Qwen QAD TP4 profile. The image must have the same local tag and image ID
on all four hosts. The profile supports cache-disabled and SparkCache selections;
the cache selection mounts model/cache data without any application-source mounts.

Published image selections and TP2 settings remain unchanged. This composition
does not promote TP2 or GLM to a different image. Each profile retains its own
serving evidence and rollback image.

## Component checks

Run these checks inside the built image on an idle GB10 GPU, mounting only the
test directory at `/checks`:

```bash
/opt/venv/bin/python -m pytest /checks/qsa_paired.py -q
/opt/venv/bin/python -m pytest /checks/ple_checkpoints.py -k internal_checkpoint -q
```

The QSA checks compare exact scalar scores, selected positions, attention outputs
and persistent state, including FP8 KV, invalid pages, large pool offsets and
CUDA graph replay. PLE checks cover interior-state export and graph replay.
The test mount is not required by serving containers.
