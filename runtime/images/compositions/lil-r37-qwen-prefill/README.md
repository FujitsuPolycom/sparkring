# R37 shared image with Qwen prefill kernels

Status: **Development**. This composition installs Qwen HC row sharding,
recurrent-checkpoint coalescing, PLE checkpoint export and paired sparse-attention
representative scoring into the shared ARM64 SparkRing image. Profiles select
HC sharding and coalescing; the paired scorer selects eligible shapes at runtime.
Model weights and site configuration remain external.

The [source provenance](provenance.json) identifies the upstream contributions
and the R37 compatibility adaptations. The [patch](runtime.patch) reconstructs
15 Python files from exact installed preimages. The [descriptor](descriptor.json)
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

An explicit local TP2 source-image trial is also supported for
`qwen38-flash-next-tp2` and `qwen38-flash-next-tp2-sparkcache`. It keeps HC
sharding off and existing TP2 features unchanged, enables recurrent-checkpoint
coalescing, and preserves the canonical 24 GiB KV allocation. The optional
`--local-kv-cache-gib 33` selects a bounded TP2 memory alternative. It cannot
select TP4-only HC fusion or the 40 GiB TP4 alternative. SparkCache uses the
packaged source lease contract and its own TP2 persistent namespace. See the
[rendering command and trial scope](../../../../docs/operations/compose.md).

Published image selections remain unchanged. This local route does not promote
TP2 or GLM or transfer TP4 serving results to TP2. Each profile retains its own
serving evidence and rollback image; TP2 performance and restart restoration
require separate measurements on the selected image.

## Promote a qualified image

The Qwen adapter supports registered source-image releases through ordinary
Docker and Compose commands. No publication is registered for this composition.
Promotion requires the actual distribution identities and profile evidence;
changing a tag or copying a parent receipt does not select this source image.

After authorized publication of the tested image:

1. Add `publication.json` in this directory using schema
   `sparkring-image-publication/v1`. Record the registry `image_reference` with
   its manifest digest, exact `image_id`, `platform: linux/arm64`, the SHA256 of
   `descriptor.json`, and `anonymous_pull_verified: true` after checking a pull.
   Include the versioned tag, serving-evidence path and qualification limits.
2. Add a distinct release selection under `runtime/releases/`. Use
   `selection: registered-source-extension-image`, the same registry reference
   as `image`, and hash-pinned inputs for this publication and descriptor, with
   the publication first. Preserve the existing release records.
3. Point only the qualified TP4 profile's `profile.json` at that release. Set
   its canonical configuration's `image_extension` to `lil-r37-qwen-prefill`.
   The adapter selects source admission and the verified source entrypoint;
   feature settings still come from the canonical configuration.
4. For the enhanced TP4 configuration, explicitly set
   `VLLM_QWEN3_8_HC_PREFILL_MODE=shard` and `VLLM_QWEN3_8_PREFILL_COALESCE=1`.
   A SparkCache selection must reference the packaged
   `/opt/sparkring/contracts/vllm-connector-jobs-r37-qwen-prefill.json` lease
   contract and its own persistent-cache namespace. Retain its checkpoint,
   context, batching and transport settings unless separately reviewed.
5. Update that profile's quickstart and evidence, then regenerate Compose
   examples with `python3 scripts/generate_compose_examples.py`. Ordinary
   `compose render PROFILE --site SITE --output DIRECTORY` now selects the
   registered image; no local source-image flags are needed.

The TP4 40 GiB and TP2 33 GiB KV alternatives remain explicit local test options. Public profiles
keep 24 GiB until their canonical settings and evidence are deliberately updated.
TP2 and GLM retain their own image selections. This registration work changes
deployment metadata and admission; it does not require rebuilding an unchanged
tested image.

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

The allocator fix preserves interior checkpoints that overlap private speculative
scratch during aligned continuation. It retains worker-visible block IDs and
reserves replacement scratch; shared or hashed blocks remain ineligible. Run
its CPU checks inside the image with `SPARKRING_TEST_SOURCE_ROOT` set to
`/opt/venv/lib/python3.12/site-packages`, selecting
`/checks/mamba_checkpoint_reservation.py` with pytest.

## Packaged serving evidence

The [bounded TP4 record](../../../../performance/records/qwen38-flash-next/r37-source-prefill-5bd9d05327d4.json)
covers this [local image build](local-build.json), its installed-source GPU checks,
ordinary Compose launch, text/tool/vision smoke, and two four-rank external
restores after process recreation. Both 7,870- and 8,194-token cold requests
published and restored a 7,200-token checkpoint. The serving configuration
uses 40 GiB FP8 KV per rank (5.22M logical KV tokens); it does not qualify the
public 24 GiB setting or other model/topology combinations on this image.

Cold-prefill medians were within 1% of the overlay control. Short C1 decode was
6.2% lower (86 versus 92 completion tokens/s); five follow-up samples retained
that level. C8 was 2.5% lower (334 versus 342 aggregate tokens/s). The cause of
the C1 difference is unestablished. These measurements use end-to-end completion
throughput, not the common-active-interval decode metric used by some upstream
reports; they do not establish decode performance parity.
