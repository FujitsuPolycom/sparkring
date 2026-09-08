# Switched TP4 quickstart

Use the [switched profile](../runtime/profiles/glm53-flash-spark-tp4-switched/README.md)
for four one-GPU nodes connected through a RoCE switch. It selects the common
SparkRing image, NVFP4-Spark, TP4/DCP1, static MTP3, coalescing, and mHC with
ordinary NCCL. SparkCache is disabled.

## Prepare the source image

The common recipe and profile must be published before a model test. From the
repository root, prepare a new source cache and context using Python 3.12+:

```bash
python3 runtime/sparkring/source_image/prepare_image.py \
  --output /tmp/sparkring-common-context \
  --source-cache /tmp/sparkring-common-sources
```

The [common recipe](../runtime/sparkring/source_image/README.md) supplies the
ARM64 build and source-verification instructions. Build that common context
once; selecting this profile does not require a separate switched image.
Record the resulting exact local image config ID as `SPARKRING_IMAGE_ID`.

Generate a CPU source/file receipt for this profile:

```bash
python3 runtime/sparkring/source_image/verify_image.py \
  --image "$SPARKRING_IMAGE_ID" \
  --context /tmp/sparkring-common-context \
  --profile glm53-flash-spark-tp4-switched-mtp3 \
  --output /tmp/glm53-switched-image-receipt.json
```

The receipt is local-image evidence, not a registry publication digest or
hardware benchmark. Keep it with the exact source checkout used to build.

## Fill private rank inputs

On each node, copy
`runtime/profiles/glm53-flash-spark-tp4-switched/rank.env.example` to a private
path and resolve `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`,
`NCCL_IB_HCA`, and `NCCL_IB_GID_INDEX`. Use the actual switch-connected
interfaces and exact HCA/port names. This profile assumes neither an uplink
count nor the direct-ring addressing layout.
The profile enables the library's extended-IPv4 and PCI-domain options while
keeping switchless and subnet-aware routing disabled; the source-specific
behavior is described in the profile README.

Provide the NVFP4-Spark checkpoint at the profile's pinned revision and a
separate writable cache directory owned by the serving user. Keep site paths,
addresses, and receipts outside version control. Ensure the existing
`sparkring-memory-guard` service is active with at least a 4 GiB floor.

## Inspect, create, and start

For rank 0, substitute the site's real values and print the plan:

```bash
python3 runtime/profiles/glm53-flash-spark-tp4-switched/launch.py plan \
  --rank 0 --master rank0.example \
  --model-dir /srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4fb16b1d37ae43d2cfd624de26ffbc832e \
  --cache-dir /srv/cache/glm53-switched \
  --env-file /srv/config/glm53-switched-rank0.env \
  --image "$SPARKRING_IMAGE_ID"
```

Repeat with ranks 1–3 and their private files. Inspect HCA selection, addresses,
cache mounts, container UID/GID, disabled custom transports, and image identity.
Use `create` with the same arguments plus
`--runtime-receipt /tmp/glm53-switched-image-receipt.json` to create stopped
containers. Existing names are preserved. Use `start` with the same inputs
to start ranks 1–3, then rank 0. Automatic restart remains disabled.

The generic startup wrapper performs bounded warmup and six sampler requests
before rank-zero readiness. Wait for Docker health, then check API `/health`
on port 8000 and `/liveness` on port 8001. The source verifier, admission
wrapper, and real readiness marker remain active for this profile.
