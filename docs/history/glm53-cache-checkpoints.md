# GLM-5.3 MTP3 with verified caching and recurrent checkpoints

This retained guide describes a retired deployment configuration. Use the
[profile catalog](../../profiles/README.md) to choose a maintained deployment.
Its evidence applies only to the pinned configuration described here.

The [maintained GLM-5.3 four-Spark guide](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md)
uses the published shared image with DCP1 by default, optional SparkCache,
and a separate DCP4 selection. This historical guide describes the separately
pinned image below with persistent caching enabled.

Status: **research-only**. The exact published image passed
[eight bounded serving checks](../../performance/records/glm53-flash/mtp3-cache-checkpoints-serving-smoke-20260906.md):
text, growing conversation, streaming, reasoning rejection, and image responses.
Its 5,308 runtime-file comparisons and 5,472-file image verification also passed.
The smoke test configured 40 GiB of persistent cache per rank but did not fill
it; no explicit persistent restore was observed. It is not a long-duration soak.

The startup-admission fix in merged
[SparkRing #237](https://github.com/FujitsuPolycom/sparkring/pull/237) is absent
from this image. Public traffic can compete with warmup before Docker readiness.
Keep client traffic off the API until readiness completes. The HTTP 503 startup
gate requires a rebuilt image and separate startup verification.

## Prerequisites

Use four NVIDIA Sparks with the [managed mesh host setup](../GLM53_SPARK_MESH_HOST_SETUP.md).
The model uses TP4/DCP4 and native MTP depth three. Keep the reviewed fabric,
driver, GID and interface settings; this guide does not reconfigure NICs.
Follow [host prerequisites](../operations/prerequisites.md#four-spark-managed-hardware-forwarded-mesh),
including optional reboot preparation after large GPU workloads. Stop model
workloads through the managed lifecycle before replacing containers or services.

Download the target `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. No external draft is required.

## Pull and verify the image

Use the same immutable image on all four ranks:

```bash
image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746'
image_id='sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a'
docker pull "$image"
test "$(docker image inspect "$image" --format '{{.Id}}')" = "$image_id"
docker run --rm --network none --entrypoint python3 "$image" \
  -S -B /opt/sparkring/bin/verify-performance.py
```

The [public image contract](../../runtime/glm53-spark-mtp3-mesh/performance/public-image.json)
is the renderer and installer input for this composition. Do not substitute the
compute parent's `image-receipt.json`. Source builds use
[the performance build instructions](../../runtime/glm53-spark-mtp3-mesh/performance/README.md).

## Extract and render

Choose unused artifact and export-container names. The following commands do
not start a model. Create `/srv/sparkring/artifacts` with operator permissions:

```bash
docker create --name mtp3-artifact-export "$image"
docker cp mtp3-artifact-export:/opt/spark-sircl /srv/sparkring/artifacts/cache-checkpoints-bundle
docker cp mtp3-artifact-export:/opt/sparkring/bin/mlx5-rdma-tx-marker /srv/sparkring/artifacts/mlx5-rdma-tx-marker
docker rm mtp3-artifact-export
cp runtime/glm53-spark-mtp3-mesh/performance/public-image.json /srv/sparkring/verified-image-receipt.json
```

Edit the private site and fabric inputs described in the
[managed deployment guide](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md).
Set the site's bundle root to the extracted bundle and its marker path to the
extracted marker. The marker SHA-256 is
`2828c07e4255c4962c77425be2c88969e7eb7dd4b1bf9e36485bc705bb5d6d64`.
Use a distinct container prefix and cache directory; do not rename entries from
another cache namespace into this one.

```bash
python runtime/glm53-spark-mtp3-mesh/profile.py render \
  --site /srv/sparkring/site/mtp3-mesh.json \
  --bundle /srv/sparkring/artifacts/cache-checkpoints-bundle \
  --image-receipt /srv/sparkring/verified-image-receipt.json \
  --output build/mtp3-cache-checkpoints-launch
```

The renderer selects the matching image, transport manifest, optimized native
placement digest, and cache namespace. Review the generated rank environments
and plan, then follow the managed guide's create-only container, installation,
and coordinated startup steps with this launch directory and verified receipt.
Do not use direct `docker start` to bypass the memory and four-rank gates.

## Source-build liveness differences

The immutable image above retains its packaged scheduler-liveness implementation.
Source builds install and attest the liveness module alongside the serving wrapper;
see [liveness packaging and timeout policy](../../runtime/glm53-spark-mtp3-mesh/performance/README.md#scheduler-liveness-packaging-and-timeout-policy).
That source-build module uses output movement or increasing KV allocation
to detect progress, with a 300-second inactivity default. A raw output gap
alone does not mark an allocating prefill unhealthy. Fully preallocated work
can remain flat while progressing. Set the private site
`liveness_output_seconds` field to an appropriate integer timeout and regenerate
launch inputs before using its HTTP 503 response for router removal or automatic
recovery. Omitting the field retains 300 seconds; the override cannot add the
rule to an image that lacks it.
This is a source-build behavior change, not an update to the published image.

## Runtime behavior and limits

SparkCache revision `48bbd2be4a7b972e56632a2d7b934bac5460f272` provides bounded
restoration, tiled CUDA placement, authenticated history reconstruction,
publication dependency protection, and publication-backlog gauges. Periodic
full captures remain off by default. They trade more writes for shorter
history processing and require a separately selected workload configuration.

The profile retains 24 GiB KV capacity per rank, an 8,192-token batch budget,
40 GiB persistent-cache maximum and 32 GiB low watermark. Those capacity
defaults are not evidence of a 40 GiB multimodal stress test. The recorded
cache-pressure soak used 2 GiB per rank and an opt-in full-capture cadence.

The image includes explicit two-checkpoint prefill, complete convolution-state
exports, stream-ordered transport, CTA publication synchronization, delayed
proxy-thread startup, and linked payload/doorbell posts. Checkpoint source and
ownership hashes are preserved by the build rather than inferred from labels.

GLM-5.3 supports `reasoning_effort=low`, `high`, or `max`. Explicit thinking-off
chat requests return HTTP 400; warmup uses supported reasoning settings. This
does not implement reasoning-free generation. Responses API behavior is not
qualified by the chat-completion checks.

Run bounded semantic, repeated-prefix, multimodal, and idle-probe checks after
deployment. Neither source equivalence nor the parent's measurements qualify
unattended availability, every cache boundary, or a universal speedup.
