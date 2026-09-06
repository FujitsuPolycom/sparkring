# ARM64 image with native-MTP3 compute and mesh bundle

Status: **research-only**. This child image packages the complete compute and
transport composition used by the GLM-5.3 Spark native-MTP3 profile. On top of
the pinned parent, it installs CUDA 13.3, the checksum-bound vLLM metadata and
proposal-head and loader/RNG patches, B12X revision `ef308bac` with the
source-checked top-k selector, the transport bundle,
the managed marker, and the readiness helper. The proposal head uses runtime
NVFP4 weights with BF16 activations; the target/verifier head retains its BF16
checkpoint representation. SparkCache and patched NCCL remain inherited.

The image sets `SPARKRING_WARMUP_TEMPERATURE=1`, so requests issued before
readiness use temperature one. The managed quickstart requires this child and
its verified receipt. The parent-image rendering mode is a separate composition
interface; it does not meet the compute, managed marker, or temperature-one
warmup contract.

For the pinned GLM vocabulary and TP4 geometry, the proposal head adds about
85.08 MiB of packed NVFP4 values and scales per rank. It does not replace the
retained BF16 target/verifier head, so this is an additional persistent
allocation rather than a net model-memory reduction.

The image contains no model weights. It does not provision NIC rules, select
network interfaces, install host services, or start a model during construction.
The site plan and native-MTP3 launch configuration remain separate inputs.

## Published image

The [public registry receipt](public-image.json) binds the published manifest
to its Linux/ARM64 config-image identity. Pull before using its local image ID:

```bash
set -euo pipefail
mtp_image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987'
mtp_image_id='sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f'
docker pull "$mtp_image"
test "$(docker image inspect "$mtp_image" --format '{{.Id}}')" = "$mtp_image_id"
```

The immutable reference is also published as tag
`glm53-spark-mtp3-c139f3670-mesh69313e19`; use the digest above for deployment.
The [compute-image equivalence record](compute-image-equivalence.json) verifies
that all 4,891 vLLM, 385 B12X, and 150 SparkCache package files and the selected
environment match compute-tested image `sha256:3b4768e5ba31cadcc882dffa06d7b667af44abdf157d5c11b7ac7fe962e80c43`.
The published config-image ID also passed its own
[GPU, serving, restart, and persistent-cache checks](../../performance/records/glm53-flash/spark-mtp3-compute-stream-safety-20260906.md).

Use the repository's [content receipt](image-receipt.json) for the renderer,
installer, and native qualification runner. This default deployment requires
no compiler or local image rebuild. The source reproduction commands below
are optional; skip to **Use and export** when using the published image.

## Inputs and outputs

| Object | Identity or location |
|---|---|
| Parent runtime | `operator_image` in `../glm53-flash-jj-r8-gb10/pins.json` |
| Compute source lock and vLLM patch | [`compute/source-lock.json`](compute/source-lock.json), [`compute/vllm-e02-to-compute.patch`](compute/vllm-e02-to-compute.patch) |
| Prepared CUDA 13.3 and B12X source payload | output of [`compute/prepare_compute_source.py`](compute/prepare_compute_source.py) |
| Target, speculation, transport, and marker pins | `pins.json` in this directory |
| Embedded transport bundle | `/opt/spark-sircl` |
| Compiled RDMA transmit marker | `/opt/sparkring/bin/mlx5-rdma-tx-marker` |
| Marker source | `/opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c` |
| Construction receipts and third-party license | `/opt/sparkring/receipts/glm53-spark-mtp3-mesh` |
| Image verification program | `/opt/sparkring/bin/verify-mtp3-mesh-image.py` |
| Readiness warmup helper | `/opt/sparkring/bin/warmup_dflash.py`; source hash and temperature recorded in the image receipt |

The marker is compiled with `cc -O2 -Wall -Wextra` and linked against
`libibverbs` and `libmlx5` supplied by the parent image. Its binary hash is
recorded in the image receipt; it is not assumed to match a binary built with
another toolchain. Merely extracting or invoking its `--help` command does not
install forwarding rules. Actual use requires the separately authorized,
device-scoped fabric plan and the [managed host service](MANAGED_MESH.md).
That service configures the hardware-forwarded ring paths, gates model
startup on four-rank readiness, and coordinates shutdown or recovery.
The bounded helper mode is for isolated diagnostics, not the serving lifecycle.

## Optional source reproduction

Use a Linux/ARM64 Docker host. Check available disk and RAM before pulling the
parent image. Do not remove resident model files or stop serving containers to
make room for this build.

From the repository root, first prepare the checksum-bound compute source. This
step downloads the pinned public CUDA component archives and Local Inference
Lab B12X source before the network-disabled image build. The output and optional
download cache are local build inputs and must use absent or empty paths:

```bash
python3 runtime/glm53-spark-mtp3-mesh/compute/prepare_compute_source.py \
  --output /var/tmp/mtp3-compute-source \
  --cache /var/tmp/mtp3-compute-downloads
```

The preparer verifies every archive checksum, B12X tree, vLLM patch hash, and
package file and writes `prepared-manifest.json`. Then pull the pinned parent
and compose its SIRCL bundle. Do not use the already-composed published child
as the parent input. Use absent output paths and an unused temporary container
name:

```bash
mtp_parent='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f'
docker pull "$mtp_parent"
test "$(docker image inspect "$mtp_parent" --format '{{.Id}}')" = \
  'sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075'
mkdir -p /var/tmp/mtp3-base-sircl
docker create --name mtp3-base-extract --entrypoint /bin/true "$mtp_parent"
docker cp mtp3-base-extract:/opt/spark-sircl/. /var/tmp/mtp3-base-sircl/
docker rm mtp3-base-extract
python3 runtime/glm53-spark-mtp3-mesh/profile.py bundle \
  --base-sircl /var/tmp/mtp3-base-sircl --output /var/tmp/mtp3-mesh-bundle
```

Then prepare the content-verified build context:

```bash
python3 runtime/glm53-spark-mtp3-mesh/build_image.py prepare \
  --bundle /var/tmp/mtp3-mesh-bundle \
  --compute-context /var/tmp/mtp3-compute-source \
  --context /var/tmp/mtp3-mesh-image-context
```

The context path must not exist. Preparation copies only manifest-listed
compute and transport files, source-pinned marker code, verification code,
pins, and the RoCEnante license and provenance. It rejects unexpected bundle
or compute files and writes a content manifest for every construction input.

Build and verify without loading a model:

```bash
python3 runtime/glm53-spark-mtp3-mesh/build_image.py build \
  --context /var/tmp/mtp3-mesh-image-context \
  --image sparkring-glm53-spark-mtp3-mesh:4204fabc \
  --receipt /var/tmp/mtp3-mesh-image-receipt.json \
  --pull-parent
```

Omit `--pull-parent` when the exact pinned parent is already local. The builder
requires its immutable image ID and Linux/ARM64 platform. It does not push an
image to a registry. The local tag is a convenience; use the receipt's full
`image_id` to identify the result.

Image construction has networking disabled. Verification runs with no host
device mounts, no Linux capabilities, no network, a read-only root filesystem,
two CPUs, and a 2 GiB memory limit. It checks the complete parent layer prefix,
package source and native-library hashes, CUDA 13.3 components, the complete
B12X `ef308bac` package with selector overrides, vLLM input/output hashes and proposal-head environment,
bundle hashes, Python syntax, readiness warmup helper hash and temperature, lazy
RoCEnante import, and marker linkage. CUDA must remain uninitialized.

## Use and export

Pass `--image-receipt runtime/glm53-spark-mtp3-mesh/image-receipt.json` to the profile's
`render` command to select the verified child image. Rendering checks its
parent identity, bundle identity, platform, and completed CPU checks. The
managed installer additionally requires the pinned managed marker source,
matching extracted binary, and temperature-one readiness attestation. For a
local reproduction, pass its generated receipt instead and retain its actual
image ID; do not relabel a rebuilt image as the published image. Without
the receipt option, rendering selects the parent image, which is not accepted
by the managed quickstart installer.

Both modes retain the read-only canonical bundle mount at `/opt/spark-sircl`.
This is redundant for the child image, but preserves the same launch contract
and an independently inspectable host bundle. Load the child on every serving
host before executing a generated launch. Do not replace the parent pin with
a child tag or disable identity checks.

To extract the packaged bundle without starting the container, create an
unused artifact directory outside `/opt/sparkring/managed-mesh`. That service
source directory must not exist before first installation:

```bash
mtp_child_image_id=$(python3 -c 'import json; print(json.load(open("runtime/glm53-spark-mtp3-mesh/image-receipt.json"))["image_id"])')
mkdir -p /srv/sparkring/artifacts
container_id=$(docker create --entrypoint /bin/true "${mtp_child_image_id}")
docker cp "$container_id:/opt/spark-sircl" /srv/sparkring/artifacts/mtp3-mesh-bundle
docker cp "$container_id:/opt/sparkring/bin/mlx5-rdma-tx-marker" /srv/sparkring/artifacts/mlx5-rdma-tx-marker
docker rm "$container_id"
```

Use a distinct destination if either output already exists. Extracted marker
linkage must be checked against the destination host's RDMA libraries before
attaching any rule; the image's linkage check does not establish host ABI
compatibility.

Set `marker_binary_sha256` in the site JSON to the SHA-256 of the exact helper
that will run on that host. The image receipt records the packaged helper's
hash as `inside_image.marker_binary_sha256`. The observed helper hash in
`pins.json` identifies a different build and is not interchangeable with an
extracted or rebuilt helper. Host attachment of the packaged helper remains
untested by image verification.

For private distribution, `docker save --output /path/to/image.tar IMAGE_ID`
and `docker load --input /path/to/image.tar` preserve the image identity. The
archive includes parent layers and can require tens of GiB of space. Verify
the loaded immutable ID on every host. Alternatively, pull the published
manifest directly on every host. The manifest digest, not its convenience
tag, identifies the deployment.

CPU checks establish packaging correctness only. Four-rank startup, graph
capture, RDMA forwarding, MTP output correctness, persistent-cache restoration,
failure containment, and model throughput require separate serving evidence.
