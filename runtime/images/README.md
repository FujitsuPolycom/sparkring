# Image construction

Installing a published deployment? Read [what the image contains](../../docs/operations/images.md)
and follow [setup](../../docs/operations/setup.md). The build commands below are
for developers; a published profile does not require a local image build.

Use `python3 scripts/build_image.py BUILDER -- BUILDER_ARGUMENTS` to print a
build command. Add `--execute` before the builder name only when you intend to
run that build on the current host. Building consumes resources and may download
dependencies; the default command planner performs neither action.
The plan is JSON. With `--execute`, builder output follows it on the same
streams; the combined output is not a single JSON document.

[builders.json](builders.json) selects existing pinned implementations. GLM,
DeepSeek and Qwen engine versions are not interchangeable. Shared image context
assembly and previously published build scripts stay at their identity-bound
paths; moving them would change relative build inputs and source receipts.
This directory owns builder selection. The selected implementation owns the
actual container assembly and its release-specific options.

A successful local build is not a published or hardware-qualified image.
See the [release procedure](../../docs/development/releases.md).

For isolated upstream catch-up trials, use the
[bounded upgrade runner](upgrades/README.md). It discovers source changes,
reconciles approved patches and can build explicitly authorized candidates.
It does not change published image selections or serving deployments.

## Container images

[Shared runtime 2026.09.3](../releases/shared-2026.09.3/README.md) is selected by
Qwen TP2/TP4 with and without SparkCache, and GLM TP2/TP4 DCP1 with SparkCache.
GLM Spark and QAD are separate checkpoint choices. The release record covers
bounded text/media and restart checks, including all-rank persistent-cache
restore. It retains full-context, stability and other-model limits.

The [shared ARM64 serving candidate](../releases/shared-2026.09.0-rc.1/README.md)
publishes pinned vLLM/B12X integration and isolated SGLang, with a verified
anonymous digest pull;
[bounded Qwen TP2/TP4 text and SparkCache restart/restore checks passed](../../performance/records/qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md).
A [DeepSeek-0731 TP2 check](../../performance/records/deepseek-v4-flash/shared-22da81ca-tp2-smoke-20260918.md)
also covers startup and one short text request without SparkCache; an isolated
SGLang TP4 check covers the same bounded scope. GLM restore results retain
unresolved limitations, not a production recommendation. Existing quickstart image selections
are unchanged. The matching GitHub prerelease provides the version landing page;
GHCR Packages stores the image. The candidate guide owns its digest, scope and limitations.

| Package / runtime | Profile | Details |
|---|---|---|
| SparkRing shared ARM64 `shared-2026.09.0-rc.1` | Bounded Qwen TP2/TP4 text/cache and DeepSeek text checks; GLM limitations retained | [Digest pull, source records and qualification](../releases/shared-2026.09.0-rc.1/README.md) |
| Local SparkRing vLLM + SGLang composition | DeepSeek-V4.1-Flash SGLang passed bounded TP4 checks; Qwen vLLM payload preserved | [Runtime isolation, build inputs and validation scope](../deepseek-v41-sglang/combined-image/README.md) |
| Published SparkRing R37 ARM64 | GLM-5.3-Flash TP4/DCP1 cache-on bounded checks; other selections experimental | [Published composition and quickstarts](compositions/lil-r37-glm-spark/README.md) |
| Published SparkRing R35 ARM64 | GLM-5.3-Flash TP2/TP4 fallback; bounded checks do not establish long-duration stability | [Pull and launch instructions](../../docs/operations/r35-local-launch.md) |
| Locally built SGLang / Mia adapter | DeepSeek-V4.1-Flash TP4/EP4; no public image | [Source-pinned build](../deepseek-v41-sglang/README.md#build-and-prepare) |
| SparkRing R33 ARM64 | Exact profiles select TP2/TP4 topology and optional components | [Source build and profile verification](../../runtime/sparkring/jovian-r33/image/README.md) |
| `gb10-vllm-serving` | Profile-specific images, including DeepSeek | [Packages](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) |
| Anemll `dspark-vllm-gx10` | DeepSeek-V4-Flash-Vision-Exp with the MiaAI-Lab recipe | [Image, recipe, and transport provenance](../../runtime/deepseek-vision-exp/profile.json) |

Use the exact digest in the selected quickstart. Images sharing a package
name are not interchangeable; a model-neutral name does not qualify every profile.
Retired profiles retain their image references in their linked guides.
The [R33 publication record](../../runtime/sparkring/jovian-r33/publication.json)
contains the download digest and profile verification scope.

The shared SGLang composition adds its runtime beside vLLM. The selected profile
chooses the Python environment, CUDA tools and NCCL library. It does not run both
model servers on the same GPU or imply that every model supports both engines.

## Bounded Python source extensions

[source_extension.py](source_extension.py) packages a pinned Git patch over a
SparkRing image that has an installed receipt. It accepts Python files under
`b12x` and `vllm`, plus additional JSON integration contracts. It preserves
native libraries, feature hooks, package versions, and inherited receipt metadata.
GPU kernels defined in Python can still require compilation during warmup;
source verification does not establish GPU compatibility or serving correctness.

A `sparkring-source-extension/v1` descriptor records:

- The parent image ID and exact installed-receipt SHA256.
- The installer SHA256, patch path and SHA256, and upstream provenance.
- Every changed package path, its inherited SHA256, and its resulting SHA256.
  A null inherited digest declares a file addition.
- Additional contract paths, repository sources, and SHA256 digests.

Prepare an isolated build context from a descriptor and this repository:

```bash
python runtime/images/source_extension.py prepare \
  --descriptor /path/to/descriptor.json \
  --repository /path/to/sparkring \
  --output /path/to/source-context
```

The context contains the compact patch, contracts, descriptor, installer, and a
generated Dockerfile. The Dockerfile accepts `PARENT_IMAGE` for a locally cached
parent tag. Before building, compare that tag's `docker image inspect` ID with
the descriptor's `parent.image_id`; installation independently verifies the
pinned receipt and every inherited file. It does not download application source.

Installation applies the patch to a temporary Git tree and verifies all resulting
files before writing the image. It rejects undeclared paths, symlinks in write
paths, native-file changes, deletions, renames, file-mode changes, and overwrites
without inherited ownership. All admission checks precede installation writes;
an interrupted filesystem write fails the build layer.

The installed entry point is `/opt/sparkring/bin/source-extension.py`:
`verify` checks the complete resulting inventory and package versions; `serve`
performs the same verification before invoking the vLLM CLI. Deployment profiles
select optional capabilities and their integration contract. Serving requires
no host source mounts.

The receipt retains the parent inventory and records the extension separately.
The original receipt, patch, and descriptor remain under
`/opt/sparkring/receipts/source-*`. Build each extension from its recorded parent;
the installer rejects applying another extension over an already extended image.
This keeps each recipe's source changes relative to an explicit shared base.

## Derived installer layers

[derived_layer.py](derived_layer.py) derives an installer image that adds or
replaces a few Python files in the serving interpreter's site-packages, in one
layer over a pinned installer image. The installer image lock pins the SHA-256
of two receipts inside the image: the external-base receipt, whose file map the
image's `verify` checks, and the toolchain receipt, which records the
external-base receipt. The derived layer records each file in the first,
re-records the second, and adds a provenance receipt that lists every path with
its inherited and resulting SHA-256.

A `sparkring-derived-layer-descriptor/v1` descriptor names the parent installer
lock, each image path with its repository source, pinned SHA-256 and inherited
SHA-256 (`null` for an addition), the provenance receipt path and the layer's
purpose. The builder refuses native libraries, the startup hooks that select the
transport, feature and status packages, and the B12X sources that the prepared
RoCE transport verifies at startup; changing those needs a new transport
manifest or composition instead.

1. On a host that holds the parent image, copy its two receipts:

   ```bash
   docker run --rm --pull never --network none --entrypoint cat PARENT_IMAGE_ID \
     /opt/sparkring/receipts/external-base-installed.json > base.json
   docker run --rm --pull never --network none --entrypoint cat PARENT_IMAGE_ID \
     /opt/sparkring/toolchain/installed.json > toolchain.json
   ```

2. Prepare the context offline. `prepare` checks both receipts against the
   parent lock and every source against its pinned SHA-256:

   ```bash
   python3 runtime/images/derived_layer.py prepare \
     --descriptor runtime/images/compositions/DESCRIPTOR_DIRECTORY/descriptor.json \
     --base-receipt base.json --toolchain-receipt toolchain.json --output CONTEXT
   ```

3. Build. BuildKit resolves a bare image ID as a registry name, so give the
   parent a local tag first:

   ```bash
   docker tag PARENT_IMAGE_ID sparkring-dev/parent:PARENT_ID_PREFIX
   docker build --build-arg PARENT_IMAGE=sparkring-dev/parent:PARENT_ID_PREFIX -t TAG CONTEXT
   ```

4. Record the lock. `record` confirms that the parent has none of the added
   paths, runs the installer's admission for every profile of the lock,
   including the image's isolated `verify`, and writes the lock:

   ```bash
   python3 runtime/images/derived_layer.py record --context CONTEXT \
     --image BUILT_IMAGE_ID --name RELEASE --output LOCK
   ```

The written lock is a development lock: its `image_reference` is the local
configuration ID and its `download_bytes` is an upper bound. Publication
replaces both from the registry and adds the release's `publication.json` and
`release.json` ([release procedure](../../docs/development/releases.md)). No
step selects the image for a profile.

| Descriptor | Parent | Adds | Status |
|---|---|---|---|
| [installer-b12x-selection-cache](compositions/installer-b12x-selection-cache/descriptor.json) | `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` | [B12X reconciled selection-cache correction](../../integrations/b12x/selection_cache/README.md) | No image built |
