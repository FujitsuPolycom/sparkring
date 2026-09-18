# Container images and downloads

Start with the [model profile catalog](../../profiles/README.md), choose the
model and node count, then follow that quickstart's exact image selection.
Do not select a serving image from a package's publication date or GitHub's
**Latest** release badge. Those labels do not establish profile compatibility.

## Packages, versions and releases

| Object | Purpose | How operators use it |
|---|---|---|
| GitHub container package | A named family of images in GHCR, such as `sparkring` | Follow the selected profile; the family name alone is insufficient |
| Image tag or registry digest | A version label or content-addressed image selection within a package | Pull the digest pinned by the quickstart; a mutable tag is not an equivalent pin |
| GitHub Release | Notes and downloadable assets attached to a Git tag | Download a host tool or build input only when the guide explicitly requires it |
| Repository `runtime/releases/` selection | A profile's published contract or pinned builder | Resolved by repository tooling; not a separate GitHub download channel |

The package count in GitHub's sidebar counts families, not individual image
versions. Model weights, site configuration and host fabric provisioning remain
separate from the serving image unless a profile explicitly states otherwise.

## Container images

SparkRing publishes through several retained package names. A model-neutral
name does not mean every model, topology, cache mode or backend is qualified.

| Package family | Role | Selection guidance |
|---|---|---|
| [`sparkring`](https://github.com/FujitsuPolycom/sparkring/pkgs/container/sparkring) | Shared ARM64 runtime compositions, including R33, R35 and R37 selections | Use the profile's digest and feature settings; see the compositions below |
| [`gb10-vllm-serving`](https://github.com/FujitsuPolycom/sparkring/pkgs/container/gb10-vllm-serving) | Separately pinned serving builds, including DeepSeek | Use the [DeepSeek serving guide](../../docs/operations/deepseek-0731.md), not a shared-image substitution |
| [`sparkring-glm53-runtime`](https://github.com/FujitsuPolycom/sparkring/pkgs/container/sparkring-glm53-runtime) | Retained GLM runtime/base compositions | Still referenced by [GLM runtime pins](../glm53-flash/pins.json); retain for dependent builds and reproduction |
| [`sparkring-glm53-sparkcache`](https://github.com/FujitsuPolycom/sparkring/pkgs/container/sparkring-glm53-sparkcache) | GLM/SparkCache compositions retained by pinned deployments and build inputs | Dependencies include the [mesh publication](../glm53-spark-mtp3-mesh/public-image.json) and [shared source-image lock](../sparkring/source_image/glm53-tp4-lock.json) |

These roles are not package-wide validation statuses. Each linked profile or
composition records its own evidence and limitations. Retained names and
digests are compatibility dependencies, not disposable duplicates.

### Shared compositions and separate runtimes

| Selection | Guide and scope |
|---|---|
| R37 shared feature image | [Published composition](compositions/lil-r37-shared/README.md); profiles select optional Qwen features and their qualified topology |
| R37 hybrid-cache extension | [Published cache64 composition](compositions/lil-r37-cache64/README.md); Qwen hybrid-cache evidence is separate from base-image checks |
| R37 GLM composition | [Published composition and quickstarts](compositions/lil-r37-glm-spark/README.md); bounded GLM TP4/DCP1 cache-on evidence |
| R35 ARM64 | [Pull and launch instructions](../../docs/operations/r35-local-launch.md); profile-specific fallback evidence |
| R33 ARM64 | [Source build and verification](../sparkring/jovian-r33/image/README.md); preserved [publication record](../sparkring/jovian-r33/publication.json) |
| SGLang / Mia adapter | [Source-pinned build](../deepseek-v41-sglang/README.md#build-and-prepare); a separate runtime, not a public combined-image recommendation |
| Anemll `dspark-vllm-gx10` | [Vision recipe and provenance](../deepseek-vision-exp/profile.json); external image selection |

## Host tools and build downloads

The [GitHub Releases page](https://github.com/FujitsuPolycom/sparkring/releases)
also distributes non-container artifacts:

- [R33 host forwarding tool](https://github.com/FujitsuPolycom/sparkring/releases/tag/r33-host-tools-c8646b0):
  the ARM64 `mlx5-rdma-tx-marker` executable maintains a hardware-forwarding
  marker on the host. It is not a serving image.
- [SM121 native runtime inputs](https://github.com/FujitsuPolycom/sparkring/releases/tag/native-runtime-sm121-aa8fa11831af):
  a checksummed NCCL/SparkCache library archive with provenance and license
  material for image assembly. It is not a ready-to-run model service.

A **Latest** badge on either download identifies GitHub Release metadata; it
does not select a Docker image or supersede a profile's pinned runtime.
Keep asset filenames, tags, hashes and URLs intact wherever builds depend on them.

## Image construction

Use `python3 scripts/build_image.py BUILDER -- BUILDER_ARGUMENTS` to print a
build command. Add `--execute` before the builder name only when you intend to
run that build on the current host. Building consumes resources and may download
dependencies; the default command planner performs neither action.
The plan is JSON. With `--execute`, builder output follows it on the same
streams; the combined output is not a single JSON document.

[builders.json](builders.json) selects pinned implementations. Engine versions
are not interchangeable. Identity-bound build scripts and source receipts stay
at their recorded paths; this directory owns selection, not a second copy of
their build logic. A local build is neither a publication nor serving
qualification. Follow the [release procedure](../../docs/development/releases.md).
