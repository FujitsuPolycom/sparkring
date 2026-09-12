# Image construction

Use `python scripts/build_image.py BUILDER -- BUILDER_ARGUMENTS` to print a
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

## Container images

| Package / runtime | Profile | Details |
|---|---|---|
| Locally built SGLang / Mia adapter | DeepSeek-V4.1-Flash TP4/EP4; no public image | [Source-pinned build](../deepseek-v41-sglang/README.md#build-and-prepare) |
| `ghcr.io/fujitsupolycom/sparkring` | Generic R33 ARM64 image; exact profiles select TP2/TP4 topology and optional components | [Source build and profile verification](../../runtime/sparkring/jovian-r33/image/README.md) |
| `gb10-vllm-serving` | Profile-specific images, including DeepSeek | [Packages](https://github.com/users/FujitsuPolycom/packages/container/package/gb10-vllm-serving) |
| Anemll `dspark-vllm-gx10` | DeepSeek-V4-Flash-Vision-Exp with the MiaAI-Lab recipe | [Image, recipe, and transport provenance](../../runtime/deepseek-vision-exp/profile.json) |

Use the exact digest in the selected quickstart. Images sharing a package
name are not interchangeable; a model-neutral name does not qualify every profile.
Retired profiles retain their image references in their linked guides.
The [R33 publication record](../../runtime/sparkring/jovian-r33/publication.json)
contains the download digest and profile verification scope.
