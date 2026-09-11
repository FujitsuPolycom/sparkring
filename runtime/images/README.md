# Image construction

Use `python scripts/build_image.py BUILDER -- BUILDER_ARGUMENTS` to print a
build command. Add `--execute` before the builder name only when you intend to
run that build on the current host. Building consumes resources and may download
dependencies; the default command planner performs neither action.

[builders.json](builders.json) selects existing pinned implementations. GLM,
DeepSeek and Qwen engine versions are not interchangeable. Shared image context
assembly and previously published build scripts stay at their identity-bound
paths; moving them would change relative build inputs and source receipts.
This directory owns builder selection. The selected implementation owns the
actual container assembly and its release-specific options.

A successful local build is not a published or hardware-qualified image.
See the [release procedure](../../docs/development/releases.md).
