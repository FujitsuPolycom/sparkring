# Runtime source inputs

Status: **implemented; distribution pending**.

The [manifest](manifest.json) pins the upstream vLLM and B12X commits, complete
patch hashes, accepted source-tree hashes and source-archive hashes. Applying
each patch to its named upstream commit reproduces the accepted source tree;
independent reconstruction was checked during image preparation.

`vllm-sources.tar.gz` and `b12x-sources.tar.gz` contain those complete source
trees. They are release assets, not model weights or container layers. Verify
their SHA-256 values against the manifest before use. Archive publication must
complete before this directory is described as a downloadable release.

The manifest identifies runtime build image `157be9a52296`. Release image
`bc16a9819d85` adds version/provenance labels only; its filesystem and other OCI
configuration are identical. The installed-receipt hash binds the runtime
payload. This source record does not establish serving qualification or a
complete rebuild of inherited native libraries; see the
[component inventory and limitations](../components.md).
