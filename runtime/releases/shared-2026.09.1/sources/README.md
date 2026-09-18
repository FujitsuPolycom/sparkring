# Exact serving source inputs

The [manifest](manifest.json) pins upstream commits, complete patches, accepted
source-tree hashes and full-source archive hashes. Independent reconstruction
applied each patch and reproduced the installed source identity. The tree hash
is SparkRing's file/mode inventory digest, not a Git commit.

Reconstruction excludes `.agents`, `.claude`, Git metadata and Python caches.
The patches retain licensing and attribution. They include sparse-attention
selection, multimodal forwarding, the startup audit and request-ID padding
initialization; they contain no model weights or deployment benchmark results.

Use the [image-upgrade tooling](../../../images/upgrades/README.md) to verify
source acceptance and native-wheel reuse. Runtime changes require their own
installed-source and cache compatibility records. Do not edit running containers
or relabel their contracts after changing source.
