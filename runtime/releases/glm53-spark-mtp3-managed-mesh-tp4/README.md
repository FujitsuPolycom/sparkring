# Published mesh bundle source inputs

`compatibility-sources.tar.gz` retains the RoCEnante C proxy, Python runtime,
collective kernels, PTX helpers, adapter,
provenance record and bundle manifest needed to reproduce published bundle
`69313e19e881ec93e9ed3bd150d2f24fc6b444488ac729a69f45d038e2243500`.
Its digest is recorded in [release.json](release.json).

The [released-profile composer](../../glm53-spark-mtp3-mesh/profile.py) checks
the archive, source hashes and complete resulting bundle identity. It uses
these frozen files in a temporary source tree while leaving development sources
unchanged. The remaining bundle files come from the pinned base image and
matching repository inputs.

The retained sources keep the released counter, gather-storage and timeout
behavior. This archive reproduces the released artifact; fixes belong to the
development source. Development bundles have a distinct source and
manifest identity and require their own validation. Published image references,
cache namespaces and historical evidence retain their original identities.

The archive contains source and provenance only, without model weights or
native binaries. The B12X source and SparkRing adapter retain their Apache-2.0
licensing; see [third-party notices](../../../THIRD_PARTY_NOTICES.md).
