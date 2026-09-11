`published-74275757-manifest.json.gz` contains the unmodified context manifest
for the image in [publication.json](../publication.json). Decompression must
produce its recorded `manifest_sha256`. The fixture contains file identities
and build metadata, not model weights or native binaries.

CPU regressions compare the default lock with that published manifest and
reconstruct its startup and profile archives from repository files. They also
check the embedded verifier tools. This guards published-image compatibility
without downloading or building the complete image context in CI. Update the
fixture from the actual prepared context when publishing another default image.
