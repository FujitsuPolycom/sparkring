# Publish the GLM-5.3 ARM64 runtime

Status: **implemented** for private GHCR publication. A registry digest remains
unqualified until four DGX Spark ranks run that exact image through the
TP4/DCP1 checks.

Build and verify the image according to [`BUILD.md`](BUILD.md). Generate an
SPDX JSON SBOM with Syft:

```bash
syft scan sparkring-glm53-runtime:da4d7be-source-arm64 \
  --output spdx-json=glm53-runtime.spdx.json
```

Authenticate to GHCR with `write:packages`, then publish privately:

```bash
python runtime/glm53-flash/publish_image.py \
  --image sparkring-glm53-runtime:da4d7be-source-arm64 \
  --destination ghcr.io/fujitsupolycom/sparkring-glm53-runtime:da4d7be-source-arm64 \
  --build-receipt glm53-runtime-image-receipt.json \
  --sbom glm53-runtime.spdx.json \
  --output glm53-runtime-publication-receipt.json
```

The receipt records the local image ID, registry digest, source-bound build
receipt, and SPDX SBOM. The semantic tag is a convenience name; operator
configuration uses only `registry_digest`.

Keep the GHCR package private until all of these conditions hold:

- the runtime contains no model weights, site configuration, credentials, or
  cache data;
- every source, patch, Git tree, parent digest, license, and build setting is
  present in `pins.json` and the build receipt;
- the SparkCache overlay is built from this runtime digest;
- the same overlay digest is pulled on all four ranks; and
- cache-enabled and cache-disabled qualification receipts pass.

Changing a GHCR package from private to public is irreversible. The public
package must retain the NVIDIA parent license and every license and notice
described in [`LICENSES/README.md`](LICENSES/README.md).
