# Frozen vLLM and B12X source inputs

Status: **implemented source reconciliation**. The [manifest](manifest.json)
binds upstream commits, exact patches and accepted source-snapshot digests.
These files reconstruct source changes; they do not announce an available image,
change profile defaults, or establish full upstream-suite compatibility.

## Reconstruct the sources

From a SparkRing checkout, choose an empty working directory outside the
repository. The commands create separate upstream checkouts and preserve their
license files and contributor notices:

```bash
SPARKRING_REPO="$PWD"
SOURCE_WORK=/path/to/empty/workspace
PATCHES="$SPARKRING_REPO/runtime/releases/shared-2026.09.0-rc.1/sources"

git clone --no-checkout https://github.com/local-inference-lab/vllm.git "$SOURCE_WORK/vllm"
git -C "$SOURCE_WORK/vllm" -c core.autocrlf=false checkout --detach 35bab057b1751a6076a457803bcc4b78809689cf
git -C "$SOURCE_WORK/vllm" apply --check "$PATCHES/vllm.patch"
git -C "$SOURCE_WORK/vllm" apply --index "$PATCHES/vllm.patch"

git clone --no-checkout https://github.com/local-inference-lab/b12x.git "$SOURCE_WORK/b12x"
git -C "$SOURCE_WORK/b12x" -c core.autocrlf=false checkout --detach a83336581a3a907076e60797df69ab66df5a2ff1
git -C "$SOURCE_WORK/b12x" apply --check "$PATCHES/b12x.patch"
git -C "$SOURCE_WORK/b12x" apply --index "$PATCHES/b12x.patch"
```

Verify patch SHA256 values against the manifest before applying. Image assembly
uses the [upgrade runner](../../../images/upgrades/README.md), not an unrecorded
Python-file replacement in a running container.

## Identity boundaries

The runner's snapshot digest includes file bytes, paths and executable flags,
after excluding upstream `.agents` and `.claude` metadata. Git metadata is not
included. It is **not a Git tree ID**; a plain checkout containing those excluded
directories is not the same materialized snapshot. Use the runner's recorded
materialization and inventory procedure when comparing accepted digests.

Likewise, a Docker image configuration ID is not a registry manifest digest.
Source pins or snapshot digests cannot substitute for a pullable image identity.
The [release record](../README.md) owns publication status.

## Qualification scope

The [bounded GPU harness and reference fixtures](../qualification/README.md)
are versioned separately. Some carried upstream tests still use retired APIs;
the source patches do not imply that the complete upstream test suite passes.
The harness adapts frozen-resolution and prepared RoPE fixtures without weakening
production checks. Its manifest distinguishes artifact identity from executed
test receipts. See the [component index](../components.md) for inherited runtimes,
native libraries, license texts and attribution.
