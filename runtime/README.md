# SparkRing runtime artifacts

`runtime/` contains pinned source inputs, image builders, artifact manifests,
and offline verification for serving profiles. It contains no model weights,
site configuration, registry credentials, or live-deployment result.

## Runtime contract

A runtime artifact is reproducible only when its parent image, source commits,
patches, dependencies, generated files, platform, and output identity are
recorded. Builders reject missing inputs and known identity drift rather than
silently substituting local state.

Building an image changes only the builder's container store. It does not place
the image on serving ranks or qualify a deployment. Distribution and serving
require separate operator authorization and evidence.

## Runtime index

The following directories are profile-specific artifacts. Their README files
define exact inputs, output identities, status, and validation.

| Path | Artifact role |
|---|---|
| [`exl3-r7/`](exl3-r7/README.md) | ARM64 EXL3 serving-image builder |
| [`glm53-flash-jj-r7-gb10/`](glm53-flash-jj-r7-gb10/README.md) | Published GB10 runtime and SparkCache image manifests and launcher |
| [`glm53-flash/`](glm53-flash/README.md) | Source-built runtime identity and attestation contract |
| [`glm53-flash-split-page-sparkcache/`](glm53-flash-split-page-sparkcache/README.md) | Split-page SparkCache artifact |
| [`glm53-flash-e10536a/`](glm53-flash-e10536a/README.md) | Embedded-speculation source composition |
| [`glm53-flash-b12x-kda-adaptive-mtp/`](glm53-flash-b12x-kda-adaptive-mtp/README.md) | Adaptive-speculation source composition |
| [`glm53-flash-dflash7-python-overlay/`](glm53-flash-dflash7-python-overlay/README.md) | External-draft source overlay |
| [`deepseek0731-gb10/`](deepseek0731-gb10/README.md) | GB10 model integration layer |
| [`qwen38/`](qwen38/README.md) | Public-source ARM64 runtime builder |

The [profile registry](../docs/profiles/README.md) maps operator-facing
deployments to these artifacts. Historical or research-only artifacts remain
separate from supported operator starts.

## Shared artifact infrastructure

| Path | Purpose |
|---|---|
| [`faststart-lock.json`](faststart-lock.json) | Immutable image, checkpoint, and platform pins |
| [`build-public-overlay.py`](build-public-overlay.py) | Builds the reviewed Python overlay bundle |
| [`public-overlay-files.json`](public-overlay-files.json) | Source-file allowlist for the overlay |
| [`test_public_overlay.py`](test_public_overlay.py) | Offline allowlist and manifest tests |

The public-overlay builder accepts only allowlisted files, places them in a
fixed runtime layout, and writes `sparkring-overlay-manifest.json` with a
SHA-256 entry for every admitted file.

```bash
python runtime/build-public-overlay.py \
  --output build/public-overlay
```

Add an overlay member only by updating the allowlist and testing the generated
manifest.

## Distribution

Use the distribution method named by the selected profile. The
[direct-fabric archive tool](../docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md)
places one checksum-bound archive across a validated cluster. Registry pull
helpers may provide profile-specific alternatives.

Distribution tools require explicit confirmation before changing hosts and
produce identity receipts. Artifact placement does not qualify model serving.

## Validation

Run the offline suites for every runtime directory changed. The complete
runtime contract suite is:

```bash
python -m pytest runtime -q
```

An image build, manifest check, or import test proves only the condition it
measures. Live serving evidence belongs to the selected profile's evidence
record.

## Safety

Overlay generation and tests are **offline**. Building changes the local
container store. Pulling, distributing, starting, or replacing an image on a
configured host mutates that host and requires explicit authorization.
