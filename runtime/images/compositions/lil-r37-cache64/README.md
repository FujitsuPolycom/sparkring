# R37 hybrid-cache extension

Status: **research-only**. This composition adds SparkCache's 64-group capture
library, profile-aware prefix hashing and Qwen hybrid profile to the published
R37 image. It changes no vLLM, B12X, NCCL, RoCEnante or SIRCL source. The
[descriptor](descriptor.json) pins the parent receipt, source inventory and
native binary. [Qwen validation](../../../../performance/records/qwen38-flash-next/r37-sparkcache.json)
states the tested conditions and limitations.

The extension image is built locally; no registry publication is implied.
Keep the published R37 image for cache-disabled rollback.

## Prepare the source and context

Run from the SparkRing repository root. Use a separate SparkCache checkout.
The included patch applies to the public base commit and includes regression
tests. Do not apply it over unrelated working-tree changes.

```bash
REPO=$PWD
git clone https://github.com/FujitsuPolycom/sparkcache.git ../sparkcache-cache64
git -C ../sparkcache-cache64 checkout --detach f220230a5a85b94af8a296187241b6aacc3ed724
git -C ../sparkcache-cache64 apply "$REPO/integrations/sparkcache/qwen38-r37.patch"
python3 runtime/images/cache_extension.py prepare \
  --descriptor runtime/images/compositions/lil-r37-cache64/descriptor.json \
  --source ../sparkcache-cache64 --output ../r37-cache64-build
```

Preparation refuses changed source, symlinks and a nonempty output directory.
Native source bytes are pinned individually. The preparer reconstructs the
specified LF/CRLF bytes from canonical Git source; it does not accept a change
in source content. CUDA compilation uses a fixed random seed to make generated
symbol names reproducible. Installation rejects a binary whose checksum differs
from the descriptor, even if compilation and layout tests pass.

## Build on each Spark

Use the same prepared context on both Linux ARM64 hosts. The parent contains
the toolchain; model files are not required for building.

```bash
PARENT_ID=sha256:beeb32253aa754cf8e22f7c054b21d659e041388a06c0427473bebe77d564f97
PARENT_REF=ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270
if ! docker image inspect "$PARENT_ID" >/dev/null 2>&1; then
  docker pull --platform linux/arm64 "$PARENT_REF"
fi
docker tag "$PARENT_ID" sparkring-extension-parent:r37-cache64
docker build --pull=false \
  --build-arg BASE_IMAGE=sparkring-extension-parent:r37-cache64 \
  -t sparkring-cache-extension:r37-cache64 ../r37-cache64-build
IMAGE_ID=$(docker image inspect --format '{{.Id}}' sparkring-cache-extension:r37-cache64)
docker run --rm --network none "$IMAGE_ID" verify
```

The build checks inherited files before replacement, runs the native C++ layout
test, checks the rebuilt library checksum and advertised capacities, and verifies
the R37 connector-job source contract. The installed receipt records the exact
extension descriptor and replacement hashes. Docker metadata IDs may differ
between independent builds; launcher admission checks their identical pinned
runtime contents rather than accepting the mutable tag as evidence.

Use the [Qwen SparkCache instructions](../../../../profiles/qwen38-flash-next-tp2/SPARKCACHE.md)
to create serving containers. Build verification alone does not qualify serving.
