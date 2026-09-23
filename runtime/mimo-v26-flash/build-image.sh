#!/bin/bash
# Assemble pinned Python packages over the SparkRing ARM64 native runtime.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:-sparkring:mimo-b12x-6afb999-4f3028}"
[ "$(uname -m)" = aarch64 ] || { echo "Build on an ARM64 Spark host" >&2; exit 1; }
CONTEXT="$(mktemp -d /var/tmp/sparkring-mimo-build.XXXXXX)"
mkdir -p "$CONTEXT/provenance"
cp "$HERE/Dockerfile" "$CONTEXT/Dockerfile"
cp "$HERE/b12x-image.json" "$CONTEXT/provenance/b12x-image.json"
for package in vllm b12x; do
  pin="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["revision"])' "$HERE/b12x-image.json" "$package")"
  repo="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["repository"])' "$HERE/b12x-image.json" "$package")"
  git init -q "$CONTEXT/source-$package"
  git -C "$CONTEXT/source-$package" fetch --depth 1 "$repo" "$pin"
  git -C "$CONTEXT/source-$package" checkout --detach FETCH_HEAD
  [ "$(git -C "$CONTEXT/source-$package" rev-parse HEAD)" = "$pin" ] || exit 1
  cp -a "$CONTEXT/source-$package/$package" "$CONTEXT/$package"
  mkdir -p "$CONTEXT/provenance/$package"
  for notice in LICENSE LICENSE.txt NOTICE NOTICE.txt; do
    if [ -f "$CONTEXT/source-$package/$notice" ]; then
      cp "$CONTEXT/source-$package/$notice" "$CONTEXT/provenance/$package/$notice"
    fi
  done
done
docker build --platform linux/arm64 -t "$IMAGE" "$CONTEXT"
python3 "$HERE/check_image.py" "$IMAGE"
echo "Built $IMAGE; retained build inputs: $CONTEXT"
