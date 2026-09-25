#!/usr/bin/env bash
# Build a toolchain-variant test image on this node from the installed decode image.
# Usage: build_toolchain_variant.sh VARIANT TAG
# The variant image re-seals /opt/sparkring/toolchain for VARIANT (cuda: CUDA 13.4.2
# with the base image's NCCL 2.31.2; nccl: the base CUDA stack with NCCL 2.32.3).
set -euo pipefail
VARIANT=$1 TAG=$2
BASE=sha256:705178360c189f8ce937b30b200754c983ddcda44052dd8aa7272e052a2e0f27
CONTEXT=/var/tmp/toolchain-variant-$VARIANT
rm -rf "$CONTEXT" && mkdir -p "$CONTEXT"
docker tag "$BASE" sparkring-dev/toolchain-base:705178360c18
docker run --rm --pull never --network none --entrypoint cat "$BASE" /opt/sparkring/toolchain/toolchain.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); d['variant']=sys.argv[1]; print(json.dumps(d, indent=2))" "$VARIANT" \
  > "$CONTEXT/toolchain.json"
ALIAS=""
if [ "$VARIANT" = cuda ]; then ALIAS=/opt/sparkring/toolchain/python/nvidia/nccl/lib; fi
if [ "$VARIANT" = nccl ]; then ALIAS=/opt/sparkring/toolchain/python/nvidia/cu13/lib; fi
cat > "$CONTEXT/Dockerfile" <<DOCKER
FROM sparkring-dev/toolchain-base:705178360c18
COPY toolchain.json /opt/sparkring/toolchain/toolchain.json
RUN rm -f /opt/sparkring/toolchain/installed.json $ALIAS && python3 /opt/sparkring/toolchain/toolchain.py seal > /dev/null
DOCKER
docker build -q -t "$TAG" "$CONTEXT"
docker run --rm --pull never --network none --entrypoint python3 "$TAG" -c "import json; d=json.load(open('/opt/sparkring/toolchain/installed.json')); print(d['variant'], d['nccl_version'], d['nvcc'].splitlines()[-1])"
