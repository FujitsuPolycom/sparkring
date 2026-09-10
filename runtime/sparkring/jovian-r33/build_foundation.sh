#!/usr/bin/env bash
# Assemble and verify the exact ARM64 Torch/NCCL foundation for R33 components.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
context="$root/artifacts"
image=${SPARKRING_R33_FOUNDATION_IMAGE:-local/sparkring:r33-arm64-foundation}
receipt="$context/foundation-image.json"

test "$(uname -m)" = aarch64
test -f "$context/nccl/build-receipt.json"
test -f "$context/nccl/lib/libnccl.so.2"
test -f "$context/Dockerfile.foundation"
test -f "$context/verify_torch.py"
set -- "$context"/torch/torch-2.13.0-*.whl
test "$#" -eq 1
test -f "$1"

docker build --progress plain --pull=false \
  --file "$context/Dockerfile.foundation" \
  --tag "$image" "$context"

image_id=$(docker image inspect "$image" --format '{{.Id}}')
docker run --rm --network none \
  "$image" python3 /opt/sparkring-build/verify_torch.py > "$receipt.tmp"
python3 -S - "$receipt.tmp" "$receipt" "$image" "$image_id" <<'PY'
import json,pathlib,sys
source=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2])
raw=source.read_text()
data=json.loads(raw[raw.index('{'):])
data.update({'image':sys.argv[3], 'image_id':sys.argv[4],
             'status':'qualified-cpu-import-only; GPU qualification pending'})
target.write_text(json.dumps(data,indent=2)+'\n')
source.unlink()
PY
printf '%s\n' "$image_id"
