#!/usr/bin/env bash
# Build the externally maintained adapter without vendoring its AGPL source.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:?Set WORK to an unused build directory path outside this checkout}
IMAGE=${IMAGE:-local/sparkring-deepseek-v41:sglang}
[ ! -e "$WORK" ] || { echo 'WORK must not already exist' >&2; exit 1; }
read_pin(){ python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$HERE/pins.json" "$1"; }
git clone --no-checkout "$(read_pin source_repository)" "$WORK"
git -C "$WORK" checkout --detach "$(read_pin source_commit)"
python3 - "$WORK/Dockerfile" "$(read_pin base_image)" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = 'FROM lmsysorg/sglang:dev-dsv41'
if s.count(old) != 1:
    raise SystemExit('unexpected upstream Dockerfile; refusing to replace base')
p.write_text(s.replace(old, 'FROM ' + sys.argv[2]))
PY
docker build --platform "$(read_pin platform)" -t "$IMAGE" "$WORK"
docker image inspect "$IMAGE" --format '{{.Id}}'
echo 'Record this image ID and the resolved NCCL library SHA256 in each private rank environment.'
