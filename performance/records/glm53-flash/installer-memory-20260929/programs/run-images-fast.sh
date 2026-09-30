#!/bin/bash
# Measures vision-load memory for one step: image_load.py under the fast fleet guard (2.0 GiB floor, 0.2 s readings on every Spark).
# CONC concurrent requests with NIMG unique images each, ROUNDS rounds, request indices from OFFSET (images no earlier
# step on the deployment used), image sizes SIZES (image_load.py's default cycle 2048x2048,3840x2160,1600x1200 when
# omitted; 2048x2048 makes every image reach the 4,096-token cap). Records NAME_start/NAME_end in OUT_ROOT/TAG/phases.txt
# and every reading in OUT_ROOT/TAG/fast-NAME.log.
# Environment: NODE_A = SSH target of Node A; WORKERS_CSV = worker addresses in rank order, comma-separated;
#              API_HOST (default 192.0.2.10), API_PORT (default 8015), MODEL; OUT_ROOT = per-configuration results.
# Usage: run-images-fast.sh TAG NAME NIMG CONC ROUNDS OFFSET [SIZES]
set -u
tag=$1 name=$2 nimg=$3 conc=$4 rounds=$5 offset=$6 sizes=${7:-2048x2048,3840x2160,1600x1200}
out=${OUT_ROOT:?set OUT_ROOT}/$tag
programs=$(cd "$(dirname "$0")" && pwd)
echo "${name}_start $(date +%s)" >> "$out/phases.txt"
LOAD_INDEX_START=$offset python "$programs/fleet_guard_fast.py" --node-a "${NODE_A:?set NODE_A}" \
  --workers "${WORKERS_CSV:?set WORKERS_CSV}" --floor 2.0 --interval 0.2 --log "$out/fast-$name.log" -- \
  python "$programs/image_load.py" --host "${API_HOST:-192.0.2.10}" --port "${API_PORT:-8015}" \
  --model "${MODEL:-GLM-5.3-Flash-NVFP4-Spark-TP4}" --images "$nimg" --sizes "$sizes" --concurrency "$conc" \
  --rounds "$rounds" --timeout 1800 --extra '{"chat_template_kwargs": {"reasoning_effort": "low"}}' > "$out/$name.txt" 2>&1
echo "${name}_end $(date +%s)" >> "$out/phases.txt"
cat "$out/$name.txt"
python "$programs/fastlows.py" "$out/fast-$name.log"
echo "Node A first reading $(grep '^r0' "$out/fast-$name.log" | head -1 | awk '{printf "%.2f", $3/1048576}') GiB, last $(grep '^r0' "$out/fast-$name.log" | tail -1 | awk '{printf "%.2f", $3/1048576}') GiB"
