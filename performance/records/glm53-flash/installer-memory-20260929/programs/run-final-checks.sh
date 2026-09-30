#!/bin/bash
# Measures one installed configuration under text, image-burst, repeated-burst and mixed load, in that order.
#   1. text load (run-load.sh: prefill 16K/64K/128K, decode 4/8/16 users x 64K/128K),
#   2. image bursts: 16 concurrent requests with NIMG (default 3) unique large images each, 2 rounds,
#   3. repeated bursts: 6 back-to-back rounds of those 16 image requests, then 60 s idle,
#   4. mixed load: 8 text streams at 64K + 16 image requests (3 images each) + 4 video requests at once, 2 rounds.
# Steps 2-4 run under guarded.py (Node A floor); run fleet_guard.sh or fleet_guard_fast.py for all Sparks alongside.
# Each step uses its own request-index offset (LOAD_INDEX_START), so every image and video is new to the deployment.
# Every step records <name>_start / <name>_end epochs in OUT_ROOT/TAG/phases.txt.
# Environment: as run-load.sh, plus BENCH_DIR/BENCH_PYTHON for mixed_load.py.
# Usage: run-final-checks.sh TAG [NIMG]
set -u
tag=$1
nimg=${2:-3}
out=${OUT_ROOT:?set OUT_ROOT}/$tag
programs=$(cd "$(dirname "$0")" && pwd)
node_a=${NODE_A:?set NODE_A to the SSH target of Node A}
workers=${WORKERS:?set WORKERS to the worker addresses reachable from Node A}
remote_dir=${REMOTE_DIR:-\$HOME/tp4mem}
extra='{"chat_template_kwargs": {"reasoning_effort": "low"}}'
common="--host ${API_HOST:-192.0.2.10} --port ${API_PORT:-8015} --model ${MODEL:-GLM-5.3-Flash-NVFP4-Spark-TP4}"
bash "$programs/run-load.sh" "$tag" > "$out/text-load-summary.txt" 2>&1
kv=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$out/rank0.log" | tail -1 | tr -dc 0-9)
step() {
  local name=$1 floor=$2
  shift 2
  echo "${name}_start $(date +%s)" >> "$out/phases.txt"
  python "$programs/guarded.py" --node "$node_a" --floor "$floor" -- "$@" > "$out/$name.txt" 2>&1
  echo "${name}_end $(date +%s)" >> "$out/phases.txt"
  sleep 30
}
export LOAD_INDEX_START=0
step img16 1.3 python "$programs/image_load.py" $common --images "$nimg" --concurrency 16 --rounds 2 --extra "$extra"
export LOAD_INDEX_START=1000
step burst6 1.3 python "$programs/image_load.py" $common --images "$nimg" --concurrency 16 --rounds 6 --extra "$extra"
sleep 30
export LOAD_INDEX_START=5000
step mixed 1.2 python "$programs/mixed_load.py" $common --kv-budget "$kv" --text-concurrency 8 \
  --text-context 64k --image-concurrency 16 --video-concurrency 4 --rounds 2 --extra "$extra"
ssh "$node_a" "WORKERS='$workers' sh $remote_dir/collect.sh $tag" > "$out/memlog.txt"
ssh "$node_a" 'ps -eo pid,rss,comm --sort=-rss | head -4; grep MemAvailable /proc/meminfo' > "$out/after-checks-ps.txt"
cat "$out/text-load-summary.txt" "$out/img16.txt" "$out/burst6.txt" "$out/mixed.txt" "$out/after-checks-ps.txt"
