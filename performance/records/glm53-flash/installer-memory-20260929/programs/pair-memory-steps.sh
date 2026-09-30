#!/bin/bash
# Measures Node A memory on a GLM two-Spark deployment: text, image, video and mixed load steps, each
# under guarded.py (stops the load when Node A's MemAvailable falls below 1.2 GiB). Every step records
# its start and end epoch in phases.txt, for cutting the one-second MemAvailable logs that memsample.sh
# writes on both Sparks. Every step starts its image and video seeds at its own LOAD_INDEX_START, so no
# step repeats media the server has already cached.
#
# Environment: NODE_A (SSH target of Node A), API_HOST, OUT_DIR, BENCH_DIR (llm-inference-bench
# checkout) and BENCH_PYTHON (the Python that runs it). Run from this directory.
set -u
here=$(cd "$(dirname "$0")" && pwd)
common="--host $API_HOST --port 8000 --model GLM-5.3-Flash-NVFP4-Spark-TP2"
extra='{"chat_template_kwargs": {"reasoning_effort": "low"}}'
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit 1

step() {  # step NAME OFFSET COMMAND...
  local name=$1 offset=$2
  shift 2
  echo "${name}_start $(date +%s)" >> phases.txt
  LOAD_INDEX_START=$offset python "$here/guarded.py" --node "$NODE_A" --floor 1.2 -- "$@" > "$name.txt" 2>&1
  echo "${name}_end $(date +%s)" >> phases.txt
  sleep 20
}

ssh "$NODE_A" 'grep MemAvailable /proc/meminfo' > idle-before.txt
echo "idle_start $(date +%s)" >> phases.txt; sleep 60; echo "idle_end $(date +%s)" >> phases.txt

step text128k 0 "$BENCH_PYTHON" "$BENCH_DIR/llm_decode_bench.py" \
  --host "http://$API_HOST" --port 8000 --model GLM-5.3-Flash-NVFP4-Spark-TP2 --dcp-size 1 --no-hw-monitor \
  --kv-budget 1530566 --temperature 1.0 --token-targeting exact --display-mode plain --concurrency 4,8 \
  --contexts 128k --skip-prefill --max-tokens 2048 --duration 60 --decode-warmup-seconds 5 \
  --cell-warmup-timeout-seconds 900 --no-resume --output "$OUT_DIR/text128k.json"
for c in 1 2 4; do
  step "img8c$c" $((c * 100)) python "$here/image_load.py" $common --images 8 --concurrency "$c" --rounds 1 --extra "$extra"
done
step mixed3 500 python "$here/mixed_load.py" $common --kv-budget 1530566 --text-concurrency 4 --text-context 64k \
  --image-concurrency 2 --images 3 --video-concurrency 1 --frames 16 --rounds 1 --extra "$extra"
step mixed8 600 python "$here/mixed_load.py" $common --kv-budget 1530566 --text-concurrency 4 --text-context 64k \
  --image-concurrency 2 --images 8 --video-concurrency 1 --frames 16 --rounds 1 --extra "$extra"
ssh "$NODE_A" 'grep MemAvailable /proc/meminfo' > idle-after.txt
