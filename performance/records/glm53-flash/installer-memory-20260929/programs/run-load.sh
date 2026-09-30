#!/bin/bash
# Measures text-load memory and speed: prefill at 16K/64K/128K and decode at 4/8/16 users x 64K/128K, with NCCL buffer counts before and after.
# Uses the KV budget from the rank-0 log ("GPU KV cache size"), records bench_start/bench_end in OUT_ROOT/TAG/phases.txt,
# and collects the 1-second MemAvailable logs afterwards.
# Environment: NODE_A = SSH target of Node A; WORKERS = worker addresses reachable from Node A as root;
#              REMOTE_DIR = directory on Node A holding nccl-all.sh and collect.sh;
#              API_HOST (default 192.0.2.10), API_PORT (default 8015), MODEL; BENCH_DIR = llm-inference-bench checkout;
#              BENCH_PYTHON = its Python; OUT_ROOT = directory for per-configuration results.
# Usage: run-load.sh TAG
set -u
tag=$1
node_a=${NODE_A:?set NODE_A to the SSH target of Node A}
workers=${WORKERS:?set WORKERS to the worker addresses reachable from Node A}
remote_dir=${REMOTE_DIR:-\$HOME/tp4mem}
host=${API_HOST:-192.0.2.10}
port=${API_PORT:-8015}
model=${MODEL:-GLM-5.3-Flash-NVFP4-Spark-TP4}
out=${OUT_ROOT:?set OUT_ROOT}/$tag
programs=$(cd "$(dirname "$0")" && pwd)
kv=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$out/rank0.log" | tail -1 | tr -dc 0-9)
echo "kv_tokens $kv" > "$out/kv.txt"
ssh "$node_a" "WORKERS='$workers' sh $remote_dir/nccl-all.sh" > "$out/nccl-maps-idle.txt" 2>&1
echo "bench_start $(date +%s)" >> "$out/phases.txt"
cd "${BENCH_DIR:?set BENCH_DIR}"
"${BENCH_PYTHON:?set BENCH_PYTHON}" llm_decode_bench.py --host "$host" --port "$port" --model "$model" \
  --concurrency 4,8,16 --contexts 64k,128k --prefill-contexts 16k,64k,128k --kv-budget "$kv" --dcp-size 1 \
  --no-hw-monitor --temperature 1.0 --token-targeting exact --display-mode plain --max-tokens 2048 --duration 20 \
  --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 900 --no-resume --output "$out/load.json" \
  < /dev/null > "$out/load.log" 2>&1
echo "bench_end $(date +%s)" >> "$out/phases.txt"
ssh "$node_a" "WORKERS='$workers' sh $remote_dir/nccl-all.sh" > "$out/nccl-maps-after-load.txt" 2>&1
ssh "$node_a" "WORKERS='$workers' sh $remote_dir/collect.sh $tag" > "$out/memlog.txt"
echo "kv_tokens $kv"
grep -o "nccl_conn_buffers *[0-9]*" "$out/nccl-maps-after-load.txt" | tr '\n' ' '; echo
python "$programs/benchsum.py" "$tag=$out/load.json"
