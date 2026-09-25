#!/usr/bin/env bash
# Usage: measure.sh BASE_URL LABEL
# Warm-up pass (discarded), then correctness checks, decode/acceptance by prompt type, and prefill.
set -u
URL=$1; LABEL=$2; OUT=results/$LABEL; mkdir -p results
python prefill_curve.py "$URL" 4096,16384,65536 1 > /dev/null 2>&1
python decode_probe.py "$URL" 128 1 > /dev/null 2>&1
{
  echo "== $LABEL checks"; python verify_api.py "$URL" '{"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | grep -v "status view"
  echo "== $LABEL decode by prompt type (512 tokens, 2 runs)"; python decode_probe.py "$URL" 512 2 2>&1
  echo "== $LABEL prefill and greedy decode"; python bench.py "$URL" "$LABEL" "$OUT.json" 2>&1 | grep -v fingerprints
} | tee "$OUT.txt"
