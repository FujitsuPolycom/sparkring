#!/usr/bin/env bash
# Draft depth 3/4/5 with probabilistic drafting on one cluster.
# Usage: ab_mtp_depth.sh SSH_HOST URL CLUSTER PREVIOUS_VARIANT
set -u
HOST=$1 URL=$2 CLUSTER=$3 previous=$4
cd "$(dirname "$0")"
for drafts in 3 4 5; do
  variant=$CLUSTER-mtp$drafts
  if [ -n "$previous" ]; then down="python3 qwab.py down $previous;"; else down=""; fi
  ssh -o BatchMode=yes "$HOST" "sudo -n systemd-run --unit qwab-$variant --collect --working-directory=/var/tmp/qwab --property=StandardOutput=truncate:/var/tmp/qwab/$variant.log --property=StandardError=inherit /bin/bash -c '$down python3 qwab.py up variants/$variant.json'"
  sleep 60
  for i in $(seq 1 240); do
    if ssh -o BatchMode=yes "$HOST" "grep -q '$variant: healthy' /var/tmp/qwab/$variant.log"; then break; fi
    if ssh -o BatchMode=yes "$HOST" "! systemctl is-active -q qwab-$variant && ! grep -q healthy /var/tmp/qwab/$variant.log"; then echo "$variant failed"; ssh -o BatchMode=yes "$HOST" "tail -n 30 /var/tmp/qwab/$variant.log"; exit 1; fi
    sleep 10
  done
  echo "== $variant checks"; python verify_api.py "$URL" '{"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | grep -v "status view" | grep "count\|math\|code"
  python decode_probe.py "$URL" 64 1 1.0 > /dev/null 2>&1
  echo "== $variant temperature 1.0"; python decode_probe.py "$URL" 512 3 1.0
  echo "== $variant temperature 0"; python decode_probe.py "$URL" 512 2 0
  python concurrency_sweep.py "$URL" "results/$variant-sweep.json" 1,4,8,16 | sed "s/^/$variant sweep /"
  previous=$variant
done
