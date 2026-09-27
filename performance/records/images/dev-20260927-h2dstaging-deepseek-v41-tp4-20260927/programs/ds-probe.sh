#!/usr/bin/env bash
# Probe the serving DeepSeek variant: startup facts, prefill, decode by prompt type, optional stress.
# Usage: WORKERS="RANK1_HOST RANK2_HOST RANK3_HOST" ds-probe.sh NAME [stress-rounds]
# Runs on rank 0 from the directory that holds the probe programs; the variant's
# rank-0 container is dsab-NAME-r0 and serves port 8015.
NAME=$1; ROUNDS=${2:-0}; URL=http://127.0.0.1:8015/v1
C=dsab-$NAME-r0
echo "== $NAME startup"
sudo -n docker logs $C 2>&1 | grep -E "KV cache size|Maximum concurrency|Loading weights took|Model loading took|Engram .*(disk|DISK)|block size|prefix caching|init engine .* took" | grep -v "Unknown vLLM" | head -12 | cut -c1-240
echo "== $NAME prefill"; python3 prefill_probe.py $URL 8192,16384,32768,65536,131072
echo "== $NAME prefill repeat"; python3 prefill_probe.py $URL 16384,65536
echo "== $NAME decode"; python3 decode_probe.py $URL 512 2 2>&1 | tail -6
echo "== $NAME ngram"; python3 ngram_probe.py $URL 16384
if [ "$ROUNDS" -gt 0 ]; then
  model=$(curl -s "$URL/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
  start=$(date +%s); BASE_URL=http://127.0.0.1:8015 MODEL="$model" python3 stress.py $NAME $ROUNDS > stress-$NAME.json
  echo "== $NAME stress $(( $(date +%s) - start )) s"
  tail -1 stress-$NAME.json | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print({k: d[k] for k in ('label', 'n', 'degen', 'wrong', 'errors')}, d['wrong_ids'])
for item in d['degen_items'][:6]: print('  degenerate:', item)"
fi
echo "== $NAME memory (GiB available per rank)"
free -g | awk '/Mem:/{print "r0", $7}'
for h in $WORKERS; do sudo -n ssh -o BatchMode=yes root@$h "free -g | awk '/Mem:/{print \"$h\", \$7}'"; done
echo "== $NAME done $(date +%T)"
