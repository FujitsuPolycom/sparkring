#!/bin/sh
# Measures available memory on all four Sparks: starts memsample.sh (1-second MemAvailable log) on Node A and each worker.
# Run on Node A. Logs land in /var/tmp/memlog-TAG on every Spark.
# Environment: WORKERS = worker addresses reachable from Node A as root over SSH
#              (for example "192.0.2.11 192.0.2.12 192.0.2.13"); REMOTE_DIR = directory holding memsample.sh.
# Usage: start-samplers.sh TAG SECONDS
tag=$1
n=$2
workers=${WORKERS:?set WORKERS to the worker addresses}
dir=${REMOTE_DIR:-$HOME/tp4mem}
cp "$dir/memsample.sh" /tmp/memsample.sh
(nohup sh /tmp/memsample.sh "/var/tmp/memlog-$tag" "$n" >/dev/null 2>&1 &)
for w in $workers; do
  sudo -n scp -q /tmp/memsample.sh "root@$w:/tmp/memsample.sh"
  sudo -n ssh "root@$w" "(nohup sh /tmp/memsample.sh /var/tmp/memlog-$tag $n >/dev/null 2>&1 &)"
done
sleep 2
echo "node-a $(wc -l < /var/tmp/memlog-$tag) lines"
for w in $workers; do
  echo "$w $(sudo -n ssh root@$w wc -l < /dev/null /var/tmp/memlog-$tag 2>&1)"
done
