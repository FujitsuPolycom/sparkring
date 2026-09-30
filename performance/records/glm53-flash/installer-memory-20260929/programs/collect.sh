#!/bin/sh
# Collects the 1-second MemAvailable logs of all four Sparks for TAG as "node epoch kB" lines (r0 = Node A, r1-r3 = workers).
# Run on Node A. Environment: WORKERS = worker addresses in rank order (for example "192.0.2.11 192.0.2.12 192.0.2.13").
# Usage: collect.sh TAG
tag=$1
workers=${WORKERS:?set WORKERS to the worker addresses}
sed "s/^/r0 /" "/var/tmp/memlog-$tag"
i=1
for w in $workers; do
  sudo -n ssh "root@$w" "cat /var/tmp/memlog-$tag" < /dev/null | sed "s/^/r$i /"
  i=$((i + 1))
done
