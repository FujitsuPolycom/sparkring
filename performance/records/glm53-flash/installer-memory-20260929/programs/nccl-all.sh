#!/bin/sh
# Measures NCCL host buffers on all four Sparks: runs nccl_maps.py on Node A and each worker and prints MemAvailable.
# Run on Node A. Environment: WORKERS = worker addresses in rank order; REMOTE_DIR = directory holding nccl_maps.py.
workers=${WORKERS:?set WORKERS to the worker addresses}
dir=${REMOTE_DIR:-$HOME/tp4mem}
echo "== r0 MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
sudo -n python3 "$dir/nccl_maps.py"
i=1
for w in $workers; do
  sudo -n scp -q "$dir/nccl_maps.py" "root@$w:/tmp/nccl_maps.py"
  sudo -n ssh "root@$w" "echo == r$i MemAvailable \$(awk '/MemAvailable/{print \$2}' /proc/meminfo) kB; python3 /tmp/nccl_maps.py" < /dev/null
  i=$((i + 1))
done
