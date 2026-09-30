#!/bin/bash
# Measures available memory on all four Sparks about every 2 s and stops local load clients when any Spark falls below FLOOR GiB.
# One reading per Spark per cycle over fresh SSH sessions (Node A directly, the workers through Node A).
# The load clients stopped are local processes whose command line mentions mm_stress, image_load,
# video_load, mixed_load or llm_decode_bench (Windows Stop-Process). Prints the lowest reading per Spark on exit.
# Environment: NODE_A = SSH target of Node A; WORKERS = worker addresses reachable from Node A as root.
# Usage: fleet_guard.sh FLOOR_GIB SECONDS
node_a=${NODE_A:?set NODE_A to the SSH target of Node A}
workers=${WORKERS:?set WORKERS to the worker addresses}
floor_kb=$(awk -v f="$1" 'BEGIN { printf "%d", f * 1048576 }')
end=$(( $(date +%s) + $2 ))
declare -A low
while [ "$(date +%s)" -lt "$end" ]; do
  readings=$(ssh -o ConnectTimeout=5 "$node_a" "echo \"r0 \$(awk '/MemAvailable/{print \$2}' /proc/meminfo)\"; i=1; for w in $workers; do echo \"r\$i \$(sudo -n ssh -o ConnectTimeout=5 root@\$w \"awk '/MemAvailable/{print \\\$2}' /proc/meminfo\" < /dev/null)\"; i=\$((i + 1)); done" 2>/dev/null)
  while read -r node kb; do
    [[ "$kb" =~ ^[0-9]+$ ]] || continue
    if [ -z "${low[$node]}" ] || [ "$kb" -lt "${low[$node]}" ]; then low[$node]=$kb; fi
    if [ "$kb" -lt "$floor_kb" ]; then
      echo "$(date +%H:%M:%S) GUARD: $node MemAvailable $((kb / 1024)) MiB below floor; stopping load clients"
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'mm_stress|image_load|video_load|mixed_load|llm_decode_bench' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    fi
  done <<< "$readings"
  sleep 1
done
for node in r0 r1 r2 r3; do echo "lowest $node $(awk -v k="${low[$node]}" 'BEGIN { printf "%.2f", k / 1048576 }') GiB"; done
