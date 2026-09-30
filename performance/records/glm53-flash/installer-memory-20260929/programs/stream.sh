#!/bin/sh
# Measures available memory at a fast interval: prints "NODE epoch.fraction MemAvailable_kB" every INTERVAL seconds until the reader goes away.
# Usage: stream.sh NODE INTERVAL
n=$1
iv=$2
while true; do
  echo "$n $(date +%s.%N) $(awk '/MemAvailable/{print $2}' /proc/meminfo)" || exit 0
  sleep "$iv"
done
