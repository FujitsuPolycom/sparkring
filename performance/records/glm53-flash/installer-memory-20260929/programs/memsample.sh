#!/bin/sh
# Measures available host memory: appends "epoch MemAvailable_kB" to FILE once a second for SECONDS seconds.
# Usage: memsample.sh FILE SECONDS
out=$1; n=$2
rm -f "$out"
i=0
while [ "$i" -lt "$n" ]; do
  echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$out"
  i=$((i + 1))
  sleep 1
done
