#!/usr/bin/env bash
# RUNBOOK.md sections 3.4 and 3.5 on one rank of a cabled pair: nccl-tests built against an NCCL header
# already in the image and libsircl's MPI shim, run with libsircl first in LD_PRELOAD, every output and
# receipt kept under OUT. Run inside the serving image, from the library tree.
#
#   tools/nccl_tests_pair.sh preflight
#       reports what the image already holds: nvcc, NCCL headers, nccl-tests binaries, LD_PRELOAD
#   tools/nccl_tests_pair.sh build SRC
#       builds the nccl-tests source tree SRC (v2.21.1) into SRC/build
#   tools/nccl_tests_pair.sh run RANK ROOT BIN OUT [milestone|further|lines FILE]
#       RANK 0 or 1; ROOT the host:port of rank 0's MPI-shim rendezvous; BIN the nccl-tests build
#       directory; milestone runs section 3.4 (default), further runs section 3.5, lines runs the rows of
#       FILE, one "<binary> <arguments>" per row (blank rows and rows starting with # skipped), in order
#   python3 tools/check_nccl_tests.py OUT_OF_RANK_0 OUT_OF_RANK_1
#       evaluates the exit criteria from both ranks' outputs and receipts
#
# run writes OUT/jobs.tsv, one row per line run (the job, its log file, its exit status), prints how many
# lines ran and failed, and exits 1 when any line failed.
#
# The NCCL header comes from NCCL_HEADER_HOME (a directory with include/nccl.h and lib/libnccl.so.2),
# else the serving image's toolchain copy (/opt/sparkring/toolchain/nccl), else an nvidia-nccl wheel.
# nccl-tests only compiles against it and links its SONAME; at run time libsircl is loaded first.
# The caller sets SIRCL_PEER_ROUTES, LIBSIRCL_TRANSPORT=verbs and SIRCL_BOOTSTRAP_IFNAME as in
# RUNBOOK.md section 3; LIB overrides the library path (default build/libsircl.so).
#
# Each line is one MPI-shim job: SIRCL_MPI_JOB is the line itself, so a rank only joins the other rank's
# run of the same line, never a neighbouring line. A rank whose binary ends outside MPI_Finalize ends the
# other rank's binary within seconds (the shim's watchdog), and every binary runs under `timeout`
# (LINE_TIMEOUT_S, default 1200 s), so a failed or stuck line costs both ranks that line only and the two
# ranks reach the next line together. SIRCL_MPI_TIMEOUT_S (default 300 s) bounds how long a rank waits for
# the other to start the same line.
#
# CPU placement as SIRCL's ring harness places its workers (sparkring_sircl.cpus, policy performance): the
# binary runs on the fastest-class cores but one (taskset), the transport's progress thread on that one
# (SIRCL_PROGRESS_CPU), so neither runs on an efficiency core or shares a core with the other.
# PLACEMENT=none leaves both to the scheduler; a SIRCL_PROGRESS_CPU already set is kept, unpinned binary.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/.." && pwd)
LIB=${LIB:-$TREE/build/libsircl.so}
SHIM=$TREE/build/mpi-shim
LINE_TIMEOUT_S=${LINE_TIMEOUT_S:-1200}
TOOLCHAIN_NCCL=/opt/sparkring/toolchain/nccl

wheel() { python3 -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))' 2>/dev/null || true; }
# "<main CPUs> <progress CPU>" for the placement above, or nothing when the cores are of one class.
placement() {
  python3 - <<'EOF' 2>/dev/null || true
import os
tiers = {0xD44: 3, 0xD48: 3, 0xD4E: 3, 0xD82: 3, 0xD85: 3, 0xD47: 2, 0xD4D: 2, 0xD81: 2, 0xD87: 2, 0xD46: 1, 0xD80: 1}
part, cpu = {}, None
for line in open("/proc/cpuinfo"):
    key, _, value = line.partition(":")
    if key.strip() == "processor":
        cpu = int(value)
    elif key.strip() == "CPU part" and cpu is not None:
        part[cpu] = tiers.get(int(value, 0), 2)
allowed = sorted(os.sched_getaffinity(0))
if all(c in part for c in allowed) and len({part[c] for c in allowed}) > 1:
    fast = [c for c in allowed if part[c] == max(part[c] for c in allowed)]
    if len(fast) >= 2:
        print(",".join(map(str, fast[:-1])), fast[-1])
EOF
}
header_home() {
  if [ -n "${NCCL_HEADER_HOME:-}" ]; then echo "$NCCL_HEADER_HOME"
  elif [ -f "$TOOLCHAIN_NCCL/include/nccl.h" ]; then echo "$TOOLCHAIN_NCCL"
  else wheel
  fi
}
# libsircl first, then the caller's LD_PRELOAD without any other libnccl.so.2 (the serving image preloads
# its own NCCL), so exactly one library with SONAME libnccl.so.2 is mapped.
preload() {
  local rest
  rest=$(printf '%s' "${LD_PRELOAD:-}" | tr ':' '\n' | grep -v '/libnccl\.so' | paste -sd: - || true)
  echo "$LIB${rest:+:$rest}"
}

command=${1:-}
shift || true
case "$command" in
preflight)
  echo "nvcc: $(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1 || echo none)"
  H=$(header_home)
  echo "NCCL header home: ${H:-none}"
  if [ -n "$H" ]; then
    ls "$H/include/nccl.h" "$H"/lib/libnccl.so* 2>&1
    grep -m3 -E '#define NCCL_(MAJOR|MINOR|PATCH) ' "$H/include/nccl.h"
  fi
  echo "LD_PRELOAD: ${LD_PRELOAD:-}"
  echo "LD_PRELOAD for the runs: $(preload)"
  echo "nccl-tests binaries:"
  find / -xdev \( -path /proc -o -path /sys \) -prune -o -type f -name all_reduce_perf -print 2>/dev/null | head -5
  ;;
build)
  SRC=${1:?nccl-tests source tree}
  H=$(header_home)
  [ -f "$H/include/nccl.h" ] || { echo "no NCCL header found (set NCCL_HEADER_HOME)"; exit 1; }
  HOME_DIR=$(mktemp -d /tmp/nccl-home-XXXX)
  mkdir -p "$HOME_DIR/lib"
  ln -sfn "$H/include" "$HOME_DIR/include"
  # The linker needs the libnccl.so link name, which an image or wheel may not ship.
  ln -sf "$(ls "$H"/lib/libnccl.so.2* | head -1)" "$HOME_DIR/lib/libnccl.so"
  make -C "$TREE" mpi-shim BUILD=build
  CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | head -1)")")}
  make -C "$SRC" -j MPI=1 MPI_HOME="$SHIM" NCCL_HOME="$HOME_DIR" CUDA_HOME="$CUDA_HOME" \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
  ls "$SRC/build/all_reduce_perf"
  ;;
run)
  RANK=${1:?rank}; ROOT=${2:?host:port}; BIN=${3:?nccl-tests build directory}; OUT=${4:?output directory}
  SECTION=${5:-milestone}
  mkdir -p "$OUT"
  RUN_PRELOAD=$(preload)
  PIN=()
  if [ "${PLACEMENT:-performance}" != none ] && [ -z "${SIRCL_PROGRESS_CPU:-}" ] && command -v taskset >/dev/null; then
    read -r MAIN_CPUS PROGRESS_CPU <<< "$(placement)" || true
    if [ -n "${MAIN_CPUS:-}" ]; then
      export SIRCL_PROGRESS_CPU=$PROGRESS_CPU
      PIN=(taskset -c "$MAIN_CPUS")
      echo "placement: binaries on CPUs $MAIN_CPUS, progress thread on CPU $PROGRESS_CPU"
    fi
  fi
  export LD_LIBRARY_PATH="$SHIM/lib:${LD_LIBRARY_PATH:-}"
  export SIRCL_MPI_SIZE=2 SIRCL_MPI_ROOT="$ROOT" SIRCL_MPI_RANK="$RANK" LIBSIRCL_RECEIPT="$OUT/receipt"
  : > "$OUT/jobs.tsv"
  RAN=0 FAILED=0
  line() {
    local name=$1; shift
    local job="$name $*" status=0 log
    log="$name$(printf -- '_%s' "$@" | tr -c 'A-Za-z0-9_.-' '_').log"
    echo "== $job"
    timeout --kill-after=30 "$LINE_TIMEOUT_S" ${PIN[@]+"${PIN[@]}"} \
      env SIRCL_MPI_JOB="$job" LD_PRELOAD="$RUN_PRELOAD" "$BIN/$name" "$@" \
      < /dev/null > "$OUT/$log" 2>&1 || status=$?
    printf '%s\t%s\t%s\n' "$job" "$log" "$status" >> "$OUT/jobs.tsv"
    RAN=$((RAN + 1))
    case $status in
      0) ;;
      124|137) echo "exit $status in $job: stopped after LINE_TIMEOUT_S=$LINE_TIMEOUT_S s"; FAILED=$((FAILED + 1)) ;;
      *) echo "exit $status in $job"; FAILED=$((FAILED + 1)) ;;
    esac
  }
  if [ "$SECTION" = lines ]; then
    LINES=${6:?a file of test lines}
    [ -f "$LINES" ] || { echo "no file $LINES"; exit 2; }
    while read -r row || [ -n "$row" ]; do
      case $row in ''|'#'*) continue ;; esac
      # shellcheck disable=SC2086
      line $row
    done < "$LINES"
  elif [ "$SECTION" = milestone ]; then
    for dtype in bfloat16 half float; do
      for graph in 0 20; do
        line all_reduce_perf -b 8 -e 256M -f 2 -d "$dtype" -o sum -c 1 -n 50 -w 10 -G "$graph"
      done
    done
  else
    for test in all_gather_perf reduce_scatter_perf broadcast_perf reduce_perf alltoall_perf sendrecv_perf \
                gather_perf scatter_perf; do
      line "$test" -b 8 -e 64M -f 4 -d bfloat16 -c 1 -n 20 -w 5
    done
    for spec in "int32 max" "int64 sum" "double prod" "float avg" "uint8 min"; do
      set -- $spec
      line all_reduce_perf -b 8 -e 16M -f 4 -d "$1" -o "$2" -c 1 -n 20 -w 5
    done
  fi
  ls "$OUT"
  echo "lines: $RAN run, $FAILED failed"
  [ "$FAILED" -eq 0 ]
  ;;
*)
  sed -n '2,33p' "$0"
  exit 2
  ;;
esac
