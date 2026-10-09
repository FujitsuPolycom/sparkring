#!/usr/bin/env bash
# tools/emulation_suite.sh [OUT]: libsircl's GPU-emulation suite on one GPU, run from the tree root, for
# example in the serving image on one Spark (python3 with torch, make, a C compiler, one CUDA GPU of
# sm_120 or sm_121). It builds the tree and runs the CPU checks, then every library-level emulation run:
# two to eight ranks as processes on the one GPU over the emulation transport, checked against the SIRCL
# session digests in tests/emulation/golden/ under the default (pair default on two ranks), pieces,
# chain and ring schedules, a relayed pair with and without a ring window, forward windows, teardown
# right after a ring collective, fail-stop (LIBSIRCL_FAIL_STOP), pipeline stage pairs of ring:8,
# point-to-point channels on communicators of more than two ranks (LIBSIRCL_P2P_CHANNELS), PyTorch's
# ProcessGroupNCCL and its pipeline exchange (lazily and eagerly initialized), the setup-failure cases and
# the timing sweeps. With
# NCCL_TESTS_BUILD (a directory of nccl-tests v2.21.1 binaries built against this tree's build/mpi-shim) it
# also runs nccl-tests on two processes; with SIRCL_PACKAGE (the
# directory holding SIRCL's sparkring_sircl package) the link pack's mixed groups against SIRCL's DSL
# kernels. It writes only build/ and OUT (default /tmp/libsircl-emulation-<time>), never touches the fabric
# or other processes, and prints one line per run; OUT/SUMMARY collects them. Exit status 1 when any run
# failed. SUITE_ONLY=<section>[,<section>...] runs the build and those sections only: library, torch, setup,
# sweeps, nccl-tests, mixed.
set -u
TREE=$(pwd)
[ -f "$TREE/VERSION" ] && [ -f "$TREE/tests/emulation/run_library.py" ] || { echo "run from the tree root"; exit 2; }
OUT=${1:-/tmp/libsircl-emulation-$(date -u +%Y%m%dT%H%M%SZ)}
PY=${PYTHON:-python3}
G=$TREE/tests/emulation/golden
LIB=$TREE/build/libsircl.so
mkdir -p "$OUT"
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_MODULE_LOADING=EAGER
failures=0
ONLY=${SUITE_ONLY:-all}
want() { [ "$ONLY" = all ] || case ",$ONLY," in *",$1,"*) true ;; *) false ;; esac; }
note() { echo "$*" | tee -a "$OUT/SUMMARY"; }
note "libsircl emulation suite, tree $TREE, $(date -u +%Y-%m-%dT%H:%MZ), $(uname -m)"
note "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"

# 1. Build and CPU checks.
if make -j BUILD=build > "$OUT/build.log" 2>&1 && make check BUILD=build > "$OUT/check.log" 2>&1 &&
   make mpi-shim BUILD=build > "$OUT/mpi-shim.log" 2>&1; then
  note "PASS build and CPU checks: $(grep 'kernel entries' "$OUT/build.log" | head -1)"
else
  note "FAIL build or CPU checks (build.log, check.log)"
  exit 1
fi
note "library $(sha256sum "$(readlink -f "$LIB")" | cut -d' ' -f1); link pack $(cat kernels/prebuilt/sircl_links.fatbin.sha256)"

# 2. Library runs: library <tag> <world> <lanes> <golden> [NAME=VALUE ...]
library() {
  local tag=$1 world=$2 lanes=$3 golden=$4; shift 4
  local envs=() kv
  for kv in "$@"; do envs+=(--env "$kv"); done
  local base="$OUT/library-$tag-w$world-l$lanes"
  timeout 3600 "$PY" tests/emulation/run_library.py --library "$LIB" --python "$PY" --world "$world" --lanes "$lanes" \
    --golden "$golden" "${envs[@]}" --json "$base.json" > "$base.log" 2>&1
  local code=$?
  local summary
  summary=$(grep "checks over" "$base.log" | tail -1)
  if [ $code -eq 0 ]; then note "PASS library $tag w$world lanes $lanes: $summary"; else
    note "FAIL library $tag w$world lanes $lanes (exit $code): ${summary:-no summary}; $base.log"
    failures=$((failures + 1))
  fi
}
CHAIN="SIRCL_LARGE_SCHEDULE=chain SIRCL_GATHER_SCHEDULE=chain SIRCL_SCATTER_SCHEDULE=chain"
RING="SIRCL_LARGE_SCHEDULE=ring SIRCL_GATHER_SCHEDULE=ring SIRCL_SCATTER_SCHEDULE=ring SIRCL_RING_MIN_BYTES=0 LIBSIRCL_RING_WINDOW=0"
WINDOWS="LIBSIRCL_FORWARD_WINDOWS=0=65536/65536,1=65536/65536,2=65536/65536,3=65536/65536"
if want library; then
library pair-default 2 1 "$G/w2.json"
library pair-default 2 2 "$G/w2.json"
library pieces 2 2 "$G/w2.json" SIRCL_LARGE_SCHEDULE=pieces
library relayed-pair 2 2 "$G/w2.json" LIBSIRCL_FORWARD_WINDOWS=0=65536/65536,1=65536/65536
# A relayed pair given a ring plan: the pair plan's ring ops and pair exchanges through the ring window.
library windowed-pair 2 2 "$G/w2.json" LIBSIRCL_FORWARD_WINDOWS=0=65536/65536,1=65536/65536 LIBSIRCL_RING_WINDOW=393216
[ -f "$G/w2-chain.json" ] && library chain 2 1 "$G/w2-chain.json" $CHAIN
[ -f "$G/w2-ring.json" ] && library ring 2 1 "$G/w2-ring.json" $RING
# Link ops of one kernel type alternating grids: every link op takes 1, 2, then 4 blocks per role.
library pair-cycle 2 2 "$G/w2.json" LIBSIRCL_LINK_BLOCKS_CYCLE=1,2,4
library pieces 3 1 "$G/w3.json"
library pieces 4 2 "$G/w4.json"
library pieces 8 2 "$G/w8.json"
[ -f "$G/w4-chain.json" ] && library chain 4 2 "$G/w4-chain.json" $CHAIN
[ -f "$G/w4-chain.json" ] && library chain-cycle 4 2 "$G/w4-chain.json" $CHAIN LIBSIRCL_LINK_BLOCKS_CYCLE=1,2,4
[ -f "$G/w4-ring.json" ] && library ring 4 2 "$G/w4-ring.json" $RING
[ -f "$G/w8-chain.json" ] && library chain 8 2 "$G/w8-chain.json" $CHAIN
[ -f "$G/w8-ring.json" ] && library ring 8 2 "$G/w8-ring.json" $RING
library windows 4 2 "$G/w4.json" $WINDOWS
[ -f "$G/w4-ring.json" ] && library ring-windows 4 2 "$G/w4-ring.json" $WINDOWS LIBSIRCL_RING_WINDOW=393216 \
  SIRCL_LARGE_SCHEDULE=ring SIRCL_GATHER_SCHEDULE=ring SIRCL_SCATTER_SCHEDULE=ring SIRCL_RING_MIN_BYTES=0

# Teardown right after a ring collective (reversed split child, immediate destroy, rank 0's writes delayed).
teardown() {  # teardown <tag> [teardown_race.py arguments]
  local tag=$1; shift
  timeout 1800 "$PY" tests/emulation/teardown_race.py --library "$LIB" --python "$PY" --lanes 2 --rounds 20     --slow-rank 0 --slow-ns 5000000 --work "$OUT/teardown-$tag" "$@" > "$OUT/teardown-$tag.log" 2>&1
  local code=$?
  if [ $code -eq 0 ]; then note "PASS teardown $tag: $(tail -1 "$OUT/teardown-$tag.log")"; else
    note "FAIL teardown $tag (exit $code): $(tail -1 "$OUT/teardown-$tag.log"); $OUT/teardown-$tag.log"
    failures=$((failures + 1))
  fi
}
teardown w4 --world 4
teardown w2-8MiB --world 2 --count 4194304
# Pipeline stage pairs of ring:8 (positions i and i + 4, every lane through relays) with the route planner's
# settings: TP4 splits and pair splits, grouped point-to-point, with and without the ring plan.
for plan in ring none; do
  args=(); [ $plan = none ] && args=(--no-ring-plan)
  timeout 1800 "$PY" tests/emulation/pp_pairs.py --library "$LIB" --python "$PY" ${args[@]+"${args[@]}"}     --work "$OUT/pp-pairs-$plan" > "$OUT/pp-pairs-$plan.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then note "PASS pipeline pairs of ring:8, ring plan $plan: $(tail -1 "$OUT/pp-pairs-$plan.log")"
  else note "FAIL pipeline pairs of ring:8, ring plan $plan (exit $code); $OUT/pp-pairs-$plan.log"
    failures=$((failures + 1)); fi
done
# Fail-stop: the pair default with LIBSIRCL_FAIL_STOP=1 (no check may end the process); and a late rank's peer
# ending its process within the wait limit (exit and abort), while without fail-stop it keeps a wrong output.
library pair-fail-stop 2 2 "$G/w2.json" LIBSIRCL_FAIL_STOP=1
timeout 900 "$PY" tests/emulation/fail_stop.py --library "$LIB" --python "$PY" --work "$OUT/fail-stop" \
  > "$OUT/fail-stop.log" 2>&1
code=$?
if [ $code -eq 0 ]; then note "PASS fail-stop: $(tail -1 "$OUT/fail-stop.log")"; else
  note "FAIL fail-stop (exit $code): $(tail -1 "$OUT/fail-stop.log"); $OUT/fail-stop.log"; failures=$((failures + 1)); fi
# Point-to-point channels on communicators of more than two ranks (LIBSIRCL_P2P_CHANNELS=on): the library run
# of four ranks with channels on (collectives as without them), then every case of p2p_channels.py: four and
# eight ranks with every ordered pair at once, a subset of pairs while the other ranks idle, a pipeline chain
# and the sendrecv ring; ring:8 under the route planner's settings with SIRCL's budget (relayed pairs refused)
# and with every relayed lane windowed; a size mismatch, a peer that is gone (fail-stop) and setup refusals.
library channels-on 4 2 "$G/w4.json" LIBSIRCL_P2P_CHANNELS=on
channels() {  # channels <tag> [p2p_channels.py arguments]
  local tag=$1; shift
  timeout 2400 "$PY" tests/emulation/p2p_channels.py --library "$LIB" --python "$PY" --work "$OUT/channels-$tag" "$@" \
    > "$OUT/channels-$tag.log" 2>&1
  local code=$?
  if [ $code -eq 0 ]; then note "PASS channels $tag: $(tail -1 "$OUT/channels-$tag.log")"; else
    note "FAIL channels $tag (exit $code): $(tail -1 "$OUT/channels-$tag.log"); $OUT/channels-$tag.log"
    failures=$((failures + 1))
  fi
}
channels w4 --world 4 --rounds 3
channels w4-l1 --world 4 --lanes 1 --rounds 2
channels w4-own-staging --world 4 --rounds 2 --env LIBSIRCL_STREAM_ORDERED_ALLOC=off
channels w8 --world 8 --rounds 2
channels ring8 --layout ring8 --rounds 2
channels ring8-alone --layout ring8-alone --rounds 2
channels size --case size
channels gone --case gone
channels setup --case setup
fi

# 3. PyTorch and setup failures.
want torch && for world in 2 4; do
  timeout 1800 "$PY" tests/emulation/torch_pg.py --launch --library "$LIB" --world $world > "$OUT/torch-w$world.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then note "PASS torch ProcessGroupNCCL w$world"; else
    note "FAIL torch ProcessGroupNCCL w$world (exit $code); $OUT/torch-w$world.log"; failures=$((failures + 1)); fi
done
# vLLM's pipeline exchange through torch.distributed (lazily created two-rank communicators per stage pair):
# a chain of four stages, and TP 4 x PP 2 under ring:8's route-planner settings; and the default group
# initialized eagerly (device_id), whose point-to-point runs on its four-rank communicator's channels.
want torch && for shape in chain tp4pp2 eager; do
  timeout 1800 "$PY" tests/emulation/torch_pp.py --launch --library "$LIB" --shape $shape --work "$OUT/torch-pp-$shape"     > "$OUT/torch-pp-$shape.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then note "PASS torch pipeline exchange $shape: $(tail -1 "$OUT/torch-pp-$shape.log")"; else
    note "FAIL torch pipeline exchange $shape (exit $code); $OUT/torch-pp-$shape.log"; failures=$((failures + 1)); fi
done
if want setup; then
timeout 1800 "$PY" tests/emulation/setup_failures.py --library "$LIB" --python "$PY" > "$OUT/setup-failures.log" 2>&1
code=$?
if [ $code -eq 0 ]; then note "PASS setup failures: $(tail -1 "$OUT/setup-failures.log")"; else
  note "FAIL setup failures (exit $code); $OUT/setup-failures.log"; failures=$((failures + 1)); fi
fi

# 4. Timing sweeps on two ranks (every size checked): the default, the chain and the ring.
want sweeps && for schedule in default chain ring; do
  D=$(mktemp -d "$OUT/perf-$schedule-XXXX")
  settings=()
  [ $schedule != default ] && settings=(SIRCL_LARGE_SCHEDULE=$schedule SIRCL_RING_MIN_BYTES=0 LIBSIRCL_RING_WINDOW=0)
  for r in 1 0; do
    env LIBSIRCL_TRANSPORT=emulation SIRCL_EMU_FABRIC=/libsircl-suite-perf-$$-$schedule LIBSIRCL_EMU_LANES=2 \
      ${settings[@]+"${settings[@]}"} timeout 2400 "$PY" tests/emulation/perf_rank.py --library "$LIB" --world 2 \
      --rank $r --dtypes bfloat16,float32 --id-file "$D/uid" --out "$D/perf$r.json" --min 16 --max $((32 << 20)) \
      --factor 4 --iters 5 --warmup 1 --graph 3 > "$D/perf$r.log" 2>&1 &
  done
  wait
  rm -f /dev/shm/libsircl-suite-perf-$$-$schedule*
  if grep -q "sizes checked, 0 wrong" "$D/perf0.log"; then note "PASS timing sweep $schedule: $(tail -1 "$D/perf0.log")"
  else note "FAIL timing sweep $schedule; $D/perf0.log"; failures=$((failures + 1)); fi
done

# 5. nccl-tests on two processes (NCCL_TESTS_BUILD), each line its own MPI-shim job; each rank reports a
# host name of its own (SIRCL_MPI_DISTINCT_HOSTS), so nccl-tests counts one rank per host and every rank uses
# device 0. NT_WORLD sets another process count and NT_ENV settings (NAME=VALUE ...) for every rank.
if [ -n "${NCCL_TESTS_BUILD:-}" ] && want nccl-tests; then
  N="$OUT/nccl-tests"
  mkdir -p "$N"
  line() {
    local test=$1; shift
    local world=${NT_WORLD:-2} tag port=$((29900 + RANDOM % 500)) r pids=() codes="" settings
    read -r -a settings <<< "${NT_ENV:-}"
    tag=$(printf '%s' "w$world $test $*" | tr -c 'A-Za-z0-9_.-' '_')
    for r in $(seq $((world - 1)) -1 0); do
      env LD_LIBRARY_PATH="$TREE/build/mpi-shim/lib:${LD_LIBRARY_PATH:-}" LIBSIRCL_TRANSPORT=emulation \
        SIRCL_EMU_FABRIC=/libsircl-suite-nt-$$ LIBSIRCL_EMU_LANES=1 SIRCL_MPI_SIZE=$world SIRCL_MPI_RANK=$r \
        SIRCL_MPI_DISTINCT_HOSTS=1 SIRCL_MPI_ROOT=127.0.0.1:$port SIRCL_MPI_TIMEOUT_S=600 \
        SIRCL_MPI_JOB="w$world $test $*" LIBSIRCL_RECEIPT="$N/receipt" ${settings[@]+"${settings[@]}"} \
        LD_PRELOAD="$(readlink -f "$LIB")" timeout --kill-after=30 1800 "$NCCL_TESTS_BUILD/$test" "$@" \
        > "$N/$tag-rank$r.log" 2>&1 &
      pids+=($!)
    done
    for r in "${pids[@]}"; do wait "$r"; codes="$codes $?"; done
    rm -f /dev/shm/libsircl-suite-nt-$$*
    # nccl-tests prints its results and check on rank 0 only; every rank must exit 0.
    if [ -z "$(echo "$codes" | tr -d ' 0')" ] && grep -q "Out of bounds values : 0 OK" "$N/$tag-rank0.log"; then
      note "PASS nccl-tests $test $* on $world processes${NT_ENV:+ ($NT_ENV)}"
    else note "FAIL nccl-tests $test $* on $world processes (exit codes, highest rank first:$codes); $N/$tag-rank0.log"
      failures=$((failures + 1)); fi
  }
  line all_reduce_perf -b 8 -e 1M -f 2 -g 1 -c 1 -n 5 -w 2
  for test in all_reduce_perf reduce_perf all_gather_perf reduce_scatter_perf broadcast_perf alltoall_perf \
              sendrecv_perf gather_perf scatter_perf; do
    line $test -b 8 -e 16M -f 4 -g 1 -c 1 -n 3 -w 1 -d bfloat16
  done
  line all_reduce_perf -b 8 -e 16M -f 4 -g 1 -c 1 -n 3 -w 1 -d float -G 2
  # Point-to-point channels on four processes with LIBSIRCL_P2P_CHANNELS=on: nccl-tests' sendrecv (every rank
  # sends to the next and receives from the previous), hypercube (exchanges with the rank across each
  # dimension) and alltoallv (one group of sends and receives with every rank), all with ncclSend and ncclRecv,
  # and its all-reduce beside them.
  for test in sendrecv_perf hypercube_perf alltoallv_perf; do
    NT_WORLD=4 NT_ENV=LIBSIRCL_P2P_CHANNELS=on line $test -b 8 -e 16M -f 4 -g 1 -c 1 -n 3 -w 1 -d bfloat16
  done
  NT_WORLD=4 NT_ENV=LIBSIRCL_P2P_CHANNELS=on line all_reduce_perf -b 8 -e 1M -f 4 -g 1 -c 1 -n 3 -w 1
fi
# 6. Mixed groups against SIRCL's DSL kernels (SIRCL_PACKAGE: the directory that holds SIRCL's
# sparkring_sircl package, with nvidia-cutlass-dsl importable): SIRCL's emulation harness with the link
# pack launched from C on every rank, under the ring schedules and SIRCL's own link checks; the ring
# all-reduce's two-pass entries too when the pack has them; ring:8 when MIXED_RING8=1.
if [ -n "${SIRCL_PACKAGE:-}" ] && want mixed; then
  make emulation-tools BUILD=build >> "$OUT/build.log" 2>&1
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  mixed() {  # mixed <tag> <layout> <lanes> [NAME=VALUE ...] [-- harness arguments]
    local tag=$1 layout=$2 lanes=$3 envs=() kv
    shift 3
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    local base="$OUT/mixed-$tag"
    env PYTHONPATH="$SIRCL_PACKAGE" SIRCL_TEST_BUILD_DIR="$OUT/sircl-build" CUTE_DSL_CACHE_DIR="$OUT/cute-cache"       CUTE_DSL_ARCH="sm_${CAP:-121}a" ${envs[@]+"${envs[@]}"} timeout 10800 "$PY" tests/emulation/mixed_group.py       --kp-lib build/libsirclkp_test.so --layout "$layout" --lanes "$lanes" --json "$base.json" "$@" > "$base.log" 2>&1
    local code=$?
    local summary
    summary=$(grep -E "checks, [0-9]+ failed" "$base.log" | tail -1)
    if [ $code -eq 0 ]; then note "PASS mixed $tag ($layout, $lanes lanes): $summary"; else
      note "FAIL mixed $tag ($layout, $lanes lanes, exit $code): ${summary:-no summary}; $base.log"
      failures=$((failures + 1))
    fi
  }
  RINGM="SIRCL_LARGE_SCHEDULE=ring SIRCL_GATHER_SCHEDULE=ring SIRCL_SCATTER_SCHEDULE=ring SIRCL_RING_MIN_BYTES=0"
  mixed ring-path_0-1 path:0-1 1 $RINGM
  mixed links-path_0-1 path:0-1 1 -- --suite sircl-links
  mixed links-path_0-3 path:0-3 2 -- --suite sircl-links
  if grep -q sircl_ring_reduce_two_pass kernels/prebuilt/sircl_links.fatbin; then
    mixed two-pass-links-path_0-1 path:0-1 1 LIBSIRCL_RING_REDUCE_PASSES=2 -- --suite sircl-links
  fi
  [ "${MIXED_RING8:-0}" = 1 ] && mixed links-ring_8 ring:8 2 -- --suite sircl-links
fi
note "$failures failed; logs in $OUT"
[ $failures -eq 0 ]
