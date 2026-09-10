#!/usr/bin/env bash
# build-image.sh — build the DeepSeek-V4.1-Flash GB10 serving image on one
# Spark *while it keeps serving another lane*.
#
# Reproduces the tonyd2wild/Kai overlay chain (build/Dockerfile.overlay,
# build_stable_ext.sh, build_overlay{3,4,5}.sh, prewarm5.py in
# https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), pinned, with
# three deliberate changes:
#   1. every compile runs inside a cgroup (--memory/--memory-swap/--cpus), so an
#      overrun kills the compiler and never the serving rank (their boot-3 wedge
#      was a 22-job runtime JIT on nodes with 10-15 GiB available);
#   2. ninja -j / MAX_JOBS default 2, NVCC threads 1, automatic retry at 1 job;
#   3. steps are idempotent (skipped when the overlay tag exists) and end in a
#      receipt + BUILD_OK / BUILD_FAILED marker in $WORK.
#
# Usage (on the build node, unattended):
#   nohup runtime/deepseek-v41-gb10/build-image.sh > build.out 2>&1 &
# Knobs: MEM (7g) CPUS (6) JOBS (2) WORK (~/deepseek-v41-build) MIN_AVAIL_GIB (12)
#        VERIFY=1 runs tools/verify5.py on the GPU at the end (only when the
#        node is not serving).
set -uo pipefail

BASE_IMG=vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c   # merge-base of deepseek-v41-feat, multi-arch
VLLM_SHA=e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba                             # vllm-project/vllm deepseek-v41-feat HEAD 2026-09-10T07:23Z
FI_SHA=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1                               # flashinfer v0.7.0rc1
FI_CUTLASS_SHA=b46b16d003484063bca4ed365e44095c4c6ed633
FI_CCCL_SHA=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
FI_SPDLOG_SHA=c3aed4b68373955e1cc94307683d44dca1515d2b
EXT_CUTLASS_TAG=v4.7.1                                                        # for _C_stable_libtorch
TAG="${TAG:-local/sparkring-deepseek-v41}"
SITE=/usr/local/lib/python3.12/dist-packages/vllm

WORK="${WORK:-$HOME/deepseek-v41-build}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# Cgroup sizing. 2026-09-10 finding: one CUTLASS TU of _C_stable_libtorch drives a
# single cicc past 7 GiB, so a 7 GiB cgroup stalls in reclaim (memory.events max
# 5.4M, no OOM kill, no progress). While a GLM mesh container serves on this node
# the safe ceiling is 7g and the build WILL stall on that TU; with the lane down
# the node has ~110 GiB free and the build should use it. Defaults pick by state.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qE "${SERVING_CONTAINER_RE:-^glm53-mtp3|^vllm_}"; then
  MEM="${MEM:-7g}"; CPUS="${CPUS:-6}"; JOBS="${JOBS:-2}"; MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-12}"
  echo "note: a serving container is running here; cgroup capped at $MEM (expect the CUTLASS TU stall unless MEM is raised)"
else
  MEM="${MEM:-40g}"; CPUS="${CPUS:-16}"; JOBS="${JOBS:-6}"; MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-60}"
fi
SRC="$WORK/vllm-src"
mkdir -p "$WORK"
LOG="$WORK/build.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
avail_gib() { echo $(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 )); }
fail() { log "FAILED: $*"; date -u +%FT%TZ > "$WORK/BUILD_FAILED"; exit 1; }
rm -f "$WORK/BUILD_OK" "$WORK/BUILD_FAILED"

# Never start a compile when the host is already tight; the cgroup only bounds
# what the build itself takes.
wait_headroom() {
  local n=0
  while [ "$(avail_gib)" -lt "$MIN_AVAIL_GIB" ]; do
    [ $n -eq 0 ] && log "waiting: MemAvailable $(avail_gib) GiB < $MIN_AVAIL_GIB GiB"
    n=$((n+1)); [ $n -gt 720 ] && fail "no headroom for 6 h"
    sleep 30
  done
}
have_tag() { docker image inspect "$1" >/dev/null 2>&1; }

# min-MemAvailable sampler for the receipt
( m=999; while [ ! -f "$WORK/BUILD_OK" ] && [ ! -f "$WORK/BUILD_FAILED" ]; do a=$(avail_gib); [ "$a" -lt "$m" ] && m=$a && echo "$m" > "$WORK/min-avail-gib"; sleep 5; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

log "=== build start host=$(hostname) MemAvailable=$(avail_gib)GiB MEM=$MEM CPUS=$CPUS JOBS=$JOBS"

# ---- 0. base image ---------------------------------------------------------
if ! have_tag "$BASE_IMG"; then
  log "pulling $BASE_IMG"
  docker pull -q "$BASE_IMG" >>"$LOG" 2>&1 || fail "pull $BASE_IMG"
fi
EP=$(docker inspect -f '{{json .Config.Entrypoint}}' "$BASE_IMG")
CMDJ=$(docker inspect -f '{{json .Config.Cmd}}' "$BASE_IMG")
[ "$EP" = null ] && EP='[]'; [ "$CMDJ" = null ] && CMDJ='[]'
log "base entrypoint=$EP cmd=$CMDJ id=$(docker inspect -f '{{.Id}}' "$BASE_IMG")"

# ---- 1. source at the pinned sha ---------------------------------------------
if [ ! -d "$SRC/.git" ] || [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$VLLM_SHA" ]; then
  rm -rf "$SRC"; git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/vllm-project/vllm.git
  git -C "$SRC" fetch -q --depth 1 origin "$VLLM_SHA" >>"$LOG" 2>&1 || fail "fetch $VLLM_SHA"
  git -C "$SRC" checkout -q FETCH_HEAD || fail "checkout"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$VLLM_SHA" ] || fail "source sha mismatch"
log "source at $VLLM_SHA"

# Helper: run a script in a bounded container from image $2, commit as $TAG:$1.
# Extra --change args may follow the script.
commit_step() {
  local name=$1 from=$2 script=$3; shift 3
  wait_headroom
  docker rm -f "deepseek-v41-$name" >/dev/null 2>&1
  log "step $name: run (MEM=$MEM CPUS=$CPUS JOBS=$JOBS)"
  docker run --name "deepseek-v41-$name" --memory "$MEM" --memory-swap "$MEM" --cpus "$CPUS" \
    --network host --entrypoint bash \
    -e MAX_JOBS="$JOBS" -e FLASHINFER_NVCC_THREADS=1 -e NVCC_THREADS=1 \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_HAS_FLASHINFER_CUBIN=1 \
    -v "$HERE/tools:/tools:ro" \
    "$from" -c "$script" >>"$LOG" 2>&1 &
  local runpid=$!
  stall_watch "deepseek-v41-$name" & local wpid=$!
  wait $runpid; local rc=$?
  kill $wpid 2>/dev/null; wait $wpid 2>/dev/null
  [ -f "$WORK/stall-deepseek-v41-$name" ] && { log "step $name: killed by the stall watchdog (cgroup at ceiling >5 min)"; rm -f "$WORK/stall-deepseek-v41-$name"; return 99; }
  if [ $rc -ne 0 ]; then
    log "step $name: rc=$rc (last lines):"; tail -15 "$LOG" | sed 's/^/    /'
    return $rc
  fi
  docker commit --change "ENTRYPOINT $EP" --change "CMD $CMDJ" "$@" "deepseek-v41-$name" "$TAG:$name" >>"$LOG" 2>&1 || return 1
  docker rm "deepseek-v41-$name" >/dev/null 2>&1
  log "step $name: committed $TAG:$name id=$(docker inspect -f '{{.Id}}' "$TAG:$name")"
}
# Kill a step whose cgroup sits at memory.max for 10 consecutive 30 s samples:
# that is reclaim thrash (2026-09-10: cicc at 7 GiB, memory.events max 5.4M, 0 OOM
# kills, zero progress). A killed step returns 99 and is NOT retried at JOBS=1,
# because the limit, not the parallelism, is what it hit.
stall_watch() {
  local c=$1 n=0 cid cg
  sleep 20
  cid=$(docker inspect -f '{{.Id}}' "$c" 2>/dev/null) || return
  cg=$(find /sys/fs/cgroup -maxdepth 3 -type d -name "*${cid:0:12}*" 2>/dev/null | head -1); [ -n "$cg" ] || return
  while sleep 30; do
    [ -f "$cg/memory.current" ] || return
    local cur max; cur=$(cat "$cg/memory.current"); max=$(cat "$cg/memory.max")
    [ "$max" = max ] && return
    if [ $(( cur * 100 / max )) -ge 98 ]; then n=$((n+1)); else n=0; fi
    if [ $n -ge 10 ]; then touch "$WORK/stall-$c"; docker kill "$c" >/dev/null 2>&1; return; fi
  done
}
retry_jobs1() { # $1 = step function name; skip the retry after a watchdog kill (rc 99)
  local last=$?
  [ "$last" = 99 ] && return 99
  if [ "$JOBS" != 1 ]; then log "retrying $1 with JOBS=1"; JOBS=1; "$1"; else return 1; fi
}

# ---- 2. FlashInfer layers on the base image (independent of the branch tree) ----
#         fi3: v0.7.0rc1 from source (0.6.18 lacks the SM120 sparse-MLA decode config
#         for V4.1's topk 1152); fi4: mxfp8_gemm_cutlass_sm120 prebuilt (its runtime
#         JIT wedged their fleet); fi5: sparse_mla_sm120 built under the runtime env.
OV3='
set -e
pip uninstall -y -q flashinfer-jit-cache flashinfer-cubin flashinfer-python || true
pip install -q ninja cmake setuptools wheel packaging
mkdir -p /opt/fi-src && curl -sL -m 900 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/'"$FI_SHA"'" | tar xz -C /opt/fi-src --strip-components=1
mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog
curl -sL -m 900 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/'"$FI_CUTLASS_SHA"'" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1
curl -sL -m 900 "https://codeload.github.com/NVIDIA/cccl/tar.gz/'"$FI_CCCL_SHA"'" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1
curl -sL -m 900 "https://codeload.github.com/gabime/spdlog/tar.gz/'"$FI_SPDLOG_SHA"'" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1
ls /opt/fi-src/3rdparty/cutlass/include/cutlass/cutlass.h /opt/fi-src/3rdparty/cccl/README.md /opt/fi-src/3rdparty/spdlog/include/spdlog/spdlog.h >/dev/null
cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation -q .
pip list 2>/dev/null | grep -iE "^flashinfer|nvidia-nccl"; rm -rf /opt/fi-src/build
python3 -c "import flashinfer; from flashinfer.mla import supported_sparse_mla_sm120_configs as f; c=f()[\"dsv4\"]; assert c.supports_decode(num_heads=16, topk=1152); print(\"fi3 python ok\", flashinfer.__version__)"
'
OV4='
set -e
timeout 5400 python3 -c "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as gen; spec = gen(); b = getattr(spec, \"build\", None); (b(verbose=True) if b else spec.build_and_load()); print(\"MXFP8-SM120-BUILT\")"
ls /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/ /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/mxfp8_gemm_cutlass_sm120/
'
OV5='set -e; timeout 5400 python3 /tools/prewarm5.py && ls /root/.cache/flashinfer/0.7.0rc1/121a/cached_ops/'
step_fi3() { commit_step fi3 "$BASE_IMG" "$OV3" --change 'ENV VLLM_HAS_FLASHINFER_CUBIN=1'; }
step_fi4() { commit_step fi4 "$TAG:fi3" "$OV4"; }
step_fi5() { commit_step fi5 "$TAG:fi4" "$OV5"; }
if ! have_tag "$TAG:fi3"; then step_fi3 || fail "fi3 (FlashInfer from source)"; else log "step fi3: exists"; fi
if ! have_tag "$TAG:fi4"; then step_fi4 || retry_jobs1 step_fi4 || fail "fi4 (mxfp8 gemm prebuild)"; else log "step fi4: exists"; fi
if ! have_tag "$TAG:fi5"; then step_fi5 || retry_jobs1 step_fi5 || fail "fi5 (sparse_mla prebuild)"; else log "step fi5: exists"; fi
if [ "${FI_ONLY:-0}" = 1 ]; then log "FI_ONLY=1: FlashInfer layers done ($TAG:fi5); ext + final image deferred"; date -u +%FT%TZ > "$WORK/FI_OK"; exit 0; fi

# ---- 3. _C_stable_libtorch rebuilt for sm_121a (the branch's kernels live there)
#         Needs a large cgroup: one CUTLASS TU takes cicc past 7 GiB.
EXT_SCRIPT='
set -o pipefail
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=12.1a
nvcc --version | tail -1 || { echo NO-NVCC; exit 2; }
command -v git >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null; }
python3 -c "import ninja, cmake" 2>/dev/null || pip install -q cmake ninja
export PATH=$(python3 -c "import sysconfig;print(sysconfig.get_path(\"scripts\"))"):$PATH
cd /src; mkdir -p build/_deps
if [ ! -f build/_deps/cutlass-src/include/cutlass/cutlass.h ]; then
  rm -rf build/_deps/cutlass-src; mkdir -p build/_deps/cutlass-src
  curl -sL -m 900 https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/'"$EXT_CUTLASS_TAG"' | tar xz -C build/_deps/cutlass-src --strip-components=1
fi
[ -f build/_deps/cutlass-src/include/cutlass/cutlass.h ] || { echo CUTLASS-MISSING; exit 2; }
cp -n CMakeLists.txt CMakeLists.txt.orig; cp CMakeLists.txt.orig CMakeLists.txt
sed -i -E "s|^(\s*)include\(cmake/external_projects/|\1# STABLE-ONLY BUILD: include(cmake/external_projects/|" CMakeLists.txt
echo "stable-only excludes: $(grep -c "STABLE-ONLY BUILD" CMakeLists.txt)"
PYPATH=$(python3 -c "import sys;print(\":\".join(p for p in sys.path if p))")
TORCH_PREFIX=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
NVRTC=$(ls /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/lib64/libnvrtc.so.* /usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so* /usr/lib/aarch64-linux-gnu/libnvrtc.so* 2>/dev/null | head -1)
echo "nvrtc=$NVRTC"
rm -rf build/CMakeCache.txt build/CMakeFiles
cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH="$PYPATH" \
  -DFETCHCONTENT_BASE_DIR=/src/build/_deps -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/src/build/_deps/cutlass-src \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNVCC_THREADS=1 -DCUDA_nvrtc_LIBRARY="$NVRTC" 2>&1 | tail -8 || exit 2
cmake --build /src/build --target _C_stable_libtorch -j "$MAX_JOBS" 2>&1 | grep -E --line-buffered "^\[[0-9]+/[0-9]+\]|error|Error|FAILED|Linking" | awk "NR%25==1 || /error|Error|FAILED|Linking/ {print; fflush()}"
rc=${PIPESTATUS[0]}; echo "ninja rc=$rc"; [ $rc -eq 0 ] || exit $rc
so=$(find /src/build -maxdepth 2 -name "_C_stable_libtorch*.so" | head -1); [ -n "$so" ] || { echo NO-SO; exit 2; }
inimg=$(find '"$SITE"' -maxdepth 1 -name "_C_stable_libtorch*" | head -1); echo "image ext: $inimg"
cp "$so" /src/vllm/$(basename "${inimg:-$so}") && sha256sum /src/vllm/_C_stable_libtorch*.so
'
step_ext() {
  wait_headroom
  docker rm -f deepseek-v41-ext >/dev/null 2>&1
  log "step ext: build _C_stable_libtorch (JOBS=$JOBS)"
  docker run --rm --name deepseek-v41-ext --memory "$MEM" --memory-swap "$MEM" --cpus "$CPUS" --network host \
    --entrypoint bash -e MAX_JOBS="$JOBS" -v "$SRC:/src" "$BASE_IMG" -c "$EXT_SCRIPT" >>"$LOG" 2>&1
}
if ls "$SRC"/vllm/_C_stable_libtorch*.so >/dev/null 2>&1; then
  log "step ext: already built ($(ls "$SRC"/vllm/_C_stable_libtorch*.so))"
else
  step_ext || retry_jobs1 step_ext || fail "stable ext build"
  ls "$SRC"/vllm/_C_stable_libtorch*.so >/dev/null 2>&1 || fail "stable ext .so missing after build"
fi
EXT_SHA=$(sha256sum "$SRC"/vllm/_C_stable_libtorch*.so | cut -c1-16)
log "stable ext sha256 $EXT_SHA"

# ---- 4. final: branch python tree + rebuilt ext over the FlashInfer layers -----
if ! have_tag "$TAG:overlay5"; then
  CTX="$WORK/final"; mkdir -p "$CTX"
  rsync -a --delete --exclude '__pycache__' "$SRC/vllm/" "$CTX/vllm/"   # own context: vLLM's .dockerignore drops vllm/*.so
  cat > "$CTX/Dockerfile" <<DF
FROM $TAG:fi5
# vllm-project/vllm deepseek-v41-feat @ $VLLM_SHA python tree + _C_stable_libtorch rebuilt for sm_121a
COPY vllm/ $SITE/
RUN find $SITE -name "__pycache__" -type d -prune -exec rm -rf {} + && python3 -c "import vllm; print('final', vllm.__version__)"
LABEL org.dgx-sparks.dsv41.vllm_sha=$VLLM_SHA org.dgx-sparks.dsv41.ext_sha256_16=$EXT_SHA org.dgx-sparks.dsv41.flashinfer_sha=$FI_SHA
DF
  log "step final: docker build"
  docker build -q -t "$TAG:overlay5" "$CTX" >>"$LOG" 2>&1 || fail "final build"
  log "step final: $(docker inspect -f '{{.Id}}' "$TAG:overlay5")"
else log "step final: exists"; fi
docker tag "$TAG:overlay5" "$TAG:20260910"

# ---- 7. optional GPU verify (not while the node serves) ------------------------
if [ "${VERIFY:-0}" = 1 ]; then
  log "verify5 on GPU"
  docker run --rm --gpus all -v "$HERE/tools/verify5.py:/v.py:ro" \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e VLLM_HAS_FLASHINFER_CUBIN=1 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
    --entrypoint bash "$TAG:overlay5" -c 'timeout 900 python3 /v.py 2>&1 | grep -E "VERIFY|Building JIT" | cut -c1-160' | tee -a "$LOG"
fi

# ---- receipt -----------------------------------------------------------------
{
  echo "built_utc=$(date -u +%FT%TZ) host=$(hostname)"
  echo "base_image=$BASE_IMG base_id=$(docker inspect -f '{{.Id}}' "$BASE_IMG")"
  echo "vllm_sha=$VLLM_SHA flashinfer_sha=$FI_SHA stable_ext_sha256_16=$EXT_SHA"
  for t in fi3 fi4 fi5 overlay5; do echo "$t=$(docker inspect -f '{{.Id}}' "$TAG:$t") size=$(docker image inspect -f '{{.Size}}' "$TAG:$t")"; done
  echo "min_memavailable_gib_during_build=$(cat "$WORK/min-avail-gib" 2>/dev/null)"
  echo "patches:"; cat "$HERE/patches/MD5SUMS" | sed 's/^/  /'
} > "$WORK/receipt.txt"
cat "$WORK/receipt.txt" | tee -a "$LOG"
date -u +%FT%TZ > "$WORK/BUILD_OK"
log "=== BUILD_OK $TAG:overlay5"
