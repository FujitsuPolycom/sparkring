#!/usr/bin/env bash
# SparkRing runtime builder — reads runtime/runtime-lock.json and drives a
# pinned, aarch64 container build of runtime/Containerfile.
#
# The lock file is the single source of truth for every version-shaped value.
# This script NEVER mutates the lock: after a successful build it prints the
# image digest and tells you exactly what to write back by hand.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/runtime/runtime-lock.json"
ENGINE="${CONTAINER_ENGINE:-docker}"

[ -f "${LOCK}" ] || { echo "FATAL: missing lock file: ${LOCK}" >&2; exit 1; }
command -v python3 >/dev/null || { echo "FATAL: python3 required to parse the lock" >&2; exit 1; }

# Fail-closed lock accessor: any missing key aborts.
lk() {
  python3 - "$LOCK" "$1" <<'EOF'
import json, sys
lock = json.load(open(sys.argv[1]))
cur = lock
for part in sys.argv[2].split("."):
    if part not in cur:
        sys.exit(f"FATAL: lock key missing: {sys.argv[2]}")
    cur = cur[part]
if not isinstance(cur, str) or not cur:
    sys.exit(f"FATAL: lock key empty/non-string: {sys.argv[2]}")
print(cur)
EOF
}

RUNTIME_ID="$(lk runtime_id)"

# Base image: prefer the pinned digest; fall back to tag only while the
# digest field is still "pending-first-build" (first successful build pins it).
base_ref() {  # $1 = lock section under base_image (builder|runtime)
  local repo tag digest
  repo="$(lk "base_image.$1.repository")"
  tag="$(lk "base_image.$1.tag")"
  digest="$(lk "base_image.$1.digest")"
  if [ "${digest}" = "pending-first-build" ]; then
    echo "WARN: base_image.$1.digest is 'pending-first-build' — building from tag ${repo}:${tag}; pin the digest after this build." >&2
    echo "${repo}:${tag}"
  else
    echo "${repo}@${digest}"
  fi
}

BASE_DEVEL_IMAGE="$(base_ref builder)"
BASE_RUNTIME_IMAGE="$(base_ref runtime)"

TORCH_VERSION="$(lk toolchain.torch_version)"
FLASHINFER_PY_VER="$(lk 'flashinfer.wheels.flashinfer-python')"
FLASHINFER_JIT_VER="$(lk flashinfer.wheels.flashinfer_jit_cache)"
DEEPGEMM_VERSION="$(lk deep_gemm.version)"

IMAGE_TAG="sparkring-runtime:${RUNTIME_ID}"

echo "== SparkRing runtime build =="
echo "   runtime_id : ${RUNTIME_ID}"
echo "   devel base : ${BASE_DEVEL_IMAGE}"
echo "   run base   : ${BASE_RUNTIME_IMAGE}"
echo "   vllm       : $(lk vllm.commit)"
echo "   tag        : ${IMAGE_TAG}"
echo "   NOTE: expect a multi-hour vLLM compile (see Containerfile header)."
echo

"${ENGINE}" build \
  --platform linux/arm64 \
  -f "${REPO_ROOT}/runtime/Containerfile" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_DEVEL_IMAGE="${BASE_DEVEL_IMAGE}" \
  --build-arg BASE_RUNTIME_IMAGE="${BASE_RUNTIME_IMAGE}" \
  --build-arg CUDA_ARCH="$(lk toolchain.compiler.cuda_architectures)" \
  --build-arg TORCH_SPEC="torch==${TORCH_VERSION}" \
  --build-arg TORCH_INDEX_URL="$(lk toolchain.torch_index_url)" \
  --build-arg VLLM_REPO="$(lk vllm.repository)" \
  --build-arg VLLM_COMMIT="$(lk vllm.commit)" \
  --build-arg SPARKINFER_REPO="$(lk sparkinfer.repository)" \
  --build-arg SPARKINFER_COMMIT="$(lk sparkinfer.commit)" \
  --build-arg FLASHINFER_COMMIT="$(lk flashinfer.commit)" \
  --build-arg FLASHINFER_PYTHON_SPEC="flashinfer-python==${FLASHINFER_PY_VER}" \
  --build-arg FLASHINFER_JIT_CACHE_SPEC="flashinfer_jit_cache==${FLASHINFER_JIT_VER}" \
  --build-arg FLASHINFER_WHEEL_INDEX="$(lk flashinfer.wheel_index)" \
  --build-arg DEEPGEMM_SPEC="deep_gemm==${DEEPGEMM_VERSION}" \
  --build-arg NCCL_REPO="$(lk nccl.repository)" \
  --build-arg NCCL_TAG="$(lk nccl.tag)" \
  --build-arg MODEL_REPO_ID="$(lk model.repository)" \
  --build-arg MODEL_REVISION="$(lk model.revision)" \
  --build-arg MODEL_CONFIG_SHA256="$(lk model.config_sha256)" \
  --build-arg RUNTIME_ID="${RUNTIME_ID}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
  "${REPO_ROOT}"

echo
echo "== build complete =="
DIGEST="$("${ENGINE}" image inspect --format '{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
IMAGE_ID="$("${ENGINE}" image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
echo "   image tag : ${IMAGE_TAG}"
echo "   image id  : ${IMAGE_ID}"
if [ -n "${DIGEST}" ]; then
  echo "   digest    : ${DIGEST}"
else
  echo "   digest    : (none yet — a registry digest exists only after push:"
  echo "                ${ENGINE} push <registry>/${IMAGE_TAG} , then inspect RepoDigests)"
fi
echo
echo "NEXT (manual — this script never edits the lock):"
echo "  1. Push the image, capture its sha256 registry digest."
echo "  2. Record it in the runtime manifest (image.digest) for this build."
echo "  3. If base_image.builder.digest / base_image.runtime.digest were"
echo "     'pending-first-build', resolve them (docker buildx imagetools inspect <ref>)"
echo "     and pin them now so the next build is digest-locked."
