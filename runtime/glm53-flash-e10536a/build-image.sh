#!/usr/bin/env bash
# Build one GLM-5.3 ARM64 runtime from public, immutable source inputs.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${here}" rev-parse --show-toplevel)"
pins="${here}/pins.json"
engine="${CONTAINER_ENGINE:-docker}"
image="${IMAGE:-sparkring-glm53-runtime:e10536a-source-arm64}"
receipt_path="${BUILD_RECEIPT:-${PWD}/glm53-runtime-image-receipt.json}"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 78
}

read_pin() {
  python3 - "${pins}" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[int(component)] if isinstance(value, list) else value[component]
print(value)
PY
}

tracked_inputs=(
  runtime/glm53-flash-e10536a
  spark_transport/nccl/nccl-2.30.7-switchless-cycle.patch
)
git -C "${repo_root}" diff --quiet HEAD -- "${tracked_inputs[@]}" ||
  fatal "builder inputs differ from the checked-out SparkRing revision"
untracked="$(git -C "${repo_root}" ls-files --others --exclude-standard -- "${tracked_inputs[@]}")"
[[ -z "${untracked}" ]] ||
  fatal "builder inputs include untracked files: ${untracked%%$'\n'*}"

arm_builder="$(read_pin public_image_build.base_images.arm_builder.reference)"
cuda_runtime="$(read_pin public_image_build.base_images.cuda_runtime.reference)"
vllm_commit="$(read_pin public_image_build.sources.vllm.commit)"
b12x_commit="$(read_pin public_image_build.sources.b12x.commit)"
nccl_commit="$(read_pin public_image_build.sources.nccl.commit)"
nccl_patched_tree="$(read_pin public_image_build.sources.nccl.patched_tree)"
nccl_patch_sha256="$(read_pin public_image_build.sources.nccl.patches.0.sha256)"
nccl_cuda_arch="$(read_pin public_image_build.toolchain.nccl_cuda_arch)"
cuda_version="$(read_pin public_image_build.toolchain.cuda)"
torch_arch_list="$(read_pin public_image_build.toolchain.vllm_torch_cuda_arch_list)"
instanttensor_version="$(read_pin public_image_build.instanttensor.version)"
instanttensor_sha256="$(read_pin public_image_build.instanttensor.sdist_sha256)"
sparkring_revision="$(git -C "${repo_root}" rev-parse HEAD)"

context="$(mktemp -d)"
cleanup() {
  # `context` is created by mktemp in this process and never accepts caller input.
  rm -rf -- "${context}"
}
trap cleanup EXIT

python3 "${here}/prepare_context.py" \
  --repo-root "${repo_root}" \
  --pins "${pins}" \
  "${context}/prepared" >/dev/null
python3 "${here}/prepare_context.py" \
  --verify --pins "${pins}" "${context}/prepared" >/dev/null
cp -a "${context}/prepared/." "${context}/"
rm -rf -- "${context}/prepared"
cp "${context}/bundle/runtime/Containerfile.seed" "${context}/Containerfile.seed"
cp "${context}/bundle/runtime/Containerfile" "${context}/Containerfile"

source_receipt_sha256="$(sha256sum "${context}/receipt.json" | cut -d' ' -f1)"
seed_image="sparkring-glm53-seed:cu130-py312-${source_receipt_sha256:0:12}"
vllm_image="sparkring-glm53-vllm:${vllm_commit:0:12}-arm64"

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/Containerfile.seed" \
  --build-arg "ARM_BUILDER=${arm_builder}" \
  --build-arg "CUDA_RUNTIME=${cuda_runtime}" \
  --build-arg "INSTANTTENSOR_VERSION=${instanttensor_version}" \
  --build-arg "INSTANTTENSOR_SDIST_SHA256=${instanttensor_sha256}" \
  --tag "${seed_image}" \
  "${context}"

"${engine}" build \
  --platform linux/arm64 \
  --target vllm-openai \
  --file "${context}/bundle/sources/vllm/docker/Dockerfile" \
  --build-arg "CUDA_VERSION=${cuda_version}" \
  --build-arg "BUILD_BASE_IMAGE=${arm_builder}" \
  --build-arg "FINAL_BASE_IMAGE=${seed_image}" \
  --build-arg "torch_cuda_arch_list=${torch_arch_list}" \
  --build-arg "max_jobs=${MAX_JOBS:-4}" \
  --build-arg "nvcc_threads=${NVCC_THREADS:-4}" \
  --build-arg GIT_REPO_CHECK=0 \
  --build-arg "VLLM_BUILD_COMMIT=${vllm_commit}" \
  --build-arg VLLM_BUILD_PIPELINE=sparkring-glm53-public \
  --build-arg "VLLM_IMAGE_TAG=${vllm_image}" \
  --tag "${vllm_image}" \
  "${context}/bundle/sources/vllm"

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/Containerfile" \
  --build-arg "VLLM_BASE=${vllm_image}" \
  --build-arg "ARM_BUILDER=${arm_builder}" \
  --build-arg "VLLM_COMMIT=${vllm_commit}" \
  --build-arg "B12X_COMMIT=${b12x_commit}" \
  --build-arg "NCCL_COMMIT=${nccl_commit}" \
  --build-arg "NCCL_PATCHED_TREE=${nccl_patched_tree}" \
  --build-arg "NCCL_PATCH_SHA256=${nccl_patch_sha256}" \
  --build-arg "NCCL_CUDA_ARCH=${nccl_cuda_arch}" \
  --build-arg "SOURCE_RECEIPT_SHA256=${source_receipt_sha256}" \
  --build-arg "SPARKRING_REVISION=${sparkring_revision}" \
  --build-arg "MAX_JOBS=${MAX_JOBS:-4}" \
  --tag "${image}" \
  "${context}"

python3 "${here}/verify_image.py" \
  --engine "${engine}" --image "${image}" --pins "${pins}" \
  --output "${receipt_path}" >/dev/null
printf 'image=%s\nreceipt=%s\n' "${image}" "${receipt_path}"
"${engine}" image inspect --format '{{.Id}}' "${image}"
