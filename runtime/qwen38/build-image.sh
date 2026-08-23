#!/usr/bin/env bash
# Build one local ARM64 Qwen3.8 image from public, immutable source inputs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${here}" rev-parse --show-toplevel)"
engine="${CONTAINER_ENGINE:-docker}"
image="${IMAGE:-sparkring-qwen38:arm64-sm121}"
pins="${here}/pins.json"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 78
}

read_pin() {
  python3 - "$pins" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[component]
print(value)
PY
}

expected_parent="$(read_pin parent_image.reference)"
base_image="${BASE_IMAGE:-${expected_parent}}"
[[ "${base_image}" == "${expected_parent}" ]] ||
  fatal "parent reference drift: expected ${expected_parent}, got ${base_image}"
base_image_id="${BASE_IMAGE_ID:?BASE_IMAGE_ID is required (docker image inspect .Id)}"
base_image_licenses="${BASE_IMAGE_LICENSES:-LicenseRef-NVIDIA-Deep-Learning-Container}"

observed_base="$("${engine}" image inspect --format '{{.Id}}' "${base_image}" 2>/dev/null || true)"
[[ -n "${observed_base}" ]] || fatal "parent image not found: ${base_image}"
[[ "${observed_base}" == "${base_image_id}" ]] ||
  fatal "parent image identity drift: expected ${base_image_id}, got ${observed_base}"

tracked_inputs=(
  runtime/qwen38
  scripts/qwen38_dgx2_serve.sh
  scripts/qwen38_dgx4_serve.sh
  spark_transport/nccl/nccl-2.30.7-skip-tree-pat.patch
  spark_transport/nccl/nccl-2.30.7-advertise-all-listener-gids.patch
)
git -C "${repo_root}" diff --quiet HEAD -- "${tracked_inputs[@]}" ||
  fatal "builder inputs differ from the checked-out SparkRing revision"
untracked="$(git -C "${repo_root}" ls-files --others --exclude-standard -- "${tracked_inputs[@]}")"
[[ -z "${untracked}" ]] || fatal "builder inputs include untracked files: ${untracked%%$'\n'*}"

context="$(mktemp -d)"
parent_build_tag=""
cleanup() {
  if [[ -n "${parent_build_tag}" ]]; then
    "${engine}" rmi "${parent_build_tag}" >/dev/null 2>&1 || true
  fi
  rm -rf -- "${context}"
}
trap cleanup EXIT

python3 "${here}/prepare_context.py" \
  --repo-root "${repo_root}" \
  --pins "${pins}" \
  "${context}/prepared" >/dev/null
python3 "${here}/prepare_context.py" \
  --verify \
  --pins "${pins}" \
  "${context}/prepared" >/dev/null
cp -a "${context}/prepared/." "${context}/"
rm -rf -- "${context}/prepared"
cp "${here}/Containerfile" "${context}/Containerfile"

source_receipt_sha="$(sha256sum "${context}/receipt.json" | awk '{print $1}')"
sparkring_revision="$(git -C "${repo_root}" rev-parse HEAD)"
parent_build_tag="sparkring-qwen38-parent:${observed_base#sha256:}"
parent_build_tag="${parent_build_tag:0:70}"
"${engine}" tag "${observed_base}" "${parent_build_tag}"

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/Containerfile" \
  --tag "${image}" \
  --build-arg "BASE_IMAGE=${parent_build_tag}" \
  --build-arg "BASE_IMAGE_ID=${observed_base}" \
  --build-arg "BASE_IMAGE_LICENSES=${base_image_licenses}" \
  --build-arg "SPARKRING_REVISION=${sparkring_revision}" \
  --build-arg "SOURCE_RECEIPT_SHA256=${source_receipt_sha}" \
  --build-arg "NCCL_LIBRARY_SHA256=$(read_pin nccl.library_sha256)" \
  --build-arg "NCCL_CUDA_ARCH=$(read_pin toolchain.nccl_cuda_arch)" \
  --build-arg "TORCH_INDEX_URL=$(read_pin toolchain.torch_index_url)" \
  --build-arg "VLLM_TORCH_CUDA_ARCH_LIST=$(read_pin toolchain.vllm_torch_cuda_arch_list)" \
  --build-arg "EXLLAMAV3_TORCH_CUDA_ARCH_LIST=$(read_pin toolchain.exllamav3_torch_cuda_arch_list)" \
  --build-arg "CUDA_TOOLKIT_PACKAGE_VERSION=$(read_pin toolchain.cuda_toolkit_package_version)" \
  --build-arg "MAX_JOBS=${MAX_JOBS:-8}" \
  "${context}"

"${engine}" image inspect --format '{{.Id}}' "${image}"
