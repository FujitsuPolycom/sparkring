#!/usr/bin/env bash
# Build the receipt-gated EXL3 R7 ARM64 image over an immutable parent image.
#
# The parent image is identified by an immutable digest/image ID, never a
# mutable tag alone. The script fails closed if the observed image ID drifts
# from the required value.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="${CONTAINER_ENGINE:-docker}"
image="${IMAGE:-sparkring-r7:arm64-sm121}"

# The parent image tag is accepted for resolution, but the build proceeds only
# after the engine resolves it to an immutable content-addressed ID that
# matches the required value.
base_image="${BASE_IMAGE:?BASE_IMAGE is required (parent image tag or digest)}"
base_image_id="${BASE_IMAGE_ID:?BASE_IMAGE_ID is required (immutable sha256 image ID)}"
base_image_licenses="${BASE_IMAGE_LICENSES:?BASE_IMAGE_LICENSES is required (SPDX expression for the parent image)}"

context="$(mktemp -d)"
trap 'rm -rf -- "${context}"' EXIT
deps_cache="${R7_DEPS_CACHE:-${here}/../../.sparkring/r7-build-deps}"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 78
}

# Resolve the parent image to its immutable ID and fail closed on drift.
observed_base="$("${engine}" image inspect --format '{{.Id}}' "${base_image}" 2>/dev/null || true)"
if [[ -z "${observed_base}" ]]; then
  fatal "parent image not found: ${base_image}"
fi
if [[ "${observed_base}" != "${base_image_id}" ]]; then
  fatal "parent image identity drift: expected ${base_image_id}, got ${observed_base}"
fi

repo_root="$(git -C "${here}" rev-parse --show-toplevel 2>/dev/null)" ||
  fatal "builder must run from a Git checkout"
sparkring_revision="$(git -C "${repo_root}" rev-parse HEAD)"
if ! git -C "${repo_root}" diff --quiet HEAD -- \
  runtime/exl3-r7 \
  runtime/build-public-overlay.py \
  runtime/public-overlay-files.json \
  spark_transport; then
  fatal "builder inputs differ from SparkRing revision ${sparkring_revision}"
fi
untracked_inputs="$(git -C "${repo_root}" ls-files --others --exclude-standard -- \
  runtime/exl3-r7 \
  runtime/build-public-overlay.py \
  runtime/public-overlay-files.json \
  spark_transport)"
if [[ -n "${untracked_inputs}" ]]; then
  fatal "builder inputs include untracked files: ${untracked_inputs%%$'\n'*}"
fi
image_licenses="${base_image_licenses} AND Apache-2.0 AND MIT AND BSD-3-Clause"

if [[ -n "${PREPARED_SOURCES:-}" ]]; then
  [[ -d "${PREPARED_SOURCES}" ]] || fatal "PREPARED_SOURCES is not a directory: ${PREPARED_SOURCES}"
  [[ -f "${PREPARED_SOURCES}/receipt.json" ]] || fatal "missing prepared-source receipt: ${PREPARED_SOURCES}/receipt.json"
  # Verify the receipt content, not just its existence.
  python3 "${here}/prepare_context.py" --verify "${PREPARED_SOURCES}" >/dev/null
  cp -a "${PREPARED_SOURCES}" "${context}/sources"
else
  python3 "${here}/prepare_context.py" "${context}/sources"
fi

cp "${here}/Containerfile" "${context}/Containerfile"
python3 "${here}/prepare_build_deps.py" "${deps_cache}"
python3 "${here}/prepare_build_deps.py" --verify "${deps_cache}"
mkdir -p "${context}/bundle/deps" "${context}/bundle/runtime"
cp -a "${deps_cache}/cutlass" "${context}/bundle/deps/cutlass"
cp -a "${deps_cache}/triton_kernels" "${context}/bundle/deps/triton_kernels"
cp "${here}/verify_runtime.py" "${context}/bundle/runtime/verify_runtime.py"
cp "${here}/entrypoint.sh" "${context}/bundle/runtime/entrypoint.sh"
cp "${here}/bake_runtime_artifacts.py" \
  "${context}/bundle/runtime/bake_runtime_artifacts.py"
cp "${here}/build_parallel_state_shared_capture_overlay.py" \
  "${context}/bundle/runtime/build_parallel_state_shared_capture_overlay.py"
cp "${here}/requirements-quack.txt" \
  "${context}/bundle/runtime/requirements-quack.txt"
cp "${here}/requirements-tvm-ffi.txt" \
  "${context}/bundle/runtime/requirements-tvm-ffi.txt"
cp "${here}/patch_sm121_cmake.py" "${context}/bundle/patch_sm121_cmake.py"
cp "${here}/../build-public-overlay.py" \
  "${context}/bundle/build-public-overlay.py"
cp "${here}/../public-overlay-files.json" \
  "${context}/bundle/public-overlay-files.json"
cp -a "${repo_root}/spark_transport" "${context}/bundle/spark_transport"
cp "${here}/exllamav3-arm64-external-collectives.patch" \
  "${context}/bundle/exllamav3-arm64-external-collectives.patch"
mv "${context}/sources/vllm" "${context}/bundle/vllm"
mv "${context}/sources/b12x" "${context}/bundle/b12x"
# Compute the source receipt hash for the OCI label before removing sources.
source_receipt_sha="$(sha256sum "${context}/sources/receipt.json" | awk '{print $1}')"
rm -rf -- "${context}/sources"

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/Containerfile" \
  --tag "${image}" \
  --build-arg "BASE_IMAGE=${observed_base}" \
  --build-arg "BASE_IMAGE_ID=${observed_base}" \
  --build-arg "BASE_IMAGE_LICENSES=${base_image_licenses}" \
  --build-arg "IMAGE_LICENSES=${image_licenses}" \
  --build-arg "SPARKRING_REVISION=${sparkring_revision}" \
  --build-arg "SOURCE_RECEIPT_SHA256=${source_receipt_sha}" \
  --build-arg "VLLM_COMMIT=e2666d9a65f41fc376607531453cbd57c4c71016" \
  --build-arg "B12X_COMMIT=7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b" \
  --build-arg "CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2" \
  --build-arg "TRITON_KERNELS_COMMIT=0add68262ab0a2e33b84524346cb27cbb2787356" \
  --build-arg CUDA_ARCH=121 \
  --build-arg "MAX_JOBS=${MAX_JOBS:-8}" \
  "${context}"

"${engine}" image inspect --format '{{.Id}}' "${image}"
