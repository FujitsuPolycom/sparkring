#!/usr/bin/env bash
# Build a GLM-5.3 SparkCache image by replacing only attested Python sources.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${here}" rev-parse --show-toplevel)"
pins="${here}/pins.json"
engine="${CONTAINER_ENGINE:-docker}"
image="${IMAGE:-sparkring-glm53-sparkcache:vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64}"
receipt_path="${BUILD_RECEIPT:-${PWD}/glm53-public-python-overlay-image-receipt.json}"

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
  runtime/glm53-flash-adaptive-mtp-python-overlay
  LICENSE
)
git -C "${repo_root}" diff --quiet HEAD -- "${tracked_inputs[@]}" ||
  fatal "builder inputs differ from the checked-out SparkRing revision"
untracked="$(git -C "${repo_root}" ls-files --others --exclude-standard -- "${tracked_inputs[@]}")"
[[ -z "${untracked}" ]] ||
  fatal "builder inputs include untracked files: ${untracked%%$'\n'*}"

public_base="$(read_pin public_base.reference)"
public_base_id="$(read_pin public_base.image_id)"
arm_builder="$(read_pin builder.arm_builder)"
vllm_native_commit="$(read_pin vllm.native_commit)"
vllm_python_commit="$(read_pin vllm.python_commit)"
vllm_python_tree="$(read_pin vllm.python_tree)"
overlay_manifest_sha256="$(read_pin vllm.overlay_manifest_sha256)"
b12x_commit="$(read_pin b12x.commit)"
b12x_tree="$(read_pin b12x.tree)"
sparkcache_commit="$(read_pin sparkcache.commit)"
sparkcache_source_sha256="$(read_pin sparkcache.source_tree_sha256)"
sparkring_revision="$(git -C "${repo_root}" rev-parse HEAD)"

"${engine}" pull --platform linux/arm64 "${public_base}"
python3 "${here}/verify_image.py" \
  --engine "${engine}" --pins "${pins}" --base-image "${public_base}" >/dev/null

context="$(mktemp -d)"
cleanup() {
  # `context` is created by mktemp in this process and never accepts caller input.
  rm -rf -- "${context}"
}
trap cleanup EXIT

python3 "${here}/prepare_context.py" \
  --repo-root "${repo_root}" "${context}" >/dev/null
python3 "${here}/prepare_context.py" --verify "${context}" >/dev/null
source_receipt_sha256="$(sha256sum "${context}/receipt.json" | cut -d' ' -f1)"
mkdir -p "${context}/base-probe"
"${engine}" run --rm --entrypoint python3 \
  --volume "${here}:/contract:ro" \
  --volume "${context}/base-probe:/out" \
  "${public_base}" \
  /contract/overlay_contract.py \
  --pins /contract/pins.json \
  --manifest /contract/vllm-python-overlay.json \
  record-base \
  --site-root /usr/local/lib/python3.12/dist-packages \
  --console-script /usr/local/bin/vllm \
  --output /out/retained-native.json >/dev/null
native_elf_manifest_sha256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["native_elf_manifest_sha256"])' "${context}/base-probe/retained-native.json")"
native_dispatch_manifest_sha256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["native_dispatch_manifest_sha256"])' "${context}/base-probe/retained-native.json")"
native_stage="sparkring-sparkcache-native:${sparkcache_commit:0:12}-${sparkring_revision:0:12}"
"${engine}" build \
  --platform linux/arm64 \
  --target sparkcache-native \
  --file "${context}/bundle/runtime/Containerfile" \
  --build-arg "ARM_BUILDER=${arm_builder}" \
  --tag "${native_stage}" \
  "${context}"
sparkcache_native_sha256="$("${engine}" run --rm --entrypoint sha256sum \
  "${native_stage}" \
  /build/sparkcache-native/build-cuda/libspark_cache_placement.so | cut -d' ' -f1)"

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/bundle/runtime/Containerfile" \
  --build-arg "PUBLIC_BASE=${public_base}" \
  --build-arg "PUBLIC_BASE_ID=${public_base_id}" \
  --build-arg "ARM_BUILDER=${arm_builder}" \
  --build-arg "VLLM_NATIVE_COMMIT=${vllm_native_commit}" \
  --build-arg "VLLM_PYTHON_COMMIT=${vllm_python_commit}" \
  --build-arg "VLLM_PYTHON_TREE=${vllm_python_tree}" \
  --build-arg "B12X_COMMIT=${b12x_commit}" \
  --build-arg "B12X_TREE=${b12x_tree}" \
  --build-arg "SPARKCACHE_COMMIT=${sparkcache_commit}" \
  --build-arg "SPARKCACHE_SOURCE_SHA256=${sparkcache_source_sha256}" \
  --build-arg "SPARKRING_REVISION=${sparkring_revision}" \
  --build-arg "SOURCE_RECEIPT_SHA256=${source_receipt_sha256}" \
  --build-arg "OVERLAY_MANIFEST_SHA256=${overlay_manifest_sha256}" \
  --build-arg "NATIVE_ELF_MANIFEST_SHA256=${native_elf_manifest_sha256}" \
  --build-arg "NATIVE_DISPATCH_MANIFEST_SHA256=${native_dispatch_manifest_sha256}" \
  --build-arg "SPARKCACHE_NATIVE_SHA256=${sparkcache_native_sha256}" \
  --tag "${image}" \
  "${context}"

python3 "${here}/verify_image.py" \
  --engine "${engine}" --pins "${pins}" --image "${image}" \
  --output "${receipt_path}" >/dev/null
printf 'image=%s\nreceipt=%s\n' "${image}" "${receipt_path}"
"${engine}" image inspect --format '{{.Id}}' "${image}"
