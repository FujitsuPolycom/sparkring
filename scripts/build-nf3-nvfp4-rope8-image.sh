#!/usr/bin/env bash
# Build the optional NVFP4-latent / FP8-RoPE layer over the pinned NF3 image.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-docker}"
CACHE_ROOT="${SPARKRING_BOOTSTRAP_CACHE:-${HOME}/.cache/sparkring/nf3-bootstrap}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-sparkring/glm52-nf3-nvfp4-rope8:local}"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null || fatal "git is required"
command -v python3 >/dev/null || fatal "python3 is required"
command -v "${ENGINE}" >/dev/null || fatal "${ENGINE} is required"
[[ "$(uname -m)" == "aarch64" ]] ||
  fatal "the NF3 image must be built natively on an ARM64 DGX Spark"
[[ -z "$(git -C "${ROOT}" status --porcelain)" ]] ||
  fatal "SparkRing checkout is dirty; commit or restore it before building"

SOURCE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
NF3_IMAGE="sparkring/glm52-nf3-fp8:${SOURCE_COMMIT:0:12}"
FASTSTART_IMAGE="sparkring/glm52-faststart:${SOURCE_COMMIT:0:12}"

OUTPUT_IMAGE="${NF3_IMAGE}" bash "${ROOT}/scripts/build-nf3-image.sh"
NF3_IMAGE_ID="$("${ENGINE}" image inspect "${NF3_IMAGE}" --format '{{.Id}}')"
MLA_IMAGE_ID="$("${ENGINE}" image inspect "${FASTSTART_IMAGE}" --format '{{.Id}}')"

write_final_receipt() {
  local final_image_id="$1"
  local receipt_dir="${CACHE_ROOT}/receipts"
  local verifier_report="${receipt_dir}/nf3-nvfp4-rope8-verification.json"
  local verifier_temporary="${verifier_report}.tmp"
  local receipt="${receipt_dir}/nf3-nvfp4-rope8-runtime.json"
  mkdir -p -- "${receipt_dir}"
  "${ENGINE}" run --rm --entrypoint /bin/cat \
    "${OUTPUT_IMAGE}" \
    /opt/sparkring/nf3-nvfp4-rope8-verification.json \
    > "${verifier_temporary}"
  mv -- "${verifier_temporary}" "${verifier_report}"
  python3 "${ROOT}/runtime/write-nf3-nvfp4-receipt.py" \
    --image "${OUTPUT_IMAGE}" \
    --image-id "${final_image_id}" \
    --nf3-image-id "${NF3_IMAGE_ID}" \
    --mla-image-id "${MLA_IMAGE_ID}" \
    --source-commit "${SOURCE_COMMIT}" \
    --verifier-report "${verifier_report}" \
    --output "${receipt}"
}

if "${ENGINE}" image inspect "${OUTPUT_IMAGE}" >/dev/null 2>&1; then
  labels="$("${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format \
    '{{index .Config.Labels "org.sparkring.kv_profile"}} {{index .Config.Labels "org.sparkring.parent.nf3_image_id"}} {{index .Config.Labels "org.sparkring.parent.mla_image_id"}} {{index .Config.Labels "org.sparkring.source_commit"}}')"
  if [[ "${labels}" == \
    "nvfp4-rope8 ${NF3_IMAGE_ID} ${MLA_IMAGE_ID} ${SOURCE_COMMIT}" ]] &&
    "${ENGINE}" run --rm --entrypoint /opt/venv/bin/python \
      "${OUTPUT_IMAGE}" /opt/sparkring/verify-nf3-nvfp4-rope8.py \
      >/dev/null; then
    EXISTING_IMAGE_ID="$(
      "${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format '{{.Id}}'
    )"
    write_final_receipt "${EXISTING_IMAGE_ID}"
    printf 'PASS: exact NVFP4/FP8-RoPE compatibility image exists; build skipped\n'
    printf 'IMAGE=%s\nIMAGE_ID=%s\n' "${OUTPUT_IMAGE}" \
      "${EXISTING_IMAGE_ID}"
    exit 0
  fi
fi

mkdir -p -- "${CACHE_ROOT}"
CONTEXT="$(mktemp -d -p "${CACHE_ROOT}" nf3-nvfp4-rope8.XXXXXXXX)"
cleanup() {
  case "$(realpath -- "${CONTEXT}")" in
    "$(realpath -- "${CACHE_ROOT}")"/nf3-nvfp4-rope8.*)
      rm -rf -- "${CONTEXT}"
      ;;
    *)
      printf 'refusing to remove unexpected context %s\n' "${CONTEXT}" >&2
      ;;
  esac
}
trap cleanup EXIT

cp -- "${ROOT}/runtime/Containerfile.nf3-nvfp4-rope8" \
  "${CONTEXT}/Containerfile"
cp -- "${ROOT}/runtime/verify-nf3-nvfp4-rope8.py" \
  "${CONTEXT}/verify-nf3-nvfp4-rope8.py"

"${ENGINE}" build \
  --platform linux/arm64 \
  --file "${CONTEXT}/Containerfile" \
  --tag "${OUTPUT_IMAGE}" \
  --build-arg "NF3_IMAGE=${NF3_IMAGE}" \
  --build-arg "MLA_IMAGE=${FASTSTART_IMAGE}" \
  --build-arg "NF3_IMAGE_ID=${NF3_IMAGE_ID}" \
  --build-arg "MLA_IMAGE_ID=${MLA_IMAGE_ID}" \
  --build-arg "SPARKRING_SOURCE_COMMIT=${SOURCE_COMMIT}" \
  "${CONTEXT}"

"${ENGINE}" run --rm --entrypoint /opt/venv/bin/python \
  "${OUTPUT_IMAGE}" /opt/sparkring/verify-nf3-nvfp4-rope8.py \
  >/dev/null

FINAL_IMAGE_ID="$(
  "${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format '{{.Id}}'
)"
write_final_receipt "${FINAL_IMAGE_ID}"

printf 'PASS: NF3 NVFP4/FP8-RoPE image built and ABI verified\n'
printf 'IMAGE=%s\nIMAGE_ID=%s\n' "${OUTPUT_IMAGE}" \
  "${FINAL_IMAGE_ID}"
