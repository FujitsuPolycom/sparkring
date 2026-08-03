#!/usr/bin/env bash
# Build the receipt-gated EXL3 layer over an exact public NF3 base image.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUILD_CONTEXT IMAGE_TAG" >&2
  exit 64
fi

context=$1
image_tag=$2
engine=${CONTAINER_ENGINE:-docker}
: "${SPARKRING_EXL3_BASE_IMAGE:?SPARKRING_EXL3_BASE_IMAGE is required}"
: "${SPARKRING_EXL3_BASE_IMAGE_ID:?SPARKRING_EXL3_BASE_IMAGE_ID is required}"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 78
}

[[ "$(uname -m)" == "aarch64" ]] || fatal "EXL3 must be built natively on an ARM64 DGX Spark"
[[ -d "${context}" ]] || fatal "missing build context: ${context}"
command -v "${engine}" >/dev/null || fatal "${engine} is required"

python3 "${context}/verify_build_context.py" --context "${context}" >/dev/null
context_sha=$(sha256sum "${context}/context-manifest.json" | awk '{print $1}')
observed_base=$("${engine}" image inspect --format '{{.Id}}' "${SPARKRING_EXL3_BASE_IMAGE}")
[[ "${observed_base}" == "${SPARKRING_EXL3_BASE_IMAGE_ID}" ]] ||
  fatal "base image mismatch: expected ${SPARKRING_EXL3_BASE_IMAGE_ID}, got ${observed_base}"

available=$(df --output=avail -B1 "${context}" | tail -n 1 | tr -d ' ')
minimum=$((24 * 1024 * 1024 * 1024))
(( available >= minimum )) || fatal "need at least ${minimum} free bytes for EXL3 build; got ${available}"

if "${engine}" image inspect "${image_tag}" >/dev/null 2>&1; then
  labels=$("${engine}" image inspect "${image_tag}" --format \
    '{{index .Config.Labels "org.sparkring.parent.image-id"}} {{index .Config.Labels "org.sparkring.exl3.context-manifest-sha256"}}')
  if [[ "${labels}" == "${observed_base} ${context_sha}" ]] &&
    "${engine}" run --rm --entrypoint /opt/venv/bin/python "${image_tag}" \
      /opt/sparkring-exl3/verify_exl3_runtime.py --phase runtime >/dev/null; then
    printf 'PASS: exact EXL3 image exists; build skipped\n'
    "${engine}" image inspect --format '{{.Id}}' "${image_tag}"
    exit 0
  fi
fi

"${engine}" build \
  --platform linux/arm64 \
  --file "${context}/Containerfile" \
  --build-arg "BASE_IMAGE=${SPARKRING_EXL3_BASE_IMAGE}" \
  --build-arg "BASE_IMAGE_ID=${observed_base}" \
  --build-arg "CONTEXT_MANIFEST_SHA256=${context_sha}" \
  --build-arg "PROFILE_ID=glm52-exl3-tr3-3.25bpw" \
  --build-arg "SPARKINFER_COMMIT=018de520e40f6bf9bd0b11c5da5517ef3364a985" \
  --build-arg "EXLLAMAV3_COMMIT=d21d9b3182e746199093b77b49a708635c1d1b5d" \
  --build-arg "VLLM_PORT_COMMIT=668275901b55230f4a70841a9aac1c0be22ef8d3" \
  --build-arg "LMCACHE_INTEGRATION_TREE=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa" \
  --build-arg "LMCACHE_COMPOSED_TREE=7dddbfde874d123e5b5785e6e56b4b7baf4baa82" \
  --build-arg "LMCACHE_TOPOLOGY_PATCH_SHA256=6eb5d3c15cd67a62dca5714fabc4d9c12c7175dfb9ab3e971dc6bc280f0aa533" \
  --tag "${image_tag}" \
  "${context}"

"${engine}" run --rm --entrypoint /opt/venv/bin/python "${image_tag}" \
  /opt/sparkring-exl3/verify_exl3_runtime.py --phase runtime >/dev/null
"${engine}" image inspect --format '{{.Id}}' "${image_tag}"
