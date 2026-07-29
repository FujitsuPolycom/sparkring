#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "FATAL: $*" >&2
  exit 78
}

: "${SPARKRING_MODEL_PATH:?SPARKRING_MODEL_PATH is required}"
: "${SPARKRING_MODEL_REPOSITORY:?SPARKRING_MODEL_REPOSITORY is required}"
: "${SPARKRING_MODEL_REVISION:?SPARKRING_MODEL_REVISION is required}"
: "${SPARKRING_MODEL_CONFIG_SHA256:?SPARKRING_MODEL_CONFIG_SHA256 is required}"
: "${SPARKRING_IMAGE_DIGEST:?SPARKRING_IMAGE_DIGEST is required}"

case "${SPARKRING_MODEL_REVISION}" in
  *[!0-9a-f]*|"") fail "model revision must be lowercase hex" ;;
esac
[ "${#SPARKRING_MODEL_REVISION}" -eq 40 ] ||
  fail "model revision must contain exactly 40 hex characters"
case "${SPARKRING_MODEL_CONFIG_SHA256}" in
  *[!0-9a-f]*|"") fail "model config sha256 must be lowercase hex" ;;
esac
[ "${#SPARKRING_MODEL_CONFIG_SHA256}" -eq 64 ] ||
  fail "model config sha256 must contain exactly 64 hex characters"
image_hex="${SPARKRING_IMAGE_DIGEST#sha256:}"
[ "${SPARKRING_IMAGE_DIGEST}" = "sha256:${image_hex}" ] ||
  fail "SPARKRING_IMAGE_DIGEST must start with sha256:"
[ "${#image_hex}" -eq 64 ] ||
  fail "SPARKRING_IMAGE_DIGEST must contain exactly 64 hex characters"
case "${image_hex}" in
  *[!0-9a-f]*|"") fail "SPARKRING_IMAGE_DIGEST must be lowercase hex" ;;
esac

config="${SPARKRING_MODEL_PATH}/config.json"
[ -f "${config}" ] || fail "model config is missing: ${config}"
actual="$(sha256sum -- "${config}" | awk '{print $1}')"
[ "${actual}" = "${SPARKRING_MODEL_CONFIG_SHA256}" ] ||
  fail "model config sha256 mismatch: expected ${SPARKRING_MODEL_CONFIG_SHA256}, got ${actual}"

/opt/venv/bin/python /opt/sparkring/verify-runtime.py \
  --manifest /opt/sparkring/runtime-manifest.json
/opt/venv/bin/python /opt/sparkring/public-model-preflight.py \
  --model-path "${SPARKRING_MODEL_PATH}"
/opt/venv/bin/python /opt/sparkring/public-headless-abi-gate.py
/opt/venv/bin/python /opt/sparkring/public-capability-gate.py

identity_dir=/run/sparkring/model-identity
install -d -m 0755 "${identity_dir}"
printf '%s\n' "${SPARKRING_MODEL_REPOSITORY}" > "${identity_dir}/repository"
printf '%s\n' "${SPARKRING_MODEL_REVISION}" > "${identity_dir}/revision"
chmod 0444 "${identity_dir}/repository" "${identity_dir}/revision"

export PYTHONPATH="/opt/spark-vllm${PYTHONPATH:+:${PYTHONPATH}}"
exec /opt/venv/bin/vllm "$@"
