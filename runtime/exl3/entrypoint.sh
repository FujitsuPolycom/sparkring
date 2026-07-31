#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "FATAL PUBLIC EXL3 PROFILE: $*" >&2
  exit 78
}

: "${SPARKRING_MODEL_PATH:?SPARKRING_MODEL_PATH is required}"
: "${SPARKRING_MODEL_REPOSITORY:?SPARKRING_MODEL_REPOSITORY is required}"
: "${SPARKRING_MODEL_REVISION:?SPARKRING_MODEL_REVISION is required}"
: "${SPARKRING_MODEL_CONFIG_SHA256:?SPARKRING_MODEL_CONFIG_SHA256 is required}"
: "${SPARKRING_MODEL_INDEX_SHA256:?SPARKRING_MODEL_INDEX_SHA256 is required}"
: "${SPARKRING_MODEL_TIER_BITMAP_SHA256:?SPARKRING_MODEL_TIER_BITMAP_SHA256 is required}"
: "${SPARKRING_IMAGE_DIGEST:?SPARKRING_IMAGE_DIGEST is required}"

pins=/opt/sparkring-exl3/pins.json
manifest=/opt/sparkring-exl3/runtime-manifest.json

expected_repo=$(/opt/venv/bin/python -S -c \
  'import json; print(json.load(open("'"${pins}"'"))["model"]["repository"])')
expected_revision=$(/opt/venv/bin/python -S -c \
  'import json; print(json.load(open("'"${pins}"'"))["model"]["revision"])')
[[ "${SPARKRING_MODEL_REPOSITORY}" == "${expected_repo}" ]] ||
  fail "model repository is not the pinned EXL3 repository"
[[ "${SPARKRING_MODEL_REVISION}" == "${expected_revision}" ]] ||
  fail "model revision is not the pinned EXL3 revision"

for pair in \
  "config.json:${SPARKRING_MODEL_CONFIG_SHA256}" \
  "model.safetensors.index.json:${SPARKRING_MODEL_INDEX_SHA256}" \
  "tier_bitmap.json:${SPARKRING_MODEL_TIER_BITMAP_SHA256}"
do
  name=${pair%%:*}
  expected=${pair#*:}
  path="${SPARKRING_MODEL_PATH}/${name}"
  [[ -f "${path}" ]] || fail "model metadata is missing: ${path}"
  observed=$(sha256sum -- "${path}" | awk '{print $1}')
  [[ "${observed}" == "${expected}" ]] ||
    fail "${name} sha256 mismatch: expected ${expected}, got ${observed}"
done

/opt/venv/bin/python /opt/sparkring/verify-runtime.py \
  --manifest "${manifest}" \
  --expect-runtime-id glm52-exl3-tr3-3.25bpw
/opt/venv/bin/python /opt/sparkring-exl3/verify_exl3_runtime.py \
  --phase runtime

identity_dir=/run/sparkring/model-identity
install -d -m 0755 "${identity_dir}"
printf '%s\n' "${SPARKRING_MODEL_REPOSITORY}" > "${identity_dir}/repository"
printf '%s\n' "${SPARKRING_MODEL_REVISION}" > "${identity_dir}/revision"
chmod 0444 "${identity_dir}/repository" "${identity_dir}/revision"

export SPARKRING_RUNTIME_MANIFEST="${manifest}"
export PYTHONPATH="/opt/spark-vllm${PYTHONPATH:+:${PYTHONPATH}}"
exec /opt/venv/bin/vllm "$@"
