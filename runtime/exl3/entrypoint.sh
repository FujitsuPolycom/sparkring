#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "FATAL PUBLIC EXL3 PROFILE: $*" >&2
  exit 78
}

if [[ -n "${SPARKRING_EXPLICITLY_UNSET:-}" ]]; then
  IFS=',' read -r -a unset_names <<<"${SPARKRING_EXPLICITLY_UNSET}"
  for name in "${unset_names[@]}"; do
    [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] ||
      fail "invalid explicitly-unset environment name: ${name}"
    unset "${name}"
  done
  unset SPARKRING_EXPLICITLY_UNSET
fi

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

/opt/venv/bin/python /opt/sparkring-exl3/verify_exl3_model.py \
  --model-path "${SPARKRING_MODEL_PATH}"

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
