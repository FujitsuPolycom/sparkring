#!/usr/bin/env bash
# Download the pinned GLM-5.2 checkpoint using the public ARM64 base image.
set -euo pipefail

BASE_IMAGE="aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7"
MODEL_REPO="aidendle94/GLM-5.2-MXFP4-Experts-GPTQ"
MODEL_REVISION="46537e0e16fcd156627800139b41b9c497fc7ee2"
CONFIG_SHA256="ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69"
OUTPUT_DIR="${1:-/srv/models/GLM-5.2-MXFP4-Experts-GPTQ}"
ENGINE="${CONTAINER_ENGINE:-docker}"

command -v "${ENGINE}" >/dev/null ||
  { echo "FATAL: ${ENGINE} is required" >&2; exit 1; }
command -v sha256sum >/dev/null ||
  { echo "FATAL: sha256sum is required" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

free_kib="$(df -Pk "${OUTPUT_DIR}" | awk 'NR==2 {print $4}')"
if [[ "${free_kib}" =~ ^[0-9]+$ ]] && (( free_kib < 450000000 )); then
  echo "WARNING: less than about 450 GB is free on the model filesystem." >&2
fi

echo "Pulling pinned ARM64 base image..."
"${ENGINE}" pull "${BASE_IMAGE}"

echo "Downloading ${MODEL_REPO}@${MODEL_REVISION} to ${OUTPUT_DIR}"
run_args=(run --rm --user "$(id -u):$(id -g)" -v "${OUTPUT_DIR}:/model")
if [[ -n "${HF_TOKEN:-}" ]]; then
  run_args+=(--env HF_TOKEN)
fi
"${ENGINE}" "${run_args[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${BASE_IMAGE}" \
  -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="aidendle94/GLM-5.2-MXFP4-Experts-GPTQ", revision="46537e0e16fcd156627800139b41b9c497fc7ee2", local_dir="/model")'

actual="$(sha256sum "${OUTPUT_DIR}/config.json" | awk '{print $1}')"
if [[ "${actual}" != "${CONFIG_SHA256}" ]]; then
  echo "FATAL: config.json hash mismatch" >&2
  echo "  expected: ${CONFIG_SHA256}" >&2
  echo "  actual:   ${actual}" >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/.sparkring-model.txt" <<EOF
repository=${MODEL_REPO}
revision=${MODEL_REVISION}
config_sha256=${CONFIG_SHA256}
EOF

echo "PASS: pinned model downloaded and config identity verified"
