#!/usr/bin/env bash
# Download the one supported SparkRing target and its small MTP draft.
set -euo pipefail

BASE_IMAGE="aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7"
MODEL_REPO="madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
MODEL_REVISION="66f3623dd8fefb5ca8046706912d5d31c8d196af"
MODEL_CONFIG_SHA256="254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef"
MODEL_INDEX_SHA256="6eb773222d932418dd0530c63aca498f86ef424da2a4526ccba76b59726da234"
MODEL_SHARDS="184"

DRAFT_REPO="aidendle94/GLM-5.2-MXFP4-Experts-GPTQ"
DRAFT_REVISION="46537e0e16fcd156627800139b41b9c497fc7ee2"
DRAFT_CONFIG_SHA256="47e27afcefcd8439cb5dcbdc9d3e11ab5069d6d8395029058141ffb56c50d9ff"
DRAFT_INDEX_SHA256="de6d6bdead79ebd556d3bbbbf56ea537b00b4fdaf3e92927ac1463328037ee1d"
DRAFT_WEIGHT_SHA256="0ade0e3da08e7e6c7b1f20e4c4e8d5d3b26b81103cea22f2ead9909c7d3d0732"

MODEL_DIR="${1:-/srv/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
DRAFT_DIR="${2:-/srv/models/GLM-5.2-NF3-MTP-Draft}"
ENGINE="${CONTAINER_ENGINE:-docker}"

fatal() {
  echo "FATAL: $*" >&2
  exit 1
}

for command_name in "${ENGINE}" sha256sum python3; do
  command -v "${command_name}" >/dev/null ||
    fatal "${command_name} is required"
done

[[ "${MODEL_DIR}" != "${DRAFT_DIR}" ]] ||
  fatal "target and MTP draft directories must be distinct"

mkdir -p "${MODEL_DIR}" "${DRAFT_DIR}"
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"
DRAFT_DIR="$(cd "${DRAFT_DIR}" && pwd)"

free_kib="$(df -Pk "${MODEL_DIR}" | awk 'NR==2 {print $4}')"
if [[ "${free_kib}" =~ ^[0-9]+$ ]] && (( free_kib < 390000000 )); then
  echo "WARNING: less than about 400 GB is free on the model filesystem." >&2
fi

echo "Pulling the pinned ARM64 download environment..."
"${ENGINE}" pull "${BASE_IMAGE}"

common_run=(
  run
  --rm
  --user "$(id -u):$(id -g)"
  --env "HOME=/tmp"
  --env "HF_HOME=/tmp/sparkring-huggingface"
  --env "HF_HUB_OFFLINE=0"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  common_run+=(--env HF_TOKEN)
fi

echo "Downloading NF3 target ${MODEL_REPO}@${MODEL_REVISION}"
"${ENGINE}" "${common_run[@]}" \
  -v "${MODEL_DIR}:/model" \
  --entrypoint /opt/venv/bin/python \
  "${BASE_IMAGE}" \
  -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid", revision="66f3623dd8fefb5ca8046706912d5d31c8d196af", local_dir="/model")'

echo "Downloading the pinned MTP draft (not the historical target weights)"
"${ENGINE}" "${common_run[@]}" \
  -v "${DRAFT_DIR}:/draft" \
  --entrypoint /opt/venv/bin/python \
  "${BASE_IMAGE}" \
  -c 'from huggingface_hub import snapshot_download; from pathlib import Path; import shutil, tempfile; root=Path(snapshot_download(repo_id="aidendle94/GLM-5.2-MXFP4-Experts-GPTQ", revision="46537e0e16fcd156627800139b41b9c497fc7ee2", allow_patterns=["mtp-draft/*"], local_dir=tempfile.mkdtemp(prefix="sparkring-draft-"))); src=root/"mtp-draft"; dst=Path("/draft"); [shutil.copy2(p, dst/p.name) for p in src.iterdir() if p.is_file()]'

check_sha256() {
  local path="$1"
  local expected="$2"
  [[ -s "${path}" ]] || fatal "missing or empty: ${path}"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] ||
    fatal "hash mismatch for ${path}; expected ${expected}, got ${actual}"
}

check_sha256 "${MODEL_DIR}/config.json" "${MODEL_CONFIG_SHA256}"
check_sha256 "${MODEL_DIR}/model.safetensors.index.json" "${MODEL_INDEX_SHA256}"
check_sha256 "${DRAFT_DIR}/config.json" "${DRAFT_CONFIG_SHA256}"
check_sha256 "${DRAFT_DIR}/model.safetensors.index.json" "${DRAFT_INDEX_SHA256}"
check_sha256 "${DRAFT_DIR}/model-mtp.safetensors" "${DRAFT_WEIGHT_SHA256}"

observed_shards="$(
  python3 - "${MODEL_DIR}/model.safetensors.index.json" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(set(document["weight_map"].values())))
PY
)"
[[ "${observed_shards}" == "${MODEL_SHARDS}" ]] ||
  fatal "expected ${MODEL_SHARDS} target shards, index names ${observed_shards}"

cat > "${MODEL_DIR}/.sparkring-model.txt" <<EOF
repository=${MODEL_REPO}
revision=${MODEL_REVISION}
config_sha256=${MODEL_CONFIG_SHA256}
index_sha256=${MODEL_INDEX_SHA256}
shards=${MODEL_SHARDS}
EOF

cat > "${DRAFT_DIR}/.sparkring-model.txt" <<EOF
repository=${DRAFT_REPO}
revision=${DRAFT_REVISION}
subdirectory=mtp-draft
config_sha256=${DRAFT_CONFIG_SHA256}
index_sha256=${DRAFT_INDEX_SHA256}
weight_sha256=${DRAFT_WEIGHT_SHA256}
EOF

echo "PASS: NF3 target and MTP draft downloaded and verified"
