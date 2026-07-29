#!/usr/bin/env bash
# Build the thin SparkRing trial image on an ARM64 DGX Spark.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAST_LOCK="${REPO_ROOT}/runtime/faststart-lock.json"
RUNTIME_LOCK="${REPO_ROOT}/runtime/runtime-lock.json"
ENGINE="${CONTAINER_ENGINE:-docker}"
MAX_JOBS="${MAX_JOBS:-8}"

command -v python3 >/dev/null ||
  { echo "FATAL: python3 is required" >&2; exit 1; }
command -v "${ENGINE}" >/dev/null ||
  { echo "FATAL: ${ENGINE} is required" >&2; exit 1; }

if [[ "$(uname -m)" != "aarch64" && "${ALLOW_CROSS_BUILD:-0}" != "1" ]]; then
  echo "FATAL: build-faststart.sh must run natively on a DGX Spark (aarch64)." >&2
  echo "Set ALLOW_CROSS_BUILD=1 only if you intentionally configured ARM64 buildx." >&2
  exit 1
fi

eval "$(
  python3 - "${FAST_LOCK}" "${RUNTIME_LOCK}" "${REPO_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import re
import shlex
import sys

fast_path = pathlib.Path(sys.argv[1])
runtime_path = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
fast = json.loads(fast_path.read_text(encoding="utf-8"))
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))

if fast.get("schema") != "sparkring-faststart-lock/v1":
    raise SystemExit("FATAL: unsupported faststart lock schema")

base = fast.get("base_image", {})
model = fast.get("model", {})
required_base = {
    "repository", "tag_for_humans", "manifest_digest", "config_digest", "platform"
}
required_model = {"repository", "revision", "config_sha256"}
if set(base) != required_base:
    raise SystemExit("FATAL: faststart base_image keys are incomplete or unknown")
if set(model) != required_model:
    raise SystemExit("FATAL: faststart model keys are incomplete or unknown")

if base["platform"] != "linux/arm64":
    raise SystemExit("FATAL: faststart base must be linux/arm64")
for key in ("manifest_digest", "config_digest"):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base[key]):
        raise SystemExit(f"FATAL: invalid base_image.{key}")
if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
    raise SystemExit("FATAL: model revision must be an immutable commit")
if not re.fullmatch(r"[0-9a-f]{64}", model["config_sha256"]):
    raise SystemExit("FATAL: model config hash must be sha256 hex")

if runtime["model"]["repository"] != model["repository"]:
    raise SystemExit("FATAL: model repository differs between locks")
if runtime["model"]["revision"] != model["revision"]:
    raise SystemExit("FATAL: model revision differs between locks")
if runtime["model"]["config_sha256"] != model["config_sha256"]:
    raise SystemExit("FATAL: model config hash differs between locks")

for section in (
    runtime["nccl"]["patches"],
    runtime["overlays"],
    runtime["public_runtime_inputs"],
):
    for record in section:
        path = root / record["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise SystemExit(f"FATAL: locked input hash mismatch: {path}")

values = {
    "BASE_IMAGE": f'{base["repository"]}@{base["manifest_digest"]}',
    "BASE_CONFIG_DIGEST": base["config_digest"],
    "BASE_DEVEL_IMAGE": (
        f'{runtime["base_image"]["builder"]["repository"]}'
        f'@{runtime["base_image"]["builder"]["digest"]}'
    ),
    "NCCL_REPO": runtime["nccl"]["repository"],
    "NCCL_COMMIT": runtime["nccl"]["commit"],
    "MODEL_REPO_ID": model["repository"],
    "MODEL_REVISION": model["revision"],
    "MODEL_CONFIG_SHA256": model["config_sha256"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

if git -C "${REPO_ROOT}" rev-parse --verify HEAD >/dev/null 2>&1; then
  SPARKRING_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
else
  echo "FATAL: faststart builds require a git checkout with a committed HEAD" >&2
  exit 1
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" \
      && "${ALLOW_DIRTY_BUILD:-0}" != "1" ]]; then
  echo "FATAL: git checkout is dirty; commit the exact source before building." >&2
  echo "Set ALLOW_DIRTY_BUILD=1 only for an explicitly unreportable local experiment." >&2
  exit 1
fi

RUNTIME_ID="${RUNTIME_ID:-glm52-gb10-faststart-${SPARKRING_GIT_COMMIT:0:12}}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-sparkring/glm52-faststart:${SPARKRING_GIT_COMMIT:0:12}}"

echo "SparkRing faststart build"
echo "  base:    ${BASE_IMAGE}"
echo "  source:  ${SPARKRING_GIT_COMMIT}"
echo "  output:  ${OUTPUT_IMAGE}"
echo "  jobs:    ${MAX_JOBS}"
echo
echo "The recovered Python patch gate runs before native compilation."
echo "Any base-image source mismatch fails closed; do not bypass it."

echo
echo "Pulling and verifying the pinned public base..."
"${ENGINE}" pull "${BASE_IMAGE}"
observed_base_id="$("${ENGINE}" image inspect "${BASE_IMAGE}" --format '{{.Id}}')"
if [[ "${observed_base_id}" != "${BASE_CONFIG_DIGEST}" ]]; then
  echo "FATAL: base-image config digest mismatch" >&2
  echo "  expected: ${BASE_CONFIG_DIGEST}" >&2
  echo "  observed: ${observed_base_id}" >&2
  exit 1
fi

"${ENGINE}" build \
  --platform linux/arm64 \
  --file "${REPO_ROOT}/runtime/Containerfile.faststart" \
  --tag "${OUTPUT_IMAGE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_DEVEL_IMAGE=${BASE_DEVEL_IMAGE}" \
  --build-arg "NCCL_REPO=${NCCL_REPO}" \
  --build-arg "NCCL_COMMIT=${NCCL_COMMIT}" \
  --build-arg "MODEL_REPO_ID=${MODEL_REPO_ID}" \
  --build-arg "MODEL_REVISION=${MODEL_REVISION}" \
  --build-arg "MODEL_CONFIG_SHA256=${MODEL_CONFIG_SHA256}" \
  --build-arg "RUNTIME_ID=${RUNTIME_ID}" \
  --build-arg "SPARKRING_GIT_COMMIT=${SPARKRING_GIT_COMMIT}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  "${REPO_ROOT}"

IMAGE_ID="$("${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format '{{.Id}}')"
echo
echo "PASS: faststart image built"
echo "  image: ${OUTPUT_IMAGE}"
echo "  id:    ${IMAGE_ID}"
echo
echo "Next: distribute this exact image to all four ranks, place the pinned model"
echo "at the same path on every rank, fill scripts/config/site.yaml, and run the"
echo "read-only preflight. See docs/QUICKSTART.md."
