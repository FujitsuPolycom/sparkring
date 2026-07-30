#!/usr/bin/env bash
# Build the pinned, thin NF3 layer on one ARM64 DGX Spark.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${ROOT}/recipes/glm52-nf3-hybrid.json"
ENGINE="${CONTAINER_ENGINE:-docker}"
CACHE_ROOT="${SPARKRING_BOOTSTRAP_CACHE:-${HOME}/.cache/sparkring/nf3-bootstrap}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-sparkring/glm52-nf3:local}"

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null || fatal "git is required"
command -v python3 >/dev/null || fatal "python3 is required"
command -v tar >/dev/null || fatal "tar is required"
command -v "${ENGINE}" >/dev/null || fatal "${ENGINE} is required"
[[ "$(uname -m)" == "aarch64" ]] ||
  fatal "the NF3 image must be built natively on an ARM64 DGX Spark"
git -C "${ROOT}" rev-parse --verify HEAD >/dev/null 2>&1 ||
  fatal "SparkRing must be a git checkout"
[[ -z "$(git -C "${ROOT}" status --porcelain)" ]] ||
  fatal "SparkRing checkout is dirty; commit or restore it before building"

eval "$(
  python3 - "${RECIPE}" <<'PY'
import json, shlex, sys
recipe = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = recipe["runtime"]
values = {
    "BASE_IMAGE": runtime["base_image"],
    "B12X_REPO": runtime["b12x_repository"],
    "B12X_COMMIT": runtime["b12x_commit"],
    "PORT_REPO": runtime["spark_port_repository"],
    "PORT_COMMIT": runtime["spark_port_commit"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

SOURCE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
FASTSTART_IMAGE="sparkring/glm52-faststart:${SOURCE_COMMIT:0:12}"

if "${ENGINE}" image inspect "${OUTPUT_IMAGE}" >/dev/null 2>&1; then
  observed_labels="$("${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format \
    '{{index .Config.Labels "org.sparkring.source_commit"}} {{index .Config.Labels "org.opencontainers.image.revision.b12x"}} {{index .Config.Labels "org.opencontainers.image.revision.spark_port"}}')"
  if [[ "${observed_labels}" == \
    "${SOURCE_COMMIT} ${B12X_COMMIT} ${PORT_COMMIT}" ]] &&
    "${ENGINE}" run --rm --entrypoint /opt/venv/bin/python \
      "${OUTPUT_IMAGE}" /opt/sparkring/verify-nf3-bootstrap.py \
      --receipt /opt/sparkring/nf3-bootstrap-input-receipt.json \
      >/dev/null; then
    printf 'PASS: exact receipt-gated NF3 image already exists; build skipped\n'
    printf 'IMAGE=%s\nIMAGE_ID=%s\n' "${OUTPUT_IMAGE}" \
      "$("${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format '{{.Id}}')"
    exit 0
  fi
fi

mkdir -p -- "${CACHE_ROOT}/sources" "${CACHE_ROOT}/receipts"
docker_root="$("${ENGINE}" info --format '{{.DockerRootDir}}')"
docker_free="$(df -PB1 "${docker_root}" | awk 'NR==2 {print $4}')"
cache_free="$(df -PB1 "${CACHE_ROOT}" | awk 'NR==2 {print $4}')"
minimum_build_free=60000000000
(( docker_free >= minimum_build_free )) ||
  fatal "Docker storage needs at least ${minimum_build_free} free bytes; found ${docker_free}"
(( cache_free >= minimum_build_free )) ||
  fatal "bootstrap cache/archive storage needs at least ${minimum_build_free} free bytes; found ${cache_free}"

sync_source() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  if [[ ! -e "${destination}" ]]; then
    git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
  fi
  [[ -d "${destination}/.git" ]] ||
    fatal "conflicting non-git source cache: ${destination}"
  [[ "$(git -C "${destination}" remote get-url origin)" == "${repository}" ]] ||
    fatal "source-cache origin mismatch: ${destination}"
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach --force "${commit}"
  [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]] ||
    fatal "source checkout did not resolve to ${commit}"
}

sync_source "${B12X_REPO}" "${B12X_COMMIT}" "${CACHE_ROOT}/sources/b12x"
sync_source "${PORT_REPO}" "${PORT_COMMIT}" "${CACHE_ROOT}/sources/spark-port"

if ! "${ENGINE}" image inspect "${FASTSTART_IMAGE}" >/dev/null 2>&1; then
  OUTPUT_IMAGE="${FASTSTART_IMAGE}" MAX_JOBS="${MAX_JOBS:-8}" \
    bash "${ROOT}/runtime/build-faststart.sh"
fi
FASTSTART_ID="$("${ENGINE}" image inspect "${FASTSTART_IMAGE}" --format '{{.Id}}')"

CONTEXT="$(mktemp -d -p "${CACHE_ROOT}" nf3-context.XXXXXXXX)"
cleanup() {
  case "$(realpath -- "${CONTEXT}")" in
    "$(realpath -- "${CACHE_ROOT}")"/nf3-context.*) rm -rf -- "${CONTEXT}" ;;
    *) printf 'refusing to remove unexpected context %s\n' "${CONTEXT}" >&2 ;;
  esac
}
trap cleanup EXIT

git -C "${CACHE_ROOT}/sources/b12x" archive "${B12X_COMMIT}" b12x |
  tar -x -C "${CONTEXT}"
mkdir -p -- "${CONTEXT}/sparkring"
git -C "${CACHE_ROOT}/sources/spark-port" archive "${PORT_COMMIT}" \
  overlay/hybrid_loader.py overlay/nf3_kernel.py overlay/nf3_replan.py \
  overlay/nvfp4_kernel.py overlay/mxfp8_tier.json |
  tar -x -C "${CONTEXT}"
cp -- "${ROOT}/spark_transport/integrations/vllm/sitecustomize.py" \
  "${ROOT}/spark_transport/integrations/vllm/spark_nf3_startup_profile_cap.py" \
  "${ROOT}/spark_transport/integrations/vllm/spark_nf3_workspace_reserve.py" \
  "${CONTEXT}/sparkring/"
cp -- "${ROOT}/runtime/Containerfile.nf3-bootstrap" "${CONTEXT}/Containerfile"
cp -- "${ROOT}/runtime/verify-nf3-bootstrap.py" "${CONTEXT}/verify-nf3-bootstrap.py"

python3 - "${CONTEXT}" "${BASE_IMAGE}" "${FASTSTART_ID}" \
  "${B12X_REPO}" "${B12X_COMMIT}" "${PORT_REPO}" "${PORT_COMMIT}" \
  "${SOURCE_COMMIT}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = {}
for group in ("b12x", "overlay", "sparkring"):
    for path in sorted((root / group).rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
receipt = {
    "schema": "sparkring-nf3-bootstrap-input/v1",
    "base_image": sys.argv[2],
    "faststart_image_id": sys.argv[3],
    "b12x": {"repository": sys.argv[4], "commit": sys.argv[5]},
    "spark_port": {"repository": sys.argv[6], "commit": sys.argv[7]},
    "sparkring_source_commit": sys.argv[8],
    "files": files,
}
(root / "input-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

RECEIPT_SHA256="$(sha256sum "${CONTEXT}/input-receipt.json" | awk '{print $1}')"
"${ENGINE}" build \
  --platform linux/arm64 \
  --file "${CONTEXT}/Containerfile" \
  --tag "${OUTPUT_IMAGE}" \
  --build-arg "FASTSTART_IMAGE=${FASTSTART_IMAGE}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "SPARK_PORT_COMMIT=${PORT_COMMIT}" \
  --build-arg "SPARKRING_SOURCE_COMMIT=${SOURCE_COMMIT}" \
  --build-arg "INPUT_RECEIPT_SHA256=${RECEIPT_SHA256}" \
  "${CONTEXT}"

IMAGE_ID="$("${ENGINE}" image inspect "${OUTPUT_IMAGE}" --format '{{.Id}}')"
python3 - "${CONTEXT}/input-receipt.json" \
  "${CACHE_ROOT}/receipts/nf3-runtime.json" "${OUTPUT_IMAGE}" "${IMAGE_ID}" <<'PY'
import json, sys
source, destination, image, image_id = sys.argv[1:]
receipt = json.load(open(source, encoding="utf-8"))
receipt.update({
    "schema": "sparkring-nf3-runtime-receipt/v1",
    "image": image,
    "image_id": image_id,
})
open(destination, "w", encoding="utf-8").write(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

printf 'PASS: NF3 image built and receipt verified\n'
printf 'IMAGE=%s\nIMAGE_ID=%s\nRECEIPT=%s\n' \
  "${OUTPUT_IMAGE}" "${IMAGE_ID}" "${CACHE_ROOT}/receipts/nf3-runtime.json"
