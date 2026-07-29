#!/usr/bin/env bash
# SparkRing runtime builder — reads runtime/runtime-lock.json and drives a
# pinned, aarch64 container build of runtime/Containerfile.
#
# The lock file is the single source of truth for every version-shaped value.
# This script NEVER mutates the lock: after a successful build it prints the
# image digest and tells you exactly what to write back by hand.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/runtime/runtime-lock.json"
ENGINE="${CONTAINER_ENGINE:-docker}"

[ -f "${LOCK}" ] || { echo "FATAL: missing lock file: ${LOCK}" >&2; exit 1; }
command -v python3 >/dev/null || { echo "FATAL: python3 required to parse the lock" >&2; exit 1; }

# Verify every file consumed by either patch pipeline before Docker sees the
# build context. In particular, runtime/patches/**/preimages.json is itself
# security-sensitive input: changing a recorded upstream preimage must require
# an explicit lock update, just like changing the patch that follows it.
python3 - "${LOCK}" "${REPO_ROOT}" <<'EOF'
import hashlib
import json
import pathlib
import re
import sys

lock_path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2]).resolve()


def fatal(message):
    raise SystemExit(f"FATAL: {message}")


try:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    fatal(f"cannot read runtime lock: {exc}")

supported_schema = "sparkring-runtime-lock/v1"
if lock.get("schema") != supported_schema:
    fatal(
        f"unsupported runtime lock schema: {lock.get('schema')!r}; "
        f"expected {supported_schema!r}"
    )


def immutable_commit(path):
    cur = lock
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    if not isinstance(cur, str) or not re.fullmatch(r"[0-9a-f]{40}", cur):
        fatal(f"{path} must be an immutable 40-hex commit; got {cur!r}")


for commit_path in (
    "model.revision",
    "vllm.commit",
    "sparkinfer.commit",
    "flashinfer.commit",
    "deep_gemm.commit_full",
    "nccl.commit",
):
    immutable_commit(commit_path)

expected_object_keys = {
    "": {
        "schema", "runtime_id", "comment", "base_image", "toolchain", "vllm",
        "sparkinfer", "flashinfer", "deep_gemm", "nccl", "model", "overlays",
        "public_runtime_inputs", "overlays_comment",
    },
    "base_image": {"builder", "runtime"},
    "base_image.builder": {"repository", "digest"},
    "base_image.runtime": {"repository", "digest"},
    "toolchain": {
        "torch_cuda_version", "torch_version", "torch_index_url",
        "python_version", "target_platform", "compiler",
    },
    "toolchain.compiler": {"cuda_architectures"},
    "vllm": {"repository", "commit"},
    "sparkinfer": {"repository", "commit"},
    "flashinfer": {
        "repository", "commit", "wheels", "wheel_sha256", "wheel_index",
    },
    "flashinfer.wheels": {"flashinfer-python", "flashinfer_jit_cache"},
    "flashinfer.wheel_sha256": {"flashinfer_jit_cache"},
    "deep_gemm": {"repository", "version", "commit_full"},
    "nccl": {"repository", "commit", "patches"},
    "model": {"repository", "revision", "config_sha256", "note"},
}

for dotted, expected in expected_object_keys.items():
    node = lock
    if dotted:
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
    if not isinstance(node, dict):
        fatal(f"lock field {dotted or '<root>'} must be an object")
    unknown = sorted(set(node) - expected)
    if unknown:
        prefix = f"{dotted}." if dotted else ""
        fatal(f"unconsumed lock field: {prefix}{unknown[0]}")
    missing = sorted(expected - set(node))
    if missing:
        prefix = f"{dotted}." if dotted else ""
        fatal(f"lock key missing: {prefix}{missing[0]}")

def require_nonempty_strings(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            require_nonempty_strings(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            require_nonempty_strings(value, f"{path}[{index}]")
    elif not isinstance(node, str) or not node:
        fatal(f"{path} must be a non-empty string")


require_nonempty_strings(lock)

jit_wheel_sha256 = lock["flashinfer"]["wheel_sha256"]["flashinfer_jit_cache"]
if not re.fullmatch(r"[0-9a-f]{64}", jit_wheel_sha256):
    fatal(
        "flashinfer JIT wheel sha256 must be 64 lowercase hex; "
        f"got {jit_wheel_sha256!r}"
    )

for image_role in ("builder", "runtime"):
    digest = lock.get("base_image", {}).get(image_role, {}).get("digest")
    if not isinstance(digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", digest
    ):
        fatal(
            f"base_image.{image_role}.digest must be sha256:<64 hex>; "
            f"got {digest!r}"
        )


def pinned_files(section, label):
    if not isinstance(section, list) or not section:
        fatal(f"{label} must be a non-empty list")
    seen = set()
    resolved = {}
    for index, entry in enumerate(section):
        if not isinstance(entry, dict):
            fatal(f"{label}[{index}] must be an object")
        expected_keys = {"path", "sha256"}
        unknown = sorted(set(entry) - expected_keys)
        if unknown:
            fatal(f"unconsumed lock field: {label}[{index}].{unknown[0]}")
        missing = sorted(expected_keys - set(entry))
        if missing:
            fatal(f"lock key missing: {label}[{index}].{missing[0]}")
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(rel, str) or not rel:
            fatal(f"{label}[{index}].path must be a non-empty string")
        if rel in seen:
            fatal(f"{label} contains duplicate path: {rel}")
        seen.add(rel)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            fatal(f"{label}[{index}].sha256 must be 64 lowercase hex characters")

        candidate = pathlib.Path(rel)
        if candidate.is_absolute():
            fatal(f"{label} path must be repository-relative: {rel}")
        try:
            path = (repo_root / candidate).resolve(strict=True)
            path.relative_to(repo_root)
        except (OSError, ValueError):
            fatal(f"{label} path is missing or escapes the repository: {rel}")
        if not path.is_file():
            fatal(f"{label} path is not a regular file: {rel}")

        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fatal(f"{label} sha256 mismatch for {rel}: {actual} != expected {expected}")
        resolved[path] = rel
    return resolved


try:
    overlay_section = lock["overlays"]
    nccl_section = lock["nccl"]["patches"]
    public_runtime_section = lock["public_runtime_inputs"]
except (KeyError, TypeError) as exc:
    fatal(f"lock key missing: {exc}")

locked_overlays = pinned_files(overlay_section, "overlays")
locked_nccl = pinned_files(nccl_section, "nccl.patches")
locked_public_runtime = pinned_files(
    public_runtime_section, "public_runtime_inputs"
)

expected_public_runtime = {
    (repo_root / path).resolve()
    for path in (
        "runtime/public-overlay-files.json",
        "runtime/build-public-overlay.py",
        "runtime/public-capability-gate.py",
        "runtime/public-entrypoint.sh",
        "runtime/public-model-preflight.py",
        "runtime/verify-runtime.py",
    )
}
if set(locked_public_runtime) != expected_public_runtime:
    locked = sorted(
        str(path.relative_to(repo_root)) for path in locked_public_runtime
    )
    expected = sorted(
        str(path.relative_to(repo_root)) for path in expected_public_runtime
    )
    fatal(
        "public_runtime_inputs do not match Containerfile inputs: "
        f"locked={locked}; expected={expected}"
    )

# These are the exact two source paths in Containerfile's explicit COPY.
# Equality makes the lock and the Docker build mutually binding: retargeting
# either side requires a reviewed change to both.
expected_nccl = {
    (
        repo_root
        / "spark_transport/experiments/nccl_switchless_ring/"
        "nccl-2.30.7-skip-tree-pat.patch"
    ).resolve(),
    (
        repo_root
        / "spark_transport/experiments/nccl_switchless_ring/"
        "nccl-2.30.7-advertise-all-listener-gids.patch"
    ).resolve(),
}
if set(locked_nccl) != expected_nccl:
    locked = sorted(str(path.relative_to(repo_root)) for path in locked_nccl)
    expected = sorted(str(path.relative_to(repo_root)) for path in expected_nccl)
    fatal(
        "nccl.patches paths do not match Containerfile COPY inputs: "
        f"locked={locked}; expected={expected}"
    )

# Match apply-patches.py exactly: direct *.patch files in each direct component
# directory, plus the preimages.json required whenever that component has a
# patch. This rejects an unlocked patch dropped into the build context and a
# locked-but-unused overlay entry with equal force.
patches_root = repo_root / "runtime" / "patches"
if not patches_root.is_dir():
    fatal(f"patches root missing: {patches_root}")
actual_inputs = set()
for component in sorted(p for p in patches_root.iterdir() if p.is_dir()):
    patches = sorted(component.glob("*.patch"))
    if not patches:
        continue
    actual_inputs.update(path.resolve() for path in patches)
    preimages = component / "preimages.json"
    if not preimages.is_file():
        fatal(f"{component.relative_to(repo_root)} has patches but no preimages.json")
    actual_inputs.add(preimages.resolve())

locked_inputs = set(locked_overlays)
if actual_inputs != locked_inputs:
    unlocked = sorted(str(p.relative_to(repo_root)) for p in actual_inputs - locked_inputs)
    unused = sorted(str(p.relative_to(repo_root)) for p in locked_inputs - actual_inputs)
    details = []
    if unlocked:
        details.append(f"unlocked patch input(s): {', '.join(unlocked)}")
    if unused:
        details.append(f"locked overlay(s) not consumed: {', '.join(unused)}")
    fatal("; ".join(details))

print(
    f"verified {len(locked_overlays)} overlay input pin(s) and "
    f"{len(nccl_section)} NCCL patch pin(s), plus "
    f"{len(locked_public_runtime)} public runtime input(s)"
)
EOF

# Fail-closed lock accessor: any missing key aborts.
lk() {
  python3 - "$LOCK" "$1" <<'EOF'
import json, sys
lock = json.load(open(sys.argv[1]))
cur = lock
for part in sys.argv[2].split("."):
    if part not in cur:
        sys.exit(f"FATAL: lock key missing: {sys.argv[2]}")
    cur = cur[part]
if not isinstance(cur, str) or not cur:
    sys.exit(f"FATAL: lock key empty/non-string: {sys.argv[2]}")
print(cur)
EOF
}

REQUIREMENTS_CHECK="$(mktemp)"
trap 'rm -f "${REQUIREMENTS_CHECK}"' EXIT
python3 "${REPO_ROOT}/runtime/prepare-public-requirements.py" \
  --freeze "${REPO_ROOT}/runtime/pip-freeze.txt" \
  --output "${REQUIREMENTS_CHECK}"

RUNTIME_ID="$(lk runtime_id)"

if [ "${1:-}" = "--verify-lock-only" ]; then
  [ "$#" -eq 1 ] || { echo "FATAL: --verify-lock-only accepts no other arguments" >&2; exit 1; }
  echo "runtime lock and patch inputs verified for ${RUNTIME_ID}"
  exit 0
elif [ "$#" -ne 0 ]; then
  echo "FATAL: unknown argument: $1" >&2
  exit 1
fi

# Base images are content-addressed; placeholders are rejected above.
base_ref() {  # $1 = lock section under base_image (builder|runtime)
  local repo digest
  repo="$(lk "base_image.$1.repository")"
  digest="$(lk "base_image.$1.digest")"
  echo "${repo}@${digest}"
}

BASE_DEVEL_IMAGE="$(base_ref builder)"
BASE_RUNTIME_IMAGE="$(base_ref runtime)"

TORCH_VERSION="$(lk toolchain.torch_version)"
FLASHINFER_PY_VER="$(lk 'flashinfer.wheels.flashinfer-python')"
FLASHINFER_JIT_VER="$(lk flashinfer.wheels.flashinfer_jit_cache)"
FLASHINFER_JIT_SHA="$(lk flashinfer.wheel_sha256.flashinfer_jit_cache)"

IMAGE_TAG="sparkring-runtime:${RUNTIME_ID}"

echo "== SparkRing runtime build =="
echo "   runtime_id : ${RUNTIME_ID}"
echo "   devel base : ${BASE_DEVEL_IMAGE}"
echo "   run base   : ${BASE_RUNTIME_IMAGE}"
echo "   vllm       : $(lk vllm.commit)"
echo "   tag        : ${IMAGE_TAG}"
echo "   NOTE: expect a multi-hour vLLM compile (see Containerfile header)."
echo

"${ENGINE}" build \
  --platform "$(lk toolchain.target_platform)" \
  -f "${REPO_ROOT}/runtime/Containerfile" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_DEVEL_IMAGE="${BASE_DEVEL_IMAGE}" \
  --build-arg BASE_RUNTIME_IMAGE="${BASE_RUNTIME_IMAGE}" \
  --build-arg CUDA_ARCH="$(lk toolchain.compiler.cuda_architectures)" \
  --build-arg PYTHON_VERSION="$(lk toolchain.python_version)" \
  --build-arg TORCH_CUDA_VERSION="$(lk toolchain.torch_cuda_version)" \
  --build-arg TORCH_SPEC="torch==${TORCH_VERSION}" \
  --build-arg TORCH_INDEX_URL="$(lk toolchain.torch_index_url)" \
  --build-arg VLLM_REPO="$(lk vllm.repository)" \
  --build-arg VLLM_COMMIT="$(lk vllm.commit)" \
  --build-arg SPARKINFER_REPO="$(lk sparkinfer.repository)" \
  --build-arg SPARKINFER_COMMIT="$(lk sparkinfer.commit)" \
  --build-arg FLASHINFER_REPO="$(lk flashinfer.repository)" \
  --build-arg FLASHINFER_COMMIT="$(lk flashinfer.commit)" \
  --build-arg FLASHINFER_PYTHON_VERSION="${FLASHINFER_PY_VER}" \
  --build-arg FLASHINFER_JIT_CACHE_SPEC="flashinfer_jit_cache==${FLASHINFER_JIT_VER}" \
  --build-arg FLASHINFER_JIT_CACHE_SHA256="${FLASHINFER_JIT_SHA}" \
  --build-arg FLASHINFER_WHEEL_INDEX="$(lk flashinfer.wheel_index)" \
  --build-arg DEEPGEMM_REPO="$(lk deep_gemm.repository)" \
  --build-arg DEEPGEMM_COMMIT="$(lk deep_gemm.commit_full)" \
  --build-arg DEEPGEMM_VERSION="$(lk deep_gemm.version)" \
  --build-arg NCCL_REPO="$(lk nccl.repository)" \
  --build-arg NCCL_COMMIT="$(lk nccl.commit)" \
  --build-arg MODEL_REPO_ID="$(lk model.repository)" \
  --build-arg MODEL_REVISION="$(lk model.revision)" \
  --build-arg MODEL_CONFIG_SHA256="$(lk model.config_sha256)" \
  --build-arg RUNTIME_ID="${RUNTIME_ID}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
  "${REPO_ROOT}"

echo
echo "== build complete =="
DIGEST="$("${ENGINE}" image inspect --format '{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || true)"
IMAGE_ID="$("${ENGINE}" image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
echo "   image tag : ${IMAGE_TAG}"
echo "   image id  : ${IMAGE_ID}"
if [ -n "${DIGEST}" ]; then
  echo "   digest    : ${DIGEST}"
else
  echo "   digest    : (none yet — a registry digest exists only after push:"
  echo "                ${ENGINE} push <registry>/${IMAGE_TAG} , then inspect RepoDigests)"
fi
echo
echo "NEXT (manual — this script never edits the lock):"
echo "  1. Push the image, capture its sha256 registry digest."
echo "  2. Record it with the launch/evidence metadata for this build."
echo "  3. Inject that value as SPARKRING_IMAGE_DIGEST at verification time."
