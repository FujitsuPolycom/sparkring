#!/usr/bin/env bash
# Install the wheel without dependencies and exercise package/native entry points without a GPU.
set -euo pipefail

wheel=$1
output=$2
foundation_image=${SPARKRING_R33_FOUNDATION_IMAGE:-local/sparkring:r33-arm64-foundation}
work_dir=$(dirname "$output")/install-smoke-$(basename "$output" .json)
test ! -e "$work_dir"
mkdir -p "$work_dir/site" "$(dirname "$output")"

docker run --rm --network none \
  -v "$(dirname "$wheel"):/wheel:ro" \
  -v "$work_dir:/smoke" \
  "$foundation_image" \
  bash --noprofile --norc -ceu '
    python3 -m pip install --disable-pip-version-check --no-deps --no-index \
      --target /smoke/site "/wheel/$1" > /smoke/pip-install.log
    PYTHONPATH=/smoke/site python3 - "$2" <<"PY"
import importlib
import json
import subprocess
import sys

output = sys.argv[1]
import vllm

modules = [
    "vllm._C_stable_libtorch",
    "vllm._flashkda_C",
    "vllm._moe_C_stable_libtorch",
    "vllm._qutlass_C",
    "vllm.cumem_allocator",
    "vllm.fs_io_C",
    "vllm.spinloop",
    "vllm.third_party.deep_gemm._C",
    "vllm.vllm_flash_attn._vllm_fa2_C",
    "vllm.vllm_flash_attn._vllm_fa3_C",
    "vllm._rust_tool_parser",
]
loaded = {}
deferred = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        loaded[name] = module.__file__
    except ImportError as error:
        # The network-disabled, GPU-free foundation intentionally lacks the host
        # driver library. CUDA-only modules may stop at that loader boundary.
        if "libcuda.so.1" not in str(error):
            raise
        deferred[name] = str(error)
version = subprocess.check_output(["/smoke/site/vllm/vllm-rs", "--version"], text=True).strip()
result = {
    "status": "wheel-install-and-driver-independent-import-smoke-passed; GPU/model-runtime-qualification-pending",
    "vllm_version": vllm.__version__,
    "loaded_modules": loaded,
    "driver_dependent_imports_deferred": deferred,
    "vllm_rs_version": version,
}
with open(output, "w") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
  ' smoke "$(basename "$wheel")" /smoke/import-smoke.json
cp "$work_dir/import-smoke.json" "$output"
