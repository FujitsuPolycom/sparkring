"""Verify a derived toolchain without rewriting the inherited RC4 receipt."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/opt/sparkring/toolchain")
PARENT_RECEIPT = Path("/opt/sparkring/receipts/external-base-installed.json")
PARENT_VERIFIER = "/opt/sparkring/bin/external-base.py"
NCCL_PATH = ROOT / "nccl/lib/libnccl.so.2"
CUDA_ROOT = "/usr/local/cuda-13.4"
CUDA_LIBRARY_NAMES = (
    "libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13",
    "libnvJitLink.so.13", "libnvrtc.so.13", "libcusparse.so.12",
    "libcusolver.so.12", "libcurand.so.10", "libcufft.so.12",
)
CUDA_SEARCH_ROOT = "/opt/sparkring/toolchain/python"


def vendor_search_aliases(lock):
    """Bind Torch's explicit vendor-package searches to selected libraries."""
    aliases = {}
    if lock["variant"] in ("combined", "cuda"):
        aliases["nvidia/cu13/lib"] = CUDA_ROOT + "/lib64"
    if lock["variant"] in ("combined", "nccl"):
        aliases["nvidia/nccl/lib"] = NCCL_PATH.parent.as_posix()
    return aliases


def prepend_unique(prefix, existing):
    return ":".join(dict.fromkeys([*prefix, *filter(None, existing.split(":"))]))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nccl_version(path):
    library = ctypes.CDLL(str(path))
    value = ctypes.c_int()
    function = library.ncclGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    require(function(ctypes.byref(value)) == 0, "ncclGetVersion failed")
    return value.value


def configure_environment(lock, environment):
    result = dict(environment)
    cuda_enabled = lock["variant"] in ("combined", "cuda")
    nccl_enabled = lock["variant"] in ("combined", "nccl")
    preloads = result.get("LD_PRELOAD", "").replace(":", " ").split()
    selected_preloads = []
    library_dirs = []
    if nccl_enabled:
        # Profiles may carry the inherited absolute library path. The candidate
        # selects its library before exec so PyTorch and vLLM resolve one runtime.
        selected = NCCL_PATH.as_posix()
        preloads = [item for item in preloads if "libnccl.so" not in Path(item).name]
        selected_preloads.append(selected)
        for name in ("VLLM_NCCL_SO_PATH", "NCCL_LOCAL_INFERENCE_PATH"):
            result[name] = selected
        result["NCCL_LIB_DIR"] = NCCL_PATH.parent.as_posix()
        result["NCCL_INCLUDE_DIR"] = (ROOT / "nccl/include").as_posix()
        result["NCCL_ROOT"] = (ROOT / "nccl").as_posix()
        library_dirs.append(NCCL_PATH.parent.as_posix())
    if cuda_enabled:
        toolkit = CUDA_ROOT
        prefixes = tuple(name.split(".so")[0] + ".so" for name in CUDA_LIBRARY_NAMES)
        preloads = [item for item in preloads if not Path(item).name.startswith(prefixes)]
        selected_preloads.extend(toolkit + "/lib64/" + name for name in CUDA_LIBRARY_NAMES)
        result["CUDA_HOME"] = result["CUDA_PATH"] = toolkit
        result["CUDA_VERSION"] = lock["cuda"]["version"]
        result["TRITON_PTXAS_PATH"] = toolkit + "/bin/ptxas"
        result["PATH"] = prepend_unique([toolkit + "/bin"], result.get("PATH", ""))
        # Torch searches sys.path for vendor libraries, independently of the
        # ELF loader. This namespace redirects that search without modifying
        # inherited wheel payloads or their metadata.
        library_dirs = [toolkit + "/compat", toolkit + "/lib64", *library_dirs]
    if vendor_search_aliases(lock):
        result["PYTHONPATH"] = prepend_unique([CUDA_SEARCH_ROOT], result.get("PYTHONPATH", ""))
    result["LD_PRELOAD"] = " ".join(dict.fromkeys([*selected_preloads, *preloads]))
    result["LD_LIBRARY_PATH"] = prepend_unique(library_dirs, result.get("LD_LIBRARY_PATH", ""))
    return result


def verify_loaded_libraries(lock, maps):
    """Reject a selected runtime that also loaded an inherited vendor copy."""
    cuda_prefixes = tuple(name.split(".so")[0] + ".so" for name in CUDA_LIBRARY_NAMES)
    selected = []
    for line in maps.splitlines():
        parts = line.split()
        if not parts or not parts[-1].startswith("/"):
            continue
        name = parts[-1]
        basename = Path(name).name
        if lock["variant"] in ("combined", "nccl") and basename.startswith("libnccl.so"):
            require(name.startswith(NCCL_PATH.parent.as_posix() + "/"), "Unselected NCCL library loaded: " + name)
            selected.append(name)
        if lock["variant"] in ("combined", "cuda") and basename.startswith(cuda_prefixes):
            require(name.startswith(CUDA_ROOT + "/"), "Unselected CUDA library loaded: " + name)
            selected.append(name)
    return sorted(set(selected))


def inventory():
    lock = json.loads((ROOT / "toolchain.json").read_text())
    subprocess.run([sys.executable, PARENT_VERIFIER, "verify"], check=True)
    files = {str(ROOT / "toolchain.py"): sha(ROOT / "toolchain.py"),
             str(ROOT / "toolchain.json"): sha(ROOT / "toolchain.json")}
    toolkit = "/usr/local/cuda-13.4" if lock["variant"] in ("combined", "cuda") else "/usr/local/cuda"
    nvcc = subprocess.check_output([toolkit + "/bin/nvcc", "--version"], text=True)
    search_tree = {}
    if lock["variant"] in ("combined", "cuda"):
        require(lock["cuda"]["nvcc_version"] in nvcc, "Unexpected CUDA compiler version")
        # Headers and NVVM inputs also affect JIT output, so bind the complete
        # copied toolkit rather than only the libraries loaded at startup.
        for path in sorted(Path(toolkit).rglob("*")):
            if path.is_file():
                files[str(path)] = sha(path)
    if vendor_search_aliases(lock):
        for relative, target in vendor_search_aliases(lock).items():
            alias = Path(CUDA_SEARCH_ROOT) / relative
            require(alias.is_symlink() and os.readlink(alias) == target,
                    "Vendor search alias differs: " + relative)
        for path in sorted(Path(CUDA_SEARCH_ROOT).rglob("*")):
            if path.is_symlink():
                search_tree[str(path)] = {"symlink": os.readlink(path)}
            elif path.is_file():
                search_tree[str(path)] = {"sha256": sha(path)}
    selected = NCCL_PATH if lock["variant"] in ("combined", "nccl") else Path("/opt/local-inference/nccl/lib/libnccl.so.2")
    version = nccl_version(selected)
    if lock["variant"] in ("combined", "nccl"):
        require(version == 23203, "Candidate does not load NCCL 2.32.3")
        for path in sorted((ROOT / "nccl").rglob("*")):
            if path.is_file():
                files[str(path)] = sha(path)
    return {"schema": "sparkring-toolchain-installed/v1", "variant": lock["variant"],
            "parent_receipt_sha256": sha(PARENT_RECEIPT), "nvcc": nvcc,
            "nccl_version": version, "files": files, "framework_native_rebuilt": False,
            "loader_search_tree": search_tree, "serving_qualified": False}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "verify"
    require(action in ("seal", "verify", "serve"), "Expected seal, verify or serve")
    lock = json.loads((ROOT / "toolchain.json").read_text())
    expected_environment = configure_environment(lock, os.environ)
    if any(os.environ.get(name) != value for name, value in expected_environment.items()):
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], expected_environment)
    observed = inventory()
    loaded = verify_loaded_libraries(lock, Path("/proc/self/maps").read_text())
    receipt = ROOT / "installed.json"
    if action == "seal":
        require(not receipt.exists(), "Toolchain receipt already exists")
        receipt.write_text(json.dumps(observed, indent=2) + "\n")
    else:
        require(json.loads(receipt.read_text()) == observed, "Toolchain differs from installed receipt")
    report = {key: value for key, value in observed.items() if key != "files"}
    report["mapped_libraries"] = loaded
    print(json.dumps(report), flush=True)
    if action == "serve":
        argv = [sys.executable, PARENT_VERIFIER, "serve", *sys.argv[2:]]
        os.execve(argv[0], argv, configure_environment(lock, os.environ))


if __name__ == "__main__":
    main()
