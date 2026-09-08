"""Build the source-locked NCCL library without GPU or network access."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from archive_utils import extract_checked, sha
from install_sources import read_manifest
from nvcc_deterministic import POLICY


def compile_jobs(value):
    jobs = int(value)
    if not 1 <= jobs <= 64:
        raise ValueError("NCCL compile jobs must be between 1 and 64")
    return jobs


def build(root, diagnostic_output=None, jobs=16):
    jobs = compile_jobs(jobs)
    if diagnostic_output is not None and diagnostic_output.exists():
        raise ValueError("Diagnostic receipt output must be absent")
    manifest = read_manifest(root)
    lock_bytes = (root / "source-lock.json").read_bytes()
    if sha(lock_bytes) != manifest["source_lock_sha256"]:
        raise RuntimeError("Source lock differs from prepared context")
    lock = json.loads(lock_bytes)
    if lock["nccl_build"].get("random_seed_policy") != POLICY:
        raise ValueError("NCCL build requires the locked deterministic seed policy")
    recipe = manifest["nccl_build"]
    archive = (root / "nccl.tar").read_bytes()
    if sha(archive) != recipe["archive_sha256"]:
        raise RuntimeError("NCCL source archive differs")
    source = Path(lock["nccl_build"]["source_directory"])
    if source != Path("/work/src-lf") or source.exists():
        raise RuntimeError("NCCL source directory must be the absent locked build path")
    extract_checked(archive, source, recipe["files"])
    build_dir = Path(lock["nccl_build"]["build_directory"])
    if build_dir != Path("/work/build"):
        raise RuntimeError("NCCL build directory differs from locked compiler path")
    build_dir.mkdir(parents=True, exist_ok=False)
    compiler = lock["nccl_build"]["cuda_home"]
    wrapper = root / "nvcc_deterministic.py"
    if sha(wrapper.read_bytes()) != lock["nccl_build"]["nvcc_wrapper_sha256"]:
        raise ValueError("NCCL compiler wrapper differs from source lock")
    commands = [
        [str(wrapper), "--version"],
        ["g++", "--version"],
        ["g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
         str(source / "tests/routing_handle/compat.cc"), "-o", str(build_dir / "compat")],
        [str(build_dir / "compat")],
        ["make", "-C", str(source), f"-j{jobs}", "src.build", f"CUDA_HOME={compiler}",
         f"CUDA_LIB={compiler}/lib", f"BUILDDIR={build_dir}", f"NVCC={wrapper}",
         "NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121"],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        logs.append({"argv": command, "returncode": result.returncode,
                     "stdout": result.stdout, "stderr": result.stderr})
        (root / "nccl-build-log.json").write_text(json.dumps(logs, indent=2))
        if result.returncode:
            raise RuntimeError("NCCL build failed; inspect nccl-build-log.json")
    built = build_dir / "lib/libnccl.so.2.30.7"
    data = built.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01" or int.from_bytes(data[18:20], "little") != 183:
        raise RuntimeError("NCCL output is not AArch64 ELF64")
    if diagnostic_output is not None:
        receipt = {"schema": "sparkring-nccl-build-diagnostic/v1", "status": "research-only",
                   "source_lock_sha256": sha(lock_bytes), "source_archive_sha256": sha(archive),
                   "nvcc_wrapper_sha256": sha(wrapper.read_bytes()), "random_seed_policy": POLICY,
                   "compiler_sha256": sha(Path(compiler, "bin/nvcc").read_bytes()),
                   "source_tree": recipe["tree"], "library_path": str(built),
                   "library_sha256": sha(data), "library_bytes": len(data),
                   "expected_runtime_sha256": lock["runtime"]["nccl_sha256"],
                   "fresh_source_and_object_directories": True,
                   "compile_jobs": jobs,
                   "cpu_routing_test_passed": True, "gpu_qualified": False,
                   "installed": False, "runtime_receipt_eligible": False,
                   "build_log_sha256": sha((root / "nccl-build-log.json").read_bytes())}
        diagnostic_output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    if sha(data) != lock["runtime"]["nccl_sha256"]:
        raise RuntimeError("Rebuilt NCCL differs from measured binary; separate qualification required")
    destination = Path(lock["runtime"]["nccl_path"])
    if destination.exists():
        raise RuntimeError("NCCL destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, destination)
    destination.chmod(0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-output", type=Path,
                        help="Write an unqualified build receipt instead of installing NCCL")
    parser.add_argument("--jobs", type=compile_jobs, default=16,
                        help="Concurrent native compile jobs (1-64; default 16)")
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parent, args.diagnostic_output, args.jobs)
    if result is not None:
        print(json.dumps(result, indent=2))
