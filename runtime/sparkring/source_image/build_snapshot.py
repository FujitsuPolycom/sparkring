"""Compile the pinned CUDA snapshot library without initializing a GPU."""
import json
from pathlib import Path
import shutil
import subprocess

from archive_utils import inventory, sha

LIBRARY = "/opt/sparkcache-native/libspark_cache_snapshot.so"
RECIPE = {"cuda_architectures": "121", "compiler": "/opt/cuda-13.3/bin/nvcc", "build": "direct-cxx-cuda/v1"}


def verify_library(data, lock):
    if data[:6] != b"\x7fELF\x02\x01" or int.from_bytes(data[18:20], "little") != 183:
        raise RuntimeError("Snapshot output is not an ARM64 ELF library")
    if (lock["runtime"]["snapshot_path"] != LIBRARY
            or sha(data) != lock["runtime"]["snapshot_sha256"]):
        raise RuntimeError("Snapshot output differs from the serving connector's locked library")


def build(root):
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("native_snapshot") != RECIPE:
        raise RuntimeError("Unsupported snapshot build recipe")
    source = root / "sparkcache-source" / "sparkcache" / "native"
    prefix = "sparkcache/native/"
    expected = {p[len(prefix):]: h for p, h in manifest["sources"]["sparkcache"]["files"].items()
                if p.startswith(prefix)}
    if not expected or inventory(source) != expected:
        raise RuntimeError("Snapshot sources differ from the pinned archive")
    build_dir = root / "snapshot-build"
    build_dir.mkdir(exist_ok=False)
    # Mirror the native CMake targets using the parent image's compilers.
    # Neither CPU executable links CUDA. Fault-injection macros remain undefined.
    include = ["-I", str(source / "include")]
    cxx = ["/usr/bin/g++", "-std=c++17", "-O3", "-fPIC", "-Wall", "-Wextra",
           "-Wpedantic", "-Werror", "-DSPARK_CACHE_SNAPSHOT_BUILD=1", *include]
    objects = [build_dir / (name + ".o") for name in ("snapshot_layout", "page_layout")]
    commands = [[RECIPE["compiler"], "--version"], ["/usr/bin/g++", "--version"]]
    for filename, obj in zip(("spark_cache_snapshot_layout.cpp", "spark_cache_page_capture_layout.cpp"), objects):
        commands.append([*cxx, "-c", str(source / "src" / filename), "-o", str(obj)])
    commands.append([
        RECIPE["compiler"], "-std=c++17", "-O3", "-DNDEBUG", "-shared", "-rdc=true",
        "--cudart=shared", "--expt-relaxed-constexpr", "-arch=sm_121", "-L/opt/cuda-13.3/lib",
        "-Xcompiler=-fPIC,-Wall,-Wextra,-Werror", "-DSPARK_CACHE_SNAPSHOT_BUILD=1",
        *include, str(source / "src/spark_cache_snapshot.cu"), *map(str, objects),
        "-o", str(build_dir / "libspark_cache_snapshot.so"),
    ])
    for name in ("snapshot_ring_test", "page_capture_layout_test"):
        executable = build_dir / name
        commands.append([*cxx, "-UNDEBUG", str(source / "tests" / (name + ".cpp")),
                         *map(str, objects), "-o", str(executable)])
        commands.append([str(executable)])
    logs = []
    for command in commands:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append({"argv": command, "returncode": result.returncode, "output": result.stdout})
        (root / "snapshot-build-log.json").write_text(json.dumps(logs, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError("Snapshot build failed; inspect snapshot-build-log.json")
    built = build_dir / "libspark_cache_snapshot.so"
    # ELF64 little-endian AArch64; inspection does not load the shared object.
    data = built.read_bytes()
    lock_bytes = (root / "source-lock.json").read_bytes()
    if sha(lock_bytes) != manifest["source_lock_sha256"]:
        raise RuntimeError("Snapshot source lock changed during compilation")
    verify_library(data, json.loads(lock_bytes))
    destination = Path(LIBRARY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("Snapshot destination already exists")
    shutil.copyfile(built, destination)
    destination.chmod(0o755)
    receipt = {
        "schema": "sparkcache-native-snapshot-build/v1",
        "source_manifest_sha256": sha(manifest_bytes),
        "sparkcache_revision": manifest["sources"]["sparkcache"]["revision"],
        "source_files": expected,
        "recipe": RECIPE,
        "files": {LIBRARY: sha(data)},
        "build_log_sha256": sha((root / "snapshot-build-log.json").read_bytes()),
        "gpu_qualified": False,
    }
    (root / "snapshot-build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
