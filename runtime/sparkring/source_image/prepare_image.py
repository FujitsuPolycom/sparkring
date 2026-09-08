"""Prepare the source-complete GLM TP4 image over its immutable ARM64 parent."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import tempfile

from archive_utils import make_archive, read_archive, sha
from prepare_context import prepare

HERE = Path(__file__).resolve().parent


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def materialize(name, record, cache, inputs=HERE, reuse=False):
    """Apply only the locked patch over its exact public source commit."""
    patch = inputs / record["patch"]
    if sha(patch.read_bytes()) != record["patch_sha256"]:
        raise ValueError(f"Source patch differs: {name}")
    destination = cache / name
    if destination.exists():
        if not reuse:
            raise ValueError(f"Source destination must be absent: {name}")
        if (git(destination, "rev-parse", "HEAD").decode().strip() != record["base_revision"]
                or git(destination, "write-tree").decode().strip() != record["result_tree"]
                or git(destination, "diff", "--name-only")
                or git(destination, "ls-files", "--others", "--exclude-standard")):
            raise ValueError(f"Cached source differs from complete locked tree: {name}")
        return destination, record["result_tree"]
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    git(destination, "config", "core.autocrlf", "false")
    git(destination, "config", "core.longpaths", "true")
    git(destination, "fetch", "--quiet", "--depth", "1", record["repository"], record["base_revision"])
    git(destination, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    if git(destination, "rev-parse", "HEAD").decode().strip() != record["base_revision"]:
        raise ValueError(f"Source base identity differs: {name}")
    git(destination, "apply", "--check", "--binary", str(patch.resolve()))
    git(destination, "apply", "--index", "--binary", str(patch.resolve()))
    tree = git(destination, "write-tree").decode().strip()
    if record.get("result_tree", tree) != tree:
        raise ValueError(f"Patched source tree differs: {name}")
    return destination, tree


def prepare_locked(output, source_cache, lock_path=HERE / "glm53-tp4-lock.json", reuse=False):
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("schema") != "sparkring-source-image-lock/v1":
        raise ValueError("Unsupported source-image lock")
    if output.exists() or (source_cache.exists() and not reuse):
        raise ValueError("Build output and source cache must be absent directories")
    sources = {}
    for name, record in lock["sources"].items():
        checkout, _ = materialize(name, record, source_cache, lock_path.parent, reuse)
        sources[name] = {**record, "checkout": str(checkout)}
    nccl, tree = materialize("nccl", lock["nccl_build"], source_cache, lock_path.parent, reuse)
    files = read_archive(git(nccl, "archive", "--format=tar", tree, "Makefile", "makefiles", "src", "tests", "LICENSE.txt", "ThirdPartyNotices.txt"))
    data = make_archive(files, lock["source_date_epoch"])
    with tempfile.TemporaryDirectory(prefix="sparkring-image-inputs-") as temporary:
        directory = Path(temporary)
        archive = directory / "nccl.tar"
        archive.write_bytes(data)
        parent_bytes = gzip.decompress((lock_path.parent / lock["parent"]["file_receipt"]).read_bytes())
        if sha(parent_bytes) != lock["parent"]["file_receipt_sha256"]:
            raise ValueError("Parent installed-file receipt differs")
        parent = directory / "parent.json"
        parent.write_bytes(parent_bytes)
        spec = {
            "base_image": lock["parent"]["reference"], "base_receipt": str(parent),
            "base_receipt_sha256": sha(parent_bytes), "source_date_epoch": lock["source_date_epoch"],
            "sources": sources, "source_lock": str(lock_path),
            "runtime": {"description": "Source-pinned GLM TP4 with retained, attested ARM64 dependencies",
                        "expected_distributions": lock["runtime"]["expected_distributions"], "required_files": {}},
            "startup_override": {**lock["startup"], "checkout": str(lock_path.parent / "startup")},
            "native_snapshot": {"cuda_architectures": "121", "compiler": "/opt/cuda-13.3/bin/nvcc", "build": "direct-cxx-cuda/v1"},
            "nccl_archive": str(archive), "nccl_build": {
                **lock["nccl_build"], "tree": tree, "archive_sha256": sha(data),
                "files": {name: sha(value[0]) for name, value in files.items()},
            },
        }
        return prepare(spec, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=HERE / "glm53-tp4-lock.json")
    parser.add_argument("--reuse-source-cache", action="store_true", help="Reuse only complete base and patched-tree matches")
    args = parser.parse_args()
    print(json.dumps(prepare_locked(args.output.resolve(), args.source_cache.resolve(), args.lock.resolve(), args.reuse_source_cache), indent=2))
