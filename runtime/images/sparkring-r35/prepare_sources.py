"""Package pinned R35 compositions and reviewed local changes without building images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASES = {
    "vllm": ("de982a50c6a3e4718e5cf9f00423a92192718da1", "6d5cc3490735a53e849a4f0909183cb5e3a22e8b"),
    "b12x": ("98086604c86ec1e78977e5023ce282ecb97ab8a7", "d963f030c72a426753d2afa412b35d29eeb045f7"),
}
R33_VLLM = "ae89131442359dc332d9c46009be3c1f8cdee0b4"


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def package(name: str, source: Path, output: Path) -> dict:
    base, expected_tree = BASES[name]
    if git(source, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError(f"{name}: checkout must start at the pinned R35 composition")
    if git(source, "rev-parse", "HEAD^{tree}").decode().strip() != expected_tree:
        raise ValueError(f"{name}: base tree mismatch")
    if git(source, "diff", "--name-only") or git(source, "ls-files", "--others", "--exclude-standard"):
        raise ValueError(f"{name}: stage intended changes before packaging")
    tree = git(source, "write-tree").decode().strip()
    archive = output / f"{name}-{tree}.tar.gz"
    # A tree has no commit timestamp; Git otherwise uses wall-clock time.
    # Use an explicit UTC date accepted by Git's archive date parser.
    # This changes archive metadata only, not the staged tree or patch identity.
    subprocess.run(["git", "-C", str(source), "-c", "core.autocrlf=false",
                    "-c", "tar.umask=0022", "archive", "--format=tar.gz",
                    "--mtime=2000-01-01T00:00:00Z", f"--output={archive}", tree], check=True)
    patch = output / f"{name}-sparkring.patch"
    patch.write_bytes(git(source, "diff", "--cached", "--binary", base))
    record = {"base_commit": base, "base_tree": expected_tree, "tree": tree,
              "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "patch": patch.name, "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
    if name == "vllm":
        native_paths = ["csrc", "CMakeLists.txt", "cmake", "requirements", "pyproject.toml", "setup.py"]
        changed = git(source, "diff", "--name-only", R33_VLLM, base, "--", *native_paths)
        if changed:
            raise ValueError("vLLM native/build inputs differ from R33; explicit rebuild review required")
        # SparkRing's requirements change belongs to both integrated versions.
        record["native_base_comparison"] = {"reference": R33_VLLM, "paths": native_paths,
                                             "unchanged": True, "runtime_verification_required": True}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {name: package(name, getattr(args, name + "_source").resolve(), output) for name in BASES}
    lock = {"schema": "sparkring-r35-source-composition/v1", "status": "candidate",
            "components": records, "qualification": "Source packaging only; image and serving validation required."}
    (output / "source-composition.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: item["tree"] for name, item in records.items()}))


if __name__ == "__main__":
    main()
