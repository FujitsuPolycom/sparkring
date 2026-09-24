"""Source identity for both a clean checkout and an installed Debian payload."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed(root, *, verify=True):
    root = Path(root).resolve()
    path = root / "distribution.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "sparkring-distribution/v1" or not record.get("files"):
        raise ValueError("Invalid installed SparkRing distribution")
    if verify:
        for name, expected in record["files"].items():
            relative = PurePosixPath(name)
            target = root / name
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or target.is_symlink() or not target.resolve().is_relative_to(root)
                    or digest(target) != expected):
                raise ValueError("Installed SparkRing file differs: " + name)
        if digest(root / "source.bundle") != record["bundle_sha256"]:
            raise ValueError("Installed source bundle differs")
    return record


def identity(root):
    root = Path(root).resolve()
    record = installed(root)
    if record:
        return record["revision"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True).strip():
        raise ValueError("Commit the operator source before initializing or executing a deployment")
    return revision


def bundle(root, output):
    record = installed(root)
    output = Path(output)
    if output.exists():
        raise ValueError("Source bundle output already exists")
    if record:
        shutil.copyfile(Path(root) / "source.bundle", output)
    else:
        subprocess.run(["git", "bundle", "create", str(output), "HEAD"], cwd=root, check=True, capture_output=True)
    output.chmod(0o600)


def tracked_files(root):
    record = installed(root)
    if record:
        return sorted(record["files"])
    return list(filter(None, subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")))
