"""Replay a retained deployment with its own verified controller source."""
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

from runtime.common import distribution, installer


def checkout(directory, cache, *, run=subprocess.run):
    directory = Path(directory).resolve()
    lock = installer.read(directory / "deployment.lock.json")
    revision, digest = lock["source_revision"], lock["bundle_sha256"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Retained deployment lacks a complete source identity")
    bundle = directory / "source.bundle"
    if bundle.is_symlink() or hashlib.sha256(bundle.read_bytes()).hexdigest() != digest:
        raise ValueError("Retained deployment source bundle differs")
    target = Path(cache) / revision
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Retained controller source path contains a symlink")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", str(bundle), str(target)], check=True)
    actual = run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = run(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout.strip()
    if actual != revision or dirty:
        raise ValueError("Retained controller source was modified")
    return target


def _operation(directory, operation):
    from runtime.common import installer
    if operation == "saved-status":
        return installer.status(directory)
    from scripts.installer_runner import Runner
    runner = Runner(directory)
    if operation in ("verify", "status"):
        result = installer.status(directory)
        if operation == "verify":
            steps = ([(row, op) for row in runner.lock["site"]["ranks"] for op in ("owned", "running")]
                     if runner.lock["backend"] != "glm-managed" else [])
            steps.append((runner.lock["site"]["ranks"][0], "smoke"))
            for row, op in steps:
                checked = runner(row["host"], ["installer", op, str(row["rank"])], 120)
                if checked["returncode"]:
                    raise ValueError(checked["stderr"])
            result["verified"] = True
        else:
            result["observations"] = []
            for row in runner.lock["site"]["ranks"]:
                value = runner(row["host"], ["installer", "status", str(row["rank"])], 120)
                import json
                result["observations"].append({"host": row["host"], "result": json.loads(value["stdout"]) if not value["returncode"] else {"error": value["stderr"]}})
            result["live_observed"] = True
        return result
    return installer.apply(directory, operation, runner=runner, execute=True)


def apply(directory, operation, *, cache, run=subprocess.run):
    if operation not in ("prepare", "up", "down", "verify", "status", "saved-status"):
        raise ValueError("Unsupported retained deployment operation")
    directory = Path(directory).resolve()
    lock = installer.read(directory / "deployment.lock.json")
    if lock["source_revision"] == distribution.identity(installer.ROOT):
        return _operation(directory, operation)
    source = checkout(directory, cache, run=run)
    code = inspect.getsource(_operation) + "\n" + """import contextlib,json,sys
sys.path.insert(0,sys.argv[1])
from runtime.host import progress
with contextlib.redirect_stdout(sys.stderr):
 with progress.run('retained deployment '+sys.argv[3]):
  result=_operation(sys.argv[2],sys.argv[3])
print(json.dumps(result))
"""
    result = run([sys.executable, "-I", "-B", "-c", code, str(source), str(directory), operation],
                 stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout)
