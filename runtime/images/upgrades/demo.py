"""Self-contained three-night trial; no model, Docker or external agent is used."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import subprocess
import sys

from .contracts import encoded, sha
from .execution import validate_gate
from .io import write_json
from .runner import run
from .sources import git, tree_digest

ORACLE = """import importlib.util, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "engine/cache.py"
spec = importlib.util.spec_from_file_location("demo_engine",path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
cases = [(1424,32,False),(1440,32,True),(0,32,False),(1440,0,False)]
passed = all(module.aligned(a,b) == expected for a,b,expected in cases)
print(json.dumps({"schema":"sparkring-upgrade-gate/v1","gate":sys.argv[2],
 "input_sha256":sys.argv[3],"variant":sys.argv[4],"subject_sha256":sys.argv[5],"assertions":len(cases),
 "skipped":0,"outcome":"passed" if passed else "failed"}))
"""
FIXED = "def aligned(tokens, chunk):\n    return tokens > 0 and chunk > 0 and tokens % chunk == 0\n"
BUG = "def aligned(tokens, chunk):\n    return True\n"
REFACTORED = "def aligned(tokens, chunk):\n    answer = True\n    return answer\n"


def diff(before, after):
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/engine/cache.py",
            tofile="b/engine/cache.py",
        )
    )


class DemoAgent:
    calls = 0

    def propose(self, request, **kwargs):
        self.calls += 1
        before = request["files"]["engine/cache.py"]["candidate"]
        return {
            "disposition": "adapt",
            "reason": "Preserve page divisibility and reject empty/invalid geometry.",
            "patch": diff(before, FIXED),
        }


class DemoExecutor:
    """Execute only the generated demonstration oracle; image builds are simulated."""

    simulation = True

    def __init__(self, oracle):
        self.oracle = oracle

    def gate(self, gate, source, output, context, deadline):
        output = Path(output)
        output.mkdir(parents=True)
        if gate["stage"] == "image":
            result = dict(
                schema="sparkring-upgrade-gate/v1",
                gate=gate["id"],
                input_sha256=context["input_sha"],
                variant=context["variant"],
                subject_sha256=context["subject_sha256"],
                assertions=1,
                skipped=0,
                outcome="passed",
            )
        else:
            raw = subprocess.check_output(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(self.oracle),
                    str(source),
                    gate["id"],
                    context["input_sha"],
                    context["variant"],
                    context["subject_sha256"],
                ],
                text=True,
            )
            result = json.loads(raw)
        write_json(output / "result.json", result)
        return validate_gate(result, gate, context)

    def action(self, kind, context, output, deadline):
        if kind != "build":
            raise ValueError("The demo cannot publish")
        bundle = json.loads(Path(context["bundle"]).read_text())
        result = {
            "schema": "sparkring-upgrade-build/v1",
            "input_sha256": context["input_sha"],
            "image_id": "sha256:" + sha(encoded(bundle)),
            "platform": "simulation",
            "installed_verified": False,
            "simulation": True,
            "source_trees": {
                key: tree_digest(Path(value["candidate_path"]))
                for key, value in bundle["sources"].items()
            },
        }
        write_json(Path(output) / "result.json", result)
        return result


def setup(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    repository = root / "upstream"
    (repository / "engine").mkdir(parents=True)
    (repository / "native").mkdir()
    (repository / "engine/cache.py").write_bytes(BUG.encode())
    (repository / "native/version.txt").write_bytes(b"1\n")
    git(repository, "init", "--initial-branch=main")
    git(repository, "config", "user.name", "Upgrade fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    git(repository, "add", "--all")
    git(repository, "commit", "-m", "Provide fixture engine")
    baseline = git(repository, "rev-parse", "HEAD").decode().strip()
    patch = diff(BUG, FIXED).encode()
    (root / "geometry.patch").write_bytes(patch)
    (root / "oracle.py").write_bytes(ORACLE.encode())
    policy = {
        "schema": "sparkring-image-upgrade/v1",
        "name": "geometry-demo",
        "local_sources": True,
        "platform": "linux/arm64",
        "required_features": [],
        "budgets": dict(
            run_seconds=120,
            command_seconds=30,
            source_bytes=2000000,
            output_bytes=1000000,
            agent_attempts=2,
        ),
        "sources": [
            {
                "id": "engine",
                "repository": str(repository.resolve()),
                "baseline": baseline,
                "ref": "refs/heads/main",
                "patch": "geometry.patch",
                "patch_sha256": sha(patch),
                "editable_paths": ["engine"],
                "native_paths": ["native"],
                "contracts": [
                    {
                        "id": "page-alignment",
                        "kind": "correctness",
                        "oracle": "geometry",
                        "invariant": "Pages divide into persistent chunks; empty pages and zero chunks are rejected.",
                    }
                ],
            }
        ],
        "gates": [
            {
                "id": "geometry",
                "stage": "oracle",
                "executor": "docker",
                "image": "sha256:" + "a" * 64,
                "argv": ["python", "/oracles/oracle.py"],
                "inputs": [{"path": "oracle.py", "sha256": sha(ORACLE.encode())}],
            },
            {
                "id": "image-contract",
                "stage": "image",
                "executor": "docker",
                "image": "candidate",
                "argv": ["verify"],
            },
        ],
        "build": {"argv": ["fixture-only"], "supports_native_rebuild": False},
        "permissions": {"candidate_publication": False},
    }
    write_json(root / "policy.json", policy)
    return root / "policy.json", repository


def trial(root):
    policy, repository = setup(root)
    agent = DemoAgent()
    executor = DemoExecutor(Path(root) / "oracle.py")
    args = dict(execute=True, build=True, executor=executor, agent=agent)
    reports = [
        run(policy, Path(root) / "state", **args),
        run(policy, Path(root) / "state", **args),
    ]
    (repository / "engine/cache.py").write_bytes(REFACTORED.encode())
    git(repository, "add", "--all")
    git(repository, "commit", "-m", "Refactor fixture return path")
    reports.append(run(policy, Path(root) / "state", **args))
    summary = {
        "simulation": True,
        "statuses": [r["status"] for r in reports],
        "agent_calls": agent.calls,
        "reports": [r["artifact_directory"] for r in reports],
    }
    write_json(Path(root) / "TRIAL.json", summary)
    return summary
