#!/usr/bin/env python3
"""Fail closed unless vLLM's pinned multi-node headless ABI is intact.

The leader owns the world-wide collective RPC writer.  A follower executor
must only create its local WorkerProc instances, which subscribe to that
writer, and then block in the worker monitor.  It must not start an EngineCore
or attempt to originate collective_rpc calls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


EXPECTED_SHA256 = {
    "entrypoints/cli/serve.py": (
        "f89771fe192e7ef262a7600244c918de4cedc426fdeebab8e8c69cfba49694d2"
    ),
    "v1/executor/multiproc_executor.py": (
        "80bde58c8336427848f87df053d756dafc88e5df6ef80df661ecf6f85d8de7e2"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("installed vllm package was not found")
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        raise RuntimeError(f"expected one installed vllm root, got {roots!r}")
    return Path(roots[0]).resolve()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"required function is missing: {name}")


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise RuntimeError(f"required method is missing: {class_name}.{method_name}")


def _source(node: ast.AST, text: str) -> str:
    segment = ast.get_source_segment(text, node)
    if segment is None:
        raise RuntimeError("could not recover source segment for ABI validation")
    return segment


def audit(vllm_root: Path) -> dict[str, object]:
    files = {relative: vllm_root / relative for relative in EXPECTED_SHA256}
    failures: list[str] = []
    hashes: dict[str, str] = {}
    for relative, path in files.items():
        if not path.is_file():
            failures.append(f"missing pinned vLLM source: {relative}")
            continue
        hashes[relative] = _sha256(path)
        if hashes[relative] != EXPECTED_SHA256[relative]:
            failures.append(
                f"source hash mismatch for {relative}: "
                f"expected {EXPECTED_SHA256[relative]}, got {hashes[relative]}"
            )

    if not failures:
        serve_text = files["entrypoints/cli/serve.py"].read_text(encoding="utf-8")
        serve_tree = ast.parse(serve_text)
        headless = _source(_function(serve_tree, "run_headless"), serve_text)
        required_headless = (
            "if parallel_config.node_rank_within_dp > 0:",
            "MultiprocExecutor(vllm_config, monitor_workers=False)",
            "executor.start_worker_monitor(inline=True)",
            "return",
            "CoreEngineProcManager(",
        )
        for token in required_headless:
            if token not in headless:
                failures.append(f"run_headless ABI token is missing: {token}")
        follower = headless.index("if parallel_config.node_rank_within_dp > 0:")
        follower_return = headless.index("return", follower)
        engine_core = headless.index("CoreEngineProcManager(")
        if not follower < follower_return < engine_core:
            failures.append(
                "follower branch must return before CoreEngineProcManager creation"
            )

        executor_text = files[
            "v1/executor/multiproc_executor.py"
        ].read_text(encoding="utf-8")
        executor_tree = ast.parse(executor_text)
        init = _source(
            _class_method(executor_tree, "MultiprocExecutor", "_init_executor"),
            executor_text,
        )
        rpc = _source(
            _class_method(executor_tree, "MultiprocExecutor", "collective_rpc"),
            executor_text,
        )
        required_executor = (
            "if self.parallel_config.node_rank_within_dp == 0:",
            "scheduler_output_handle = self.rpc_broadcast_mq.export_handle()",
            "input_shm_handle=scheduler_output_handle",
        )
        for token in required_executor:
            if token not in init:
                failures.append(f"MultiprocExecutor ABI token is missing: {token}")
        if "collective_rpc should not be called on follower node" not in rpc:
            failures.append("follower collective_rpc assertion is missing")

    return {
        "schema": "sparkring-headless-abi-audit/v1",
        "ok": not failures,
        "vllm_root": str(vllm_root),
        "files": hashes,
        "failures": failures,
        "contract": (
            "leader-originated world-wide RPC; followers host WorkerProc subscribers "
            "and never originate collective_rpc"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    try:
        report = audit(
            args.vllm_root.resolve() if args.vllm_root else _find_vllm_root()
        )
    except Exception as exc:
        report = {
            "schema": "sparkring-headless-abi-audit/v1",
            "ok": False,
            "failures": [str(exc)],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
