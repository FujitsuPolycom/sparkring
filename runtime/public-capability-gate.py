#!/usr/bin/env python3
"""Fail closed unless the installed vLLM exposes the GLM-5.2 Spark surface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

REQUIRED_CLI_FLAGS = (
    "--attention-backend",
    "--decode-context-parallel-size",
    "--dcp-comm-backend",
    "--enable-auto-tool-choice",
    "--kv-cache-dtype",
    "--reasoning-parser",
    "--speculative-config",
    "--tool-call-parser",
)
REQUIRED_SOURCE_TOKENS = ("B12X_MLA_SPARSE", "nvfp4_ds_mla", "glm47")


def vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vLLM package is not importable")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def source_token_presence(root: Path, tokens: tuple[str, ...]) -> dict[str, bool]:
    remaining = set(tokens)
    found = {token: False for token in tokens}
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for token in tuple(remaining):
            if token in text:
                found[token] = True
                remaining.remove(token)
        if not remaining:
            break
    return found


def evaluate(vllm_command: str) -> dict:
    root = vllm_root()
    result = subprocess.run(
        [vllm_command, "serve", "--help=all"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    help_text = result.stdout + "\n" + result.stderr
    flags = {flag: flag in help_text for flag in REQUIRED_CLI_FLAGS}
    tokens = source_token_presence(root, REQUIRED_SOURCE_TOKENS)
    failures = [
        f"CLI flag absent: {flag}" for flag, present in flags.items() if not present
    ]
    failures.extend(
        f"runtime token absent: {token}"
        for token, present in tokens.items()
        if not present
    )
    if result.returncode != 0:
        failures.append(
            f"`{vllm_command} serve --help=all` exited {result.returncode}"
        )
    return {
        "schema": "sparkring-public-capability-gate/v1",
        "passed": not failures,
        "vllm_root": str(root),
        "cli_flags": flags,
        "source_tokens": tokens,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-command", default="/opt/venv/bin/vllm")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(args.vllm_command)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        report = {
            "schema": "sparkring-public-capability-gate/v1",
            "passed": False,
            "failures": [str(exc)],
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["passed"]:
        print("PASS: public GLM-5.2 capability surface is present")
    else:
        print("FAIL: " + "; ".join(report["failures"]))
    return 0 if report["passed"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
