"""Bind the paired vLLM coordinator before running protected source checks."""
import argparse
import hashlib
import os
from pathlib import Path
import re

try:
    from .standalone_gate import run
except ImportError:
    from standalone_gate import run


def bind_peer(root, expected):
    root = Path(root).resolve()
    path = root / "vllm/v1/worker/b12x_startup.py"
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected an approved coordinator SHA-256")
    if (not path.is_file() or any(p.is_symlink() for p in [path, *path.parents])
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError("Paired vLLM coordinator differs from the approved input")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "suite", "result", "peer-vllm-root", "peer-vllm-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    peer = bind_peer(args.peer_vllm_root, args.peer_vllm_sha256)
    os.environ["SPARKRING_VLLM_SOURCE_ROOT"] = str(peer)
    result = run(args.source, args.baseline, args.suite, args.result)
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
