"""Build a manifest-bound local Qwen prefill source bundle."""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import pprint
import re
from qwen4_prefill_bootstrap import IMAGE_SOURCES


def package(destination, source_bindings=None):
    bindings = IMAGE_SOURCES if source_bindings is None else source_bindings
    suffixes = (
        "vllm/models/qwen4_exp/nvidia/hyperconnection.py",
        "b12x/sequence/mtp_feedback/_kernels.py",
    )
    roots = (
        "/opt/venv/lib/python3.12/site-packages/",
        "/usr/local/lib/python3.12/dist-packages/",
    )
    if (
        not isinstance(bindings, dict)
        or len(bindings) != len(suffixes)
        or {next((suffix for suffix in suffixes if name.endswith(suffix)), None)
            for name in bindings} != set(suffixes)
        or any(
            name not in {root + suffix for root in roots for suffix in suffixes}
            or ".." in PurePosixPath(name).parts
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in bindings.items()
        )
    ):
        raise ValueError("Unsupported Qwen prefill source bindings")
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    names = (
        "qwen4_prefill_bootstrap.py",
        "qwen4_hc_fusion.py",
        "qwen4_mtp_gemm.py",
        "qwen4_fused_gate_kernel.py",
    )
    files = {name: (root / name).read_bytes() for name in names}
    if source_bindings is not None:
        name = "qwen4_prefill_bootstrap.py"
        text = files[name].decode()
        assignments = [
            node for node in ast.parse(text).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "IMAGE_SOURCES"
                    for target in node.targets)
        ]
        if len(assignments) != 1:
            raise ValueError("Prefill bootstrap source-binding declaration changed")
        node = assignments[0]
        lines = text.splitlines(keepends=True)
        files[name] = (
            "".join(lines[:node.lineno - 1])
            + "IMAGE_SOURCES = " + pprint.pformat(bindings, sort_dicts=True) + "\n"
            + "".join(lines[node.end_lineno:])
        ).encode()
    files["qwen4_prefill.pth"] = (
        "/opt/sparkring/qwen4-prefill\n"
        "import qwen4_prefill_bootstrap; qwen4_prefill_bootstrap.install()\n"
    ).encode()
    for name, data in files.items():
        (destination / name).write_bytes(data)
    manifest = {
        "schema": "sparkring-qwen4-prefill/v1",
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        "image_source_preimages": bindings,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (destination / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-bindings", type=Path)
    args = parser.parse_args()
    bindings = json.loads(args.source_bindings.read_text()) if args.source_bindings else None
    digest = package(args.destination, bindings)
    print(
        json.dumps(
            {
                "manifest_sha256": digest,
                "cache_namespace": "qwen4-prefill-" + digest[:12],
            }
        )
    )
