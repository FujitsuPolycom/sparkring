"""Build a manifest-bound local Qwen prefill source bundle."""

import argparse
import hashlib
import json
from pathlib import Path
from qwen4_prefill_bootstrap import IMAGE_SOURCES


def package(destination):
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    names = (
        "qwen4_prefill_bootstrap.py",
        "qwen4_hc_fusion.py",
        "qwen4_mtp_gemm.py",
        "qwen4_fused_gate_kernel.py",
    )
    files = {name: (root / name).read_bytes() for name in names}
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
        "image_source_preimages": IMAGE_SOURCES,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (destination / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    digest = package(parser.parse_args().destination)
    print(
        json.dumps(
            {
                "manifest_sha256": digest,
                "cache_namespace": "qwen4-prefill-" + digest[:12],
            }
        )
    )
