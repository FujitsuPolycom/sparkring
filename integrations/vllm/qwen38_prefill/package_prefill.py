"""Build a manifest-bound local Qwen prefill source bundle."""

import argparse
import hashlib
import json
from pathlib import Path
from prefill_bootstrap import IMAGE_SOURCES


def package(destination):
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    names = (
        "prefill_bootstrap.py",
        "qwen_hc_fusion.py",
        "qwen_mtp_gemm.py",
        "fused_gate_kernel.py",
    )
    files = {name: (root / name).read_bytes() for name in names}
    files["qwen_prefill.pth"] = (
        "/opt/sparkring/qwen38-prefill\n"
        "import prefill_bootstrap; prefill_bootstrap.install()\n"
    ).encode()
    for name, data in files.items():
        (destination / name).write_bytes(data)
    manifest = {
        "schema": "sparkring-qwen-prefill/v1",
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
                "cache_namespace": "qwen-prefill-" + digest[:12],
            }
        )
    )
