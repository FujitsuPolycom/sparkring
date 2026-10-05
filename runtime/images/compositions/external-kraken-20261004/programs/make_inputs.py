"""Write the sparkring-external-inputs/v1 manifest for the kraken software layer.

Run from the build directory after base-inventory.json and assets/ exist.
Usage: python3 make_inputs.py BASE_CONFIG_ID
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
config_id = sys.argv[1]


def pin(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


controller = ROOT / "controller/qwen4_prefill"
prefill_files = ("package_prefill.py", "qwen4_prefill_bootstrap.py", "qwen4_hc_fusion.py",
                 "qwen4_mtp_gemm.py", "qwen4_fused_gate_kernel.py")
manifest = {
    "schema": "sparkring-external-inputs/v1",
    "platform": "linux/arm64",
    "base": {
        "reference": "eugr/spark-vllm-b12x@sha256:141f46a4a2c3751798f16759cc859648784be430be852a84a21f0c4c427b4052",
        "config_id": config_id,
    },
    "base_inventory": pin(ROOT / "base-inventory.json"),
    "installer": pin(ROOT / "controller/external_base.py"),
    "parent_cache_contract": pin(ROOT / "parent-cache-contract.json"),
    "sources": {
        "vllm": {
            "commit": "d51b4181acc06622357c81dc8e0be46ad96a0779",
            "upstream": "ad3743e35876d895a26a04e88702a260bc79b07b",
            "baseline_commit": "ab86b70734009c34e91db456bb96340c3faf3d4e",
            "archive": pin(ROOT / "vllm-kraken-d51b4181.tar"),
            "baseline_archive": pin(ROOT / "vllm-base-ab86b707.tar"),
        },
        "b12x": {
            "commit": "a850d8ebde9b23674e0b76949c2e87161baf3ec4",
            "upstream": "dfe61efc19862d764872bf2d831200e40b3ce561",
            "baseline_commit": "e4084d2eef4932e0fa06db3f7a94deb83ad132d7",
            "archive": pin(ROOT / "b12x-kraken-a850d8eb.tar"),
            "baseline_archive": pin(ROOT / "b12x-base-e4084d2e.tar"),
        },
    },
    "integration_sources": {
        "vllm": pin(ROOT / "integration/vllm-source-m3.tar"),
        "b12x": pin(ROOT / "integration/b12x-source-m3.tar"),
    },
    "assets": {
        "root": "assets",
        "export_manifest": pin(ROOT / "assets/export.json"),
        "parent_receipt": pin(ROOT / "assets/parent-receipt.json"),
    },
    "prefill_controller": {
        "root": "controller/qwen4_prefill",
        "files": {name: pin(controller / name)["sha256"] for name in prefill_files},
    },
    "runtime_status": {
        "wheel": pin(ROOT / "status/sparkring_runtime_status-0.3.4-py3-none-any.whl"),
        "source_archive": pin(ROOT / "status/sparkring-runtime-status-0.3.4-source.tar.gz"),
    },
    "deployment_addons": {
        "manifest": pin(ROOT / "addons/manifest.json"),
        "archive": pin(ROOT / "addons/addons.tar"),
    },
}
(ROOT / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"inputs": str(ROOT / "inputs.json")}))
