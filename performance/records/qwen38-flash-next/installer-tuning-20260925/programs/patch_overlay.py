#!/usr/bin/env python3
"""Build a container overlay that applies a Qwen decode patch directory to an image.

Usage: patch_overlay.py IMAGE_ID PATCH_DIR OVERLAY_DIR
PATCH_DIR holds envs.py, low_latency_gemm.py, model.py and mtp.py from
qwen_decode_patch.py. OVERLAY_DIR receives those files plus the image's
external-base receipt with their hashes re-recorded and the toolchain receipt
re-bound to it, so the image's installed-tree verification accepts the mounts.
volumes.json lists the mounts for a qwab variant.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

IMAGE, PATCH, OVERLAY = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm/"
TARGETS = {"envs.py": VLLM + "envs.py",
           "low_latency_gemm.py": VLLM + "models/qwen4_exp/nvidia/low_latency_gemm.py",
           "model.py": VLLM + "models/qwen4_exp/nvidia/model.py",
           "mtp.py": VLLM + "models/qwen4_exp/nvidia/mtp.py"}
BASE_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"


def image_file(path):
    return subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "cat",
                           IMAGE, path], capture_output=True, check=True).stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


OVERLAY.mkdir(parents=True, exist_ok=True)
base = json.loads(image_file(BASE_RECEIPT))
volumes = []
for name, path in TARGETS.items():
    data = (PATCH / name).read_bytes()
    if path not in base["files"]:
        raise SystemExit("The image does not record " + path)
    base["files"][path] = sha(data)
    (OVERLAY / name).write_bytes(data)
    volumes.append({"source": str(OVERLAY / name), "target": path})
payload = (json.dumps(base, indent=2, sort_keys=True) + "\n").encode()
(OVERLAY / "external-base-installed.json").write_bytes(payload)
toolchain = json.loads(image_file(TOOLCHAIN_RECEIPT))
toolchain["parent_receipt_sha256"] = sha(payload)
(OVERLAY / "toolchain-installed.json").write_text(json.dumps(toolchain, indent=2) + "\n")
volumes += [{"source": str(OVERLAY / "external-base-installed.json"), "target": BASE_RECEIPT},
            {"source": str(OVERLAY / "toolchain-installed.json"), "target": TOOLCHAIN_RECEIPT}]
(OVERLAY / "volumes.json").write_text(json.dumps(volumes, indent=1) + "\n")
print(f"overlay with {len(TARGETS)} files; base receipt {sha(payload)[:16]}")
