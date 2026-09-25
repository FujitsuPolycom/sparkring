#!/usr/bin/env python3
"""Derive a SparkRing serving image with the Qwen4Exp SM121 decode patch.

Run on a host holding the parent image:
  build_qwen_decode.py PARENT_IMAGE_ID TAG PATCH_DIR PARENT_LOCK OUTPUT_LOCK NAME

PATCH_DIR holds envs.py, low_latency_gemm.py, model.py and mtp.py from
qwen_decode_patch.py. The derived layer replaces those vLLM files and re-records
their hashes in the external-base receipt; the toolchain receipt records the
rewritten base receipt. /opt/sparkring/receipts/derived-qwen-decode.json lists
each replaced path with its inherited and resulting SHA-256. The image's isolated `verify`
must accept the result. The output lock copies the parent lock's transport,
composition and status fields and records the derived image.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PARENT, TAG, PATCH, PARENT_LOCK, OUTPUT_LOCK, NAME = sys.argv[1:7]
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm/"
TARGETS = {"envs.py": VLLM + "envs.py",
           "low_latency_gemm.py": VLLM + "models/qwen4_exp/nvidia/low_latency_gemm.py",
           "model.py": VLLM + "models/qwen4_exp/nvidia/model.py",
           "mtp.py": VLLM + "models/qwen4_exp/nvidia/mtp.py"}
BASE_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"
PROVENANCE = "/opt/sparkring/receipts/derived-qwen-decode.json"
CONTEXT = Path("/var/tmp/qwen-decode-context")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_file(path):
    return subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "cat",
                           PARENT, path], capture_output=True, check=True).stdout


if CONTEXT.exists():
    shutil.rmtree(CONTEXT)
(CONTEXT / "files").mkdir(parents=True)
base_bytes = image_file(BASE_RECEIPT)
base = json.loads(base_bytes)
provenance = {"schema": "sparkring-derived-layer/v1", "parent_image_id": PARENT,
              "parent_receipt_sha256": sha(base_bytes),
              "purpose": ("Qwen4Exp decode on GB10 (SM121): measured low-latency GEMM plans for BF16 projections "
                          "beside B12X, and MXFP8 hyper-connection mixing projections under VLLM_QWEN4_EXP_MXFP8_HC"),
              "files": {}}
for name, path in TARGETS.items():
    data = (Path(PATCH) / name).read_bytes()
    inherited = base["files"].get(path)
    if inherited is None:
        raise SystemExit("Replaced path is not recorded by the parent: " + path)
    if sha(image_file(path)) != inherited:
        raise SystemExit("Parent file differs from its receipt: " + path)
    base["files"][path] = sha(data)
    provenance["files"][path] = {"inherited_sha256": inherited, "sha256": sha(data)}
    target = CONTEXT / "files" / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
new_base = (json.dumps(base, indent=2, sort_keys=True) + "\n").encode()
toolchain = json.loads(image_file(TOOLCHAIN_RECEIPT))
toolchain["parent_receipt_sha256"] = sha(new_base)
new_toolchain = (json.dumps(toolchain, indent=2) + "\n").encode()
provenance["receipts"] = {BASE_RECEIPT: sha(new_base), TOOLCHAIN_RECEIPT: sha(new_toolchain)}
for path, data in ((BASE_RECEIPT, new_base), (TOOLCHAIN_RECEIPT, new_toolchain),
                   (PROVENANCE, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())):
    target = CONTEXT / "files" / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
# BuildKit resolves a bare image ID as a registry name, so the parent is
# referenced through a local tag.
parent_tag = "sparkring-dev/parent:" + PARENT.removeprefix("sha256:")[:12]
subprocess.run(["docker", "tag", PARENT, parent_tag], check=True)
(CONTEXT / "Dockerfile").write_text(f"FROM {parent_tag}\nCOPY files/ /\n")
subprocess.run(["docker", "build", "-q", "-t", TAG, str(CONTEXT)], check=True)
image = json.loads(subprocess.run(["docker", "image", "inspect", TAG], capture_output=True, check=True).stdout)[0]
isolated = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "python3", image["Id"],
            "/opt/sparkring/toolchain/toolchain.py", "verify"]
subprocess.run(isolated, check=True, capture_output=True)
lock = json.loads(Path(PARENT_LOCK).read_text())
lock.update(name=NAME, image_id=image["Id"], image_reference=image["Id"],
            parent_receipt_sha256=sha(new_base), toolchain_receipt_sha256=sha(new_toolchain),
            image_bytes=image["Size"])
Path(OUTPUT_LOCK).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
print(json.dumps({"image_id": image["Id"], "tag": TAG, "changed": len(provenance["files"]),
                  "parent_receipt_sha256": sha(new_base), "toolchain_receipt_sha256": sha(new_toolchain)}))
