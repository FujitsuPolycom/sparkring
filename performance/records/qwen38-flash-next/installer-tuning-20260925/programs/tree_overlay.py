#!/usr/bin/env python3
"""Build a container overlay that mounts a tree of changed Python files over an image.

Usage: tree_overlay.py IMAGE_ID FILES_DIR OVERLAY_DIR
FILES_DIR mirrors the image's dist-packages root (vllm/..., b12x/...). Every file
under it is mounted at /usr/local/lib/python3.12/dist-packages/<relative path>.
The image's external-base receipt records each mounted file's SHA-256 (replaced
files are re-hashed, new files are added), and the toolchain receipt is re-bound
to the rewritten base receipt, so the image's installed-tree verification accepts
the mounts. volumes.json lists the mounts for a qwab variant.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

IMAGE, FILES, OVERLAY = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
ROOT = "/usr/local/lib/python3.12/dist-packages/"
BASE_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
TOOLCHAIN_RECEIPT = "/opt/sparkring/toolchain/installed.json"


def image_file(path):
    return subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "cat",
                           IMAGE, path], capture_output=True, check=True).stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


if OVERLAY.exists():
    shutil.rmtree(OVERLAY)
(OVERLAY / "files").mkdir(parents=True)
base = json.loads(image_file(BASE_RECEIPT))
volumes, added = [], []
for source in sorted(p for p in FILES.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
    relative = source.relative_to(FILES).as_posix()
    target = ROOT + relative
    data = source.read_bytes()
    if target not in base["files"]:
        added.append(target)
    base["files"][target] = sha(data)
    copy = OVERLAY / "files" / relative
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(data)
    volumes.append({"source": str(copy), "target": target})
payload = (json.dumps(base, indent=2, sort_keys=True) + "\n").encode()
(OVERLAY / "external-base-installed.json").write_bytes(payload)
toolchain = json.loads(image_file(TOOLCHAIN_RECEIPT))
toolchain["parent_receipt_sha256"] = sha(payload)
(OVERLAY / "toolchain-installed.json").write_text(json.dumps(toolchain, indent=2) + "\n")
volumes += [{"source": str(OVERLAY / "external-base-installed.json"), "target": BASE_RECEIPT},
            {"source": str(OVERLAY / "toolchain-installed.json"), "target": TOOLCHAIN_RECEIPT}]
(OVERLAY / "volumes.json").write_text(json.dumps(volumes, indent=1) + "\n")
print(json.dumps({"mounted": len(volumes) - 2, "new_files": added, "base_receipt": sha(payload)[:16]}))
