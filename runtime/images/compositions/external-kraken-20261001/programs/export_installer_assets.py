"""Export receipt-bound integration assets from an installer image without starting it.

Usage: python3 export_installer_assets.py IMAGE_ID EXPECTED_RECEIPT_SHA256 OUTPUT_DIR
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

image, expected_receipt, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])
assert image.startswith("sha256:") and len(image) == 71, image
output.mkdir(exist_ok=False)
container = subprocess.check_output(
    ["docker", "create", "--network", "none", "--entrypoint", "/bin/true", image], text=True).strip()
items = {
    "/opt/sparkring/transports": "transports",
    "/opt/sparkring/features": "features",
    "/opt/sparkring/sparkcache": "sparkcache-native",
    "/opt/sparkring/licenses": "licenses",
    "/opt/local-inference/nccl/lib": "nccl-lib",
    "/usr/local/lib/python3.12/dist-packages/sparkcache": "sparkcache",
    "/opt/sparkring/receipts/external-base-installed.json": "parent-receipt.json",
}
try:
    for source, destination in items.items():
        subprocess.run(["docker", "cp", container + ":" + source, str(output / destination)], check=True)
finally:
    subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)
(output / "sparkcache-overrides").mkdir()
receipt_raw = (output / "parent-receipt.json").read_bytes()
assert hashlib.sha256(receipt_raw).hexdigest() == expected_receipt, "Installed receipt differs from the expected image"
parent = json.loads(receipt_raw)
files = {}
for source, destination in items.items():
    if destination == "parent-receipt.json":
        continue
    for path in sorted((output / destination).rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        name = path.relative_to(output / destination).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        original = source + "/" + name
        expected = parent["files"].get(original)
        if destination == "nccl-lib" and path.is_symlink():
            expected = parent["files"].get(str(Path(source) / path.resolve().name))
        assert expected == digest, (original, expected, digest)
        files[destination + "/" + name] = {"origin": original, "sha256": digest}
record = {
    "parent_image": image,
    "parent_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    "files": files,
    "scope": "Installer-image integration assets only; no framework package sources or binaries copied.",
}
(output / "export.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"files": len(files), "output": str(output)}))
