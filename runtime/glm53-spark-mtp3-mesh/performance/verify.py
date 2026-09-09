"""Verify the complete installed performance image file inventory."""

import hashlib
import json
from pathlib import Path

receipt = json.loads(Path("/opt/sparkring/receipts/mtp3-performance.json").read_text())
for name, expected in receipt["files"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Runtime source differs from image receipt: {name}")
print(
    json.dumps(
        {
            "verified_files": len(receipt["files"]),
            "sparkcache_commit": receipt["sparkcache_commit"],
        }
    )
)
