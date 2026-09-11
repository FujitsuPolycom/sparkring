"""Check installed package files against the supplied package manifest.

The manifest is trusted input: this detects file drift, not changes to both
files and manifest, and does not attest source provenance or GPU behavior.
package_image.py installs this tool as verify-runtime-package.py. The separate
source_image/verify_image.py verifies source-image receipts against a lock.
No CUDA initialization is performed.
"""
import hashlib
import json
from pathlib import Path


def verify(manifest_path=Path('/opt/sparkring/runtime-manifest.json')):
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        path = Path(item['destination'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError(f'Packaged file mismatch: {path}')
    print('Packaged files verified; GPU behavior remains unqualified.')


if __name__ == '__main__':
    verify()
