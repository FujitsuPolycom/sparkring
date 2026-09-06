"""Verify packaged overlay bytes; deliberately does not initialize CUDA."""
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
