"""Inventory installed eugr package bytes without importing GPU code."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform

names = {dist.metadata['Name'] for dist in metadata.distributions()
         if dist.metadata.get('Name')}
record = {'architecture': platform.machine(),
          'versions': {name: metadata.version(name) for name in sorted(names)},
          'packages': {}}
for component in ('vllm', 'b12x'):
    distribution = metadata.distribution(component)
    root = Path(distribution.locate_file(component)).resolve()
    assert root.is_dir() and str(root).startswith('/usr/local/lib/python3.12/dist-packages/')
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix == '.pyc':
            continue
        assert not path.is_symlink(), path
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[path.relative_to(root).as_posix()] = {'sha256': digest, 'bytes': path.stat().st_size}
    record['packages'][component] = {'root': str(root), 'version': distribution.version, 'files': files}
Path('/candidate/base-inventory.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'packages': {k: len(v['files']) for k, v in record['packages'].items()}}))
