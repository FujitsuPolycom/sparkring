import hashlib
import importlib.util
from pathlib import Path
import tarfile

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mesh_indexer_patch', HERE / 'patch_indexer_barrier.py')
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def test_pinned_selector_archive_produces_the_checked_barrier_source():
    with tarfile.open(HERE.parent / 'compute/b12x-selector-files.tar.gz') as archive:
        source = archive.extractfile(patch.RELATIVE).read()
    assert hashlib.sha256(source).hexdigest() == patch.BEFORE
    result = patch.patch_bytes(source)
    assert hashlib.sha256(result).hexdigest() == patch.AFTER
    assert result.count(b'\r\n') == source.count(b'\r\n') + 2
    compile(result.decode('utf-8'), patch.RELATIVE, 'exec')
    assert patch.patch_bytes(result) == result


def test_unknown_source_is_rejected_without_mutation(tmp_path):
    target = tmp_path / 'indexer.py'
    target.write_bytes(b'unsupported source\n')
    with pytest.raises(ValueError, match='Unsupported mesh indexer'):
        patch.apply(target)
    assert target.read_bytes() == b'unsupported source\n'
