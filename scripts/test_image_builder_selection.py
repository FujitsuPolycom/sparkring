"""Reject ambiguous builder dispatch without invoking a build."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.images.build import plan  # noqa: E402


@pytest.mark.parametrize('rows', [
    [{'id': 'sample', 'kind': 'pyhton', 'path': 'build.py', 'reason': 'fixture'}],
    [{'id': 'sample', 'kind': 'python', 'path': 'build.py', 'reason': 'fixture'}] * 2,
])
def test_invalid_dispatch_is_rejected(tmp_path, rows):
    directory = tmp_path / 'runtime/images'
    directory.mkdir(parents=True)
    (directory / 'builders.json').write_text(json.dumps({
        'schema': 'sparkring-image-builders/v1', 'builders': rows,
    }))
    (tmp_path / 'build.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='unique IDs and a supported interpreter'):
        plan('sample', root=tmp_path)


def test_catalog_plans_preserve_arguments_without_execution():
    from runtime.common.profiles import ROOT, read_json
    for row in read_json(ROOT / 'runtime/images/builders.json')['builders']:
        result = plan(row['id'], ['argument with spaces', '--execute'])
        assert result['command'][0] == (sys.executable if row['kind'] == 'python' else 'bash')
        assert Path(result['command'][1]) == (ROOT / row['path']).resolve()
        assert result['command'][2:] == ['argument with spaces', '--execute']
