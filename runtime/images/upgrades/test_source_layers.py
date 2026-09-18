"""Carried source layers preserve exact baseline feature sets without approval drift."""

import pytest

from .contracts import Refused, sha
from .source_layers import compose_extension
from .sources import apply_patch, git


def fixture(tmp_path):
    root = tmp_path / 'baseline'
    (root / 'vllm').mkdir(parents=True)
    path = root / 'vllm/demo.py'
    path.write_bytes(b'value = 1\n')
    git(root, 'init')
    git(root, 'add', '--all')
    path.write_bytes(b'value = 2\n')
    patch = b'diff --git a/vllm/demo.py b/vllm/demo.py\n--- a/vllm/demo.py\n+++ b/vllm/demo.py\n@@ -1 +1 @@\n-value = 2\n+value = 3\n'
    descriptor = dict(schema='sparkring-source-extension/v1', id='test-extension',
                      patch={'sha256': sha(patch)}, sources={'vllm/demo.py': {
                          'parent_sha256': sha(b'value = 2\n'), 'sha256': sha(b'value = 3\n')}})
    return root, descriptor, patch


def test_combines_prior_patch_and_extension_without_changing_baseline(tmp_path):
    root, descriptor, patch = fixture(tmp_path)
    result, proof = compose_extension(root, tmp_path/'result', 'vllm', descriptor, patch)
    assert (root/'vllm/demo.py').read_bytes() == b'value = 2\n'
    fresh = tmp_path/'fresh'
    (fresh/'vllm').mkdir(parents=True)
    (fresh/'vllm/demo.py').write_bytes(b'value = 1\n')
    git(fresh, 'init')
    assert apply_patch(fresh, result)[0]
    assert (fresh/'vllm/demo.py').read_bytes() == b'value = 3\n'
    assert proof['serving_qualified'] is False


@pytest.mark.parametrize('failure', ['preimage', 'postimage', 'patch'])
def test_refuses_drift(tmp_path, failure):
    root, descriptor, patch = fixture(tmp_path)
    if failure == 'preimage':
        (root/'vllm/demo.py').write_bytes(b'wrong\n')
    elif failure == 'postimage':
        descriptor['sources']['vllm/demo.py']['sha256'] = '0'*64
    else:
        patch += b'changed'
    with pytest.raises(Refused):
        compose_extension(root, tmp_path/'result', 'vllm', descriptor, patch)
