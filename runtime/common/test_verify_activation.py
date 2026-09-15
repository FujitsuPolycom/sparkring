"""Validate retained activation evidence without loading a model or changing hosts."""

import copy
import json
import subprocess
import sys

import pytest

from runtime.common import verify_activation


RECEIPT = verify_activation.ROOT / 'runtime/sparkring/jovian-r33/profiles/evidence/tp4-dcp4-sparkcache-activation-20260911.json'


def test_stored_receipt_passes_maintained_cli():
    result = subprocess.run([sys.executable, str(verify_activation.__file__), '--receipt', str(RECEIPT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'profile': 'tp4-dcp4-sparkcache', 'ranks': 4, 'checks_passed': True}


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'bool', 'float', 'cache-off', 'cache-absent', 'image'])
def test_inconsistent_receipt_is_rejected(change):
    document = json.loads(RECEIPT.read_text())
    if change == 'duplicate':
        document['ranks'].append(copy.deepcopy(document['ranks'][0]))
    elif change == 'missing':
        document['ranks'].pop()
    elif change in ('bool', 'float'):
        document['ranks'][0]['rank'] = False if change == 'bool' else 0.0
    elif change == 'cache-off':
        document['sparkcache']['enabled'] = False
    elif change == 'cache-absent':
        document['sparkcache'].pop('enabled')
    else:
        document['image']['checks_passed'] = False
    with pytest.raises(ValueError):
        verify_activation.validate(document)
