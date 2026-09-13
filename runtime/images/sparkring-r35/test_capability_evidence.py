"""Evidence must identify both its summary and the underlying component-test bytes."""
import json
from pathlib import Path
import shutil

import pytest

from verify_image import verify_capability_evidence


@pytest.fixture
def profile(tmp_path):
    root = Path(__file__).parent/'contracts'
    shutil.copy2(root/'tp2-sparkcache-capabilities.json', tmp_path)
    shutil.copytree(root/'evidence', tmp_path/'evidence')
    return tmp_path


def test_bundled_component_evidence(profile):
    verify_capability_evidence(profile)


@pytest.mark.parametrize('summary', [True, False])
def test_changed_summary_or_artifact_is_rejected(profile, summary):
    path = profile/'evidence/managed_b12x_loader.evidence.json'
    if not summary:
        path = path.parent/json.loads(path.read_text())['artifact']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        verify_capability_evidence(profile)
