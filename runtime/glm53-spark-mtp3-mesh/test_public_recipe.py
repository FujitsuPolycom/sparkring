"""Keep the published recipe, registry receipt and executable profile pins aligned."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def test_recipe_matches_canonical_inputs():
    recipe = json.loads((ROOT / 'recipes/glm53-spark-mtp3-managed-mesh-tp4.json').read_text())
    pins = json.loads((HERE / 'pins.json').read_text())
    public = json.loads((HERE / 'public-image.json').read_text())
    assert recipe['model']['repository'] == pins['target']['repository']
    assert recipe['model']['revision'] == pins['target']['revision']
    assert recipe['model']['config_sha256'] == pins['target']['config_sha256']
    assert recipe['model']['weight_index_sha256'] == pins['target']['index_sha256']
    assert recipe['runtime']['image'] == public['public_reference']
    assert recipe['runtime']['image_id'] == public['config_image_id']
    assert recipe['serving_common']['cudagraph_capture_sizes'] == pins['capture_sizes']
    assert recipe['transport']['bundle_manifest_sha256'] == pins['canonical_bundle_manifest_sha256']
    assert recipe['transport']['managed_marker_expiry'] is None
    assert recipe['sparkcache']['enabled'] is True
    assert recipe['status'] == 'research-only'


def test_public_content_receipt_is_accepted_by_renderer():
    spec = importlib.util.spec_from_file_location('public_recipe_profile', HERE / 'profile.py')
    profile = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = profile
    spec.loader.exec_module(profile)
    receipt = profile.load_image_receipt(HERE / 'image-receipt.json')
    public = json.loads((HERE / 'public-image.json').read_text())
    assert receipt['image_id'] == public['config_image_id']
    assert public['checks_passed'] and public['anonymous_pull']
    assert public['all_layer_diff_ids_match_tested_image']


def test_published_warmup_receipt_does_not_follow_checkout_source(monkeypatch):
    spec = importlib.util.spec_from_file_location('public_warmup_profile', HERE / 'profile.py')
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    monkeypatch.setattr(profile, "sha", lambda path: "f" * 64)
    receipt = profile.load_image_receipt(HERE / 'image-receipt.json')
    assert receipt['inside_image']['readiness_warmup']['helper_sha256'] != "f" * 64


@pytest.mark.parametrize("use_published_identity", [True, False])
def test_published_identity_cannot_authorize_forged_warmup(tmp_path, use_published_identity):
    spec = importlib.util.spec_from_file_location('forged_warmup_profile', HERE / 'profile.py')
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    document = json.loads((HERE / 'image-receipt.json').read_text())
    document['inside_image']['readiness_warmup']['helper_sha256'] = "f" * 64
    if not use_published_identity:
        document['image_id'] = document['image_reference'] = "sha256:" + "a" * 64
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Published image receipt|sampling warmup helper"):
        profile.load_image_receipt(path)
