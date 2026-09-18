"""Fail-closed native release admission without Docker or GPUs."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from runtime.common import native_candidate as native


def fixture():
    trees = {"vllm": "a" * 64, "b12x": "b" * 64}
    raw = json.dumps({"compiler": {"source_trees": trees}, "files": {"/owned": "c" * 64}}).encode()
    image = "sha256:" + "d" * 64
    record = dict(image_id=image, release="shared-2026.09.0", source_trees=trees,
                  installed_receipt_sha256=hashlib.sha256(raw).hexdigest(), input_sha256="e" * 64,
                  image_reference="ghcr.io/fujitsupolycom/sparkring@sha256:" + "f" * 64,
                  features=["qwen4-prefill"])
    inspection = dict(Id=image, Os="linux", Architecture="arm64",
                      Config={"Entrypoint": ["/opt/venv/bin/python", native.ENTRYPOINT]})
    verified = dict(schema="sparkring-native-verification/v1", source_trees=trees,
                    receipt_sha256=record["installed_receipt_sha256"], input_sha256="e" * 64,
                    files_verified=1, features=["qwen4-prefill"])
    return record, image, inspection, raw, verified


def test_exact_native_receipt_is_accepted():
    record, image, info, raw, result = fixture()
    assert native.validate(record, image, info, raw, result)["image_id"] == image


@pytest.mark.parametrize("field,value", [
    ("receipt_sha256", "0" * 64), ("source_trees", {}),
    ("input_sha256", "0" * 64), ("files_verified", 0), ("features", []),
])
def test_mismatched_native_verification_is_rejected(field, value):
    record, image, info, raw, result = fixture()
    result[field] = value
    with pytest.raises(ValueError):
        native.validate(record, image, info, raw, result)


def test_changed_receipt_or_entrypoint_is_rejected():
    record, image, info, raw, result = fixture()
    with pytest.raises(ValueError):
        native.validate(record, image, info, raw + b" ", result)
    info = copy.deepcopy(info)
    info["Config"]["Entrypoint"] = ["/bin/sh"]
    with pytest.raises(ValueError):
        native.validate(record, image, info, raw, result)


def test_release_path_escape_is_rejected():
    with pytest.raises(ValueError):
        native.publication("../../unregistered")


def test_bad_native_platform_is_rejected_before_running_any_container(monkeypatch):
    record, image, info, _, _ = fixture()
    info['Architecture'] = 'amd64'
    monkeypatch.setattr(native, 'publication', lambda *args, **kwargs: record)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[1:3] == ['image', 'inspect']
        return SimpleNamespace(stdout=json.dumps([info]))

    with pytest.raises(ValueError, match='platform'):
        native.verify_image(image, record['release'], run=run)
    assert len(calls) == 1
