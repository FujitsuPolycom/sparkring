import hashlib
import json
from types import SimpleNamespace

from runtime.host import assets


def test_cache_candidate_requires_exact_metadata_and_every_shard(tmp_path):
    config = b'{"model_type":"fixture"}\n'
    index = json.dumps({"weight_map": {"weight": "weights.safetensors"}}).encode()
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors.index.json").write_bytes(index)
    contract = {"config_sha256": hashlib.sha256(config).hexdigest(), "index_sha256": hashlib.sha256(index).hexdigest()}
    assert not assets.metadata_matches(tmp_path, contract)
    (tmp_path / "weights.safetensors").write_bytes(b"fixture")
    assert assets.metadata_matches(tmp_path, contract)
    (tmp_path / "config.json").write_bytes(b"wrong version")
    assert not assets.metadata_matches(tmp_path, contract)


def test_discovery_does_not_download_or_call_gpu_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.setup, "selection", lambda p: {"profile": p, "model_repository": "org/model", "model_revision": "a" * 40})
    monkeypatch.setattr(assets.installer, "checkpoint_contract", lambda c: {})
    monkeypatch.setattr(assets, "metadata_matches", lambda path, contract: str(path) == str(tmp_path))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    found = assets.discover("fixture", run=run, extra_roots=[tmp_path])
    assert found["model_path"] == str(tmp_path)
    assert found["full_shard_verification"] == "required-before-launch"
    assert calls == [["docker", "ps", "-aq"]]
