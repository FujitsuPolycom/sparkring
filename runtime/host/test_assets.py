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
    assert calls == [["docker", "--context", "default", "ps", "-aq"]]


def test_portable_discovery_uses_head_pins_without_worker_profile_imports(tmp_path, monkeypatch, capsys):
    import subprocess
    config = b'{}'
    index = b'{"weight_map":{"weight":"weights.safetensors"}}'
    (tmp_path / 'config.json').write_bytes(config)
    (tmp_path / 'model.safetensors.index.json').write_bytes(index)
    (tmp_path / 'weights.safetensors').write_bytes(b'fixture')
    card = {'profile': 'new-profile-unknown-to-worker', 'model_repository': 'org/model', 'model_revision': 'a' * 40}
    contract = {'config_sha256': hashlib.sha256(config).hexdigest(), 'index_sha256': hashlib.sha256(index).hexdigest()}
    def run(argv, **kwargs):
        assert argv[:3] == ['docker', '--context', 'default']
        data = json.dumps([{'Mounts': [{'Type': 'bind', 'Source': str(tmp_path), 'Destination': '/models/target'}]}]) if 'inspect' in argv else 'container\n'
        return SimpleNamespace(stdout=data, returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    code = assets.probe_code(card, contract)
    assert 'from runtime' not in code and '/usr/bin/sparkring' not in code
    exec(compile(code, '<portable-asset-probe>', 'exec'), {})
    assert json.loads(capsys.readouterr().out)['model_path'] == str(tmp_path)


def test_hub_named_folder_without_revision_is_a_candidate(tmp_path, monkeypatch):
    import hashlib as _hashlib
    import json as _json
    from runtime.host import assets as _assets
    root = tmp_path / "models" / "Example--Model"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(_json.dumps({"weight_map": {"w": "w.safetensors"}}))
    (root / "w.safetensors").write_text("x")
    contract = {"config_sha256": _hashlib.sha256(b"{}").hexdigest(),
                "index_sha256": _hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()}
    card = {"profile": "p", "model_repository": "Example/Model", "model_revision": "a" * 40}
    result = _assets.discover_contract(card, contract, run=lambda *a, **k: SimpleNamespace(stdout=""),
                                       model_roots=(str(tmp_path / "models"),))
    assert result["model_path"] == str(root)
