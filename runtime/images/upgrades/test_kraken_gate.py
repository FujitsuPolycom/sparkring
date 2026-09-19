import hashlib
import json

import pytest

from .kraken_gate import bind_peer
from . import kraken_gate


def test_coordinator_identity_is_required(tmp_path):
    path = tmp_path / "vllm/v1/worker/b12x_startup.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"coordinator = 1\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert bind_peer(tmp_path, expected) == tmp_path.resolve()
    path.write_bytes(b"coordinator = 2\n")
    with pytest.raises(ValueError, match="differs"):
        bind_peer(tmp_path, expected)


@pytest.mark.parametrize("digest", ["", "short", "G" * 64, "0" * 64])
def test_absent_or_unidentified_coordinator_is_rejected(tmp_path, digest):
    with pytest.raises(ValueError):
        bind_peer(tmp_path, digest)


@pytest.mark.parametrize("outcome", ["passed", "failed"])
def test_completed_oracle_outcome_is_reported_not_an_execution_failure(
    tmp_path, monkeypatch, capsys, outcome
):
    path = tmp_path / "vllm/v1/worker/b12x_startup.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"coordinator = 1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("SPARKRING_VLLM_SOURCE_ROOT", "")
    monkeypatch.setattr("sys.argv", ["kraken_gate.py", "--source", "/source",
        "--baseline", "/baseline", "--suite", "/suite", "--result", "/result",
        "--peer-vllm-root", str(tmp_path), "--peer-vllm-sha256", digest])
    monkeypatch.setattr(kraken_gate, "run", lambda *args: {"outcome": outcome})
    assert kraken_gate.main() == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == outcome
