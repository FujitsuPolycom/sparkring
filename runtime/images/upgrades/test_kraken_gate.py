import hashlib

import pytest

from .kraken_gate import bind_peer


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
