"""Cache discovery and transfer decisions on the actual installer asset path."""
import io
from types import SimpleNamespace

import pytest

from runtime.host import install_assets as assets
from runtime.host.install_errors import NeedsInput


class Transport:
    hosts = [{"rank": n} for n in range(4)]
    mode = "fiber-ssh"
    def argv(self, rank):
        return ["ssh", "-F", "/head/ssh_config", "spark" + str(rank)]
    def command(self, rank, args):
        return ["node" + str(rank), *args]


def test_missing_private_image_asks_for_input_without_contacting_registry(tmp_path):
    current = assets.Assets(Transport(), tmp_path, run=lambda *a, **k: pytest.fail("Unexpected download"))
    current.remote = lambda *a, **k: {"present": False}
    with pytest.raises(NeedsInput, match="not cached"):
        current.images({"image_id": "sha256:a", "image_reference": "sha256:a"})


def test_peer_cached_image_streams_over_fabric_without_archive_on_head(tmp_path):
    calls = []
    class Child:
        stdout = io.BytesIO(b"docker image bytes")
        returncode = 0
        def wait(self, **kw): return 0
        def poll(self): return 0
    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["stdin"].read() == b"docker image bytes"
        return SimpleNamespace(returncode=0, stdout=b'{"image_id":"sha256:a"}', stderr=b"")
    current = assets.Assets(Transport(), tmp_path, run=run, popen=lambda argv, **kwargs: calls.append(argv) or Child())
    def remote(rank, fn, *args, **kwargs):
        assert fn == assets.image_probe
        return {"present": rank != 2, "image_id": "sha256:a", "size_bytes": 10, "free_bytes": 100 * 1024**3}
    current.remote = remote
    result = current.images({"image_id": "sha256:a", "image_reference": "sha256:a"})
    assert result["donor_rank"] == 0 and result["imported"][0]["rank"] == 2
    assert calls[0][:5] == ["node0", "docker", "--context", "default", "image"]
    assert calls[1][:2] == ["node2", "sudo"]
    assert not list(tmp_path.glob("*.tar"))


@pytest.mark.parametrize("cached,donor,copies", [([0, 1, 2, 3], 0, []), ([2], 2, [0, 1, 3]), ([], 0, [1, 2, 3])])
def test_checkpoint_uses_one_donor_or_one_download(tmp_path, cached, donor, copies):
    calls = []
    rows = [{"rank": n, "model": "/models/pinned"} for n in range(4)]
    class Runner:
        def remote(self, rank, op, **kwargs):
            calls.append((rank, op))
            if op == "model-present":
                return {"present": rank in cached}
            if op == "model-transfer-manifest":
                return {"files": {"weight.safetensors": "a" * 64}}
            return {"ok": True}
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    current = assets.Assets(Transport(), tmp_path, run=run)
    report = current.models({"site": {"ranks": rows}}, Runner())
    assert report["donor_rank"] == donor
    assert [c[0] for c in calls if isinstance(c, tuple) and c[1] == "model"] == [donor]
    assert [c[0] for c in calls if isinstance(c, tuple) and c[1] == "model-transfer-complete"] == copies
    for call in (c for c in calls if isinstance(c, list)):
        assert call[0] == "rsync" and "--delete" not in call and "-e" in call
        assert "/head/ssh_config" in call[call.index("-e") + 1]
