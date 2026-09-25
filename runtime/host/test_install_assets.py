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
    def forwarded(self, rank, remote, local, args):
        return ["node" + str(rank), f"forward:{remote}->{local}", *args]


REGISTRY_CARD = {"image_id": "sha256:a", "image_reference": "ghcr.io/owner/image@sha256:" + "b" * 64,
                 "image_bytes": 30 * 1024**3, "download_bytes": 15 * 1024**3}


class Relay:
    port = 5255
    instances = []
    def __init__(self, reference, directory):
        self.closed = False
        Relay.instances.append(self)
    def reference(self, port=None):
        return f"127.0.0.1:{port or self.port}/owner/image@sha256:" + "b" * 64
    def close(self):
        self.closed = True


@pytest.fixture
def relay(monkeypatch):
    Relay.instances = []
    monkeypatch.setattr(assets.registry_relay, "Relay", Relay)
    return Relay


def probe(present, free=200 * 1024**3):
    def remote(rank, fn, *args, **kwargs):
        assert fn == assets.image_probe
        return {"present": rank in present, "image_id": "sha256:a", "size_bytes": 30 * 1024**3, "free_bytes": free}
    return remote


def test_every_missing_node_pulls_through_one_relay_on_node_a(tmp_path, relay):
    present, calls = {1, 3}, []
    def run(argv, **kwargs):
        calls.append(argv)
        rank = 0 if argv[0] == "docker" else int(argv[0][-1])
        if "forward:5255->5255" in argv:
            return SimpleNamespace(returncode=255, stderr="Error: remote port forwarding failed for listen port 5255")
        present.add(rank)
        return SimpleNamespace(returncode=0, stderr="")
    current = assets.Assets(Transport(), tmp_path, run=run, popen=lambda *a, **k: pytest.fail("archive copy"))
    current.remote = probe(present)
    result = current.images(REGISTRY_CARD)
    assert sorted(result["relayed_ranks"]) == [0, 2] and result["reused_ranks"] == [1, 3]
    assert ["docker", "--context", "default", "pull", "-q", "--platform", "linux/arm64", relay.instances[0].reference()] in calls
    retried = [c for c in calls if c[0] == "node2"]
    assert len(retried) == 2 and retried[1][1] != "forward:5255->5255"
    assert retried[1][-1] == relay.instances[0].reference(int(retried[1][1].split(":")[1].split("-")[0]))
    assert retried[1][2:4] == ["sudo", "-n"] and relay.instances[0].closed


def test_relay_storage_shortfall_asks_for_input_before_downloading(tmp_path, relay):
    current = assets.Assets(Transport(), tmp_path, run=lambda *a, **k: pytest.fail("Unexpected download"))
    current.remote = probe({1, 2, 3}, free=40 * 1024**3)
    with pytest.raises(NeedsInput) as error:
        current.images(REGISTRY_CARD)
    assert error.value.field == "storage" and error.value.details["rank"] == 0
    assert not relay.instances


def test_unreachable_registry_falls_back_to_copying_between_nodes(tmp_path, relay):
    present, calls = {0}, []
    class Child:
        stdout = io.BytesIO(b"docker image bytes")
        returncode = 0
        def wait(self, **kw): return 0
        def poll(self): return 0
    def run(argv, **kwargs):
        calls.append(argv)
        if "pull" in argv:
            return SimpleNamespace(returncode=1, stderr="unauthorized: authentication required")
        present.add(int(argv[0][-1]))
        return SimpleNamespace(returncode=0, stdout=b'{"image_id":"sha256:a"}', stderr=b"")
    current = assets.Assets(Transport(), tmp_path, run=run, popen=lambda argv, **kwargs: calls.append(argv) or Child())
    current.remote = probe(present)
    result = current.images(REGISTRY_CARD)
    assert result["donor_rank"] == 0 and sorted(i["rank"] for i in result["imported"]) == [1, 2, 3]
    assert relay.instances[0].closed


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
