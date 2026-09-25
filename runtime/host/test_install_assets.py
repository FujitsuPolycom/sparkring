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
    repository = "owner/image"
    instances = []
    layout = None
    def __init__(self, reference, directory):
        self.closed = False
        self.fetched = []
        self.directory = directory
        Relay.instances.append(self)
    def reference(self, port=None):
        return f"127.0.0.1:{port or self.port}/owner/image@sha256:" + "b" * 64
    def image(self, image_id):
        if Relay.layout is None:
            raise ValueError("fixture registry publishes no layer list")
        return Relay.layout
    def blob(self, digest):
        self.fetched.append(digest)
        path = self.directory / digest.removeprefix("sha256:")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"compressed " + digest.encode())
        return path
    def close(self):
        self.closed = True


@pytest.fixture
def relay(monkeypatch):
    Relay.instances = []
    Relay.layout = None
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


LAYERS = [("sha256:" + c * 64, "sha256:" + d * 64, 100) for c, d in (("1", "4"), ("2", "5"), ("3", "6"))]
CONFIG = b'{"rootfs": {"diff_ids": ["sha256:1..", "sha256:2..", "sha256:3.."]}}'


class Sink(io.BytesIO):
    def close(self):
        self.data = self.getvalue()
        super().close()


def test_nodes_holding_leading_layers_load_only_the_layers_they_lack(tmp_path, relay):
    import json
    import tarfile
    relay.layout = (CONFIG, LAYERS)
    card = {**REGISTRY_CARD, "release": "dev-20260925-cuda1342-nccl2323-status031"}
    held = {0: 0, 2: 2, 3: 3}
    present, pulls, loads = {1}, [], {}
    def run(argv, **kwargs):
        pulls.append(argv)
        present.add(0 if argv[0] == "docker" else int(argv[0][-1]))
        return SimpleNamespace(returncode=0, stderr="")
    class Loader:
        def __init__(self, argv, **kwargs):
            self.rank, self.argv = int(argv[0][-1]), argv
            self.stdin, self.stdout = Sink(), io.BytesIO(b'{"image_id": "sha256:a"}')
            loads[self.rank] = self
        def wait(self, timeout=None):
            present.add(self.rank)
            return 0
    def remote(rank, fn, *args, **kwargs):
        if fn == assets.layer_prefix:
            assert args == ([diff for diff, _, _ in LAYERS],)
            return held[rank]
        assert fn == assets.image_probe
        return {"present": rank in present, "image_id": "sha256:a", "size_bytes": 30 * 1024**3, "free_bytes": 200 * 1024**3}
    current = assets.Assets(Transport(), tmp_path, run=run, popen=Loader)
    current.remote = remote
    result = current.images(card)
    assert sorted(result["relayed_ranks"]) == [0, 2, 3]
    # Node 0 holds no leading layer and pulls; nodes 2 and 3 load only what they lack.
    assert [argv[-1] for argv in pulls] == [relay.instances[0].reference()]
    assert sorted(loads) == [2, 3] and relay.instances[0].fetched == [LAYERS[2][1]]
    for rank, expected in ((2, ["3" * 64 + "/layer.tar"]), (3, [])):
        assert loads[rank].argv[1:5] == ["sudo", "-n", "python3", "-I"]
        assert "receive_image(*('sha256:a'," in loads[rank].argv[-1]
        with tarfile.open(fileobj=io.BytesIO(loads[rank].stdin.data)) as archive:
            names = archive.getnames()
            manifest = json.load(archive.extractfile("manifest.json"))
            assert archive.extractfile("a.json").read() == CONFIG
        assert sorted(names) == sorted(["a.json", "manifest.json", *expected])
        assert manifest == [{"Config": "a.json", "Layers": [d[7:] + "/layer.tar" for d, _, _ in LAYERS],
                             "RepoTags": ["127.0.0.1:5255/owner/image:dev-20260925-cuda1342-nccl2323-status031"]}]
    assert relay.instances[0].closed


def test_layer_prefix_counts_the_longest_locally_held_chain(monkeypatch):
    import json
    import subprocess
    images = {"sha256:x": ["sha256:1", "sha256:2", "sha256:9"], "sha256:y": ["sha256:1"]}
    def check_output(argv, text=True):
        if argv[3:4] == ["info"]:
            return '[["Backing Filesystem","extfs"]]'
        if argv[3:5] == ["image", "ls"]:
            return "\n".join(images) + "\n"
        return json.dumps([{"RootFS": {"Layers": images[name]}} for name in argv[5:]])
    monkeypatch.setattr(subprocess, "check_output", check_output)
    assert assets.layer_prefix(["sha256:1", "sha256:2", "sha256:3"]) == 2
    assert assets.layer_prefix(["sha256:7"]) == 0
    monkeypatch.setattr(subprocess, "check_output",
                        lambda argv, text=True: '[["driver-type","io.containerd.snapshotter.v1"]]')
    assert assets.layer_prefix(["sha256:1"]) == 0


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
