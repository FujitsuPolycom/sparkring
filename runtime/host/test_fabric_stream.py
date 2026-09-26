"""Direct fabric checkpoint streams over loopback, and the cable-ordered copy plan."""
import hashlib
import queue
import threading
from types import SimpleNamespace

import pytest

from runtime.host import fabric_stream, install_assets
from runtime.host.test_fabric_ssh import cluster

TOKEN = b"t" * 32


def test_copies_follow_cables_outward_from_the_donor():
    assert fabric_stream.tree(2, 1) == [[(1, 0)]]
    assert fabric_stream.tree(4, 0) == [[(0, 1), (0, 3)], [(1, 2)]]
    assert fabric_stream.tree(4, 2) == [[(2, 3), (2, 1)], [(3, 0)]]


@pytest.mark.parametrize("size", [2, 4])
def test_links_pair_each_shared_fabric_subnet(size):
    hosts = cluster(size)["plan"]["spec"]["hosts"]
    pairs = fabric_stream.links(hosts, 0, 1)
    assert len(pairs) == 2 and all(a != b for a, b in pairs)
    if size == 4:
        assert fabric_stream.links(hosts, 0, 2) == []


def test_balance_spreads_bytes_across_streams():
    sizes = {"a": 10, "b": 9, "c": 2, "d": 1}
    assert fabric_stream.balance(sizes, sizes, 2) == [["a", "d"], ["b", "c"]]


def tree_files(root, contents):
    files = {}
    for name, data in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = [len(data), hashlib.sha256(data).hexdigest()]
    return files


def run_stream(source_root, target_root, files, *, streams=2, token=TOKEN, groups=None):
    offers, result = queue.Queue(), {}
    addresses = ["127.0.0.1"] * streams

    def receiver():
        try:
            result["value"] = fabric_stream.receive(addresses, addresses, str(target_root), files,
                                                    token=TOKEN, announce=offers.put)
        except Exception as error:  # noqa: BLE001 - surfaced to the test thread
            result["error"] = error
            offers.put(None)

    thread = threading.Thread(target=receiver)
    thread.start()
    offer = offers.get(timeout=30)
    sent = None
    if offer and offer["needed"]:
        sizes = {name: value[0] for name, value in files.items()}
        plan = groups or fabric_stream.balance(offer["needed"], sizes, streams)
        try:
            sent = fabric_stream.send(addresses, addresses, offer["ports"], str(source_root), plan, token=token)
        except OSError:
            pass
    thread.join(timeout=30)
    return offer, sent, result


def test_stream_sends_only_missing_or_differing_files(tmp_path):
    contents = {"config.json": b"{}", "weights/model-00001.safetensors": b"x" * 300000,
                "weights/model-00002.safetensors": b"y" * 200000}
    files = tree_files(tmp_path / "source", contents)
    target = tmp_path / "target"
    tree_files(target, {"config.json": b"{}", "weights/model-00002.safetensors": b"z" * 200000})
    offer, sent, result = run_stream(tmp_path / "source", target, files)
    assert offer["needed"] == ["weights/model-00001.safetensors", "weights/model-00002.safetensors"]
    assert sent == {"sent": 500000} and result["value"] == {"received": 2, "bytes": 500000}
    for name, data in contents.items():
        assert (target / name).read_bytes() == data


def test_stream_rejects_a_sender_without_the_token(tmp_path):
    files = tree_files(tmp_path / "source", {"a.safetensors": b"a" * 1000})
    offer, _, result = run_stream(tmp_path / "source", tmp_path / "target", files, streams=1, token=b"x" * 32)
    assert "token" in str(result["error"])
    assert not (tmp_path / "target" / "a.safetensors").exists()


def test_stream_rejects_files_outside_the_plan(tmp_path):
    files = tree_files(tmp_path / "source", {"a.safetensors": b"a" * 1000})
    (tmp_path / "source" / "extra.py").write_bytes(b"print(1)")
    offer, _, result = run_stream(tmp_path / "source", tmp_path / "target", files, streams=1,
                                  groups=[["extra.py"]])
    assert "unplanned" in str(result["error"])
    assert not (tmp_path / "target" / "extra.py").exists()


class Transport:
    def __init__(self, size):
        self.hosts = cluster(size)["plan"]["spec"]["hosts"]
        self.mode = "fabric-control"

    def argv(self, rank):
        return ["ssh", "node" + str(rank)]

    def command(self, rank, args):
        return ["node" + str(rank), *args]


class Runner:
    def __init__(self, cached):
        self.cached, self.calls = cached, []

    def remote(self, rank, op, **kwargs):
        self.calls.append((rank, op))
        if op == "model-present":
            return {"present": rank in self.cached}
        if op == "model-transfer-manifest":
            return {"files": {"w.safetensors": "a" * 64}, "sizes": {"w.safetensors": 1}}
        return {"ok": True}


def test_checkpoint_copies_stream_along_the_ring_without_rsync(tmp_path):
    rows = [{"rank": n, "model": "/models/pinned"} for n in range(4)]
    current = install_assets.Assets(Transport(4), tmp_path, run=lambda *a, **k: pytest.fail("rsync"))
    edges = []
    current.stream_checkpoint = lambda runner, rows, manifest, source, target: edges.append((source, target)) or target
    report = current.models({"site": {"ranks": rows}}, Runner([2]))
    assert sorted(edges[:2]) == [(2, 1), (2, 3)] and edges[2] == (3, 0)
    assert report["donor_rank"] == 2 and report["copied_ranks"] == [0, 1, 3]


def test_failed_stream_leaves_remaining_ranks_to_rsync(tmp_path):
    rows = [{"rank": n, "model": "/models/pinned"} for n in range(4)]
    copies = []
    def run(argv, **kwargs):
        copies.append(argv[-1])
        return SimpleNamespace(returncode=0)
    current = install_assets.Assets(Transport(4), tmp_path, run=run)
    def stream(runner, rows, manifest, source, target):
        if target == 3:
            raise ValueError("connection refused")
        return target
    current.stream_checkpoint = stream
    runner = Runner([0])
    report = current.models({"site": {"ranks": rows}}, runner)
    assert report["copied_ranks"] == [1, 2, 3]
    rsynced = [rank for rank, op in runner.calls if op == "model-transfer-complete"]
    assert 3 in rsynced and 2 in rsynced and 1 not in rsynced
