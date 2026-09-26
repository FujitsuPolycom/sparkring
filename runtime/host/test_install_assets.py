"""Image distribution and checkpoint orchestration on the actual installer asset path."""
import concurrent.futures
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from runtime.host import checkpoint_plan, install_assets as assets
from runtime.host.install_errors import NeedsInput
from runtime.host.test_fabric_ssh import cluster


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


# Checkpoint orchestration ------------------------------------------------------------
#
# A small pinned checkpoint on two or four Sparks. Rank operations are simulated by
# the names each Spark holds; the fabric receiver, the sender and rsync are
# simulated at the process boundary, so the controller code of ``models``,
# ``stream_checkpoint`` and ``copy`` runs unchanged.

GIB = 1024 ** 3
REPO = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REV = "629bc3218833a38b475b719f34aa571666f4a03e"
W1, W2, W3 = (f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3))
SIZES = {"config.json": 1000, "tokenizer.json": 2000, W1: 3 * GIB, W2: 3 * GIB, W3: 3 * GIB}
REQUIRED = set(SIZES)
PINS = {"schema": "sparkring-checkpoint-pins/v1", "repository": REPO, "revision": REV,
        "index": "tokenizer.json", "weights": [W1, W2, W3], "optional": ["README.md"],
        "files": {**{name: {"size": size, "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "git_blob": hashlib.sha1(name.encode()).hexdigest()} for name, size in SIZES.items()},
                  "README.md": {"size": 10, "sha256": "0" * 64, "git_blob": "0" * 40}}}
OWNED = "/srv/sparkring/test/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/" + REV
POLICY = {"cache_bytes": 32 * GIB, "image_bytes": 68 * GIB}
NOW = datetime.datetime(2026, 9, 26, 2, 0, tzinfo=datetime.timezone.utc)
DEVICE, MOUNT = 66306, 29


def ranks(count, **changes):
    rows = [{"rank": rank, "host": f"root@198.51.100.{rank + 1}", "model": OWNED} for rank in range(count)]
    for rank, change in changes.items():
        rows[int(rank.removeprefix("r"))].update(change)
    return rows


def lock(rows):
    return {"id": "qwen-test", "selection": {"model_repository": REPO, "model_revision": REV}, "site": {"ranks": rows}}


def candidate(path, names, *, state="match", mount_id=MOUNT, device=DEVICE):
    files = {name: {"state": state, "evidence": "hashed", "size": SIZES[name], "source": f"{path}/{name}",
                    "kind": "file", "owner": "code", "mode": 0o644, "mount_id": mount_id,
                    "identity": [device, int(hashlib.sha256(f"{path}/{name}".encode()).hexdigest()[:8], 16)]}
             for name in names}
    return {"path": path, "layout": "plain", "found_by": ["folder"], "commit": None, "branches": [], "home": None,
            "sparkring": False, "mount_id": mount_id, "device": device, "rotational": False, "files": files,
            "counts": {"match": len(names)}}


def survey(rank, *candidates):
    return {"schema": "sparkring-checkpoint-survey/v1", "host": f"spark-{rank:04x}", "repository": REPO,
            "revision": REV, "operator": "code", "docker": {"userns": False, "driver": "overlay2"},
            "owned": {"path": OWNED, "probe_path": "/srv/sparkring/test", "mount_point": "/", "mount_id": MOUNT,
                      "device": DEVICE, "fstype": "ext4", "free_bytes": 10 ** 13, "files": {}},
            "search": {"complete": True, "passes": 1, "seconds": 1.0, "entries": 100, "unvisited": {},
                       "skipped_mounts": []},
            "candidates": list(candidates), "not_used": [], "named": []}


def approve(surveys, rows):
    plan = checkpoint_plan.plan(PINS, surveys, rows, policy=POLICY, now=NOW)
    assert plan["problems"] == []
    return {**plan, "approval": "command-line"}


FOLDER = "/var/tmp/models/qwen"


class Spark:
    """Rank operations of every Spark, simulated by the set of pinned names each holds.

    ``adopt`` maps a rank to a function of the ``model-adopt`` input returning
    ``(held names, extra result fields)``; by default a Spark holds every name its
    plan entry lists. Transfers complete when ``model-transfer-complete`` runs.
    """

    def __init__(self, count, adopt=None, barrier=None):
        self.held = {rank: set() for rank in range(count)}
        self.adopt, self.barrier = adopt or {}, barrier
        self.events, self.pending, self.guard = [], {}, threading.Lock()

    def event(self, *item):
        with self.guard:
            self.events.append(item)

    def remote(self, rank, op, data=None):
        payload = json.loads(data) if data else None
        self.event(op, rank, payload)
        if op == "model-reuse-receipt":
            return {"reused": False}
        if op == "model-adopt":
            if self.barrier:
                self.barrier.wait()
            if rank in self.adopt:
                held, extra = self.adopt[rank](payload)
            else:
                held, extra = set(payload["files"]), {}
            self.held[rank] = set(held)
            return {"complete": self.held[rank] == REQUIRED, "verified": {n: [1, 2, 3, 4, 5] for n in held},
                    "missing": sorted(REQUIRED - self.held[rank]), "differs": [],
                    "linked": sum(1 for e in payload["files"].values() if e["action"] == "link"),
                    "copied": sum(1 for e in payload["files"].values() if e["action"] == "copy"),
                    "bytes_written": sum(e["size"] for e in payload["files"].values() if e["action"] == "copy"),
                    "refreshed": list(payload["receipts"]), **extra}
        if op == "model-transfer-prepare":
            self.pending[rank] = sorted(set(payload["files"]) - self.held[rank])
            return {"ok": True, "needed": self.pending[rank]}
        if op == "model-transfer-complete":
            self.held[rank] |= set(payload["files"])
            return {"ok": True, "complete": self.held[rank] == REQUIRED, "missing": sorted(REQUIRED - self.held[rank])}
        if op == "model-fetch":
            assert rank == 0
            self.held[0] |= set(payload["names"])
            return {"complete": self.held[0] == REQUIRED, "placed": payload["names"]}
        raise AssertionError("unexpected rank operation " + op)


class Fabric:
    """Process boundary of transfers: the fabric receiver, its sender and rsync."""

    def __init__(self, spark, count):
        self.spark, self.count = spark, count
        self.transport = FabricTransport(count)

    def popen(self, argv, **kwargs):
        target = int(argv[0].removeprefix("node"))
        assert "receive_checkpoint(" in argv[-1] and "def place_staged(" in argv[-1]
        needed = self.spark.pending[target]
        self.spark.event("receive", target, needed)

        class Token:
            def write(self, data):
                assert len(data) == 32

            def close(self):
                pass

        class Receiver:
            stdin = Token()
            stdout = io.BytesIO((json.dumps({"ports": [4000, 4001], "needed": needed}) + "\n"
                                 + json.dumps({"received": len(needed)}) + "\n").encode())

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0
        return Receiver()

    def run(self, argv, **kwargs):
        if argv[0] == "rsync":
            # rsync reads the names to send from stdin.
            assert "--files-from=-" in argv
            self.spark.event("rsync", argv[-2], argv[-1], kwargs["input"].decode().split())
        else:
            self.spark.event("send", int(argv[0].removeprefix("node")), argv[-1])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def assets(self, directory):
        return assets.Assets(self.transport, directory, run=self.run, popen=self.popen)


class FabricTransport(Transport):
    def __init__(self, count):
        self.hosts = cluster(count)["plan"]["spec"]["hosts"]

    def argv(self, rank):
        return ["ssh", "-F", "/head/ssh_config", "spark" + str(rank)]


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(assets.installer, "checkpoint_pins", lambda card: PINS)


@pytest.fixture
def steps(monkeypatch):
    labels = []

    @contextlib.contextmanager
    def step(message):
        labels.append(message)
        yield {"failed": False}
    monkeypatch.setattr(assets.progress, "step", step)
    return labels


def transfers(spark):
    return [event for event in spark.events if event[0] in ("receive", "send", "rsync", "model-fetch",
                                                              "model-transfer-prepare")]


def test_models_adopts_every_rank_then_distributes_from_a_complete_donor(tmp_path, pinned, steps):
    rows = ranks(4)
    plan = approve([survey(0), survey(1), survey(2, candidate(FOLDER, SIZES)), survey(3)], rows)
    assert plan["distribution"]["donor"] == 2 and plan["hub_files"] == []
    spark = Spark(4, barrier=threading.Barrier(4, timeout=5))
    fabric = Fabric(spark, 4)
    receipts = ["/srv/sparkring/test/qwen-retained/installer/model.json"]

    result = fabric.assets(tmp_path).models(lock(rows), spark, plan=plan, receipts=receipts)

    adopted = [event for event in spark.events if event[0] == "model-adopt"]
    assert sorted(rank for _, rank, _ in adopted) == [0, 1, 2, 3]
    assert {rank: payload for _, rank, payload in adopted} == {
        rank: checkpoint_plan.adoption(plan, rank, receipts) for rank in range(4)}
    first_transfer = spark.events.index(transfers(spark)[0])
    assert all(spark.events.index(event) < first_transfer for event in adopted)
    receives = [(event[1], event[2]) for event in spark.events if event[0] == "receive"]
    assert sorted(receives[:2]) == [(1, sorted(SIZES)), (3, sorted(SIZES))] and receives[2] == (0, sorted(SIZES))
    senders = [event[1] for event in spark.events if event[0] == "send"]
    assert sorted(senders[:2]) == [2, 2] and senders[2] == 3
    assert not any(event[0] in ("rsync", "model-fetch") for event in spark.events)
    assert all(held == REQUIRED for held in spark.held.values())
    assert result["donor_rank"] == 2 and result["downloaded"] == [] and result["pooled"] == []
    assert [(item["source"], item["target"], item["transport"]) for item in result["received"]] == [
        (2, 1, "fabric"), (2, 3, "fabric"), (3, 0, "fabric")]
    assert result["adopted"][2]["refreshed"] == receipts
    assert json.loads((tmp_path / "checkpoint-result.json").read_text()) == result


def pooling_plan(rows, *, other_disk=False):
    small = candidate("/data/tokenizers", ["config.json", "tokenizer.json"],
                      **({"mount_id": 31, "device": 66310} if other_disk else {}))
    return approve([survey(0), survey(1, candidate(FOLDER, [W1])), survey(2, small), survey(3, candidate(FOLDER, [W2]))],
                   rows)


def test_node_a_pools_from_adjacent_peers_by_fabric_and_others_by_rsync(tmp_path, pinned, steps):
    rows = ranks(4)
    plan = pooling_plan(rows)
    assert plan["hub_files"] == [W3]
    spark = Spark(4)

    result = Fabric(spark, 4).assets(tmp_path).models(lock(rows), spark, plan=plan)

    moves = [event for event in transfers(spark) if event[0] != "model-transfer-prepare"]
    assert moves[:5] == [("receive", 0, [W1]), ("send", 1, moves[1][2]), ("receive", 0, [W2]),
                         ("send", 3, moves[3][2]),
                         ("rsync", "spark2:" + OWNED + "/",
                          "/srv/sparkring/test/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/." + REV
                          + ".sparkring/receive/rsync/", ["config.json", "tokenizer.json"])]
    assert moves[5] == ("model-fetch", 0, {"names": [W3]})
    # Node A is then the donor: ranks 1 and 3 by the cables, rank 2 from rank 1.
    receives = [(event[1], event[2]) for event in moves[6:] if event[0] == "receive"]
    assert sorted(receives[:2]) == [(1, sorted(REQUIRED - {W1})), (3, sorted(REQUIRED - {W2}))]
    assert receives[2] == (2, sorted(REQUIRED - {"config.json", "tokenizer.json"}))
    assert result["downloaded"] == [W3] and result["donor_rank"] == 0
    assert [(item["source"], item["transport"]) for item in result["pooled"]] == [(1, "fabric"), (3, "fabric"),
                                                                                  (2, "rsync")]
    assert all(held == REQUIRED for held in spark.held.values())


def differing_weights(payload):
    return set(payload["files"]) - {W1, W2}, {"differs": [{"name": W1, "source": f"{FOLDER}/{W1}"},
                                                          {"name": W2, "source": f"{FOLDER}/{W2}"}]}


def test_unplanned_download_raises_needs_input_before_any_fetch(tmp_path, pinned, steps):
    rows = ranks(2)
    plan = approve([survey(0, candidate(FOLDER, SIZES)), survey(1, candidate(FOLDER, SIZES))], rows)
    assert plan["hub_files"] == []
    spark = Spark(2, adopt={0: differing_weights, 1: differing_weights})
    fabric = Fabric(spark, 2)

    with pytest.raises(NeedsInput) as error:
        fabric.assets(tmp_path).models(lock(rows), spark, plan=plan)

    assert error.value.field == "checkpoint"
    message = str(error.value)
    assert f"2 files in {FOLDER} are not the pinned model's ({W1}, {W2})" in message
    assert "6.00 GiB from huggingface.co, which the approved plan did not include" in message
    assert message.endswith("Nothing was downloaded and the running model was not changed. Review the resulting "
                            "plan with sudo sparkring install --plan.")
    assert error.value.details["items"][0]["names"] == [W1, W2]
    assert transfers(spark) == [] and [e[0] for e in spark.events] == ["model-adopt", "model-adopt"]
    assert not (tmp_path / "checkpoint-result.json").exists()


def test_local_writes_beyond_the_plan_raise_needs_input(tmp_path, pinned, steps):
    rows = ranks(2)
    plan = approve([survey(0, candidate(FOLDER, SIZES)), survey(1, candidate(FOLDER, SIZES))], rows)

    def cross_mount(payload):
        weights = sorted(name for name in payload["files"] if name.endswith(".safetensors"))
        return set(payload["files"]) - set(weights), {"missing": [
            {"name": name, "reason": "EXDEV", "source": f"{FOLDER}/{name}"} for name in weights]}
    spark = Spark(2, adopt={1: cross_mount})

    with pytest.raises(NeedsInput) as error:
        Fabric(spark, 2).assets(tmp_path).models(lock(rows), spark, plan=plan)

    assert error.value.field == "checkpoint"
    assert str(error.value) == (
        f"Node 1 spark-0001: hard links from {FOLDER} into {OWNED} failed (they are different mounts of one "
        "filesystem), so 9.0 GiB must be received over the fabric instead, which the approved plan did not "
        "include. Nothing more was written; the running model was not changed. Review the resulting plan with "
        "sudo sparkring install --plan.")
    assert transfers(spark) == []


def test_checkpoint_steps_have_progress_labels(tmp_path, pinned, steps):
    rows = ranks(4)
    spark = Spark(4)
    Fabric(spark, 4).assets(tmp_path).models(lock(rows), spark, plan=pooling_plan(rows, other_disk=True))
    # Node 0 holds no file of its own: its adoption only prepares SparkRing's directory.
    assert sorted(steps[:4]) == ["Node 0: Prepare SparkRing's checkpoint directory",
                                 "Node 1: Link and verify local checkpoint files",
                                 "Node 2: Copy checkpoint files from other disks",
                                 "Node 3: Link and verify local checkpoint files"]
    assert steps[4:8] == [
        "Node 0: Copy checkpoint files from Node 1 over the fabric",
        "Node 0: Copy checkpoint files from Node 3 over the fabric",
        "Node 0: Copy checkpoint files from Node 2 over fiber-ssh",
        "Node 0: Download missing checkpoint files from huggingface.co"]
    assert steps[10:] == ["Node 2: Copy checkpoint files from Node 1 over the fabric"]
    # The two receives of one fabric level run in parallel, in either order.
    assert sorted(steps[8:10]) == ["Node 1: Copy checkpoint files from Node 0 over the fabric",
                                   "Node 3: Copy checkpoint files from Node 0 over the fabric"]


def test_a_failed_rank_operation_names_the_spark_and_its_cause(tmp_path, pinned, steps):
    rows = ranks(2)
    plan = approve([survey(0), survey(1)], rows)
    assert plan["hub_files"] == sorted(REQUIRED)
    spark = Spark(2)
    remote = spark.remote

    def failing(rank, op, data=None):
        if op == "model-fetch":
            # A rank operation's failure arrives as the remote traceback on the SSH error output.
            raise RuntimeError("root@198.51.100.1: Traceback (most recent call last):\n"
                               '  File "<string>", line 9, in <module>\n'
                               "ValueError: Downloading 5 files from huggingface.co failed: 503 Service Unavailable. "
                               "Nothing was placed for those files.")
        return remote(rank, op, data)
    spark.remote = failing
    with pytest.raises(RuntimeError) as error:
        Fabric(spark, 2).assets(tmp_path).models(lock(rows), spark, plan=plan)
    # One line that names the Spark and the cause, without the traceback or the exception's name.
    assert str(error.value) == ("Node 0: Downloading 5 files from huggingface.co failed: 503 Service Unavailable. "
                                "Nothing was placed for those files.")
    assert assets.cause(RuntimeError("scripts.installer_host.CommandError: docker run exited with status 1: x")) == \
        "docker run exited with status 1: x"
    assert assets.cause(ValueError("Node 1: already short")) == "Node 1: already short"


def test_rows_served_in_place_are_not_adopted_and_can_donate(tmp_path, pinned, steps):
    usb = "/mnt/usb/qwen"
    rows = ranks(2, r0={"model": usb, "reuse_verified_model": True})
    exact = {**candidate(usb, SIZES, mount_id=40, device=2049), "exact": True}
    plan = approve([survey(0, exact), survey(1)], rows)
    assert plan["nodes"][0]["mode"] == "in-place"
    spark = Spark(2)

    result = Fabric(spark, 2).assets(tmp_path).models(lock(rows), spark, plan=plan)

    assert [(e[0], e[1]) for e in spark.events if e[0] == "model-adopt"] == [("model-adopt", 1)]
    sender = next(e for e in spark.events if e[0] == "send")
    assert sender[1] == 0 and repr(usb) in sender[2]
    assert result["in_place_ranks"] == [0] and spark.held[1] == REQUIRED


def test_plans_for_other_sparks_paths_or_pins_are_refused(tmp_path, pinned, steps):
    rows = ranks(2)
    plan = approve([survey(0, candidate(FOLDER, SIZES)), survey(1)], rows)
    spark = Spark(2)
    current = Fabric(spark, 2).assets(tmp_path)
    moved = ranks(2, r1={"model": "/srv/sparkring/other/checkpoints/x/" + REV})
    with pytest.raises(ValueError, match="other Sparks, checkpoint paths or modes"):
        current.models(lock(moved), spark, plan=plan)
    with pytest.raises(ValueError, match="another checkpoint revision or pin manifest"):
        current.models(lock(rows), spark, plan={**plan, "pins_sha256": "0" * 64})
    problem = {"field": "storage", "rank": 1, "message": "Node 1 spark-0001 needs 134.9 GiB free on /"}
    with pytest.raises(NeedsInput) as error:
        current.models(lock(rows), spark, plan={**plan, "problems": [problem]})
    assert error.value.field == "storage" and error.value.details == {"problems": [problem]}
    assert spark.events == []


def test_without_a_plan_nothing_is_downloaded(tmp_path, pinned, steps):
    rows = ranks(2)
    spark = Spark(2)
    with pytest.raises(NeedsInput) as error:
        Fabric(spark, 2).assets(tmp_path).models(lock(rows), spark)
    assert error.value.field == "checkpoint" and "which the approved plan did not include" in str(error.value)
    assert [e[2]["files"] for e in spark.events if e[0] == "model-adopt"] == [{}, {}]
    assert transfers(spark) == []


def test_a_target_left_incomplete_stops_the_install(tmp_path, pinned, steps):
    rows = ranks(2)
    plan = approve([survey(0, candidate(FOLDER, SIZES)), survey(1)], rows)
    spark = Spark(2)
    remote = spark.remote

    def lossy(rank, op, data=None):
        result = remote(rank, op, data)
        if op == "model-transfer-complete":
            spark.held[rank].discard(W3)
            return {"ok": True, "complete": False, "missing": [W3]}
        return result
    spark.remote = lossy
    with pytest.raises(ValueError, match=f"Node 1: the checkpoint is incomplete after receiving from Node 0 \\(missing {W3}\\)"):
        Fabric(spark, 2).assets(tmp_path).models(lock(rows), spark, plan=plan)


# The prepared runner ---------------------------------------------------------------------

def prepared(monkeypatch, tmp_path, rows, images, spark, *, plan=None, models=None):
    """The real PreparedRunner, with its lock supplied directly and rank operations simulated by ``spark``."""
    from scripts import installer_runner

    def init(self, directory):
        self.directory, self.lock = Path(directory), lock(rows)
    monkeypatch.setattr(installer_runner.Runner, "__init__", init)
    monkeypatch.setattr(installer_runner.Runner, "remote", lambda self, rank, op, data=None: spark.remote(rank, op, data))
    monkeypatch.setattr(installer_runner.Runner, "_call", lambda self, target, argv, timeout: {
        "returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False})
    current = Fabric(spark, len(rows)).assets(tmp_path / "assets")
    if models:
        current.models = models
    return current.runner(tmp_path, None, images, plan=plan, receipts=["/srv/sparkring/test/qwen-retained/installer/model.json"])


def test_needs_input_keeps_its_type_through_the_real_prepared_runner(tmp_path, monkeypatch, pinned, steps):
    from scripts import deploy_engine
    rows = ranks(2)
    plan = approve([survey(0, candidate(FOLDER, SIZES)), survey(1, candidate(FOLDER, SIZES))], rows)
    spark = Spark(2, adopt={0: differing_weights, 1: differing_weights})
    images = concurrent.futures.Future()
    images.set_result({"image_id": "sha256:a"})
    runner = prepared(monkeypatch, tmp_path, rows, images, spark, plan=plan)
    phase = {"id": "model", "actions": [
        {"host": row["host"], "argv": ["installer", "model", str(row["rank"])], "risk": "mutates-host",
         "timeout": 60, "verify": {"argv": ["installer", "model-check", str(row["rank"])], "stdout": "ok"}}
        for row in rows]}
    sealed = deploy_engine.seal_plan({"schema": "sparkring-deploy-plan/v1", "deployment": "qwen-test",
                                      "operation": "prepare", "phases": [phase]})

    with pytest.raises(RuntimeError, match="model: phase failed"):
        deploy_engine.execute_plan(sealed, tmp_path / "receipt.json", sealed["sha256"], runner=runner,
                                   allow_model_actions=True)

    assert isinstance(runner.needs_input, NeedsInput) and runner.needs_input.field == "checkpoint"
    assert "not the pinned model's" in str(runner.needs_input)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    # Each rank's action reports only that the plan needs a decision; the request is printed once, at the end.
    assert {action["result"]["stderr"] for action in receipt["actions"].values()} == {assets.STOPPED}
    # Checkpoint preparation ran once for both ranks, with the approved plan.
    assert [e[1] for e in spark.events if e[0] == "model-adopt"] in ([0, 1], [1, 0])
    assert runner.checkpoint is None


def test_checkpoint_work_waits_for_image_distribution(tmp_path, monkeypatch, pinned, steps):
    rows = ranks(2)
    started = threading.Event()

    def models(lock, runner, previous=None, **options):
        started.set()
        if not images.done():
            raise AssertionError("checkpoint work started before image distribution finished")
        return {"donor_rank": 0, "options": sorted(options)}
    images = concurrent.futures.Future()
    runner = prepared(monkeypatch, tmp_path, rows, images, Spark(2), models=models)
    outcome = {}
    worker = threading.Thread(target=lambda: outcome.update(result=runner._call(rows[0]["host"], ["installer", "model", "0"], 60)))
    worker.start()
    assert not started.wait(1), "checkpoint work started before image distribution finished"
    images.set_result({"image_id": "sha256:a"})
    worker.join(10)
    assert outcome["result"]["returncode"] == 0 and started.is_set()
    assert runner.checkpoint == {"donor_rank": 0, "options": ["receipts"]}

    failing = concurrent.futures.Future()
    failing.set_exception(NeedsInput("Node 1: insufficient image-import space.", field="storage"))
    started.clear()
    runner = prepared(monkeypatch, tmp_path, rows, failing, Spark(2), models=models)
    result = runner._call(rows[1]["host"], ["installer", "model", "1"], 60)
    assert result["returncode"] == 1 and result["stderr"].startswith("Image distribution failed")
    assert not started.is_set() and runner.needs_input is None
