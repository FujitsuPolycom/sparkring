"""`sparkring storage`: the per-Spark storage report and the release of unreferenced SparkRing data.

Tests marked ``linux`` build a ``/srv/sparkring`` tree below ``tmp_path`` with
real hard links and SparkRing checkpoint directories placed by
``runtime/host/checkpoint_place.py`` under a synthetic ext4 mount table, with
installed meshes' configurations in ``etc/sparkring`` below ``tmp_path``, and
answer Node A's per-Spark calls with this module's own node functions and a
canned Docker. The others drive Node A's command with canned per-Spark answers.
"""
import io
import json
import os
from pathlib import Path
import posixpath
import re
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from runtime.common import installer, installer_image, profiles, qwen_flash_next
from runtime.host import checkpoint_place as place
from runtime.host import checkpoints, storage
from runtime.host.test_checkpoints import REVISION, SLUG, WEIGHT, adopt, tree_state, user_copy, write_json

CLUSTER = "tp4"
HOSTS = ["root@192.0.2.10", "root@192.0.2.11"]
LIST = ["sudo", "-n", "/usr/bin/sparkring", "node", "storage"]
DEFAULT_IMAGE = installer_image.default_lock()["image_id"]
# The installed deployment runs an explicit development lock's image; its caches are found from its containers.
DEV_IMAGE = "sha256:" + "d" * 64
# The image of the dev-20260925-qwendecode installer image lock, which no profile selects by default.
OLD_IMAGE = "sha256:4100e1d2bd038f885d92f8c0021d482b23f9a38a003cd7bfc3e700e7e0afa971"
OLD_LOCK = "dev-20260925-qwendecode-cuda1342-nccl2323-status031"
OTHER_REVISION = "60215d26cf5e42c2db6128774032d57fc62678da"
INSTALLED_CACHE = f"qwen-flash-next-{DEV_IMAGE[7:19]}-{REVISION[:12]}"
COMPILE_CACHE = f"qwen-flash-next-cuda{installer_image.CUDA_VERSION}-{REVISION[:12]}"
PROFILE_CACHE = f"qwen-flash-next-{DEFAULT_IMAGE[7:19]}-{OTHER_REVISION[:12]}"
STALE_CACHE = "glm53-flash-nvfp4-spark-4100e1d2bd03-a608241037e4"
LEFTOVER_CACHE = f"qwen-flash-next-5ce6ce267d80-{REVISION[:12]}"
INSTALLED = "qwen38-flash-next-tp2-iaaaaaaaaaaaa"
STALE = "qwen38-flash-next-tp2-ibbbbbbbbbbbb"
WITH_MODELS = "qwen38-flash-next-tp2-idddddddddddd"
QWEN = ["qwen38-flash-next-qad-tp4", "qwen38-flash-next-tp2"]
# Deployments of a four-Spark ring in installation order: the first created the ring's mesh and the others reused it.
FIRST = "qwen38-flash-next-qad-tp4-i111111111111"
SECOND = "qwen38-flash-next-qad-tp4-i222222222222"
THIRD = "qwen38-flash-next-qad-tp4-i333333333333"
# The site name of FIRST's mesh, which controller.model_site derives from the cluster, profile and instance.
MESH = "tp4-qwen38-flash-next--1a2b3c"
# The mesh's host marker binary and bundle root in the workspace of the deployment that created the mesh.
MARKER = "mesh/artifacts/mlx5-rdma-tx-marker"
BUNDLE = "mesh/artifacts"
linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="the fake Spark uses Linux hard links, /proc/self/fd, O_NOFOLLOW and flock")


@pytest.fixture(autouse=True)
def administrator(monkeypatch):
    """Node A's command runs under sudo; the tests run it as the invoking account."""
    monkeypatch.setattr(storage, "_administrator", lambda: True)


@pytest.fixture
def host_records(tmp_path, monkeypatch):
    """Synthetic ext4 mount table and private record directories for real placements."""
    table = tmp_path / "mountinfo"
    table.write_text("29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
    monkeypatch.setattr(place, "MOUNTINFO", str(table))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records/files"))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS", str(tmp_path / "records"))
    (tmp_path / "records").mkdir()


class Docker:
    """Canned ``docker --context default`` for one host: images, and containers with their mounts."""

    def __init__(self, images=(), containers=(), root="/var/lib/docker"):
        self.images, self.containers, self.root = list(images), list(containers), root

    def __call__(self, argv, **kwargs):
        assert argv[:3] == ["docker", "--context", "default"]
        command = argv[3:]
        if command[0] == "info":
            output = self.root + "\n"
        elif command[:2] == ["image", "ls"]:
            output = "".join(image["Id"] + "\n" for image in self.images)
        elif command[:2] == ["image", "inspect"]:
            output = json.dumps([image for image in self.images if image["Id"] in command[2:]])
        elif command[0] == "ps":
            output = "".join(item["Id"] + "\n" for item in self.containers
                             if "--all" in command or item["State"]["Running"])
        else:
            assert command[:2] == ["container", "inspect"]
            output = json.dumps([item for item in self.containers if item["Id"] in command[2:]])
        return SimpleNamespace(returncode=0, stdout=output, stderr="")


def container(name, image=DEV_IMAGE, *, running=True, mounts=(), env=()):
    return {"Id": name + "0" * 8, "Name": "/" + name, "Image": image, "State": {"Running": running},
            "Mounts": [{"Source": source, "Destination": destination} for source, destination in mounts],
            "Config": {"Env": list(env), "Cmd": [], "Entrypoint": []}, "Args": [], "Path": ""}


def image(identifier, tags=(), size=10 ** 9):
    return {"Id": identifier, "RepoTags": list(tags), "RepoDigests": [], "Size": size}


def put(path, size=0, content=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else b"x" * size)
    return path


def workspace(path, deployment):
    """A deployment workspace as the installer's source step leaves it."""
    put(path / ".installer-owner.json", content=json.dumps({"deployment": deployment}).encode())
    put(path / "source-111111111111/README.md", 3000)
    put(path / "installer/model.json", content=b"{}")
    return path


def du(path, *, outside=()):
    """Allocated bytes of the distinct inodes below ``path``, and those whose inode is not in ``outside``."""
    seen, total, frees = set(), 0, 0
    shared = {os.lstat(item).st_ino for item in outside}
    for directory, _, files in os.walk(path):
        size = os.lstat(directory).st_blocks * 512
        total, frees = total + size, frees + size
        for name in files:
            info = os.lstat(os.path.join(directory, name))
            if info.st_ino not in seen:
                seen.add(info.st_ino)
                total += info.st_blocks * 512
                frees += 0 if info.st_ino in shared else info.st_blocks * 512
    return total, frees


@pytest.fixture
def spark(tmp_path, host_records):
    """One Spark's /srv/sparkring: the cluster's checkpoint, caches and workspaces, and directories SparkRing did not create."""
    srv = tmp_path / "srv/sparkring"
    cluster = srv / CLUSTER
    folder = user_copy(srv / "image020/models/qwen")
    owned = adopt(cluster / "checkpoints" / SLUG / REVISION, folder)
    cache = cluster / "cache"
    put(cache / INSTALLED_CACHE / "triton/kernel.bin", 12000)
    put(cache / COMPILE_CACHE / "b12x/kernel.bin", 9000)
    put(cache / PROFILE_CACHE / "vllm/graph.bin", 7000)
    graph = put(cache / STALE_CACHE / "vllm/graph.bin", 20000)
    os.link(graph, srv / "image020/graph-copy.bin")
    put(cache / STALE_CACHE / "inductor/unique.bin", 5000)
    put(cache / "jit/rank0/state.json", 100)
    put(cache / f".{LEFTOVER_CACHE}.sparkring-releasing/triton/kernel.bin", 4000)
    workspace(cluster / INSTALLED, "id-" + INSTALLED)
    stale = workspace(cluster / STALE, "id-" + STALE)
    os.link(put(stale / "containers/a.bin", 30000), stale / "containers/b.bin")
    inner = adopt(workspace(cluster / WITH_MODELS, "id-" + WITH_MODELS) / "models" / REVISION, folder)
    put(cluster / "notes/todo.txt", 500)
    workspace(srv / "tp2-prefresh-20260925/qwen38-flash-next-tp2-icccccccccccc", "id-old")
    docker = Docker(images=[image(DEV_IMAGE, ["ghcr.io/fujitsupolycom/sparkring-installer:dev"], 29 * 10 ** 9),
                            image(DEFAULT_IMAGE, ["ghcr.io/fujitsupolycom/sparkring-installer:shared"], 30 * 10 ** 9),
                            image(OLD_IMAGE, [], 28 * 10 ** 9), image("sha256:" + "e" * 64, [], 10 ** 9)],
                    containers=[container("sr-qwen-r0", mounts=[(str(cache), "/cache"), (owned, "/models/target")],
                                          env=[f"XDG_CACHE_HOME=/cache/{INSTALLED_CACHE}",
                                               f"B12X_COMPILE_CACHE_DIR=/cache/{COMPILE_CACHE}/b12x"]),
                                container("scratch", "sha256:" + "e" * 64, running=False)])
    return SimpleNamespace(root=str(tmp_path), empty=str(tmp_path / "empty"), srv=srv, cluster=cluster, cache=cache,
                           owned=owned, inner=inner, folder=folder, docker=docker)


def spec(workspace_path, model, cache, names):
    """``rank<N>/container.json`` of an installer deployment that uses cache directories ``names``."""
    return {"name": "sr-qwen-r0", "image_id": DEV_IMAGE, "entrypoint": ["python3"],
            "command": ["serve", "/models/target", "--speculative-config", json.dumps({"model": "/models/target"})],
            "environment": {"XDG_CACHE_HOME": f"/cache/{names[0]}", "TRITON_CACHE_DIR": f"/cache/{names[0]}/triton",
                            "B12X_COMPILE_CACHE_DIR": f"/cache/{names[1]}/b12x",
                            "SPARKRING_RUNTIME_BINDING": "/run/sparkring/runtime-binding.json"},
            "mounts": [{"source": model, "target": "/models/target", "read_only": True},
                       {"source": cache, "target": "/cache", "read_only": False},
                       {"source": f"{workspace_path}/containers/lock/runtime-binding.json",
                        "target": "/run/sparkring/runtime-binding.json", "read_only": True}],
            "security_opt": [f"seccomp={workspace_path}/source-111111111111/runtime/common/loader-seccomp.json"]}


def controller(tmp_path, spark, *, hosts=HOSTS[:1], active=INSTALLED, specifications=True):
    """Node A's controller state: the cluster record, the installed deployment and a retained one.

    The fake Spark is this host's own filesystem, so the cluster has one Spark.
    """
    state = tmp_path / "controller"
    write_json(state / "cluster.json", {"name": CLUSTER, "plan": {"spec": {"hosts": [{"host": h} for h in hosts]}}})
    for name, image_id, names in ((INSTALLED, DEV_IMAGE, [INSTALLED_CACHE, COMPILE_CACHE]),
                                  (STALE, OLD_IMAGE, [STALE_CACHE, COMPILE_CACHE])):
        path = str(spark.cluster / name)
        rows = [{"rank": rank, "host": host, "model": spark.owned, "cache": str(spark.cache),
                 "repository": path + "/source-111111111111", "deployment_root": path + "/containers",
                 "reuse_verified_model": False} for rank, host in enumerate(hosts)]
        write_json(state / "deployments" / name / "deployment.lock.json",
                   {"id": "id-" + name, "selection": {"image_id": image_id},
                    "site": {"workspace": path, "ranks": rows}})
        if specifications:
            for rank in range(len(hosts)):
                write_json(state / "deployments" / name / f"rank{rank}" / "container.json",
                           spec(path, spark.owned, str(spark.cache), names))
    if active:
        write_json(state / "active.json", {"path": str(state / "deployments" / active)})
    return state


class Invoke:
    """Node A's SSH calls, answered by this module's node functions on the fake Spark (Node 0) or an empty one."""

    def __init__(self, spark, **docker):
        self.spark, self.calls = spark, []
        self.docker = {HOSTS[0]: spark.docker, HOSTS[1]: Docker(), **docker}

    def __call__(self, host, argv, *, data=None, timeout=None):
        request = json.loads(data)
        self.calls.append((host, argv, request))
        root = self.spark.root if host == HOSTS[0] else self.spark.empty
        if argv == LIST:
            return json.dumps(storage.list_local(request, root=root, run=self.docker[host]))
        assert argv[:-1] == LIST + ["--release"] and timeout
        return json.dumps(storage.release_local(argv[-1], request, root=root, run=self.docker[host], table=[]))

    def releases(self):
        return [(host, argv[-1], request) for host, argv, request in self.calls if argv != LIST]


@linux
def test_report_classifies_every_item_and_image_of_a_spark(tmp_path, spark, capsys):
    state = controller(tmp_path, spark)
    invoke = Invoke(spark)
    assert storage.main(["--json"], state_root=state, invoke=invoke) == 0
    result = json.loads(capsys.readouterr().out)
    [first] = result["nodes"]
    cache, cluster, srv = spark.cache, spark.cluster, spark.srv
    assert {item["path"]: (item["kind"], item["class"]) for item in first["items"]} == {
        spark.owned: ("checkpoint", "installed"),
        spark.inner: ("checkpoint", "profile"),
        str(cache / INSTALLED_CACHE): ("cache", "installed"),
        str(cache / COMPILE_CACHE): ("cache", "installed"),
        str(cache / PROFILE_CACHE): ("cache", "profile"),
        str(cache / STALE_CACHE): ("cache", "unreferenced"),
        str(cache / LEFTOVER_CACHE): ("releasing", "unreferenced"),
        str(cache / "jit"): ("other", "unmanaged"),
        str(cluster / INSTALLED): ("workspace", "installed"),
        str(cluster / STALE): ("workspace", "unreferenced"),
        str(cluster / WITH_MODELS): ("workspace", "unreferenced"),
        str(cluster / "notes"): ("other", "unmanaged"),
        str(srv / "image020"): ("other", "unmanaged"),
        str(srv / "tp2-prefresh-20260925"): ("other", "unmanaged")}
    items = {item["path"]: item for item in first["items"]}
    # Hard links count once; an inode also linked from outside the item is not freed by removing it.
    assert (items[str(cluster / STALE)]["bytes"], items[str(cluster / STALE)]["frees_bytes"]) == du(cluster / STALE)
    assert (items[str(cache / STALE_CACHE)]["bytes"], items[str(cache / STALE_CACHE)]["frees_bytes"]) == \
        du(cache / STALE_CACHE, outside=[srv / "image020/graph-copy.bin"])
    assert (items[str(srv / "image020")]["bytes"], items[str(srv / "image020")]["frees_bytes"]) == \
        du(srv / "image020", outside=[Path(spark.owned) / WEIGHT, cache / STALE_CACHE / "vllm/graph.bin"])
    assert all(item["complete"] for item in first["items"])
    assert items[str(cache / STALE_CACHE)]["deployments"] == [{"name": STALE}]
    assert items[str(cache / INSTALLED_CACHE)]["deployments"] == [{"name": INSTALLED, "role": "active"}]
    assert items[str(cache / COMPILE_CACHE)]["deployments"] == [{"name": INSTALLED, "role": "active"}, {"name": STALE}]
    assert items[str(cache / PROFILE_CACHE)]["profiles"] == QWEN
    assert {path for path, item in items.items() if item.get("containers")} == {
        spark.owned, str(cache / INSTALLED_CACHE), str(cache / COMPILE_CACHE)}
    assert items[str(cache / INSTALLED_CACHE)]["containers"] == ["sr-qwen-r0"]
    assert items[str(cluster / WITH_MODELS)]["checkpoints"] == [spark.inner]
    assert items[str(cache / LEFTOVER_CACHE)]["location"] == str(cache / f".{LEFTOVER_CACHE}.sparkring-releasing")
    released = [str(cache / STALE_CACHE), str(cluster / STALE), str(cache / LEFTOVER_CACHE)]
    assert {path: item["release"] for path, item in items.items() if item["release"]} == {
        path: "sudo sparkring storage --release " + path for path in released}
    assert first["proposed"] == {"frees_bytes": sum(items[path]["frees_bytes"] for path in released),
                                 "commands": ["sudo sparkring storage --release " + path
                                              for path in sorted(released, key=lambda p: (-items[p]["bytes"], p))]}
    assert {entry["id"]: (entry["class"], entry["locks"]) for entry in first["docker"]["images"]} == {
        DEV_IMAGE: ("installed", []), DEFAULT_IMAGE: ("profile", [installer_image.default_lock()["name"]]),
        OLD_IMAGE: ("unreferenced", [OLD_LOCK]), "sha256:" + "e" * 64: ("unreferenced", [])}
    assert [entry["role"] for entry in first["filesystems"][0]["paths"]][:1] == ["sparkring"]
    # Each Spark was asked for the paths the retained deployments name there.
    [request] = [request for host, _, request in invoke.calls if host == HOSTS[0]]
    assert request == {"cluster": CLUSTER, "workspaces": [str(cluster / INSTALLED), str(cluster / STALE)],
                       "caches": [str(cache)], "models": [spark.owned], "budget_seconds": storage.BUDGET_SECONDS}

    assert storage.main([], state_root=state, invoke=invoke) == 0
    printed = capsys.readouterr().out.splitlines()
    hostname = socket.gethostname()
    assert printed[0] == f"Node 0 {hostname}"
    assert any(line.endswith(f"unreferenced  cache       {cache / STALE_CACHE}") for line in printed)
    start = next(index for index, line in enumerate(printed) if line.endswith(str(cache / STALE_CACHE)))
    assert printed[start + 1].strip() == f"used by {STALE} (retained)"
    start = next(index for index, line in enumerate(printed) if line.endswith(str(cluster / WITH_MODELS)))
    assert printed[start + 1].strip() == (f"holds checkpoint directory {spark.inner}; release it first with sudo "
                                          f"sparkring checkpoints --release {spark.inner}")
    assert any(line.endswith(f"unreferenced  image       {OLD_IMAGE[7:19]} <untagged> (installer image lock "
                             f"{OLD_LOCK})") for line in printed)
    assert any(line.endswith("installed     image       dddddddddddd ghcr.io/fujitsupolycom/sparkring-installer:dev "
                             "(running in sr-qwen-r0)") for line in printed)
    assert any(line.endswith("(stopped container scratch)") for line in printed)
    proposal = printed.index(f"    Proposed releases on this Spark free {storage._size(first['proposed']['frees_bytes'])}:")
    assert printed[proposal + 1:proposal + 4] == ["        " + command for command in first["proposed"]["commands"]]
    assert printed[-2] == storage.LEGEND


@linux
def test_release_removes_unreferenced_data_after_approval(tmp_path, spark, capsys, monkeypatch):
    state = controller(tmp_path, spark)
    invoke = Invoke(spark)
    stale = str(spark.cache / STALE_CACHE)
    _, frees = du(stale, outside=[spark.srv / "image020/graph-copy.bin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert storage.main(["--release", stale], state_root=state, invoke=invoke) == 3
    printed = capsys.readouterr().out.splitlines()
    hostname = socket.gethostname()
    assert printed == [f"Release {stale}:",
                       f"    Node 0 {hostname}: removes the cache directory; frees {storage._size(frees)}",
                       f"Retained deployments that use it need sudo sparkring install again: {STALE}",
                       "Review the release above, then repeat with --yes to apply it. Nothing was released."]
    assert invoke.releases() == [] and os.path.isdir(stale)
    # Only the measured path is walked while a release is prepared.
    assert all(request.get("measure") == [stale] for _, _, request in invoke.calls)

    assert storage.main(["--release", stale, "--yes", "--json"], state_root=state, invoke=invoke) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "released" and result["deployments"] == [STALE]
    [answer] = result["nodes"]
    assert (answer["rank"], answer["state"], answer["kind"], answer["freed_bytes"]) == (0, "released", "cache", frees)
    assert not os.path.lexists(stale) and not os.path.lexists(spark.cache / f".{STALE_CACHE}.sparkring-releasing")
    kept = spark.srv / "image020/graph-copy.bin"
    assert kept.read_bytes() == b"x" * 20000 and os.lstat(kept).st_nlink == 1
    # The Spark was told what the installed deployment uses, to check again before it removed anything.
    [(host, path, request)] = invoke.releases()
    assert (host, path) == (HOSTS[0], stale) and request["opaque"] == []
    assert {str(spark.cache / INSTALLED_CACHE / "triton"), str(spark.cache / COMPILE_CACHE / "b12x"), spark.owned,
            str(spark.cluster / INSTALLED)} <= set(request["in_use"])
    assert not any(storage._inside(value, stale) for value in request["in_use"])

    for path, kind in ((str(spark.cluster / STALE), "deployment workspace"),
                       (str(spark.cache / LEFTOVER_CACHE), "remainder of an interrupted release")):
        assert storage.main(["--release", path, "--yes"], state_root=state, invoke=invoke) == 0
        printed = capsys.readouterr().out.splitlines()
        assert printed[1].startswith(f"    Node 0 {hostname}: removes the {kind}; frees ")
        assert printed[-1].startswith(f"Node 0 {hostname}: released; freed ")
        assert not os.path.lexists(path)
    assert not os.path.lexists(spark.cache / f".{LEFTOVER_CACHE}.sparkring-releasing")

    refusals = {
        str(spark.cache / INSTALLED_CACHE): f"the installed deployment {INSTALLED} (active) uses",
        str(spark.cluster / INSTALLED): f"the installed deployment {INSTALLED} (active) uses",
        str(spark.cache / PROFILE_CACHE): ("is referenced by the installer profiles qwen38-flash-next-qad-tp4, "
                                           "qwen38-flash-next-tp2 of the installed package"),
        spark.owned: f"is a SparkRing checkpoint directory; sudo sparkring checkpoints --release {spark.owned}",
        str(spark.cluster / WITH_MODELS): f"holds model files ({spark.inner}); release checkpoint directories first",
        str(spark.srv / "image020"): "was not created by SparkRing's installer; SparkRing does not remove it",
        str(spark.cache / "jit"): "was not created by SparkRing's installer",
        str(spark.cluster / "missing"): "is not a SparkRing cache directory or deployment workspace on any Spark",
    }
    before = tree_state(spark.srv)
    for path, message in refusals.items():
        assert storage.main(["--release", path, "--yes"], state_root=state, invoke=invoke) == 2, path
        assert message in capsys.readouterr().err, path
    assert tree_state(spark.srv) == before
    assert len(invoke.releases()) == 3


@linux
def test_each_spark_checks_again_before_it_removes_anything(spark, monkeypatch):
    path = str(spark.cache / STALE_CACHE)
    hidden = spark.cache / f".{STALE_CACHE}.sparkring-releasing"
    request = {"cluster": CLUSTER}
    before = tree_state(spark.srv)
    user = container("manual", mounts=[(str(spark.cache), "/cache")], env=[f"XDG_CACHE_HOME=/cache/{STALE_CACHE}"])
    refusals = [({"in_use": [path + "/vllm"]}, Docker(), [], "The installed deployment uses"),
                ({"opaque": [str(spark.cache)]}, Docker(), [], "The installed deployment uses"),
                ({}, Docker(), [{"point": path + "/vllm"}], "holds mount points"),
                ({}, Docker(containers=[user]), [], rf"Running containers on this Spark use {path} \(manual\)")]
    for extra, docker, table, message in refusals:
        with pytest.raises(ValueError, match=message):
            storage.release_local(path, {**request, **extra}, root=spark.root, run=docker, table=table)
    for other in (spark.owned, str(spark.srv / "image020"), str(spark.cluster / WITH_MODELS)):
        with pytest.raises(ValueError, match="Nothing was released"):
            storage.release_local(other, request, root=spark.root, run=Docker(), table=[])
    assert tree_state(spark.srv) == before

    # A release interrupted after the rename leaves a remainder that the listing reports and a repeat removes.
    original = storage._remove_tree

    def interrupted(*arguments):
        raise OSError("interrupted")

    monkeypatch.setattr(storage, "_remove_tree", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        storage.release_local(path, request, root=spark.root, run=Docker(), table=[])
    assert hidden.is_dir() and not os.path.lexists(path)
    listing = storage.list_local(request, root=spark.root, run=Docker())
    [entry] = [item for item in listing["items"] if item["path"] == path]
    assert (entry["kind"], entry["location"]) == ("releasing", str(hidden))
    monkeypatch.setattr(storage, "_remove_tree", original)
    assert storage.release_local(path, request, root=spark.root, run=Docker(), table=[])["state"] == "released"
    assert not hidden.exists()
    assert storage.release_local(path, request, root=spark.root, run=Docker(), table=[]) == {
        "path": path, "state": "absent", "kind": None, "freed_bytes": 0}


def test_running_containers_are_found_through_their_mounts_and_environment():
    cache = "/srv/sparkring/tp4/cache"
    entry = cache + "/" + STALE_CACHE
    cases = [(container("direct", mounts=[(entry + "/", "/cache/x")]), True),
             (container("named", mounts=[(cache, "/cache")], env=[f"XDG_CACHE_HOME=/cache/{STALE_CACHE}/vllm"]), True),
             # A cache variable that names the mount point itself may create any directory below it.
             (container("whole", mounts=[(cache, "/cache/jit")], env=["XDG_CACHE_HOME=/cache/jit"]), True),
             (container("other", mounts=[(cache, "/cache")], env=[f"XDG_CACHE_HOME=/cache/{INSTALLED_CACHE}"]), False),
             (container("sibling", mounts=[(entry + "-copy", "/cache")]), False)]
    docker = Docker(containers=[item for item, _ in cases])
    assert storage.containers_using(entry, run=docker) == sorted(item["Name"][1:] for item, used in cases if used)
    stopped = Docker(containers=[container("stopped", running=False, mounts=[(entry, "/cache")])])
    assert storage.containers_using(entry, run=stopped) == []

    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")
    assert storage.containers_using(entry, run=missing) == []

    def refused(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="permission denied while trying to connect\n")
    with pytest.raises(ValueError, match=r"could not list this host's running containers \(permission denied"):
        storage.containers_using(entry, run=refused)
    assert storage.docker_state(run=missing) is None
    assert storage.docker_state(run=refused) == {"root": None, "images": [],
                                                  "error": "permission denied while trying to connect"}


@linux
def test_disk_use_stops_at_its_time_limit(tmp_path):
    put(tmp_path / "tree/a/b/c.bin", 5000)
    measured = storage.disk_use(str(tmp_path / "tree"), deadline=time.monotonic() - 1)
    assert measured["complete"] is False and measured["files"] == 0
    assert measured["bytes"] == os.lstat(tmp_path / "tree").st_blocks * 512
    assert storage.disk_use(str(tmp_path / "tree"), deadline=time.monotonic() + 60)["complete"] is True
    assert storage.disk_use(str(tmp_path / "absent"), deadline=time.monotonic() + 60)["bytes"] == 0


@pytest.mark.parametrize("profile", installer_image.SUPPORTED)
def test_profile_cache_names_are_the_caches_installer_containers_use(profile):
    from runtime.common.test_compose_installer import install_site
    metadata, _ = profiles.load(profile)
    configuration = qwen_flash_next.read(profiles.ROOT / metadata["configuration"]["path"])
    _, names = qwen_flash_next.checkpoint_names(configuration)
    used = set()
    for variant in names or [None]:
        # Every rank uses the cluster's checkpoint directory, as installations do; a derived checkpoint needs it.
        site = install_site(4 if profile.endswith("-tp4") else 2)
        for row in site["hosts"]:
            row["model"] = installer.checkpoint_directory("parity", installer.setup.selection(profile, variant))
        lock = installer.make_lock(profile, site, "1" * 40, "2" * 64,
                                   variant, image_runtime=installer_image.for_profile(profile))
        root = lock["site"]["ranks"][0]["cache"]
        for specification in installer.specifications(lock):
            used |= {posixpath.relpath(path, root).split("/")[0] for path in storage.spec_paths(specification.document())
                     if storage._inside(path, root) and path != root}
    references = storage.profile_references()
    assert used == {name for name, listed in references["caches"].items() if profile in listed}
    assert all(storage.CACHE_NAME.fullmatch(name) for name in used)
    # The example of an image that no profile selects by default: its caches are unreferenced.
    assert STALE_CACHE not in references["caches"] and OLD_LOCK in references["locks"][OLD_IMAGE]


def local(items=(), *, revision="a" * 40, hostname="spark-e", filesystems=(), docker=None):
    return {"schema": storage.LOCAL_SCHEMA, "hostname": hostname, "package_revision": revision,
            "filesystems": list(filesystems), "items": [dict(item) for item in items], "docker": docker,
            "measurement": {"budget_seconds": 60, "seconds": 1.0, "complete": True}}


STALE_ITEM = {"path": f"/srv/sparkring/{CLUSTER}/cache/{STALE_CACHE}", "kind": "cache",
              "cache_root": f"/srv/sparkring/{CLUSTER}/cache", "bytes": 12 * 1024 ** 3, "frees_bytes": 12 * 1024 ** 3,
              "files": 40, "complete": False, "mounts": []}


class Spark:
    """Canned ``discovery.ssh``: per-host listings, and recorded release requests."""

    def __init__(self, listings):
        self.listings, self.calls = listings, []

    def __call__(self, host, argv, *, data=None, timeout=None):
        self.calls.append((host, argv))
        answer = self.listings[host]
        if isinstance(answer, Exception):
            raise answer
        assert argv == LIST
        return json.dumps(answer)


def canned_controller(tmp_path):
    state = tmp_path / "controller"
    write_json(state / "cluster.json", {"name": CLUSTER, "plan": {"spec": {"hosts": [{"host": h} for h in HOSTS]}}})
    return state


def test_report_text_names_filesystems_and_marks_incomplete_sizes(tmp_path, capsys):
    size, free = int(3.7e12), 149 * 1024 ** 3
    used = size - free - 190 * 1024 ** 3
    filesystem = {"mount_point": "/", "fstype": "ext4", "size_bytes": size, "used_bytes": used, "free_bytes": free,
                  "paths": [{"role": "sparkring", "path": "/srv/sparkring"},
                            {"role": "docker", "path": "/var/lib/docker"}, {"role": "root", "path": "/"}]}
    spark = Spark({HOSTS[0]: local([STALE_ITEM], filesystems=[filesystem],
                                   docker={"root": "/var/lib/docker", "images": [], "error": None}),
                   HOSTS[1]: RuntimeError("root@192.0.2.11: Connection timed out")})
    assert storage.main([], state_root=canned_controller(tmp_path), invoke=spark) == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[:4] == [
        "Node 0 spark-e",
        "    / (ext4): 3.4 TiB, 149.0 GiB free, 96% used; holds /srv/sparkring, Docker's data root /var/lib/docker",
        "         SIZE  CLASS         KIND        PATH",
        f"    12.0 GiB+  unreferenced  cache       {STALE_ITEM['path']}"]
    assert printed[4:8] == ["    Docker images in /var/lib/docker:",
                            "    Proposed releases on this Spark free 12.0 GiB:",
                            f"        sudo sparkring storage --release {STALE_ITEM['path']}",
                            "Node 1 root@192.0.2.11: not reachable: root@192.0.2.11: Connection timed out"]


def test_an_unreferenced_checkpoint_directory_is_released_by_the_checkpoints_command(tmp_path, capsys):
    path = f"/srv/sparkring/{CLUSTER}/checkpoints/{SLUG}/" + "f" * 40
    entry = {"path": path, "kind": "checkpoint", "repository": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
             "revision": "f" * 40, "state": "ok", "bytes": 100 * 1024 ** 3, "frees_bytes": 2 * 1024 ** 3,
             "complete": True, "containers": []}
    replaced = {**entry, "path": path[:-1] + "e", "state": "replaced"}
    spark = Spark({HOSTS[0]: local([entry, replaced]), HOSTS[1]: local(hostname="spark-d")})
    state = canned_controller(tmp_path)
    assert storage.main(["--json"], state_root=state, invoke=spark) == 0
    node = json.loads(capsys.readouterr().out)["nodes"][0]
    assert {item["path"]: (item["class"], item["release"]) for item in node["items"]} == {
        path: ("unreferenced", "sudo sparkring checkpoints --release " + path),
        replaced["path"]: ("unreferenced", None)}
    assert node["proposed"] == {"frees_bytes": 2 * 1024 ** 3, "commands": ["sudo sparkring checkpoints --release " + path]}
    assert storage.main(["--release", path, "--yes"], state_root=state, invoke=spark) == 2
    assert f"sudo sparkring checkpoints --release {path} releases it. Nothing was released." in capsys.readouterr().err


def test_a_running_container_keeps_an_unreferenced_item_out_of_the_proposal(tmp_path, capsys):
    busy = {**STALE_ITEM, "containers": ["rehearsal-r0"]}
    spark = Spark({HOSTS[0]: local([busy]), HOSTS[1]: local(hostname="spark-d")})
    state = canned_controller(tmp_path)
    assert storage.main(["--json"], state_root=state, invoke=spark) == 0
    [entry] = json.loads(capsys.readouterr().out)["nodes"][0]["items"]
    assert (entry["class"], entry["release"]) == ("unreferenced", None)
    assert storage.main(["--release", busy["path"], "--yes"], state_root=state, invoke=spark) == 2
    assert (f"Node 0 spark-e: running containers use {busy['path']} (rehearsal-r0); stop them first, then repeat "
            "the release. Nothing was released.") in capsys.readouterr().err


def test_release_refuses_mixed_package_revisions_and_unreachable_sparks(tmp_path, capsys):
    state = canned_controller(tmp_path)
    spark = Spark({HOSTS[0]: local([STALE_ITEM]), HOSTS[1]: local(revision="b" * 40, hostname="spark-d")})
    assert storage.main(["--release", STALE_ITEM["path"], "--yes"], state_root=state, invoke=spark) == 2
    error = capsys.readouterr().err
    assert "The Sparks run different SparkRing package revisions (Node 0 aaaaaaaaaaaa, Node 1 bbbbbbbbbbbb)" in error
    assert "Nothing was released." in error
    spark = Spark({HOSTS[0]: local([STALE_ITEM]), HOSTS[1]: RuntimeError("root@192.0.2.11: timed out")})
    assert storage.main(["--release", STALE_ITEM["path"], "--yes"], state_root=state, invoke=spark) == 2
    assert "Node 1 root@192.0.2.11 could not be listed: root@192.0.2.11: timed out" in capsys.readouterr().err
    assert all(argv == LIST for _, argv in spark.calls)
    assert storage.main(["--release", "relative/path", "--yes"], state_root=state, invoke=spark) == 2
    assert "absolute path" in capsys.readouterr().err
    # Without a cluster record the report covers this Spark alone and a release is refused.
    alone = tmp_path / "alone"
    alone.mkdir()
    assert storage.main(["--release", STALE_ITEM["path"], "--yes"], state_root=alone, invoke=spark) == 2
    assert "from Node A of a configured cluster" in capsys.readouterr().err


def test_container_specification_paths_are_translated_through_binds():
    document = spec("/srv/sparkring/tp4/ws", "/srv/sparkring/tp4/checkpoints/x/" + REVISION,
                    "/srv/sparkring/tp4/cache", [INSTALLED_CACHE, COMPILE_CACHE])
    assert storage.spec_paths(document) == {
        "/srv/sparkring/tp4/checkpoints/x/" + REVISION, "/srv/sparkring/tp4/cache",
        "/srv/sparkring/tp4/ws/containers/lock/runtime-binding.json",
        f"/srv/sparkring/tp4/cache/{INSTALLED_CACHE}", f"/srv/sparkring/tp4/cache/{INSTALLED_CACHE}/triton",
        f"/srv/sparkring/tp4/cache/{COMPILE_CACHE}/b12x",
        "/srv/sparkring/tp4/ws/source-111111111111/runtime/common/loader-seccomp.json"}


@linux
def test_a_deployment_without_container_specifications_uses_its_whole_cache_root(tmp_path, spark, capsys):
    state = controller(tmp_path, spark, specifications=False)
    assert storage.main(["--json"], state_root=state, invoke=Invoke(spark)) == 0
    items = {item["path"]: item for item in json.loads(capsys.readouterr().out)["nodes"][0]["items"]}
    assert {items[str(spark.cache / name)]["class"] for name in (STALE_CACHE, PROFILE_CACHE, INSTALLED_CACHE)} == {
        "installed"}
    assert items[str(spark.cluster / STALE)]["class"] == "unreferenced"


def install_mesh(root, name, workspace_path, *, model, cache, config=None):
    """A mesh installed on the Spark below ``root``, with the files that ``native_mesh.install_local`` writes.

    ``config`` is the configuration directory, by default the named layout's
    of ``name``. Its site names ``workspace_path``'s marker binary and bundle
    root, as ``native_mesh.definitions`` places them. Returns the site's path.
    """
    config = config or f"/etc/sparkring/deployments/{name}"
    local = Path(root) / config.lstrip("/")
    site = {"schema": "sparkring-glm53-mtp3-mesh-site/v1", "topology_file": "fabric.json",
            "management_addresses": [f"192.0.2.{10 + rank}" for rank in range(4)],
            "model_roots": [model] * 4, "cache_roots": [cache] * 4,
            "bundle_root": f"{workspace_path}/{BUNDLE}", "container_prefix": "sr-" + name,
            "marker_binary": f"{workspace_path}/{MARKER}", "marker_binary_sha256": "5" * 64,
            "state_root": "/run/sparkring-" + name}
    write_json(local / "site.json", site)
    write_json(local / "fabric.json", {"ranks": []})
    write_json(local / "service.json", {
        "schema": "sparkring-managed-mesh/v1", "site_path": config + "/site.json", "rank": 0,
        "key_file": config + "/health.key", "epoch": "a" * 32, "health_port": 9976, "state_dir": "/run/sparkring-" + name,
        "container_id": "c" * 64, "container_image": DEV_IMAGE,
        **({"deployment_name": name} if config.startswith("/etc/sparkring/deployments/") else {})})
    return config + "/site.json"


def ring_controller(tmp_path, spark):
    """Node A's state after two model switches on a four-Spark ring, with the mesh files on the fake Spark.

    FIRST created the ring's mesh MESH, whose marker binary and bundle lie in
    FIRST's workspace; SECOND and THIRD reused that mesh, so every rank row
    names its site as the fabric. THIRD is active and SECOND is the rollback
    target. The fake Spark stands for each rank, which holds the same
    workspaces and mesh configuration.
    """
    state = tmp_path / "controller"
    write_json(state / "cluster.json", {"name": CLUSTER, "plan": {"spec": {"hosts": [{"host": HOSTS[0]}]}}})
    first = spark.cluster / FIRST
    site = install_mesh(spark.root, MESH, str(first), model=spark.owned, cache=str(spark.cache))
    for name in (FIRST, SECOND, THIRD):
        path = str(workspace(spark.cluster / name, "id-" + name))
        row = {"rank": 0, "host": HOSTS[0], "model": spark.owned, "cache": str(spark.cache),
               "repository": path + "/source-111111111111", "deployment_root": path + "/containers",
               "reuse_verified_model": False,
               "fabric": {"site_path": site, "site_sha256": "6" * 64, "plan_sha256": "7" * 64}}
        write_json(state / "deployments" / name / "deployment.lock.json",
                   {"id": "id-" + name, "selection": {"image_id": DEV_IMAGE}, "site": {"workspace": path, "ranks": [row]}})
        write_json(state / "deployments" / name / "rank0" / "container.json",
                   spec(path, spark.owned, str(spark.cache), [INSTALLED_CACHE, COMPILE_CACHE]))
    put(first / MARKER, 4096)
    put(first / BUNDLE / "manifest.json", 200)
    write_json(state / "active.json", {"path": str(state / "deployments" / THIRD)})
    write_json(state / "transaction.json", {"previous": str(state / "deployments" / SECOND),
                                            "candidate": str(state / "deployments" / THIRD), "state": "complete"})
    return state, site


@linux
def test_the_workspace_holding_the_ring_mesh_files_stays_installed_after_two_model_switches(tmp_path, spark, capsys):
    state, site = ring_controller(tmp_path, spark)
    invoke = Invoke(spark)
    first = str(spark.cluster / FIRST)
    unit = f"sparkring-{MESH}-mesh.service"
    named = [f"{first}/{BUNDLE}", f"{first}/{MARKER}"]
    assert storage.main(["--json"], state_root=state, invoke=invoke) == 0
    [node] = json.loads(capsys.readouterr().out)["nodes"]
    items = {item["path"]: item for item in node["items"]}
    # FIRST is neither active nor the rollback target, but the mesh that SECOND and THIRD run needs its files.
    assert items[first]["deployments"] == [{"name": FIRST}]
    assert (items[first]["class"], items[first]["release"]) == ("installed", None)
    assert items[first]["meshes"] == [{"unit": unit, "site": site, "paths": named}]
    assert {items[str(spark.cluster / name)]["class"] for name in (SECOND, THIRD)} == {"installed"}
    assert all(first not in command for command in node["proposed"]["commands"])
    assert node["meshes"] == [{"unit": unit, "config": f"/etc/sparkring/deployments/{MESH}", "site": site,
                               "paths": sorted({*named, spark.owned, str(spark.cache), "/run/sparkring-" + MESH}),
                               "readable": True, "container": "c" * 64}]
    # The mesh's cache roots name the cluster cache; its entries stay classified by the deployments that use them.
    assert (items[str(spark.cache / STALE_CACHE)]["class"], items[str(spark.cache / STALE_CACHE)]["meshes"]) == (
        "unreferenced", [])
    assert items[str(spark.cluster / STALE)]["class"] == "unreferenced"

    assert storage.main([], state_root=state, invoke=invoke) == 0
    printed = capsys.readouterr().out.splitlines()
    start = next(index for index, line in enumerate(printed) if line.endswith(f"installed     workspace   {first}"))
    assert [line.strip() for line in printed[start + 1:start + 3]] == [
        f"used by {FIRST} (retained)",
        f"used by the installed mesh {unit}, whose site {site} names {named[0]}, {named[1]}"]

    before = tree_state(spark.srv)
    assert storage.main(["--release", first, "--yes"], state_root=state, invoke=invoke) == 2
    assert (f"Node 0 {socket.gethostname()}: the installed mesh {unit}, whose site {site} names {named[0]}, "
            f"{named[1]}, uses {first}; SparkRing does not release it while that mesh is installed. Nothing was "
            "released.") in capsys.readouterr().err
    assert invoke.releases() == [] and tree_state(spark.srv) == before

    # Without the mesh's configuration in /etc/sparkring, as native_mesh.set_aside leaves a replaced mesh, the
    # workspace is unreferenced and released.
    replaced = Path(spark.root) / "var/lib/sparkring/replaced-meshes" / f"{MESH}-1"
    replaced.parent.mkdir(parents=True)
    os.rename(Path(spark.root) / "etc/sparkring/deployments" / MESH, replaced)
    assert storage.main(["--json"], state_root=state, invoke=invoke) == 0
    items = {item["path"]: item for item in json.loads(capsys.readouterr().out)["nodes"][0]["items"]}
    assert (items[first]["class"], items[first]["release"]) == ("unreferenced", "sudo sparkring storage --release " + first)
    assert storage.main(["--release", first, "--yes"], state_root=state, invoke=invoke) == 0
    assert not os.path.lexists(first)


@linux
def test_each_spark_refuses_to_release_what_an_installed_mesh_uses(tmp_path, spark):
    stale, cache = str(spark.cluster / STALE), str(spark.cache / STALE_CACHE)
    # Node A's request names nothing in use; each Spark reads its own installed meshes.
    request = {"cluster": CLUSTER}
    default = install_mesh(spark.root, "glm", stale, model=spark.owned, cache=str(spark.cache),
                           config="/etc/sparkring/managed-mesh")
    before = tree_state(spark.srv)
    with pytest.raises(ValueError, match="^" + re.escape(
            f"The installed mesh sparkring-mesh.service, whose site {default} names {stale}/{BUNDLE}, "
            f"{stale}/{MARKER}, uses {stale}; SparkRing does not release it while that mesh is installed. "
            "Nothing was released.") + "$"):
        storage.release_local(stale, request, root=spark.root, run=Docker(), table=[])
    # A mesh whose site cannot be read may name any path, so it keeps every cache directory and workspace.
    site = install_mesh(spark.root, MESH, str(spark.cluster / FIRST), model=spark.owned, cache=str(spark.cache))
    (Path(spark.root) / site.lstrip("/")).write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="^" + re.escape(
            f"The installed mesh sparkring-{MESH}-mesh.service, whose site {site} cannot be read, uses {cache}; "
            "SparkRing does not release it while that mesh is installed. Nothing was released.") + "$"):
        storage.release_local(cache, request, root=spark.root, run=Docker(), table=[])
    assert tree_state(spark.srv) == before
    # With both meshes readable, a cache entry that no site path lies in is released.
    install_mesh(spark.root, MESH, str(spark.cluster / FIRST), model=spark.owned, cache=str(spark.cache))
    assert storage.release_local(cache, request, root=spark.root, run=Docker(), table=[])["state"] == "released"


def test_installed_meshes_are_read_from_every_managed_mesh_layout(tmp_path):
    root = str(tmp_path)
    workspace_path = f"/srv/sparkring/{CLUSTER}/{FIRST}"
    named = install_mesh(root, MESH, workspace_path, model="/srv/models/a", cache=f"/srv/sparkring/{CLUSTER}/cache")
    default = install_mesh(root, "glm", "/srv/sparkring/glm-managed", model="/srv/models/b",
                           cache="/srv/sparkring/glm-managed/cache", config="/etc/sparkring/managed-mesh")
    # A configuration directory holding only its service file whose site cannot be read, and one without any.
    write_json(tmp_path / "etc/sparkring/deployments/partial/service.json",
               {"site_path": "/etc/sparkring/deployments/partial/site.json", "deployment_name": "partial"})
    (tmp_path / "etc/sparkring/deployments/empty").mkdir()
    meshes = {mesh["config"]: mesh for mesh in storage.installed_meshes(root)}
    assert set(meshes) == {"/etc/sparkring/managed-mesh", f"/etc/sparkring/deployments/{MESH}",
                           "/etc/sparkring/deployments/partial"}
    assert (meshes["/etc/sparkring/managed-mesh"]["unit"], meshes["/etc/sparkring/managed-mesh"]["site"]) == (
        "sparkring-mesh.service", default)
    mesh = meshes[f"/etc/sparkring/deployments/{MESH}"]
    assert (mesh["unit"], mesh["site"], mesh["readable"]) == (f"sparkring-{MESH}-mesh.service", named, True)
    assert mesh["paths"] == sorted({"/srv/models/a", f"/srv/sparkring/{CLUSTER}/cache", f"{workspace_path}/{BUNDLE}",
                                    f"{workspace_path}/{MARKER}", "/run/sparkring-" + MESH})
    assert meshes["/etc/sparkring/deployments/partial"]["readable"] is False
    assert storage._mesh_users([mesh], workspace_path) == [
        {"unit": mesh["unit"], "site": named, "paths": [f"{workspace_path}/{BUNDLE}", f"{workspace_path}/{MARKER}"]}]
    assert storage._mesh_users([mesh], f"/srv/sparkring/{CLUSTER}/cache/{STALE_CACHE}") == []
    assert storage._mesh_users([meshes["/etc/sparkring/deployments/partial"]], workspace_path) == [
        {"unit": "sparkring-partial-mesh.service", "site": "/etc/sparkring/deployments/partial/site.json",
         "paths": [], "readable": False}]


def test_the_report_and_a_release_name_the_mesh_that_uses_a_workspace(tmp_path, capsys):
    path = f"/srv/sparkring/{CLUSTER}/{FIRST}"
    site = f"/etc/sparkring/deployments/{MESH}/site.json"
    unit = f"sparkring-{MESH}-mesh.service"
    entry = {"path": path, "kind": "workspace", "deployment": "id-" + FIRST, "holds_models": False,
             "bytes": 3 * 1024 ** 2, "frees_bytes": 3 * 1024 ** 2, "files": 9, "complete": True, "mounts": [],
             "checkpoints": [], "containers": [], "meshes": [{"unit": unit, "site": site, "paths": [path + "/" + MARKER]}]}
    spark = Spark({HOSTS[0]: local([entry]), HOSTS[1]: local([{**entry, "meshes": []}], hostname="spark-d")})
    state = canned_controller(tmp_path)
    assert storage.main(["--json"], state_root=state, invoke=spark) == 0
    nodes = json.loads(capsys.readouterr().out)["nodes"]
    assert [(node["items"][0]["class"], node["items"][0]["release"]) for node in nodes] == [
        ("installed", None), ("unreferenced", "sudo sparkring storage --release " + path)]
    assert storage.main([], state_root=state, invoke=spark) == 0
    printed = capsys.readouterr().out.splitlines()
    start = next(index for index, line in enumerate(printed) if line.endswith(f"installed     workspace   {path}"))
    assert printed[start + 1].strip() == f"used by the installed mesh {unit}, whose site {site} names {path}/{MARKER}"
    assert storage.main(["--release", path, "--yes"], state_root=state, invoke=spark) == 2
    assert (f"Node 0 spark-e: the installed mesh {unit}, whose site {site} names {path}/{MARKER}, uses {path}; "
            "SparkRing does not release it while that mesh is installed. Nothing was released.") in capsys.readouterr().err
    assert all(argv == LIST for _, argv in spark.calls)


def test_commands_route_to_the_storage_module(monkeypatch, capsys):
    from scripts import sparkring, sparkring_node
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    seen = []
    monkeypatch.setattr(storage, "main", lambda argv: seen.append(argv) or 0)
    assert sparkring.main(["storage", "--release", STALE_ITEM["path"], "--yes"]) == 0
    assert seen == [["--release", STALE_ITEM["path"], "--yes"]]
    monkeypatch.setattr(storage, "release_local", lambda path, request: {"path": path, "request": request})
    monkeypatch.setattr(storage, "list_local", lambda request: {"request": request})
    request = {"cluster": CLUSTER, "in_use": ["/srv/sparkring/tp4/cache/x"]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert sparkring_node.main(["storage", "--release", STALE_ITEM["path"]]) == 0
    assert json.loads(capsys.readouterr().out) == {"path": STALE_ITEM["path"], "request": request}
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert sparkring_node.main(["storage"]) == 0
    assert json.loads(capsys.readouterr().out) == {"request": {}}


def test_a_derived_checkpoint_is_installed_while_its_deployment_is_and_otherwise_kept_for_its_profile(tmp_path, capsys):
    from runtime.common import derived_checkpoint
    from runtime.common.test_derived_checkpoint import lock as derived_lock
    value = derived_lock("qwen38-flash-next-tp2")
    manifest = derived_checkpoint.load(value["selection"])
    base = value["site"]["ranks"][0]["model"]
    served = derived_checkpoint.directory(base, manifest)
    state = canned_controller(tmp_path)
    directory = state / "deployments" / "qwen-derived"
    write_json(directory / "deployment.lock.json", {**value, "site": {**value["site"], "ranks": [
        {**row, "host": host} for row, host in zip(value["site"]["ranks"], HOSTS)]}})
    for rank, specification in enumerate(installer.specifications(value)):
        write_json(directory / f"rank{rank}" / "container.json", specification.document())
    item = {"path": served, "kind": "checkpoint", "repository": manifest["repository"], "revision": manifest["revision"],
            "state": "ok", "bytes": 100 * 1024 ** 3, "frees_bytes": 6 * 1024 ** 3, "complete": True, "containers": []}
    spark = Spark({HOSTS[0]: local([item]), HOSTS[1]: local(hostname="spark-d")})
    write_json(state / "active.json", {"path": str(directory)})
    assert storage.main(["--json"], state_root=state, invoke=spark) == 0
    [entry] = json.loads(capsys.readouterr().out)["nodes"][0]["items"]
    assert (entry["class"], entry["release"]) == ("installed", None)
    assert entry["deployments"] == [{"name": "qwen-derived", "role": "active"}]
    assert entry["derived"] == {"name": "qad-step5500-mxfp8-attention", "base": manifest["base"],
                                "donor": {key: manifest["donor"][key] for key in ("repository", "revision")}}
    # Retained but not active, it stays for the profiles that list it; its details say what releasing frees.
    (state / "active.json").unlink()
    assert storage.main([], state_root=state, invoke=spark) == 0
    printed = capsys.readouterr().out.splitlines()
    row = next(index for index, line in enumerate(printed) if line.endswith(served))
    assert " profile " in printed[row] and "checkpoint" in printed[row]
    assert printed[row + 2].strip() == (
        f"derived checkpoint qad-step5500-mxfp8-attention of {manifest['base']['repository']} at "
        f"{manifest['base']['revision'][:12]} with files of {manifest['donor']['revision'][:12]}; its unchanged files "
        "are hard links to the base's, so releasing it frees only the files its recipe wrote")
    assert "referenced by installer profiles qwen38-flash-next-qad-tp4, qwen38-flash-next-tp2" in "\n".join(printed)
