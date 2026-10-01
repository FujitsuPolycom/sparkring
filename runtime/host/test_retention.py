"""Automatic release of what older deployments leave on the Sparks (``runtime/host/retention.py``).

The policy tests feed ``retention.plan`` canned deployment records and Spark
listings. Tests marked ``linux`` run Node A's release against the fake Spark
of ``test_storage.py`` (a ``/srv/sparkring`` tree below ``tmp_path`` with real
checkpoint directories and hard links), answering Node A's SSH calls with
storage's own node functions and a canned Docker that also inspects
containers by name and removes stopped ones.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from runtime.common import installer
from runtime.host import checkpoints, controller, retention, settings, storage
from runtime.host import test_storage as fake
from runtime.host.test_checkpoints import REVISION, adopt, tree_state, write_json
from runtime.host.test_storage import (CLUSTER, COMPILE_CACHE, DEV_IMAGE, HOSTS, INSTALLED_CACHE, LEFTOVER_CACHE, LIST,
                                       PROFILE_CACHE, STALE_CACHE, linux)

# The fake Spark and its record directories, and Node A's command run as administrator (test_storage's fixtures).
spark, host_records, administrator = fake.spark, fake.host_records, fake.administrator
PROFILE = "qwen38-flash-next-tp2"
OTHER = "mimo-v26-flash-mopd-tp2"
# The package revision of OLD, which ring() releases completely; the other deployments use "1" * 40.
OLD_REVISION = "2" * 40
DOWN = {"generation": 3, "operation": "down", "complete": True}


def identity(name):
    return hashlib.sha256(name.encode()).hexdigest()


# Policy -------------------------------------------------------------------------

def record(name, *, profile=PROFILE, state=DOWN, age=0, backend="compose", hosts=HOSTS):
    """A ``retention.deployments`` record; ``age`` is seconds since its last operation."""
    workspace = f"/srv/sparkring/{CLUSTER}/{name}"
    rows = [{"rank": rank, "host": host, "model": f"/srv/sparkring/{CLUSTER}/checkpoints/x/{REVISION}",
             "cache": f"/srv/sparkring/{CLUSTER}/cache", "repository": workspace + "/source-1",
             "deployment_root": workspace + "/containers", "reuse": False} for rank, host in enumerate(hosts)]
    return {"name": name, "directory": "/var/lib/sparkring/controller/deployments/" + name, "id": identity(name),
            "workspace": workspace, "image_id": DEV_IMAGE, "rows": rows,
            "references": {host: {"paths": [workspace, f"/srv/sparkring/{CLUSTER}/cache/{name}-cache"], "opaque": []}
                           for host in hosts},
            "profile": profile, "backend": backend, "state": state, "time": None if state is None else 10 ** 9 - age,
            "containers": [{"host": host, "rank": rank, "name": f"sr-{name[-12:]}-r{rank}"}
                           for rank, host in enumerate(hosts)],
            "released": None}


def listing(records, *, host=HOSTS[0], running=(), meshes=(), extra=()):
    """A classified listing of ``host``: each record's stopped container, workspace and private cache."""
    containers = [{"id": identity(entry["name"] + host), "name": value["name"], "deployment": entry["id"],
                   "rank": str(value["rank"]), "running": entry["name"] in running, "status": "exited",
                   "bytes": 300 * 10 ** 6}
                  for entry in records for value in entry["containers"] if value["host"] == host]
    items = []
    for entry in records:
        items.append({"path": entry["workspace"], "kind": "workspace", "deployment": entry["id"],
                      "class": "installed" if entry["name"] in meshes else "unreferenced",
                      "meshes": [{"unit": "sparkring-mesh.service"}] if entry["name"] in meshes else [],
                      "release": None if entry["name"] in meshes else "sudo sparkring storage --release "
                      + entry["workspace"], "frees_bytes": 100 * 10 ** 6, "complete": True})
        items.append({"path": f"/srv/sparkring/{CLUSTER}/cache/{entry['name']}-cache", "kind": "cache",
                      "class": "unreferenced", "meshes": [], "frees_bytes": 10 ** 9, "complete": True,
                      "release": f"sudo sparkring storage --release /srv/sparkring/{CLUSTER}/cache/{entry['name']}-cache"})
    return {"rank": HOSTS.index(host), "host": host, "hostname": "spark-" + host[-2:], "model_containers": containers,
            "items": items + list(extra)}


def roles(**names):
    return {"/var/lib/sparkring/controller/deployments/" + name: role for role, name in names.items()}


def kept(decided):
    return {entry["name"]: entry["reasons"] for entry in decided["deployments"] if entry["kept"]}


def test_the_policy_keeps_each_protected_deployment_and_releases_the_others():
    records = [record("active-aaaaaaaaaaaa", age=10, state={"generation": 9, "operation": "up", "complete": True}),
               record("rollback-bbbbbbbbbbbb", age=20),
               record("switching-cccccccccccc", age=30, state={"generation": 1, "operation": "prepare", "complete": True}),
               record("running-dddddddddddd", age=40),
               record("mesh-eeeeeeeeeeee", age=50),
               record("unfinished-ffffffffffff", age=60, state={"generation": 2, "operation": "up", "complete": False}),
               record("unreadable-111111111111", age=70, state={"unreadable": True}),
               record("started-222222222222", age=80, state={"generation": 4, "operation": "up", "complete": True}),
               record("managed-333333333333", age=90, backend="glm-managed"),
               record("old-444444444444", age=100),
               record("older-555555555555", age=110),
               record("never-666666666666", state=None),
               record("other-777777777777", profile=OTHER, age=1000),
               record("other-888888888888", profile=OTHER, age=2000),
               record("other-999999999999", profile=OTHER, age=3000)]
    role = roles(active="active-aaaaaaaaaaaa", rollback="rollback-bbbbbbbbbbbb", switching="switching-cccccccccccc")
    nodes = [listing(records, running=["running-dddddddddddd"], meshes=["mesh-eeeeeeeeeeee"]),
             listing(records, host=HOSTS[1])]
    decided = retention.plan(records, role, nodes, 2)
    assert kept(decided) == {
        "active-aaaaaaaaaaaa": ["active", "started", "recent"], "rollback-bbbbbbbbbbbb": ["rollback", "recent"],
        "switching-cccccccccccc": ["switching", "prepared"], "running-dddddddddddd": ["running"],
        "mesh-eeeeeeeeeeee": ["mesh"],
        "unfinished-ffffffffffff": ["unfinished"], "unreadable-111111111111": ["unfinished"],
        "started-222222222222": ["started"], "managed-333333333333": ["backend"],
        "other-777777777777": ["recent"], "other-888888888888": ["recent"]}
    # A deployment that never ran an operation holds nothing on the Sparks and is neither kept nor released.
    assert decided["released"] == ["old-444444444444", "older-555555555555", "other-999999999999"]
    # Recency counts only deployments that ran an operation, by their last operation.
    assert kept(retention.plan(records, {}, [], 1))["active-aaaaaaaaaaaa"] == ["started", "recent"]
    assert "rollback-bbbbbbbbbbbb" not in kept(retention.plan(records, {}, [], 1))
    # 0 keeps only the deployments the policy always keeps.
    assert "recent" not in {code for codes in kept(retention.plan(records, role, nodes, 0)).values() for code in codes}
    # Without Spark listings, Node A's records alone cannot see running containers or installed meshes.
    assert {"running-dddddddddddd", "mesh-eeeeeeeeeeee"} <= set(retention.plan(records, role, [], 2)["released"])


def test_the_policy_releases_only_released_deployments_containers_workspaces_and_unused_caches():
    old, older, active = record("old-444444444444", age=100), record("older-555555555555", age=110), \
        record("active-aaaaaaaaaaaa", age=10)
    records = [active, record("recent-bbbbbbbbbbbb", age=20), old, older]
    cache = f"/srv/sparkring/{CLUSTER}/cache"
    extra = [
        # A cache that only the released deployments use goes; one the active deployment also uses stays.
        {"path": f"{cache}/shared", "kind": "cache", "class": "unreferenced", "release": "x", "frees_bytes": 5},
        {"path": f"{cache}/profile", "kind": "cache", "class": "profile", "release": None, "frees_bytes": 7},
        {"path": f"{cache}/busy", "kind": "cache", "class": "unreferenced", "release": None, "containers": ["manual"]},
        {"path": f"{cache}/.gone", "kind": "releasing", "class": "unreferenced", "release": "y", "frees_bytes": 3},
        {"path": f"/srv/sparkring/{CLUSTER}/checkpoints/x/{REVISION}", "kind": "checkpoint", "class": "unreferenced",
         "release": "sudo sparkring checkpoints --release ...", "frees_bytes": 10 ** 11},
        # Workspaces of unknown deployments and of released ones holding model files stay.
        {"path": f"/srv/sparkring/{CLUSTER}/unknown", "kind": "workspace", "deployment": "f" * 64,
         "class": "unreferenced", "release": "z", "frees_bytes": 1},
        {"path": f"/srv/sparkring/{CLUSTER}/notes", "kind": "other", "class": "unmanaged", "release": None}]
    active["references"][HOSTS[0]]["paths"].append(f"{cache}/shared/triton")
    node_entry = listing(records, extra=extra)
    # storage.classify proposes no release of a workspace holding model files.
    node_entry["items"].append({"path": old["workspace"] + "-models", "kind": "workspace", "deployment": old["id"],
                                "class": "unreferenced", "release": None, "holds_models": True, "frees_bytes": 9})
    # Only the container names that the released deployment's lock gives its ranks are removed.
    node_entry["model_containers"].append({"id": "1" * 64, "name": "sr-unexpected-r0", "deployment": old["id"],
                                           "running": False, "bytes": 1})
    [action] = retention.plan(records, roles(active="active-aaaaaaaaaaaa"), [node_entry], 2)["nodes"]
    assert action["containers"] == [{"deployment": old["id"], "name": "sr-444444444444-r0", "bytes": 300 * 10 ** 6},
                                    {"deployment": older["id"], "name": "sr-555555555555-r0", "bytes": 300 * 10 ** 6}]
    assert [(entry["path"], entry["kind"]) for entry in action["paths"]] == [
        (old["workspace"], "workspace"), (older["workspace"], "workspace"),
        (f"{cache}/old-444444444444-cache", "cache"), (f"{cache}/older-555555555555-cache", "cache"),
        (f"{cache}/.gone", "releasing")]
    assert action["frees_bytes"] == 2 * 300 * 10 ** 6 + 2 * 100 * 10 ** 6 + 2 * 10 ** 9 + 3
    assert action["complete"] is True
    # A kept deployment whose container specification cannot be read keeps its whole cache root.
    active["references"][HOSTS[0]]["opaque"] = [cache]
    [action] = retention.plan(records, roles(active="active-aaaaaaaaaaaa"), [node_entry], 2)["nodes"]
    assert [entry["kind"] for entry in action["paths"]] == ["workspace", "workspace", "releasing"]


def test_settings_accept_off_or_a_count_of_recent_deployments(tmp_path):
    assert [settings.retain_deployments(text) for text in ("off", "OFF", "0", "2", "99")] == [None, None, 0, 2, 99]
    for text in ("", "-1", "100", "two", "2.5", "none"):
        with pytest.raises(ValueError, match="use off, or the number of recent deployments"):
            settings.retain_deployments(text)
    path = tmp_path / "settings.env"
    path.write_text("SPARKRING_RETAIN_DEPLOYMENTS=off\n")
    assert settings.load(path)["SPARKRING_RETAIN_DEPLOYMENTS"] == "off"
    assert settings.load()["SPARKRING_RETAIN_DEPLOYMENTS"] == ""
    # The installation checks the value where it reads it, so that the refusal names this key.
    path.write_text("SPARKRING_RETAIN_DEPLOYMENTS=many\n")
    assert settings.load(path)["SPARKRING_RETAIN_DEPLOYMENTS"] == "many"
    from runtime.host import install_workflow
    from runtime.host.install_errors import NeedsInput
    with pytest.raises(NeedsInput, match="use off, or the number") as error:
        install_workflow.retain_preference(SimpleNamespace(env=path))
    assert error.value.field == "retain_deployments"
    path.write_text("SPARKRING_RETAIN_DEPLOYMENTS=3\n")
    assert install_workflow.retain_preference(SimpleNamespace(env=path)) == "3"
    assert install_workflow.retain_preference(SimpleNamespace(env=None)) is None
    assert retention.setting(tmp_path) == (2, "2")
    assert retention.save_setting(tmp_path, "Off") is None
    assert retention.setting(tmp_path) == (None, "off")
    (tmp_path / retention.SETTING_FILE).write_text("{")
    with pytest.raises(ValueError, match="automatic release does not run until"):
        retention.setting(tmp_path)


def test_the_summary_names_what_was_released():
    base = {"released": [], "freed_bytes": 0, "errors": [], "caches": 0, "remainders": 0}
    gib = 1024 ** 3
    assert retention.summary({**base, "released": [f"d{n}" for n in range(7)], "freed_bytes": int(9.8 * gib),
                              "caches": 3}) == ["Released 7 older deployments' containers, workspaces and caches: 9.8 GiB"]
    assert retention.summary({**base, "released": ["d"], "freed_bytes": int(1.5 * gib)}) == [
        "Released 1 older deployment's containers, workspaces and caches: 1.5 GiB"]
    assert retention.summary({**base, "caches": 2, "freed_bytes": 4 * gib}) == [
        "Released 2 unused compile caches: 4.0 GiB"]
    assert retention.summary({**base, "sources": ["2" * 40], "freed_bytes": 80 * 10 ** 6}) == [
        "Released Node A's checkouts of 1 older SparkRing source: 80.0 MB"]
    assert retention.summary(base) == []
    assert [retention.reason_text("recent", retain) for retain in (1, 3)] == [
        "most recent of its profile", "3 most recent of its profile"]
    errors = [f"Node 1 spark-931e: refusal {n}" for n in range(7)]
    assert retention.summary({**base, "released": ["d"], "freed_bytes": gib, "errors": errors}) == [
        "Released 1 older deployment's containers, workspaces and caches: 1.0 GiB",
        "Some data stays on the Sparks; sudo sparkring storage lists it:",
        *("  " + error for error in errors[:5]), "  and 2 more"]
    assert retention.summary({**base, "errors": errors[:1]}) == [
        "Older deployments were not released; sudo sparkring storage lists it:", "  " + errors[0]]


# Release on the fake Spark -------------------------------------------------------------

class Docker(fake.Docker):
    """Canned Docker that also inspects containers by name with sizes and removes stopped ones."""

    def __init__(self, *arguments, **options):
        super().__init__(*arguments, **options)
        self.removed = []

    def __call__(self, argv, **kwargs):
        command = argv[3:]
        if command[:2] == ["container", "inspect"] and "--size" in command:
            wanted = [value for value in command[2:] if value != "--size"]
            found = [item for item in self.containers if item["Id"] in wanted or item["Name"][1:] in wanted]
            if not found:
                return SimpleNamespace(returncode=1, stdout="[]", stderr=f"Error: No such container: {wanted[0]}\n")
            return SimpleNamespace(returncode=0, stdout=json.dumps(found), stderr="")
        if command[:2] == ["container", "rm"]:
            assert command[2] == "--volumes"
            [item] = [item for item in self.containers if item["Id"] == command[3]]
            if item["State"]["Running"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="Error: cannot remove a running container\n")
            self.containers.remove(item)
            self.removed.append(item["Name"][1:])
            return SimpleNamespace(returncode=0, stdout=command[3] + "\n", stderr="")
        return super().__call__(argv, **kwargs)


def model_container(name, deployment, *, running=False, size=400 * 10 ** 6):
    item = fake.container(name, running=running)
    item.update(Id=identity(name), SizeRw=size)
    item["Config"]["Labels"] = {storage.DEPLOYMENT_LABEL: deployment, storage.RANK_LABEL: "0"}
    item["State"]["Status"] = "running" if running else "exited"
    return item


def deployment(state, spark, name, *, profile=PROFILE, operation="down", complete=True, age=0, caches=None,
               models=False, revision="1" * 40):
    """A deployment on Node A and its workspace on the fake Spark; returns ``(lock ID, container name)``."""
    identifier = identity(name)
    path = spark.cluster / name
    fake.workspace(path, identifier)
    if models:
        adopt(path / "models" / REVISION, spark.folder)
    site = "tp4-" + name.rsplit("-", 1)[-1]
    rows = [{"rank": 0, "host": HOSTS[0], "model": spark.owned, "cache": str(spark.cache),
             "repository": f"{path}/source-111111111111", "deployment_root": f"{path}/containers",
             "reuse_verified_model": False}]
    directory = state / "deployments" / name
    write_json(directory / "deployment.lock.json",
               {"id": identifier, "backend": "compose", "selection": {"image_id": DEV_IMAGE, "profile": profile},
                "source_revision": revision, "site": {"name": site, "workspace": str(path), "ranks": rows}})
    write_json(directory / "rank0" / "container.json",
               fake.spec(str(path), spark.owned, str(spark.cache), caches or [INSTALLED_CACHE, COMPILE_CACHE]))
    if operation is not None:
        write_json(directory / "state.json", {"generation": 3, "operation": operation, "complete": complete})
        moment = time.time() - age
        os.utime(directory / "state.json", (moment, moment))
    return identifier, f"sr-{site}-r0"


class Invoke:
    """Node A's SSH calls to the fake Spark, answered by storage's node functions."""

    def __init__(self, spark, docker):
        self.spark, self.docker, self.calls = spark, docker, []

    def __call__(self, host, argv, *, data=None, timeout=None):
        request = json.loads(data)
        self.calls.append((argv, request))
        assert host == HOSTS[0] and timeout
        if argv == LIST:
            return json.dumps(storage.list_local(request, root=self.spark.root, run=self.docker))
        assert argv == retention.BATCH
        return json.dumps(storage.release_batch_local(request, root=self.spark.root, run=self.docker, table=[]))


def ring(tmp_path, spark):
    """Node A's deployments on the fake Spark, in order of their last operation, and the Docker that holds them.

    ACTIVE serves; ROLLBACK is the rollback target; RECENT is the second most
    recent of the profile after ACTIVE; OLD and OLDER are older. OLDER's
    workspace holds model files. OTHER is the only deployment of another
    profile, PLANNED never ran an operation and UNFINISHED's last start did
    not complete.
    """
    state = tmp_path / "controller"
    write_json(state / "cluster.json", {"name": CLUSTER, "plan": {"spec": {"hosts": [{"host": HOSTS[0]}]}}})
    made = {
        "active": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa1", operation="up", age=10),
        "recent": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa2", age=20),
        "rollback": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa3", age=30),
        "old": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa4", age=40, caches=[STALE_CACHE, COMPILE_CACHE],
                          revision=OLD_REVISION),
        "older": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa5", age=50, models=True),
        "unfinished": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa6", operation="up", complete=False, age=60),
        "planned": deployment(state, spark, PROFILE + "-iaaaaaaaaaaa7", operation=None),
        "other": deployment(state, spark, OTHER + "-iaaaaaaaaaaa8", profile=OTHER, age=5000)}
    write_json(state / "active.json", {"path": str(state / "deployments" / (PROFILE + "-iaaaaaaaaaaa1"))})
    write_json(state / "transaction.json", {"previous": str(state / "deployments" / (PROFILE + "-iaaaaaaaaaaa3")),
                                            "candidate": str(state / "deployments" / (PROFILE + "-iaaaaaaaaaaa1")),
                                            "state": "complete"})
    containers = [model_container(name, identifier, running=key == "active") for key, (identifier, name) in made.items()
                  if key != "planned"]
    docker = Docker(images=spark.docker.images, containers=[*spark.docker.containers, *containers])
    return state, made, docker


@linux
def test_release_removes_older_deployments_containers_workspaces_and_caches_and_keeps_the_rest(tmp_path, spark, capsys):
    state, made, docker = ring(tmp_path, spark)
    invoke = Invoke(spark, docker)
    names = {key: PROFILE + f"-iaaaaaaaaaaa{number}" for number, key in enumerate(made, 1) if key != "other"}
    old, older = spark.cluster / names["old"], spark.cluster / names["older"]
    _, frees_old = fake.du(old)
    _, frees_stale = fake.du(spark.cache / STALE_CACHE, outside=[spark.srv / "image020/graph-copy.bin"])
    _, frees_leftover = fake.du(spark.cache / f".{LEFTOVER_CACHE}.sparkring-releasing")
    # Node A's checkouts of deployment sources: only OLD uses OLD_REVISION's, and a removal of another stopped
    # after its rename.
    sources = state / retention.SOURCES
    leftover = "." + "3" * 40 + ".removing"
    for name in (OLD_REVISION, "1" * 40, "notes", leftover):
        fake.put(sources / name / "README.md", 5000)
    frees_source = fake.du(sources / OLD_REVISION)[1] + fake.du(sources / leftover)[1]
    checkpoint_before = tree_state(Path(spark.owned).parent)
    # sparkring storage names what the next release removes.
    view = storage.list_cluster(state, invoke)["retention"]
    assert [entry["containers"] for entry in view["nodes"]] == [
        [{"deployment": made[key][0], "name": made[key][1], "bytes": 400 * 10 ** 6} for key in ("old", "older")]]
    assert [line for line in retention.describe(view) if "next install" in line] == [
        "    The next install or up releases the containers, workspaces and caches of 2 older deployments: "
        + storage._measured(view["frees_bytes"], view["complete"])]
    lines = []
    result = retention.release(state, invoke, 2, write=lines.append)

    # OLD and OLDER are released: their stopped containers on the Spark, OLD's workspace and the cache only OLD used.
    assert sorted(docker.removed) == sorted([made["old"][1], made["older"][1]])
    assert not os.path.lexists(old) and not os.path.lexists(spark.cache / STALE_CACHE)
    assert not os.path.lexists(spark.cache / f".{LEFTOVER_CACHE}.sparkring-releasing")
    # Kept: the active, rollback, recent, unfinished and other-profile deployments, and what they use.
    for key in ("active", "recent", "rollback", "unfinished", "planned", "other"):
        assert (spark.cluster / (names.get(key) or OTHER + "-iaaaaaaaaaaa8")).is_dir(), key
    assert {item["Name"][1:] for item in docker.containers} >= {made[key][1] for key in
                                                               ("active", "recent", "rollback", "unfinished", "other")}
    for name in (INSTALLED_CACHE, COMPILE_CACHE, PROFILE_CACHE):
        assert (spark.cache / name).is_dir()
    # Never released: checkpoint directories, OLDER's workspace with model files, and items it did not create.
    assert tree_state(Path(spark.owned).parent) == checkpoint_before
    assert older.is_dir() and (older / "models" / REVISION).is_dir()
    assert (spark.srv / "image020").is_dir() and (spark.cache / "jit").is_dir()

    assert result["state"] == "released" and result["released"] == sorted([names["old"], names["older"]])
    assert result["sources"] == [OLD_REVISION] and sorted(path.name for path in sources.iterdir()) == ["1" * 40, "notes"]
    assert result["freed_bytes"] == frees_old + frees_stale + frees_leftover + frees_source + 2 * 400 * 10 ** 6
    assert lines == [f"Released 2 older deployments' containers, workspaces and caches: "
                     f"{storage._size(result['freed_bytes'])}"]
    # Each Spark was told what every kept deployment uses, to check each path again before removing it.
    [(_, request)] = [call for call in invoke.calls if call[0] == retention.BATCH]
    assert {str(spark.cluster / names[key]) for key in ("active", "recent", "rollback", "unfinished")} <= set(
        request["in_use"])
    assert request["containers"] == [{"deployment": made[key][0], "name": made[key][1]} for key in ("old", "older")]
    # OLDER keeps its workspace, so its release is not complete and the next release lists the Sparks again.
    for key, complete in (("old", True), ("older", False)):
        record = installer.read(state / "deployments" / names[key] / retention.RELEASED_FILE)
        assert record == {"schema": retention.RELEASED_SCHEMA, "generation": 3, "complete": complete,
                          "removed": True}
    invoke.calls.clear()
    assert retention.release(state, invoke, 2, write=lines.append)["state"] == "nothing"
    assert [argv for argv, _ in invoke.calls] == [LIST]
    # Once OLDER's workspace is gone too, the next release asks no Spark.
    shutil.rmtree(older)
    retention.release(state, invoke, 2, write=lines.append)
    invoke.calls.clear()
    assert retention.release(state, invoke, 2, write=lines.append)["state"] == "nothing" and invoke.calls == []
    adopt(older / "models" / REVISION, spark.folder)
    fake.workspace(older, made["older"][0])

    # sparkring storage shows what the policy keeps and why.
    assert storage.main([], state_root=state, invoke=invoke) == 0
    printed = capsys.readouterr().out.splitlines()
    start = printed.index("Deployments on Node A: 8, 1 never started")
    width = len(OTHER + "-iaaaaaaaaaaa8")
    assert printed[start + 1:start + 7] == [
        f"    kept  {OTHER + '-iaaaaaaaaaaa8':<{width}}  2 most recent of its profile",
        f"    kept  {names['active']:<{width}}  active, model container running, started and not stopped, 2 most "
        "recent of its profile",
        f"    kept  {names['recent']:<{width}}  2 most recent of its profile",
        f"    kept  {names['rollback']:<{width}}  rollback target",
        f"    kept  {names['unfinished']:<{width}}  last operation incomplete",
        f"    1 older deployment keeps a workspace or container that automatic release leaves on the Sparks; the "
        f"details above say why: {names['older']}"]
    assert printed[start + 7] == retention.setting_text(2)


@linux
def test_release_never_removes_a_checkpoint_or_what_an_installed_mesh_uses(tmp_path, spark):
    state, made, docker = ring(tmp_path, spark)
    names = {key: PROFILE + f"-iaaaaaaaaaaa{number}" for number, key in enumerate(made, 1)}
    old = spark.cluster / names["old"]
    # A mesh installed on the Spark names OLD's workspace and starts OLD's container.
    fake.install_mesh(spark.root, "tp4-aaaaaaaaaaa4", str(old), model=spark.owned, cache=str(spark.cache))
    service = Path(spark.root) / "etc/sparkring/deployments/tp4-aaaaaaaaaaa4/service.json"
    write_json(service, {**json.loads(service.read_text()), "container_id": identity(made["old"][1])})
    before = tree_state(Path(spark.owned).parent)
    # With no recent deployment kept, OLD stays for the mesh alone.
    result = retention.release(state, Invoke(spark, docker), 0, write=lambda line: None)
    assert old.is_dir() and made["old"][1] not in docker.removed
    assert names["old"] not in result["released"] and names["recent"] in result["released"]
    assert tree_state(Path(spark.owned).parent) == before
    view = storage.list_cluster(state, Invoke(spark, docker))["retention"]
    assert {entry["name"]: entry["reasons"] for entry in view["deployments"]}[names["old"]] == ["mesh"]

    # Each Spark refuses on its own: the container that the mesh starts, a running one and one of another deployment.
    meshes = storage.installed_meshes(spark.root)
    refusals = [({"deployment": made["old"][0], "name": made["old"][1]}, "The installed mesh .* starts"),
                ({"deployment": made["active"][0], "name": made["active"][1]}, "is running; SparkRing removes only"),
                ({"deployment": made["old"][0], "name": made["rollback"][1]}, "does not carry deployment"),
                ({"deployment": made["old"][0], "name": "scratch"}, "is not a SparkRing model container")]
    for entry, message in refusals:
        with pytest.raises(ValueError, match=message):
            storage.remove_container(entry, run=docker, meshes=meshes)
    assert storage.remove_container({"deployment": made["old"][0], "name": "sr-absent-r0"}, run=docker,
                                    meshes=meshes)["state"] == "absent"
    # A workspace stays when its deployment's container on that Spark was not removed.
    answer = storage.release_batch_local({"cluster": CLUSTER, "containers": [{"deployment": made["old"][0],
                                                                               "name": made["old"][1]}],
                                          "paths": [str(old), str(spark.owned)]}, root=spark.root, run=docker, table=[])
    assert "starts" in answer["containers"][0]["error"]
    assert answer["paths"][0] == {"path": str(old), "error": f"{old} stays because its deployment's model container on "
                                                             "this Spark was not removed"}
    assert "is a SparkRing checkpoint directory" in answer["paths"][1]["error"]
    assert old.is_dir() and tree_state(Path(spark.owned).parent) == before


@linux
def test_the_off_switch_and_a_preference_stop_automatic_release(tmp_path, spark, capsys, monkeypatch):
    state, made, docker = ring(tmp_path, spark)
    invoke = Invoke(spark, docker)
    before = tree_state(spark.srv)
    assert retention.after_operation(state, invoke, preference="off", write=print) == {
        "schema": retention.RESULT_SCHEMA, "state": "off"}
    assert invoke.calls == [] and tree_state(spark.srv) == before and docker.removed == []
    assert retention.setting(state) == (None, "off")
    # sparkring storage reports the setting and changes it.
    assert storage.main([], state_root=state, invoke=invoke) == 0
    assert retention.setting_text(None) in capsys.readouterr().out.splitlines()
    assert storage.main(["--retain-deployments", "3", "--json"], state_root=state, invoke=invoke) == 0
    assert json.loads(capsys.readouterr().out) == {"schema": storage.SCHEMA, "state": "saved", "retain_deployments": 3}
    assert retention.setting(state) == (3, "3")
    assert storage.main(["--retain-deployments", "off"], state_root=state, invoke=invoke) == 0
    assert capsys.readouterr().out.splitlines() == [retention.setting_text(None)]
    invoke.calls.clear()
    assert retention.after_operation(state, invoke, write=print)["state"] == "off" and invoke.calls == []
    assert storage.main(["--retain-deployments", "some"], state_root=state, invoke=invoke) == 2
    assert "use off, or the number of recent deployments" in capsys.readouterr().err
    # A failure is reported and never raised: the model operation before it completed.
    retention.save_setting(state, "2")
    monkeypatch.setattr(storage, "_survey", lambda *a, **k: [{"rank": 0, "host": HOSTS[0], "error": "timed out"}])
    result = retention.after_operation(state, invoke, write=print)
    assert result["state"] == "failed" and tree_state(spark.srv) == before
    assert capsys.readouterr().out == (f"Older deployments were not released: Node 0 {HOSTS[0]} could not be listed: "
                                       "timed out; nothing was released. sudo sparkring storage lists what they hold.\n")
    # A Spark whose Docker cannot be read lists no container, so a running one would look stopped.
    monkeypatch.setattr(storage, "_survey", lambda *a, **k: [{
        "rank": 0, "host": HOSTS[0], "hostname": "spark-aa42", "package_revision": checkpoints.package_revision(),
        "items": [],
        "model_containers": [], "docker": {"root": None, "images": [], "error": "permission denied"}}])
    assert retention.after_operation(state, invoke, write=print)["state"] == "failed"
    assert f"Docker on Node 0 {HOSTS[0]} could not be read: permission denied; nothing was released" in \
        capsys.readouterr().out
    # A Spark running another package lacks the batch release, and an earlier one lists no model containers.
    for other in ({"package_revision": "b" * 40}, {"model_containers": None}):
        monkeypatch.setattr(storage, "_survey", lambda *a, **k: [{
            "rank": 0, "host": HOSTS[0], "hostname": "spark-aa42", "package_revision": checkpoints.package_revision(),
            "items": [], "model_containers": [], "docker": {"root": None, "images": [], "error": None}, **other}])
        assert retention.after_operation(state, invoke, write=print)["state"] == "failed"
        assert "the Sparks do not all run Node A's SparkRing package revision; nothing was released" in \
            capsys.readouterr().out
    assert tree_state(spark.srv) == before


def test_down_of_a_released_deployment_reports_it_instead_of_verifying_a_removed_workspace(tmp_path, monkeypatch,
                                                                                           capsys):
    from runtime.host import retained_source
    monkeypatch.setattr(controller, "STATE", tmp_path)
    calls = []
    monkeypatch.setattr(retained_source, "review", lambda directory, operation, **k: calls.append("review-" + operation)
                        or {"profile": PROFILE, "hosts": [HOSTS[0]], "phases": ["owned", "stop"]})
    monkeypatch.setattr(retained_source, "apply", lambda directory, operation, **k: calls.append(operation) or {})
    directory = tmp_path / "deployments" / (PROFILE + "-iaaaaaaaaaaa4")
    write_json(directory / "deployment.lock.json", {"selection": {"profile": PROFILE}})
    write_json(directory / "state.json", DOWN)
    write_json(directory / retention.RELEASED_FILE, {"generation": 3, "complete": True, "removed": True})
    assert controller.lifecycle(["down", PROFILE, "--instance", "iaaaaaaaaaaa4", "--execute"]) == 0
    assert calls == [] and capsys.readouterr().out == (
        f"{directory.name} is stopped, and automatic release removed its containers and workspaces from the Sparks; "
        f"there is nothing to stop. sudo sparkring up {PROFILE} --instance iaaaaaaaaaaa4 starts it again.\n")
    # A release that did not finish on every Spark still removed the workspace that the stop would read somewhere.
    write_json(directory / retention.RELEASED_FILE, {"generation": 3, "complete": False, "removed": True})
    assert controller.lifecycle(["down", PROFILE, "--instance", "iaaaaaaaaaaa4", "--execute"]) == 0
    assert calls == [] and "removed its containers and workspaces from some Sparks, and sudo sparkring storage " \
        "lists what stays; there is nothing to stop." in capsys.readouterr().out
    # Only a completed down is released: a completed preparation resumes its receipt when it is repeated.
    write_json(directory / "state.json", {**DOWN, "operation": "prepare"})
    assert retention.release_record(directory) is None
    # Once another operation ran, the record no longer applies and down runs the deployment's own stop.
    write_json(directory / "state.json", {**DOWN, "generation": 4})
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: None)
    assert controller.lifecycle(["down", PROFILE, "--instance", "iaaaaaaaaaaa4", "--execute"]) == 0
    assert calls == ["review-down", "down"]
    assert retention.release_record(directory) is None
    write_json(directory / "state.json", {"generation": 3, "operation": "up", "complete": True})
    assert retention.release_record(directory) is None


def test_up_releases_older_deployments_only_after_it_completes(tmp_path, monkeypatch, capsys):
    from runtime.host import recovery, retained_source
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: None)
    monkeypatch.setattr(recovery, "started", lambda directory, enabled=None: None)
    monkeypatch.setattr(retained_source, "review", lambda directory, operation, **k: {
        "profile": PROFILE, "hosts": [HOSTS[0]], "phases": ["source", "start"]})
    outcome = {"complete": True}
    monkeypatch.setattr(retained_source, "apply", lambda directory, operation, **k: dict(outcome))
    released = []
    monkeypatch.setattr(retention, "after_operation", lambda state_root, invoke, **k: released.append(state_root) or {
        "state": "nothing"})
    directory = tmp_path / "deployments" / PROFILE
    write_json(directory / "deployment.lock.json", {"selection": {"profile": PROFILE}, "site_input": {}})
    assert controller.lifecycle(["up", PROFILE, "--execute", "--json"]) == 0
    # The steps printed before the result precede its JSON document.
    out = capsys.readouterr().out
    assert released == [tmp_path] and json.loads(out[out.index("{"):])["retention"] == {"state": "nothing"}
    outcome["complete"] = False
    assert controller.lifecycle(["up", PROFILE, "--execute"]) == 0
    assert released == [tmp_path]


def test_node_routes_a_batch_release(monkeypatch, capsys):
    from scripts import sparkring_node
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(storage, "release_batch_local", lambda request: {"batch": request})
    request = {"cluster": CLUSTER, "paths": ["/srv/sparkring/tp4/cache/x"], "containers": []}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert sparkring_node.main(["storage", "--release-batch"]) == 0
    assert json.loads(capsys.readouterr().out) == {"batch": request}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert sparkring_node.main(["storage", "--release-batch", "--release", "/srv/sparkring/tp4/cache/x"]) == 2
    assert "Give --release PATH or --release-batch, not both" in capsys.readouterr().err


def test_model_containers_carry_writable_layer_sizes_when_docker_answers_in_time():
    import subprocess
    stopped, running = model_container("sr-tp4-a-r0", "a" * 64), model_container("sr-tp4-b-r0", "b" * 64, running=True)
    unlabeled = fake.container("scratch", running=False)

    def run(argv, **kwargs):
        assert argv[-2:] == ["--size", stopped["Id"]] and kwargs["timeout"] == storage.SIZE_SECONDS
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"Id": stopped["Id"], "SizeRw": 5}]), stderr="")
    listed = storage.model_containers([running, unlabeled, stopped], run=run, sizes=True)
    assert [(entry["name"], entry["deployment"], entry["running"], entry["bytes"]) for entry in listed] == [
        ("sr-tp4-a-r0", "a" * 64, False, 5), ("sr-tp4-b-r0", "b" * 64, True, None)]

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    assert storage.model_containers([stopped], run=slow, sizes=True)[0]["bytes"] is None
    assert storage.model_containers([stopped], run=lambda *a, **k: pytest.fail("sizes"), sizes=False)[0]["bytes"] is None


@linux
def test_a_remainder_is_released_without_the_directory_now_at_its_path(spark):
    path = spark.cache / LEFTOVER_CACHE
    hidden = spark.cache / f".{LEFTOVER_CACHE}.sparkring-releasing"
    fake.put(path / "vllm/graph.bin", 3000)
    answer = storage.release_batch_local({"cluster": CLUSTER, "remainders": [str(path)]}, root=spark.root,
                                         run=Docker(), table=[])
    assert answer == {"containers": [], "paths": [{"path": str(path), "state": "released", "kind": "releasing",
                                                   "freed_bytes": answer["paths"][0]["freed_bytes"]}]}
    assert not hidden.exists() and (path / "vllm/graph.bin").is_file()
    assert storage.release_batch_local({"cluster": CLUSTER, "remainders": [str(path)]}, root=spark.root,
                                       run=Docker(), table=[])["paths"][0]["state"] == "absent"


def test_the_policy_never_releases_a_derived_checkpoint_or_its_base_even_after_its_deployment_is_released():
    """A released deployment's workspace and caches go; the derived and base directories stay for a later install."""
    old = record("old-444444444444", age=100)
    derived = f"/srv/sparkring/{CLUSTER}/checkpoints/sparkring-derived--Fixture/{'f' * 40}"
    base = f"/srv/sparkring/{CLUSTER}/checkpoints/x/{REVISION}"
    old["references"][HOSTS[0]]["paths"] += [derived, base]
    records = [record("active-aaaaaaaaaaaa", age=10), record("recent-bbbbbbbbbbbb", age=20),
               record("older-555555555555", age=30), old]
    extra = [{"path": path, "kind": "checkpoint", "class": "unreferenced", "meshes": [], "frees_bytes": 6 * 10 ** 9,
              "complete": True, "release": "sudo sparkring checkpoints --release " + path,
              **({"derived": {"name": "fixture-mxfp8"}} if path == derived else {})} for path in (derived, base)]
    decided = retention.plan(records, roles(active="active-aaaaaaaaaaaa"), [listing(records, extra=extra)], 2)
    assert "old-444444444444" in decided["released"]
    paths = {entry["path"] for entry in decided["nodes"][0]["paths"]}
    assert old["workspace"] in paths and not {derived, base} & paths
