"""`sparkring checkpoints`: listing and releasing SparkRing checkpoint directories.

Tests marked ``linux`` build real checkpoint directories with
``runtime/host/checkpoint_place.py`` (hard links, ``/proc/self/fd``, ``flock``)
under a synthetic ext4 mount table. The others drive Node A's command with
canned per-Spark answers.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time

import pytest

from runtime.host import checkpoint_place as place
from runtime.host import checkpoints

REPOSITORY = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REVISION = "629bc3218833a38b475b719f34aa571666f4a03e"
SLUG = "local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
DIRECTORY = f"/srv/sparkring/tp2/checkpoints/{SLUG}/{REVISION}"
HOSTS = ["root@192.0.2.10", "root@192.0.2.11"]
WEIGHT = "model-00001-of-00001.safetensors"
FILES = {WEIGHT: b"w" * 8192, "config.json": b'{"model_type": "qwen"}', "tokenizer/vocab.json": b'{"a": 1}'}
LIST = ["sudo", "-n", "/usr/bin/sparkring", "node", "checkpoints"]
linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="checkpoint directories use Linux hard links, /proc/self/fd, O_NOFOLLOW and flock")


# Each Spark's check for running containers, kept before the fixture below replaces it.
RUNNING_CONTAINERS = checkpoints.running_containers


@pytest.fixture(autouse=True)
def administrator(monkeypatch):
    """Node A's command runs under sudo; the tests run it as the invoking account, with no container running."""
    monkeypatch.setattr(checkpoints, "_administrator", lambda: True)
    monkeypatch.setattr(checkpoints, "running_containers", lambda path, **kwargs: [])


@pytest.fixture
def host_records(tmp_path, monkeypatch):
    """Synthetic ext4 mount table and private record directories for real placements."""
    table = tmp_path / "mountinfo"
    table.write_text("29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
    monkeypatch.setattr(place, "MOUNTINFO", str(table))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records/files"))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS", str(tmp_path / "records"))
    (tmp_path / "records").mkdir()
    return tmp_path / "records"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def user_copy(folder):
    """An operator's plain checkpoint folder holding every fixture file."""
    for name, content in FILES.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return folder


def adopt(path, folder):
    """Fill SparkRing checkpoint directory ``path`` as model adoption does: link the weight, copy the rest."""
    with place.claim(str(path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        receive = place.staging(claimed, "receive", empty=True)
        try:
            for name, content in FILES.items():
                source = str(folder / name)
                fd, before = place.open_source(source, None, len(content))
                try:
                    if name == WEIGHT:
                        sha = place.hash_descriptor(fd, len(content))
                        place.place_link(claimed.dir_fd, fd, name, journal, sha256=sha, before=before, source=source)
                    else:
                        part, sha = place.copy_part(fd, len(content), receive, name + ".part")
                        place.place_staged(claimed.dir_fd, receive, name + ".part", name, journal, fd=part,
                                           sha256=sha, origin="copy", source=source)
                        os.close(part)
                finally:
                    os.close(fd)
        finally:
            os.close(receive)
    return str(path)


def tree_state(root):
    """Every entry below ``root`` with mode, owner, size, inode, link count, modification time and content."""
    state = {}
    for directory, directories, files in os.walk(root):
        for name in directories + files:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            content = Path(path).read_bytes() if stat.S_ISREG(info.st_mode) else None
            state[os.path.relpath(path, root)] = (info.st_mode, info.st_uid, info.st_size, info.st_ino,
                                                  info.st_nlink, info.st_mtime_ns, content)
    return state


def five(path):
    return place.stats(os.lstat(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def controller_state(tmp_path, deployments, *, hosts=HOSTS, active=None, transaction=None):
    """Node A's controller directory: the cluster record and one lock per retained deployment.

    ``deployments`` maps a deployment name to ``(model, reuse, workspace)``;
    every rank row names that model.
    """
    state = tmp_path / "controller"
    write_json(state / "cluster.json", {"name": "tp2", "plan": {"spec": {"hosts": [{"host": host} for host in hosts]}}})
    for name, (model, reuse, workspace) in deployments.items():
        write_json(state / "deployments" / name / "deployment.lock.json",
                   {"id": "id-" + name, "site": {"workspace": workspace,
                                                  "ranks": [{"host": host, "model": model, "reuse_verified_model": reuse}
                                                            for host in hosts]}})
    if active:
        write_json(state / "active.json", {"path": str(state / "deployments" / active)})
    if transaction:
        write_json(state / "transaction.json", {key: str(state / "deployments" / value) if key in ("candidate", "previous") else value
                                                for key, value in transaction.items()})
    return state


def local_listing(directories=(), revision="a" * 40, hostname="spark-aa42"):
    return {"schema": checkpoints.LOCAL_SCHEMA, "hostname": hostname, "package_revision": revision,
            "directories": list(directories)}


def listed(path=DIRECTORY, **values):
    entry = {"path": path, "repository": REPOSITORY, "revision": REVISION, "state": "ok", "files": 48,
             "bytes": 105895988120, "frees_bytes": 56497920, "shared_bytes": 105839492200, "staging_bytes": 0,
             "shared": [{"path": "/var/tmp/models/qwen", "files": 36, "bytes": 105839492200}]}
    return {**entry, **values}


class Spark:
    """Canned ``discovery.ssh``: per-host listings, and recorded release requests."""

    def __init__(self, listings, released=None):
        self.listings, self.released, self.calls = listings, released or {}, []

    def __call__(self, host, argv, *, data=None, timeout=None):
        self.calls.append((host, argv, json.loads(data) if data else None))
        if argv == LIST:
            answer = self.listings[host]
            if isinstance(answer, Exception):
                raise answer
            return json.dumps(answer)
        assert argv[:-1] == LIST + ["--release"]
        return json.dumps(self.released.get(host, {"path": argv[-1], "state": "released", "unlinked": 48,
                                                   "freed_bytes": 56497920, "kept_bytes": 105839492200,
                                                   "refreshed": []}))

    def releases(self):
        return [(host, argv[-1], request) for host, argv, request in self.calls if argv != LIST]


@linux
def test_listing_names_deployments_and_shared_user_copies(tmp_path, host_records, capsys):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    # A site-file deployment keeps its checkpoint outside /srv/sparkring; its lock names the path.
    site = adopt(tmp_path / "site/qwen/models" / REVISION, folder)
    state = controller_state(tmp_path, {
        "qwen-tp2-iaaaa": (owned, False, "/srv/sparkring/tp2/qwen-tp2-iaaaa"),
        "qwen-tp2-ibbbb": (owned, False, "/srv/sparkring/tp2/qwen-tp2-ibbbb"),
        "qwen-site": (site, False, str(tmp_path / "site/qwen")),
        "qwen-in-place": (str(folder), True, "/srv/sparkring/tp2/qwen-in-place")}, active="qwen-tp2-iaaaa")

    def spark(host, argv, *, data=None, timeout=None):
        assert argv == LIST and timeout
        request = json.loads(data)
        if host == HOSTS[0]:
            assert request["paths"] == sorted([owned, site])
            return json.dumps(checkpoints.list_local(request["paths"], root=str(tmp_path)))
        return json.dumps(checkpoints.list_local(root=str(tmp_path / "empty")))

    assert checkpoints.main([], state_root=state, invoke=spark) == 0
    printed = capsys.readouterr().out.splitlines()
    hostname = socket.gethostname()
    start = printed.index("    " + owned)
    assert printed[start + 1:start + 5] == [
        f"        {REPOSITORY} at {REVISION[:12]}: 3 files, 8.22 KB",
        "        used by qwen-tp2-iaaaa (active), qwen-tp2-ibbbb",
        f"        shares 1 file (8.19 KB) with {folder}; deleting that copy does not free these 8.19 KB while this "
        "directory exists",
        "        releasing it frees 30 bytes"]
    assert printed[printed.index("    " + site) + 2] == "        used by qwen-site"
    assert printed[printed.index(f"Node 1 {hostname}") + 1] == "    no SparkRing checkpoint directories"
    assert not any("qwen-in-place" in line for line in printed)

    # The same facts as JSON; a copy that no longer holds the linked inode is no longer named.
    assert checkpoints.main(["--json"], state_root=state, invoke=spark) == 0
    [entry] = [item for item in json.loads(capsys.readouterr().out)["nodes"][0]["directories"] if item["path"] == owned]
    weight, others = len(FILES[WEIGHT]), sum(len(content) for name, content in FILES.items() if name != WEIGHT)
    assert entry["files"] == 3 and entry["state"] == "ok" and entry["bytes"] == weight + others
    assert entry["shared"] == [{"path": str(folder), "files": 1, "bytes": weight}]
    assert (entry["shared_bytes"], entry["frees_bytes"]) == (weight, others)
    assert entry["deployments"] == [{"name": "qwen-tp2-iaaaa", "role": "active"}, {"name": "qwen-tp2-ibbbb"}]
    os.unlink(folder / WEIGHT)
    os.unlink(Path(site) / WEIGHT)
    entry = checkpoints.list_local([owned], root=str(tmp_path / "empty"))["directories"][0]
    assert entry["shared"] == [] and entry["frees_bytes"] == weight + others


@pytest.mark.parametrize("role, transaction, message", [
    ("active", None, "the active deployment qwen-tp2-iaaaa"),
    ("rollback", {"candidate": "qwen-tp2-ibbbb", "previous": "qwen-tp2-iaaaa", "state": "complete"},
     "the rollback target recorded in transaction.json qwen-tp2-iaaaa"),
    ("switching", {"candidate": "qwen-tp2-iaaaa", "previous": "glm-tp2-icccc", "state": "verifying"},
     "the deployment of an unfinished model switch qwen-tp2-iaaaa"),
])
def test_release_refuses_the_active_and_rollback_directories(tmp_path, capsys, role, transaction, message):
    deployments = {"qwen-tp2-iaaaa": (DIRECTORY, False, "/srv/sparkring/tp2/qwen-tp2-iaaaa"),
                   "qwen-tp2-ibbbb": ("/srv/sparkring/tp2/checkpoints/other/" + "b" * 40, False, "/srv/sparkring/tp2/b"),
                   "glm-tp2-icccc": ("/srv/sparkring/tp2/checkpoints/glm/" + "c" * 40, False, "/srv/sparkring/tp2/c")}
    state = controller_state(tmp_path, deployments, active="qwen-tp2-iaaaa" if role == "active" else "glm-tp2-icccc",
                             transaction=transaction)
    spark = Spark({host: local_listing([listed()]) for host in HOSTS})
    assert checkpoints.main(["--release", DIRECTORY, "--yes"], state_root=state, invoke=spark) == 2
    assert f"{DIRECTORY} is the checkpoint directory of {message}; SparkRing does not release it" in capsys.readouterr().err
    assert spark.calls == []

    # Once the switch has settled and another deployment is active, the directory can be released.
    state = controller_state(tmp_path / "settled", deployments, active="glm-tp2-icccc",
                             transaction={"candidate": "glm-tp2-icccc", "previous": "qwen-tp2-ibbbb",
                                          "state": "failed-recovered"})
    assert checkpoints.main(["--release", DIRECTORY, "--yes", "--json"], state_root=state, invoke=spark) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "released" and result["deployments"] == ["qwen-tp2-iaaaa"]
    assert [host for host, path, _ in spark.releases() if path == DIRECTORY] == HOSTS
    workspaces = {item["workspace"] for _, _, request in spark.releases() for item in request["receipts"]}
    assert workspaces == {"/srv/sparkring/tp2/qwen-tp2-iaaaa", "/srv/sparkring/tp2/b", "/srv/sparkring/tp2/c"}


@linux
def test_release_removes_only_journaled_names_and_refreshes_other_receipts(tmp_path, host_records, capsys):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    untouched = tree_state(folder)
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    state_directory = Path(owned).parent / ("." + REVISION + ".sparkring")
    linked = str(folder / WEIGHT)
    hashes = {name: digest(content) for name, content in FILES.items()}

    def receipt(workspace, deployment, stats):
        write_json(workspace / ".installer-owner.json", {"deployment": deployment})
        write_json(workspace / "installer/model.json",
                   {"repository": REPOSITORY, "revision": REVISION, "path": str(folder), "files": hashes,
                    "file_stats": stats, "origin": "in-place-verified-copy"})
        return workspace / "installer/model.json"

    current = {name: five(str(folder / name)) for name in FILES}
    # A deployment serving the copy in place, whose receipt SparkRing's link already refreshed.
    served = receipt(tmp_path / "srv/sparkring/tp2/in-place", "id-qwen-in-place", current)
    # A receipt that was already out of date, and one in a workspace owned by another deployment.
    stale = receipt(tmp_path / "srv/sparkring/tp2/stale", "id-qwen-stale",
                    {**current, WEIGHT: current[WEIGHT][:4] + [current[WEIGHT][4] - 1]})
    foreign = receipt(tmp_path / "srv/sparkring/tp2/foreign", "someone-else", current)
    path_record = host_records / (hashlib.sha256(str(folder).encode()).hexdigest() + ".json")
    path_record.write_bytes(served.read_bytes())
    own_record = host_records / (hashlib.sha256(owned.encode()).hexdigest() + ".json")
    write_json(own_record, {"path": owned, "files": hashes})
    fixed = {path: path.read_bytes() for path in (stale, foreign)}
    seen = {json.loads(path.read_text()).get("seen_as") for path in (host_records / "files").iterdir()}
    assert seen == {linked, owned + "/config.json", owned + "/tokenizer/vocab.json"}
    state = controller_state(tmp_path, {
        "qwen-in-place": (str(folder), True, str(served.parent.parent)),
        "qwen-stale": (str(folder), True, str(stale.parent.parent)),
        "qwen-foreign": (str(folder), True, str(foreign.parent.parent)),
        "qwen-owned": (owned, False, str(tmp_path / "srv/sparkring/tp2/owned"))}, hosts=HOSTS[:1], active="qwen-in-place")

    def spark(host, argv, *, data=None, timeout=None):
        request = json.loads(data)
        if argv == LIST:
            return json.dumps(checkpoints.list_local(request["paths"], root=str(tmp_path)))
        return json.dumps(checkpoints.release_local(argv[-1], request["receipts"]))

    def replace_config():
        os.rename(Path(owned, "config.json"), Path(owned, ".saved-config.json"))
        Path(owned, "config.json").write_bytes(FILES["config.json"])

    def restore_config():
        os.unlink(Path(owned, "config.json"))
        os.rename(Path(owned, ".saved-config.json"), Path(owned, "config.json"))

    def snapshot():
        return tree_state(owned), tree_state(state_directory), tree_state(folder)

    # Anything in the directory that SparkRing did not place stops the release before any change.
    unplaced = {"a file": (lambda: Path(owned, "added_tokens.json").write_text("{}"),
                           lambda: Path(owned, "added_tokens.json").unlink()),
                "a directory holding a file": (lambda: (Path(owned, "extra").mkdir(), Path(owned, "extra/notes").write_text("")),
                                               lambda: (Path(owned, "extra/notes").unlink(), Path(owned, "extra").rmdir())),
                "a replaced name": (replace_config, restore_config)}
    for label, (add, undo) in unplaced.items():
        before = snapshot()
        add()
        changed = snapshot()
        assert checkpoints.main(["--release", owned, "--yes"], state_root=state, invoke=spark) == 2, label
        printed = capsys.readouterr()
        assert "holds files SparkRing did not place" in printed.out + printed.err, label
        assert snapshot() == changed, label
        undo()
        assert snapshot() == before, label

    time.sleep(0.05)  # the release's unlink must land in a later change-time tick than the adoption's link
    assert checkpoints.main(["--release", owned, "--yes", "--json"], state_root=state, invoke=spark) == 0
    printed = capsys.readouterr()
    result = json.loads(printed.out)
    assert printed.err.splitlines()[-1] == f"Node 0 {socket.gethostname()}: released; freed 30 bytes, refreshed 2 receipts"
    [released] = result["nodes"]
    assert released["state"] == "released" and released["kept_bytes"] == len(FILES[WEIGHT])
    assert sorted(released["refreshed"]) == sorted([str(served), str(path_record)])
    assert result["deployments"] == ["qwen-owned"]
    assert not os.path.lexists(owned) and not state_directory.exists()
    assert os.listdir(Path(owned).parent) == []
    assert tree_state(folder) == untouched
    refreshed = json.loads(served.read_text())
    assert refreshed["file_stats"][WEIGHT] == five(linked)
    assert refreshed["file_stats"][WEIGHT][4] != current[WEIGHT][4]
    assert {name: values for name, values in refreshed["file_stats"].items() if name != WEIGHT} == \
        {name: values for name, values in current.items() if name != WEIGHT}
    assert json.loads(path_record.read_text())["file_stats"] == refreshed["file_stats"]
    assert {path: path.read_bytes() for path in fixed} == fixed
    assert not own_record.exists()
    assert [json.loads(p.read_text()).get("seen_as") for p in (host_records / "files").iterdir()] == [linked]
    assert checkpoints.release_local(owned) == {"path": owned, "state": "absent", "unlinked": 0, "freed_bytes": 0,
                                                "kept_bytes": 0, "refreshed": []}


@linux
@pytest.mark.parametrize("step", ["_rmdir_below", "_remove_state"])
def test_an_interrupted_release_continues_when_repeated(tmp_path, host_records, monkeypatch, step):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    state_directory = Path(owned).parent / ("." + REVISION + ".sparkring")
    original = getattr(checkpoints, step)

    def interrupted(*arguments):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(checkpoints, step, interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        checkpoints.release_local(owned)
    assert (state_directory / "owner.json").is_file()
    [entry] = checkpoints.list_local(root=str(tmp_path))["directories"]
    if step == "_rmdir_below":
        # Every name is unlinked; only the emptied tokenizer/ directory is left in the directory.
        assert os.listdir(owned) == ["tokenizer"] and entry["state"] == "ok" and entry["files"] == 0
    else:
        assert not os.path.lexists(owned) and entry["state"] == "missing"
    monkeypatch.setattr(checkpoints, step, original)
    assert checkpoints.release_local(owned)["state"] == "released"
    assert os.listdir(Path(owned).parent) == []
    assert [json.loads(p.read_text()).get("seen_as") for p in (host_records / "files").iterdir()] == [str(folder / WEIGHT)]
    assert tree_state(folder)[WEIGHT][4] == 1


@linux
def test_receipt_refresh_needs_valid_stats_unless_the_content_was_just_hashed(tmp_path, host_records):
    folder = user_copy(tmp_path / "copy")
    path = folder / WEIGHT
    before = five(str(path))
    workspace = tmp_path / "srv/sparkring/tp2/in-place"
    write_json(workspace / ".installer-owner.json", {"deployment": "id-in-place"})
    receipt = workspace / "installer/model.json"
    write_json(receipt, {"path": str(folder), "files": {WEIGHT: digest(FILES[WEIGHT])},
                         "file_stats": {WEIGHT: before[:4] + [before[4] - 1]}})
    (tmp_path / "alias").symlink_to(workspace)
    requests = [{"workspace": str(workspace), "deployment": "id-in-place"},
                {"workspace": str(workspace), "deployment": "another"},
                {"workspace": str(tmp_path / "alias"), "deployment": "id-in-place"}]
    assert checkpoints.receipt_paths(requests) == [str(receipt)]
    time.sleep(0.05)
    os.link(path, tmp_path / "linked")
    changed = {tuple(before[:2]): {"sha256": digest(FILES[WEIGHT]), "before": before}}
    # The entry was out of date before the link, so a release that reads no content keeps it.
    assert checkpoints.refresh_receipts(changed, [str(receipt)]) == []
    wrong = {key: {**value, "sha256": digest(b"other")} for key, value in changed.items()}
    assert checkpoints.refresh_receipts(wrong, [str(receipt)], content_verified=True) == []
    assert checkpoints.refresh_receipts(changed, [str(receipt)], content_verified=True) == [str(receipt)]
    assert json.loads(receipt.read_text())["file_stats"][WEIGHT] == five(str(path))


@pytest.mark.parametrize("role", ["active", "rollback"])
def test_release_refuses_a_directory_a_protected_deployment_serves_in_place(tmp_path, capsys, role):
    # sparkring up --model-path <D>, or a site row with reuse_verified_model, serves D in place.
    deployments = {"qwen-up-main": (DIRECTORY, True, "/srv/sparkring/tp2/qwen-up-main"),
                   "glm-tp2-icccc": ("/srv/sparkring/tp2/checkpoints/glm/" + "c" * 40, False, "/srv/sparkring/tp2/c")}
    if role == "active":
        state = controller_state(tmp_path, deployments, active="qwen-up-main")
        reason = "the active deployment"
    else:
        state = controller_state(tmp_path, deployments, active="glm-tp2-icccc",
                                 transaction={"candidate": "glm-tp2-icccc", "previous": "qwen-up-main",
                                              "state": "complete"})
        reason = "the rollback target recorded in transaction.json"
    spark = Spark({target: local_listing([listed()]) for target in HOSTS})
    assert checkpoints.main(["--release", DIRECTORY, "--yes"], state_root=state, invoke=spark) == 2
    assert (f"{DIRECTORY} is the copy served in place by {reason} qwen-up-main; SparkRing does not release it. "
            "Nothing was released.") in capsys.readouterr().err
    assert spark.calls == []
    # Another retained deployment serving it in place is named among those that need installing again.
    settled = controller_state(tmp_path / "settled", {**deployments, "qwen-up-main": (DIRECTORY, True, "/w")},
                               active="glm-tp2-icccc")
    assert checkpoints.main(["--release", DIRECTORY, "--yes", "--json"], state_root=settled, invoke=spark) == 0
    assert json.loads(capsys.readouterr().out)["deployments"] == ["qwen-up-main"]
    # The listing names it as a user of the directory as well.
    entry = checkpoints.list_cluster(settled, spark)["nodes"][0]["directories"][0]
    assert entry["deployments"] == [{"name": "qwen-up-main"}]


def test_running_containers_are_found_by_their_mounts():
    inspected = [{"Name": "/qwen-manual-r0", "Mounts": [{"Source": DIRECTORY + "/", "Destination": "/models/target"}]},
                 {"Name": "/other", "Mounts": [{"Source": DIRECTORY + "-copy", "Destination": "/m"}]},
                 {"Name": "/below", "Mounts": [{"Source": DIRECTORY + "/nested", "Destination": "/m"}]}]

    def run(argv, **kwargs):
        from types import SimpleNamespace
        assert argv[:3] == ["docker", "--context", "default"]
        if argv[3] == "ps":
            return SimpleNamespace(returncode=0, stdout="a\nb\nc\n", stderr="")
        assert argv[3:] == ["inspect", "a", "b", "c"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(inspected), stderr="")
    assert RUNNING_CONTAINERS(DIRECTORY, run=run) == ["below", "qwen-manual-r0"]

    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")
    assert RUNNING_CONTAINERS(DIRECTORY, run=missing) == []

    def refused(argv, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(returncode=1, stdout="", stderr="permission denied while trying to connect\n")
    with pytest.raises(ValueError, match="could not list this host's running containers "
                                         r"\(permission denied while trying to connect\); nothing was released"):
        RUNNING_CONTAINERS(DIRECTORY, run=refused)


@linux
def test_release_refuses_a_directory_that_a_running_container_mounts(tmp_path, host_records, monkeypatch):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    before = tree_state(owned)
    monkeypatch.setattr(checkpoints, "running_containers", lambda path, **kwargs: ["qwen-manual-r0"])
    with pytest.raises(ValueError, match=r"is mounted by running containers on this Spark \(qwen-manual-r0\)"):
        checkpoints.release_local(owned)
    assert tree_state(owned) == before


@linux
def test_release_removes_only_directories_the_journal_records(tmp_path, host_records):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    state_directory = Path(owned).parent / ("." + REVISION + ".sparkring")
    # Placement recorded the directory it created for the nested name before creating it.
    assert json.loads((state_directory / "journal.json").read_text())["directories"] == ["tokenizer"]
    Path(owned, "operator-notes").mkdir()
    before = tree_state(owned)
    with pytest.raises(ValueError, match="holds files SparkRing did not place: operator-notes"):
        checkpoints.release_local(owned)
    assert tree_state(owned) == before
    Path(owned, "operator-notes").rmdir()
    assert checkpoints.release_local(owned)["state"] == "released"
    assert os.listdir(Path(owned).parent) == []


@linux
def test_release_refreshes_a_receipt_only_while_the_file_keeps_its_post_unlink_stats(tmp_path, host_records,
                                                                                    monkeypatch):
    folder = user_copy(tmp_path / "var/tmp/models/qwen")
    owned = adopt(tmp_path / "srv/sparkring/tp2/checkpoints" / SLUG / REVISION, folder)
    workspace = tmp_path / "srv/sparkring/tp2/in-place"
    write_json(workspace / ".installer-owner.json", {"deployment": "id-in-place"})
    receipt = workspace / "installer/model.json"
    write_json(receipt, {"path": str(folder), "files": {name: digest(content) for name, content in FILES.items()},
                         "file_stats": {name: five(str(folder / name)) for name in FILES}})
    time.sleep(0.05)
    refresh = checkpoints.refresh_receipts

    def rewrite_then_refresh(changed, receipts, **kwargs):
        # The owner rewrites the file in place after the release unlinked SparkRing's name and before the
        # receipts are refreshed, keeping its size and modification time.
        time.sleep(0.05)
        info = os.stat(folder / WEIGHT)
        with open(folder / WEIGHT, "r+b") as stream:
            stream.write(b"W")
        os.utime(folder / WEIGHT, ns=(info.st_atime_ns, info.st_mtime_ns))
        return refresh(changed, receipts, **kwargs)
    monkeypatch.setattr(checkpoints, "refresh_receipts", rewrite_then_refresh)
    result = checkpoints.release_local(owned, [{"workspace": str(workspace), "deployment": "id-in-place"}])
    assert result["state"] == "released" and result["refreshed"] == []
    # The receipt still records the stats from before the change, so the next start hashes the file again.
    assert json.loads(receipt.read_text())["file_stats"][WEIGHT] != five(str(folder / WEIGHT))


def test_release_refuses_mixed_package_revisions(tmp_path, capsys):
    state = controller_state(tmp_path, {"qwen-tp2-ibbbb": (DIRECTORY, False, "/srv/sparkring/tp2/b")},
                             active=None)
    spark = Spark({HOSTS[0]: local_listing([listed()], revision="a" * 40),
                   HOSTS[1]: local_listing([listed()], revision="b" * 40, hostname="spark-931e")})
    assert checkpoints.main(["--release", DIRECTORY, "--yes"], state_root=state, invoke=spark) == 2
    error = capsys.readouterr().err
    assert "The Sparks run different SparkRing package revisions (Node 0 aaaaaaaaaaaa, Node 1 bbbbbbbbbbbb)" in error
    assert "Nothing was released." in error and spark.releases() == []

    # A Spark that cannot be listed also stops the release: its holdings and revision are unknown.
    spark = Spark({HOSTS[0]: local_listing([listed()]), HOSTS[1]: RuntimeError("root@192.0.2.11: timed out")})
    assert checkpoints.main(["--release", DIRECTORY, "--yes"], state_root=state, invoke=spark) == 2
    assert "Node 1 root@192.0.2.11 could not be listed" in capsys.readouterr().err
    assert spark.releases() == []


def test_release_names_retained_deployments_and_needs_yes_without_a_terminal(tmp_path, capsys, monkeypatch):
    state = controller_state(tmp_path, {"qwen-tp2-ibbbb": (DIRECTORY, False, "/srv/sparkring/tp2/b"),
                                        "glm-tp2-icccc": ("/srv/sparkring/tp2/checkpoints/glm/" + "c" * 40, False,
                                                          "/srv/sparkring/tp2/c")}, active="glm-tp2-icccc")
    spark = Spark({HOSTS[0]: local_listing([listed()]), HOSTS[1]: local_listing([], hostname="spark-931e")})
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert checkpoints.main(["--release", DIRECTORY], state_root=state, invoke=spark) == 3
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        f"Release {DIRECTORY}:",
        "    Node 0 spark-aa42: removes 48 files; frees 56.5 MB; 98.6 GiB stays on disk in /var/tmp/models/qwen",
        "Retained deployments that use it need sudo sparkring install again: qwen-tp2-ibbbb",
        "SparkRing removes only the names it placed there; it never writes, moves or deletes the copies they were "
        "linked from.",
        "Review the release above, then repeat with --yes to apply it. Nothing was released."]
    assert spark.releases() == []
    assert checkpoints.main(["--release", DIRECTORY, "--yes"], state_root=state, invoke=spark) == 0
    assert [host for host, _, _ in spark.releases()] == [HOSTS[0]]
    assert capsys.readouterr().out.splitlines()[-1] == "Node 0 spark-aa42: released; freed 56.5 MB"
    assert checkpoints.main(["--release", "/srv/sparkring/tp2/checkpoints/none", "--yes"], state_root=state,
                            invoke=spark) == 2
    assert "is not a SparkRing checkpoint directory on any Spark" in capsys.readouterr().err
    assert checkpoints.main(["--release", "relative/path", "--yes"], state_root=state, invoke=spark) == 2
    assert "absolute path" in capsys.readouterr().err


def test_listing_reports_a_spark_that_cannot_be_reached(tmp_path, capsys):
    state = controller_state(tmp_path, {})
    spark = Spark({HOSTS[0]: local_listing([listed(shared=[])]),
                   HOSTS[1]: RuntimeError("root@192.0.2.11: Connection timed out")})
    assert checkpoints.main([], state_root=state, invoke=spark) == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[:5] == ["Node 0 spark-aa42", "    " + DIRECTORY,
                           f"        {REPOSITORY} at {REVISION[:12]}: 48 files, 98.6 GiB",
                           "        used by no retained deployment",
                           "        98.6 GiB is also linked from other paths"]
    assert "Node 1 root@192.0.2.11: not reachable: root@192.0.2.11: Connection timed out" in printed


def test_commands_route_to_the_checkpoints_module(tmp_path, monkeypatch, capsys):
    from scripts import sparkring, sparkring_node
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    seen = []
    monkeypatch.setattr(checkpoints, "main", lambda argv: seen.append(argv) or 0)
    assert sparkring.main(["checkpoints", "--release", DIRECTORY, "--yes"]) == 0
    assert seen == [["--release", DIRECTORY, "--yes"]]

    monkeypatch.setattr(checkpoints, "release_local", lambda path, receipts: {"path": path, "receipts": receipts})
    monkeypatch.setattr(checkpoints, "list_local", lambda paths: {"paths": paths})
    request = {"receipts": [{"workspace": "/srv/sparkring/tp2/b", "deployment": "id-b"}]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert sparkring_node.main(["checkpoints", "--release", DIRECTORY]) == 0
    assert json.loads(capsys.readouterr().out) == {"path": DIRECTORY, "receipts": request["receipts"]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert sparkring_node.main(["checkpoints"]) == 0
    assert json.loads(capsys.readouterr().out) == {"paths": []}
