"""Fabric checkpoint streams over loopback, rsync into staging, and the cable-ordered copy plan.

Receives run against real files, hard links and SparkRing checkpoint directories
claimed with ``checkpoint_place`` under a synthetic ext4 mount table. The rsync
tests run the real ``rsync`` binary through a stand-in SSH command that runs the
remote side locally.
"""
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from runtime.host import checkpoint_place as place, fabric_stream, install_assets
from runtime.host.test_fabric_ssh import cluster

TOKEN = b"t" * 32
REPOSITORY = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REVISION = "629bc3218833a38b475b719f34aa571666f4a03e"
CONFIG, FIRST, SECOND = "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
OWNED = "/srv/sparkring/tp4/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/" + REVISION
STAGED = ("/srv/sparkring/tp4/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/." + REVISION
          + ".sparkring/receive/rsync/")
FORBIDDEN_RSYNC = ("-t", "-a", "--times", "--archive", "--inplace", "--append", "--append-verify", "--partial",
                   "--checksum", "-c")
linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="checkpoint placement uses Linux hard links, /proc/self/fd, O_NOATIME and flock")


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


# Checkpoint directories and fixtures -------------------------------------------

MOUNTINFO = "29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"


@pytest.fixture
def mount_table(tmp_path, monkeypatch):
    """A synthetic mount table whose root filesystem is ext4, and per-inode records under ``tmp_path``."""
    path = tmp_path / "mountinfo"
    path.write_text(MOUNTINFO)
    monkeypatch.setattr(place, "MOUNTINFO", str(path))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records"))
    return path


def checkpoint(tmp_path, cluster_name="tp2"):
    return str(tmp_path / "srv" / cluster_name / "checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4" / REVISION)


def state(path):
    return Path(path).parent / ("." + Path(path).name + ".sparkring")


def prepare(path):
    """What ``model-transfer-prepare`` leaves: a claimed directory with an empty, marked ``receive/``."""
    with place.claim(path, REPOSITORY, REVISION) as claimed:
        os.close(place.staging(claimed, "receive", empty=True))


def journal(path):
    return json.loads((state(path) / "journal.json").read_text())["files"]


def write(root, contents):
    """Write ``contents`` below ``root``; returns the pinned ``[size, sha256]`` of each name."""
    files = {}
    for name, data in contents.items():
        target = Path(root) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = [len(data), hashlib.sha256(data).hexdigest()]
    return files


def link_user_file(path, name, user_file):
    """Hard-link a user's file into claimed directory ``path`` as adoption does: hashed descriptor, journal first."""
    size = os.path.getsize(user_file)
    with place.claim(path, REPOSITORY, REVISION) as claimed:
        records = place.journal_load(claimed)
        info = os.lstat(user_file)
        fd, before = place.open_source(user_file, [info.st_dev, info.st_ino], size)
        try:
            digest = place.hash_descriptor(fd, size)
            place.place_link(claimed.dir_fd, fd, name, records, sha256=digest, before=before, source=user_file)
        finally:
            os.close(fd)


def tree_state(root):
    """Every entry at and below ``root`` with its metadata and content, excluding access and change times."""
    result = {}
    for directory, directories, files in os.walk(root):
        for name in [".", *directories, *files]:
            path = os.path.normpath(os.path.join(directory, name))
            info = os.lstat(path)
            content = hashlib.sha256(Path(path).read_bytes()).hexdigest() if stat.S_ISREG(info.st_mode) else None
            result[os.path.relpath(path, root)] = (info.st_mode, info.st_uid, info.st_size, info.st_ino,
                                                   info.st_nlink, info.st_mtime_ns, content)
    return result


def run_stream(source_root, target, files, *, streams=2, token=TOKEN, groups=None):
    """Receive ``files`` into ``target`` over loopback while ``send`` streams them from ``source_root``."""
    offers, result = queue.Queue(), {}
    addresses = ["127.0.0.1"] * streams

    def receiver():
        try:
            result["value"] = fabric_stream.receive_checkpoint(addresses, addresses, target, REPOSITORY, REVISION,
                                                               files, token=TOKEN, announce=offers.put)
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
    assert not thread.is_alive()
    return offer, sent, result


# Fabric receive ------------------------------------------------------------------

@linux
def test_receive_places_new_inodes_and_never_writes_through_hard_links(tmp_path, mount_table):
    source = tmp_path / "source"
    weights = {FIRST: b"x" * 300000, "audio/" + SECOND: b"y" * 200000}
    files = write(source, {CONFIG: b'{"pinned": true}', **weights})
    user = tmp_path / "home/code/qwen"
    write(user, {CONFIG: b'{"edited": 1}   '})
    target = checkpoint(tmp_path)
    prepare(target)
    # SparkRing's directory holds a hard link to the user's file under a name the
    # transfer lists; the pinned hash differs, as for a file edited after linking.
    link_user_file(target, CONFIG, str(user / CONFIG))
    (state(target) / "receive" / (FIRST + ".part")).write_bytes(b"stale part of an interrupted stream")
    before = tree_state(user)

    offer, sent, result = run_stream(source, target, files)

    assert "error" not in result, result
    assert offer["needed"] == sorted(weights)
    assert sent == {"sent": 500000}
    assert result["value"] == {"received": 2, "bytes": 500000, "placed": sorted(weights)}
    assert tree_state(user) == before
    assert os.lstat(Path(target) / CONFIG).st_ino == os.lstat(user / CONFIG).st_ino
    recorded = journal(target)
    for name, data in weights.items():
        placed = os.lstat(Path(target) / name)
        assert Path(target, name).read_bytes() == data
        assert placed.st_nlink == 1 and placed.st_ino != os.lstat(source / name).st_ino
        assert stat.S_IMODE(placed.st_mode) & 0o022 == 0
        entry = recorded[name]
        assert (entry["state"], entry["origin"], entry["sha256"]) == ("placed", "fabric", files[name][1])
        assert entry["stats"] == place.stats(placed)
        record = json.loads((tmp_path / "records" / f"{placed.st_dev}-{placed.st_ino}.json").read_text())
        assert record["sha256"] == files[name][1]
    assert sorted(p.name for p in (state(target) / "receive").iterdir()) == [place.STAGING_MARKER, "audio"]
    assert not list((state(target) / "receive" / "audio").iterdir())


@linux
def test_receive_hashes_while_writing_and_rejects_mismatches_without_placing(tmp_path, mount_table):
    source = tmp_path / "source"
    write(source, {FIRST: b"Q" * 64})  # a fine-tune's shard of the pinned size
    files = {FIRST: [64, hashlib.sha256(b"P" * 64).hexdigest()]}
    target = checkpoint(tmp_path)
    prepare(target)

    offer, _, result = run_stream(source, target, files, streams=1)

    assert offer["needed"] == [FIRST]
    assert "different bytes for " + FIRST in str(result["error"])
    assert not (Path(target) / FIRST).exists()
    assert journal(target) == {}
    assert sorted(p.name for p in (state(target) / "receive").iterdir()) == [place.STAGING_MARKER]
    assert not (tmp_path / "records").exists()


@linux
def test_stream_rejects_a_sender_without_the_token(tmp_path, mount_table):
    files = write(tmp_path / "source", {FIRST: b"a" * 1000})
    target = checkpoint(tmp_path)
    prepare(target)
    _, _, result = run_stream(tmp_path / "source", target, files, streams=1, token=b"x" * 32)
    assert "token" in str(result["error"])
    assert not (Path(target) / FIRST).exists() and journal(target) == {}


@linux
def test_stream_rejects_files_outside_the_plan(tmp_path, mount_table):
    files = write(tmp_path / "source", {FIRST: b"a" * 1000})
    (tmp_path / "source" / "extra.py").write_bytes(b"print(1)")
    target = checkpoint(tmp_path)
    prepare(target)
    _, _, result = run_stream(tmp_path / "source", target, files, streams=1, groups=[["extra.py"]])
    assert "unplanned" in str(result["error"])
    assert os.listdir(target) == [] and journal(target) == {}


@linux
@pytest.mark.parametrize("case", ["absent", "not claimed"])
def test_receiver_writes_only_into_a_prepared_checkpoint_directory(case, tmp_path, mount_table):
    target = checkpoint(tmp_path)
    if case == "not claimed":
        Path(target).mkdir(parents=True)
    files = {FIRST: [3, hashlib.sha256(b"abc").hexdigest()]}
    with pytest.raises(ValueError, match="was not prepared for receiving"):
        fabric_stream.receive_checkpoint(["127.0.0.1"], ["127.0.0.1"], target, REPOSITORY, REVISION, files,
                                         token=TOKEN, announce=lambda value: pytest.fail("announced"))
    assert not state(target).exists()
    assert os.path.exists(target) == (case == "not claimed")


def test_receiver_source_is_self_contained():
    """The shipped receiver imports only the standard library and names only what its own source defines."""
    source = fabric_stream.receiver_source(["198.18.0.2"], ["198.18.0.1"], OWNED, REPOSITORY, REVISION,
                                           {FIRST: [3, "0" * 64]})
    tree = ast.parse(source)
    defined = set(dir(builtins))
    for statement in tree.body:
        for node in ast.walk(statement) if isinstance(statement, ast.Try) else [statement]:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, ast.Assign):
                defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name in ("receive", "receive_checkpoint"):
            local = set()
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    local.add(node.id)
                elif isinstance(node, ast.arg):
                    local.add(node.arg)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    local.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
                elif isinstance(node, ast.FunctionDef):
                    local.add(node.name)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    local.add(node.name)
            used = {node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)}
            assert used - local - defined == set(), statement.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in sys.stdlib_module_names
    compile(source, "receiver", "exec")
    # The remote shell receives the whole program as one argument, which Linux
    # limits to 128 KiB.
    assert len(shlex.join(["sudo", "-n", "python3", "-I", "-c", source])) < 120 * 1024


@linux
def test_shipped_receiver_runs_in_a_fresh_interpreter(tmp_path, mount_table):
    source = tmp_path / "source"
    files = write(source, {FIRST: b"q" * 70000, "nested/" + SECOND: b"r" * 50000})
    target = checkpoint(tmp_path)
    prepare(target)
    program = fabric_stream.receiver_source(["127.0.0.1"], ["127.0.0.1"], target, REPOSITORY, REVISION, files)
    program = program.replace('MOUNTINFO = "/proc/self/mountinfo"', f"MOUNTINFO = {str(mount_table)!r}", 1)
    program = program.replace('RECORDS = "/var/lib/sparkring/checkpoints/files"',
                              f"RECORDS = {str(tmp_path / 'records')!r}", 1)
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", program], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={}, cwd=tmp_path)
    try:
        child.stdin.write(TOKEN)
        child.stdin.close()
        line = child.stdout.readline()
        assert line, child.stderr.read().decode()
        offer = json.loads(line)
        assert offer["needed"] == sorted(files)
        sent = fabric_stream.send(["127.0.0.1"], ["127.0.0.1"], offer["ports"], str(source), [offer["needed"]],
                                  token=TOKEN)
        output, errors = child.stdout.read(), child.stderr.read()
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 0, errors.decode()
    assert json.loads(output) == {"received": 2, "bytes": 120000, "placed": sorted(files)}
    assert sent == {"sent": 120000}
    for name in files:
        assert Path(target, name).read_bytes() == (source / name).read_bytes()
        assert journal(target)[name]["origin"] == "fabric"


@linux
def test_sender_reads_with_o_noatime_when_permitted(tmp_path, monkeypatch):
    files = write(tmp_path / "source", {FIRST: b"f" * 5000, SECOND: b"s" * 3000})
    opened, real_open = [], os.open

    def recording_open(path, flags, *args, **kwargs):
        if str(path).startswith(str(tmp_path / "source")):
            opened.append((Path(path).name, bool(flags & os.O_NOATIME), bool(flags & os.O_NOFOLLOW)))
            if Path(path).name == SECOND and flags & os.O_NOATIME:
                raise PermissionError(1, "Operation not permitted")
        return real_open(path, flags, *args, **kwargs)
    server = socket.create_server(("127.0.0.1", 0))
    received = {}

    def accept():
        connection, _ = server.accept()
        with connection, connection.makefile("rb") as stream:
            received["data"] = stream.read()
    thread = threading.Thread(target=accept)
    thread.start()
    monkeypatch.setattr(os, "open", recording_open)
    try:
        sent = fabric_stream.send(["127.0.0.1"], ["127.0.0.1"], [server.getsockname()[1]], str(tmp_path / "source"),
                                  [[FIRST, SECOND]], token=TOKEN)
    finally:
        monkeypatch.undo()
        thread.join(timeout=30)
        server.close()
    assert sent == {"sent": 8000}
    assert opened == [(FIRST, True, True), (SECOND, True, True), (SECOND, False, True)]
    assert received["data"].startswith(TOKEN) and len(received["data"]) > 8000
    assert len(files) == 2


# rsync fallback --------------------------------------------------------------------

class Transport:
    """Two or four Sparks whose SSH command is ``ssh`` (or a stand-in) with the alias ``nodeN``."""

    def __init__(self, size, ssh="ssh"):
        self.hosts = cluster(size)["plan"]["spec"]["hosts"]
        self.mode = "fiber-ssh"
        self.ssh = ssh

    def argv(self, rank):
        return [] if rank == 0 else [self.ssh, "-F", "/head/ssh_config", "node" + str(rank)]

    def command(self, rank, args):
        return list(args) if rank == 0 else ["node" + str(rank), *args]


class TransferRunner:
    """Rank operations of a transfer: prepare reports the needed names; complete runs ``place_staged``."""

    def __init__(self, needed, complete=None):
        self.needed, self.complete, self.calls = needed, complete, []

    def remote(self, rank, op, data=None):
        self.calls.append((rank, op, json.loads(data) if data else None))
        if op == "model-transfer-prepare":
            return {"ok": True, "needed": self.needed}
        if self.complete:
            return self.complete(rank, json.loads(data))
        return {"ok": True, "complete": True}


def manifest(files):
    return {"repository": REPOSITORY, "revision": REVISION, "files": {n: v[1] for n, v in files.items()},
            "sizes": {n: v[0] for n, v in files.items()}}


def test_rsync_fallback_transfers_only_needed_names(tmp_path):
    runs = []

    def run(argv, **kwargs):
        # The names to send arrive on rsync's stdin; nothing is written to name them.
        assert argv[argv.index("-e") - 1] == "--files-from=-"
        runs.append((argv, kwargs["input"].decode()))
        return SimpleNamespace(returncode=0)
    rows = [{"rank": n, "model": OWNED} for n in range(4)]
    rows[2]["model"] = "/mnt/usb/qwen"
    assets = install_assets.Assets(Transport(4), tmp_path / "assets", run=run)
    files = {CONFIG: [10, "c" * 64], FIRST: [20, "a" * 64], SECOND: [30, "b" * 64]}

    runner = TransferRunner([SECOND, CONFIG])
    result = assets.copy(runner, rows, manifest(files), 0, 3)
    assert result == {"ok": True, "complete": True}
    argv, listing = runs[0]
    assert listing == CONFIG + "\n" + SECOND + "\n"
    assert argv[:9] == ["rsync", "-r", "--perms", "--no-owner", "--no-group", "--chmod=D0755,F0644",
                        "--protect-args", "--open-noatime", "--rsync-path=sudo -n rsync"]
    assert not any(flag in argv for flag in FORBIDDEN_RSYNC)
    assert not any(a.startswith(("--append", "--partial", "--inplace")) for a in argv)
    assert argv[argv.index("-e") + 1] == "ssh -F /head/ssh_config"
    assert argv[-2] == rows[0]["model"] + "/"
    assert argv[-1] == "node3:" + STAGED
    assert [(rank, op) for rank, op, _ in runner.calls] == [(3, "model-transfer-prepare"), (3, "model-transfer-complete")]
    assert runner.calls[0][2] == manifest(files)

    # Pooling into Node A reads the other Spark's folder and names only the pooled files.
    runner = TransferRunner([FIRST])
    assets.copy(runner, rows, manifest(files), 2, 0, [FIRST])
    argv, listing = runs[1]
    assert listing == FIRST + "\n"
    assert argv[-2:] == ["node2:/mnt/usb/qwen/", STAGED]
    assert runner.calls[0][2]["files"] == {FIRST: "a" * 64}

    # Nothing needed: no rsync, and the target still verifies its files.
    runner = TransferRunner([])
    assets.copy(runner, rows, manifest(files), 0, 1)
    assert len(runs) == 2 and [op for _, op, _ in runner.calls][-1] == "model-transfer-complete"
    assert not list((tmp_path / "assets").iterdir())
    with pytest.raises(ValueError, match="between Node A and one other Spark"):
        assets.copy(runner, rows, manifest(files), 1, 3)


def test_rsync_reads_without_changing_access_times_and_sets_no_times():
    from scripts.test_installer_adopt import _forbidden_argv
    assert "--open-noatime" in install_assets.RSYNC and "--perms" in install_assets.RSYNC
    assert "--chmod=D0755,F0644" in install_assets.RSYNC
    assert _forbidden_argv(install_assets.RSYNC) is None
    assert not any(flag in install_assets.RSYNC for flag in FORBIDDEN_RSYNC)


def test_receiver_program_sets_its_own_umask():
    source = fabric_stream.receiver_source(["198.18.0.2"], ["198.18.0.1"], OWNED, REPOSITORY, REVISION,
                                           {FIRST: [3, "0" * 64]})
    # Received files get mode 0644 whatever the umask of the session that started the receiver.
    assert source.rstrip().splitlines()[-2] == "os.umask(0o022)"


def stand_in_commands(tmp_path, monkeypatch):
    """``ssh`` that runs the remote command locally, and ``sudo -n`` that runs its command."""
    directory = tmp_path / "bin"
    directory.mkdir()
    (directory / "ssh").write_text('#!/bin/sh\nwhile [ "$1" = "-F" ]; do shift 2; done\nshift\nexec sh -c "$*"\n')
    (directory / "sudo").write_text('#!/bin/sh\n[ "$1" = "-n" ] && shift\nexec "$@"\n')
    for name in ("ssh", "sudo"):
        (directory / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    return str(directory / "ssh")


@pytest.fixture
def standard_umask():
    """New files take the process umask; sudo's default, 0022, gives SparkRing's files mode 0644."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


@linux
@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs the rsync binary")
def test_rsync_fallback_writes_only_staging_and_leaves_linked_files_untouched(tmp_path, mount_table, monkeypatch,
                                                                             standard_umask):
    ssh = stand_in_commands(tmp_path, monkeypatch)
    donor = checkpoint(tmp_path / "node0")
    files = write(donor, {CONFIG: b'{"pinned": true}', FIRST: b"w" * 40000, "audio/" + SECOND: b"v" * 30000})
    user = tmp_path / "home/code/qwen"
    write(user, {FIRST: b"W" * 40000})
    target = checkpoint(tmp_path / "node1")
    prepare(target)
    # The target already links a user's file under a name the transfer lists.
    link_user_file(target, FIRST, str(user / FIRST))
    before, linked = tree_state(user), os.lstat(Path(target) / FIRST).st_ino
    staged = Path(state(target)) / "receive" / "rsync"
    seen = {}

    def complete(rank, subset):
        # model-transfer-complete: hash each staged file through its descriptor and place it.
        seen["staged"] = sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file())
        seen["target"] = tree_state(target)
        with place.claim(target, REPOSITORY, REVISION) as claimed:
            records = place.journal_load(claimed)
            directory = os.open(str(staged), os.O_RDONLY | os.O_DIRECTORY)
            try:
                for name in ("audio/" + SECOND, CONFIG):
                    fd, measured = place.open_source(name, None, subset["sizes"][name], dir_fd=directory)
                    try:
                        digest = place.hash_descriptor(fd, subset["sizes"][name])
                        assert digest == subset["files"][name]
                        place.place_staged(claimed.dir_fd, directory, name, name, records, fd=fd, sha256=digest,
                                           origin="rsync", before=measured)
                    finally:
                        os.close(fd)
            finally:
                os.close(directory)
        return {"ok": True, "complete": True}
    runs = []

    def run(argv, **kwargs):
        runs.append(argv)
        return subprocess.run(argv, **kwargs)
    rows = [{"rank": 0, "model": donor}, {"rank": 1, "model": target}]
    assets = install_assets.Assets(Transport(2, ssh), tmp_path / "assets", run=run)
    # A prepare that listed an already placed name still cannot reach the checkpoint directory.
    runner = TransferRunner(sorted(files), complete)

    assert assets.copy(runner, rows, manifest(files), 0, 1) == {"ok": True, "complete": True}

    assert seen["staged"] == sorted(files)
    assert tree_state(user) == before
    assert os.lstat(Path(target) / FIRST).st_ino == linked
    assert set(seen["target"]) == {".", FIRST}
    for name in (CONFIG, "audio/" + SECOND):
        placed = os.lstat(Path(target) / name)
        assert Path(target, name).read_bytes() == Path(donor, name).read_bytes()
        assert placed.st_nlink == 1 and stat.S_IMODE(placed.st_mode) == 0o644
        assert journal(target)[name]["origin"] == "rsync"
    assert len(runs) == 1 and not any(flag in runs[0] for flag in FORBIDDEN_RSYNC)


def aged(path, days=3):
    """Set ``path``'s access time into the past and return it, so that a later read would move it."""
    info = os.stat(path)
    os.utime(path, ns=(info.st_atime_ns - days * 86400 * 10 ** 9, info.st_mtime_ns))
    return os.stat(path).st_atime_ns


@linux
@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs the rsync binary")
def test_rsync_sender_keeps_access_times_and_received_files_get_mode_0644(tmp_path):
    donor = tmp_path / "donor"
    write(donor, {CONFIG: b"x" * 4096, "audio/" + SECOND: b"y" * 2048})
    before = {name: aged(donor / name) for name in (CONFIG, "audio/" + SECOND)}
    staged = tmp_path / "receive/rsync"
    staged.mkdir(parents=True)
    argv = [value for value in install_assets.RSYNC if not value.startswith("--rsync-path")]
    previous = os.umask(0o077)
    try:
        subprocess.run([*argv, "--files-from=-", str(donor) + "/", str(staged) + "/"],
                       input=f"{CONFIG}\naudio/{SECOND}\n".encode(), check=True, capture_output=True)
    finally:
        os.umask(previous)
    # The sender read the donor's files without changing their access times.
    assert {name: os.stat(donor / name).st_atime_ns for name in before} == before
    # --perms with --chmod gives received files mode 0644 even under a restrictive umask.
    assert stat.S_IMODE(os.stat(staged / CONFIG).st_mode) == 0o644
    assert stat.S_IMODE(os.stat(staged / "audio" / SECOND).st_mode) == 0o644


# Orchestration of transfers ----------------------------------------------------------

class RingRunner:
    def __init__(self):
        self.calls = []

    def remote(self, rank, op, data=None):
        self.calls.append((rank, op))
        return {"ok": True, "needed": list(json.loads(data)["files"]), "complete": True}


RING = {"repository": REPOSITORY, "revision": REVISION, "files": {"w.safetensors": "a" * 64},
        "sizes": {"w.safetensors": 1}}


def receives(count, donor):
    return [{"source": s, "target": t, "names": ["w.safetensors"], "level": level, "transport": "fabric"}
            for level, edges in enumerate(fabric_stream.tree(count, donor)) for s, t in edges]


def test_checkpoint_copies_stream_along_the_ring_without_rsync(tmp_path):
    rows = [{"rank": n, "model": OWNED} for n in range(4)]
    current = install_assets.Assets(Transport(4), tmp_path, run=lambda *a, **k: pytest.fail("rsync"))
    edges = []
    current.stream_checkpoint = lambda runner, rows, manifest, source, target, names: (
        edges.append((source, target)) or {"ok": True, "complete": True})
    complete = {2}
    sent = current.distribute(RingRunner(), rows, RING, 2, receives(4, 2), complete)
    assert sorted(edges[:2]) == [(2, 1), (2, 3)] and edges[2] == (3, 0)
    assert complete == {0, 1, 2, 3} and {item["transport"] for item in sent} == {"fabric"}


def test_failed_stream_leaves_remaining_ranks_to_rsync_through_node_a(tmp_path):
    rows = [{"rank": n, "model": OWNED} for n in range(4)]
    copies = []

    def run(argv, **kwargs):
        copies.append((argv[-2], argv[-1]))
        return SimpleNamespace(returncode=0)
    current = install_assets.Assets(Transport(4), tmp_path, run=run)

    def stream(runner, rows, manifest, source, target, names):
        if target == 3:
            raise ValueError("connection refused")
        return {"ok": True, "complete": True}
    current.stream_checkpoint = stream
    runner = RingRunner()
    complete = {2}
    sent = current.distribute(runner, rows, RING, 2, receives(4, 2), complete)
    assert complete == {0, 1, 2, 3}
    assert [(item["source"], item["target"], item["transport"]) for item in sent] == [
        (2, 1, "fabric"), (2, 0, "rsync"), (0, 3, "rsync")]
    assert copies[0][0].startswith("node2:") and not copies[0][1].startswith("node")
    assert copies[1][1].startswith("node3:")
    assert [rank for rank, op in runner.calls if op == "model-transfer-complete"] == [0, 3]
