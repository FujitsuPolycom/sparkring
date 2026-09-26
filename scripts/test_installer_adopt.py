"""Checkpoint rank operations against real files, hard links, a fake Docker and an audit hook.

Every source layout a Spark can hold (Hugging Face cache, download folder,
plain folder, git clone with LFS objects) is built from real files. The audit
hook refuses, while a test declares source trees protected, every write, rename,
removal, mode, owner or time change under them and every subprocess argv that
carries a forbidden rsync option or a read-write Docker mount other than a
staging directory. ``tree_state`` then compares each protected tree before and
after: only linked weight files may show a new link count and change time.
"""
import ast
import contextlib
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from runtime.host import checkpoint_place as place
from scripts import installer_host as host, installer_runner

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rank operations use Linux hard links, /proc/self/fd, O_NOFOLLOW, O_NOATIME, flock and change times")

REPOSITORY = "fixture-owner/fixture-model"
REVISION = "c0ffee" + "0" * 34
MAIN = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
OPTIONAL = [".gitattributes", "README.md"]
DEPLOYMENT = "d" * 64
IMAGE = "sha256:" + "a" * 64


# Fixture checkpoint --------------------------------------------------------------

def contents():
    """File contents of the fixture revision; ``.safetensors`` files are the weight files."""
    index = json.dumps({"metadata": {}, "weight_map": {"a": SHARDS[0], "b": SHARDS[1]}}).encode()
    return {
        ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        "README.md": b"# Fixture model\n",
        "audio_tokenizer/config.json": b'{"sampling_rate": 24000}',
        "audio_tokenizer/model.safetensors": bytes(range(256)) * 40,
        "chat_template.jinja": b"{% for m in messages %}{{ m.content }}{% endfor %}",
        "config.json": b'{"model_type": "fixture"}',
        SHARDS[0]: b"first shard " * 3000,
        SHARDS[1]: b"second shard" * 2500,
        "model.safetensors.index.json": index,
        "tokenizer.json": b'{"model": {"vocab": {}}}',
    }


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def pins(data):
    files = {name: {"size": len(value), "sha256": sha256(value), "git_blob": git_blob(value)}
             for name, value in sorted(data.items())}
    for name in files:
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            files[name]["lfs"] = True
    return {"schema": "sparkring-checkpoint-pins/v1", "repository": REPOSITORY, "revision": REVISION,
            "index": "model.safetensors.index.json", "weights": list(SHARDS), "optional": list(OPTIONAL),
            "files": files}


def required(data):
    return sorted(name for name in data if name not in OPTIONAL)


def weights(data):
    return sorted(name for name in required(data) if name.endswith(".safetensors"))


def key(data, name):
    """The HF cache blob name of ``name``: SHA-256 for LFS files, the git blob id otherwise."""
    value = data[name]
    return sha256(value) if name.endswith(".safetensors") or name.endswith(".index.json") else git_blob(value)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def plain_folder(root, data, names=None, changed=None):
    """A plain folder of ``names`` (default: every file); ``changed`` replaces some contents."""
    for name in names if names is not None else data:
        write(Path(root) / name, (changed or {}).get(name, data[name]))
    return Path(root)


def local_dir(root, data, names=None):
    """A ``hf download --local-dir`` folder with the client's per-file download records."""
    root = plain_folder(root, data, names)
    for name in names if names is not None else data:
        metadata = root / ".cache/huggingface/download" / (name + ".metadata")
        write(metadata, f"{REVISION}\n{key(data, name)}\n1700000000.0\n".encode())
    return root


def hf_cache(hub, data, names=None, commit=MAIN):
    """An HF cache repository folder whose snapshot links point at content-addressed blobs."""
    repo = Path(hub) / ("models--" + REPOSITORY.replace("/", "--"))
    for name in names if names is not None else data:
        blob = write(repo / "blobs" / key(data, name), data[name])
        link = repo / "snapshots" / commit / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    write(repo / "refs/main", commit.encode())
    return repo


def git_lfs(root, data, names=None):
    """A git clone whose LFS files live in ``.git/lfs/objects`` and whose worktree holds pointer files."""
    root = Path(root)
    for name in names if names is not None else data:
        value = data[name]
        if name.endswith(".safetensors"):
            digest = sha256(value)
            write(root / ".git/lfs/objects" / digest[:2] / digest[2:4] / digest, value)
            write(root / name, f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(value)}\n".encode())
        else:
            write(root / name, value)
    write(root / ".git/HEAD", b"ref: refs/heads/main\n")
    return root


def lfs_object(root, data, name):
    digest = sha256(data[name])
    return Path(root) / ".git/lfs/objects" / digest[:2] / digest[2:4] / digest


def entry(action, path, data, name):
    info = os.lstat(path)
    return {"action": action, "source": str(path), "identity": [info.st_dev, info.st_ino], "size": len(data[name])}


# Environment -----------------------------------------------------------------------

def environment(tmp_path, monkeypatch, *, reuse=False, model=None, backend="compose", hosts=("root@spark0",)):
    """A deployment lock of the fixture revision and patched host globals; returns a namespace.

    ``call(operation, document, number)`` runs a rank operation with
    ``document`` as its JSON stdin.
    """
    data = contents()
    pinned = pins(data)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
    monkeypatch.setattr(place, "MOUNTINFO", str(mountinfo))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records/files"))
    monkeypatch.setattr(host, "CHECKPOINTS", tmp_path / "records")
    monkeypatch.setattr(host, "POSIX_STATS", os.name == "posix")

    def checkpoint_pins(card, **kwargs):
        assert (card["model_repository"], card["model_revision"]) == (REPOSITORY, REVISION)
        return pinned
    monkeypatch.setattr(host.installer, "checkpoint_pins", checkpoint_pins)
    monkeypatch.setattr(host.installer, "checkpoint_contract", lambda card: {
        "config_sha256": sha256(data["config.json"]), "index_sha256": sha256(data["model.safetensors.index.json"])})
    monkeypatch.setattr(host.installer, "validate", lambda value: value)
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(total=1 << 50, used=0, free=1 << 45))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".installer-owner.json").write_text(json.dumps({"deployment": DEPLOYMENT}))
    model = Path(model) if model is not None else (
        tmp_path / "srv/sparkring/tp2/checkpoints" / REPOSITORY.replace("/", "--") / REVISION)
    lock = {"id": DEPLOYMENT, "backend": backend, "site_input": {},
            "selection": {"profile": "fixture", "nodes": len(hosts), "image_id": IMAGE, "image_reference": IMAGE,
                          "model_repository": REPOSITORY, "model_revision": REVISION},
            "site": {"workspace": str(workspace), "ranks": [
                {"rank": rank, "host": target, "model": str(model), "cache": str(tmp_path / "cache"),
                 "repository": str(workspace / "source"), "reuse_verified_model": reuse}
                for rank, target in enumerate(hosts)]}}

    def call(operation, document=None, number=0):
        monkeypatch.setattr(sys, "stdin", io.StringIO("" if document is None else json.dumps(document)))
        return host.perform(operation, lock, number)
    return SimpleNamespace(data=data, pins=pinned, lock=lock, model=model, state=workspace / "installer",
                           tmp=tmp_path, call=call, row=lock["site"]["ranks"][0])


def state_directory(model):
    return Path(model).parent / ("." + Path(model).name + ".sparkring")


def journal(model):
    return json.loads((state_directory(model) / "journal.json").read_text())["files"]


FAKE_DOCKER = r'''#!{python}
"""Stand-in for the docker CLI: answers inspections and emulates the pinned image's hf_hub_download."""
import json, os, sys
config = json.load(open(os.environ["FAKE_DOCKER"]))
args = sys.argv[1:]
if args[:2] == ["--context", "default"]:
    args = args[2:]
with open(config["log"], "a") as log:
    log.write(json.dumps(args) + "\n")
if args[:2] == ["container", "inspect"]:
    sys.exit(0 if args[2] in config["containers"] else 1)
if args[:2] == ["image", "inspect"]:
    if args[2] in config["images"]:
        print(json.dumps([{"Id": args[2]}]))
        sys.exit(0)
    sys.exit(1)
if args[:1] == ["run"]:
    fields = dict(item.split("=", 1) for item in args[args.index("--mount") + 1].split(","))
    target = fields["src"]
    start = args.index("-c") + 2
    repository, revision, *names = args[start:]
    for count, name in enumerate(names):
        if config["fail_after"] is not None and count >= config["fail_after"]:
            print("connection reset by peer", file=sys.stderr)
            sys.exit(1)
        data = bytes.fromhex(config["files"][name])
        path = os.path.join(target, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".incomplete", "wb") as output:
            output.write(data)
        os.replace(path + ".incomplete", path)
        metadata = os.path.join(target, ".cache", "huggingface", "download", name + ".metadata")
        os.makedirs(os.path.dirname(metadata), exist_ok=True)
        with open(metadata, "w") as output:
            output.write(revision + "\n" + "etag\n1700000000.0\n")
    sys.exit(0)
sys.exit(2)
'''


class Docker:
    """A fake ``docker`` executable on PATH; ``runs()`` returns the argv of every ``docker run``."""

    def __init__(self, tmp_path, monkeypatch, data):
        self.directory = tmp_path / "fake-docker"
        self.directory.mkdir()
        self.config_path = self.directory / "config.json"
        self.log = self.directory / "log.jsonl"
        self.config = {"log": str(self.log), "containers": [], "images": [IMAGE], "fail_after": None,
                       "files": {name: value.hex() for name, value in data.items()}}
        self.save()
        program = self.directory / "docker"
        program.write_text(FAKE_DOCKER.replace("{python}", sys.executable))
        program.chmod(0o755)
        monkeypatch.setenv("PATH", str(self.directory) + os.pathsep + os.environ.get("PATH", ""))
        monkeypatch.setenv("FAKE_DOCKER", str(self.config_path))

    def save(self):
        self.config_path.write_text(json.dumps(self.config))

    def serve(self, name, value):
        self.config["files"][name] = value.hex()
        self.save()

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def runs(self):
        return [argv for argv in self.calls() if argv[:1] == ["run"]]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return environment(tmp_path, monkeypatch)


@pytest.fixture
def docker(tmp_path, monkeypatch, env):
    return Docker(tmp_path, monkeypatch, env.data)


# Audit hook ------------------------------------------------------------------------

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_AUDIT = {"installed": False, "protected": [], "violations": []}
_LOCAL = threading.local()
_OPEN = os.open


def _resolve(path, dir_fd=None):
    if isinstance(path, int):
        return os.readlink(f"/proc/self/fd/{path}")
    path = os.fsdecode(path)
    if not os.path.isabs(path):
        if isinstance(dir_fd, int) and dir_fd >= 0:
            path = os.path.join(os.readlink(f"/proc/self/fd/{dir_fd}"), path)
        else:
            path = os.path.join(os.getcwd(), path)
    return os.path.normpath(path)


def _forbidden_argv(argv):
    """The option of a subprocess argv that could change files SparkRing did not create, or ``None``."""
    if not isinstance(argv, (list, tuple)) or not argv:
        return None
    program, args = os.path.basename(os.fsdecode(argv[0])), [os.fsdecode(a) for a in argv[1:]]
    if program == "rsync":
        for value in args:
            if value in ("--inplace", "--times", "--archive") or value.startswith("--append"):
                return value
            if value.startswith("-") and not value.startswith("--") and set(value[1:]) & {"t", "a"}:
                return value
    if program == "docker" and "run" in args:
        for position, value in enumerate(args):
            if value in ("-v", "--volume") or value.startswith("--volume="):
                return value
            if value == "--mount" or value.startswith("--mount="):
                mount = args[position + 1] if value == "--mount" else value.split("=", 1)[1]
                items = mount.split(",")
                fields = dict(item.split("=", 1) for item in items if "=" in item)
                if "readonly" in items or fields.get("readonly") == "true":
                    continue
                source = fields.get("src") or fields.get("source") or ""
                if not (os.path.basename(source) == "fetch" and os.path.basename(os.path.dirname(source)).endswith(".sparkring")):
                    return mount
    return None


def _hook(event, args):
    if not _AUDIT["protected"] or getattr(_LOCAL, "busy", False):
        return
    _LOCAL.busy = True
    try:
        paths = []
        if event == "open":
            path, _, flags = args
            if isinstance(flags, int) and flags & WRITE_FLAGS:
                paths.append(_resolve(path, getattr(_LOCAL, "dir_fd", None)))
        elif event in ("os.remove", "os.rmdir", "shutil.rmtree"):
            paths.append(_resolve(args[0], args[1]))
        elif event == "os.mkdir":
            paths.append(_resolve(args[0], args[2]))
        elif event == "os.rename":
            paths += [_resolve(args[0], args[2]), _resolve(args[1], args[3])]
        elif event == "os.link":
            paths.append(_resolve(args[1], args[3]))
        elif event == "os.symlink":
            paths.append(_resolve(args[1], args[2]))
        elif event in ("os.chmod", "os.chown", "os.utime"):
            paths.append(_resolve(args[0], args[-1]))
        elif event == "os.truncate":
            paths.append(_resolve(args[0]))
        elif event in ("shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree", "shutil.move"):
            paths.append(_resolve(args[1]))
        elif event == "subprocess.Popen":
            option = _forbidden_argv(args[1])
            if option is not None:
                _AUDIT["violations"].append(f"{event}: {option} in {args[1]}")
                raise PermissionError(f"audit: forbidden subprocess option {option}")
        for path in paths:
            if any(path == root or path.startswith(root + os.sep) for root in _AUDIT["protected"]):
                _AUDIT["violations"].append(f"{event}: {path}")
                raise PermissionError(f"audit: {event} under a protected tree: {path}")
    finally:
        _LOCAL.busy = False


def _open(path, flags, mode=0o777, *, dir_fd=None):
    # The "open" audit event omits dir_fd, so the hook learns it from this wrapper.
    _LOCAL.dir_fd = dir_fd
    try:
        return _OPEN(path, flags, mode, dir_fd=dir_fd)
    finally:
        _LOCAL.dir_fd = None


@pytest.fixture(scope="session")
def audit_hook():
    """Install the audit hook once per session; it stays idle while no tree is protected."""
    if not _AUDIT["installed"]:
        sys.addaudithook(_hook)
        _AUDIT["installed"] = True
    return _AUDIT


@pytest.fixture
def audited(audit_hook, monkeypatch, tmp_path):
    """``with audited(*roots):`` refuses and records any change under ``roots`` during the block."""
    monkeypatch.setattr(os, "open", _open)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)

    @contextlib.contextmanager
    def protect(*roots):
        audit_hook["violations"].clear()
        audit_hook["protected"][:] = [os.path.normpath(str(root)) for root in roots]
        try:
            yield audit_hook["violations"]
        finally:
            audit_hook["protected"][:] = []
        assert audit_hook["violations"] == []
    yield protect
    audit_hook["protected"][:] = []


def tree_state(root):
    """Metadata and content of every entry at and below ``root``, without following symlinks or atime."""
    root = str(root)
    paths = [root]
    for directory, directories, files in os.walk(root):
        paths.extend(os.path.join(directory, name) for name in directories + files)
    result = {}
    for path in paths:
        info = os.lstat(path)
        item = {"ino": info.st_ino, "nlink": info.st_nlink, "size": info.st_size, "mode": info.st_mode,
                "uid": info.st_uid, "gid": info.st_gid, "mtime": info.st_mtime_ns, "ctime": info.st_ctime_ns}
        if stat.S_ISLNK(info.st_mode):
            item["target"] = os.readlink(path)
        elif stat.S_ISREG(info.st_mode):
            item["sha256"] = sha256(Path(path).read_bytes())
        result[os.path.relpath(path, root)] = item
    return result


def changes(before, after):
    """``{name: {field, ...}}`` of every changed entry; added or removed entries map to ``{"added"}``/``{"removed"}``."""
    result = {}
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            result[name] = {"added" if name not in before else "removed"}
            continue
        fields = {field for field in set(before[name]) | set(after[name]) if before[name].get(field) != after[name].get(field)}
        if fields:
            result[name] = fields
    return result


def only_links_changed(before, after, linked):
    """Whether only the ``linked`` entries changed, each in its link count and at most its change time.

    Adding a hard link always raises the link count; the change time moves with
    the kernel's coarse clock, so it may still equal a creation time from the
    same tick.
    """
    found = changes(before, after)
    return set(found) == set(linked) and all("nlink" in fields and fields <= {"nlink", "ctime"} for fields in found.values())


# Tests -----------------------------------------------------------------------------

def layouts(env):
    """Every source layout, each holding part of the fixture revision; returns roots and the plan entry."""
    data, root = env.data, env.tmp / "host"
    hub = root / "home/code/.cache/huggingface/hub"
    repo = hf_cache(hub, data, [SHARDS[0], "tokenizer.json", "README.md"])
    download = local_dir(root / "var/tmp/models/Qwen copy", data, [SHARDS[1], "chat_template.jinja", "config.json"])
    folder = plain_folder(root / "data/qwen", data, ["audio_tokenizer/config.json"])
    clone = git_lfs(root / "srv/git/fixture-model", data, ["audio_tokenizer/model.safetensors", "model.safetensors.index.json"])
    files = {
        SHARDS[0]: entry("link", repo / "blobs" / key(data, SHARDS[0]), data, SHARDS[0]),
        "tokenizer.json": entry("copy", repo / "blobs" / key(data, "tokenizer.json"), data, "tokenizer.json"),
        SHARDS[1]: entry("link", download / SHARDS[1], data, SHARDS[1]),
        "chat_template.jinja": entry("copy", download / "chat_template.jinja", data, "chat_template.jinja"),
        "config.json": entry("copy", download / "config.json", data, "config.json"),
        "audio_tokenizer/config.json": entry("copy", folder / "audio_tokenizer/config.json", data, "audio_tokenizer/config.json"),
        "audio_tokenizer/model.safetensors": entry("link", lfs_object(clone, data, "audio_tokenizer/model.safetensors"),
                                                   data, "audio_tokenizer/model.safetensors"),
        "model.safetensors.index.json": entry("copy", clone / "model.safetensors.index.json", data,
                                              "model.safetensors.index.json"),
    }
    return {"hf": hub, "local-dir": download, "plain": folder, "git": clone}, files


def test_adoption_never_changes_sources(env, audited):
    roots, files = layouts(env)
    before = {kind: tree_state(root) for kind, root in roots.items()}
    with audited(*roots.values()):
        result = env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": 0})
    assert result["complete"] and result["missing"] == [] and result["differs"] == []
    assert sorted(result["linked"]) == weights(env.data)
    assert result["bytes_written"] == sum(len(env.data[name]) for name in required(env.data) if name not in weights(env.data))
    linked_sources = {files[name]["source"] for name in weights(env.data)}
    for kind, root in roots.items():
        allowed = {os.path.relpath(source, root) for source in linked_sources if source.startswith(str(root) + os.sep)}
        assert only_links_changed(before[kind], tree_state(root), allowed), (kind, changes(before[kind], tree_state(root)))
    receipt = json.loads((env.state / "model.json").read_text())
    assert receipt["origin"] == "adopted-local-copy" and receipt["scope"] == "required-files"
    assert sorted(receipt["files"]) == required(env.data)
    assert {name: entry["origin"] for name, entry in journal(env.model).items()} == {
        name: "link" if name in weights(env.data) else "copy" for name in required(env.data)}


def test_a_differing_source_is_never_linked_and_is_recorded(env, audited):
    changed = {SHARDS[1]: env.data[SHARDS[1]].upper()}
    assert len(changed[SHARDS[1]]) == len(env.data[SHARDS[1]])
    folder = plain_folder(env.tmp / "lookalike", env.data, required(env.data), changed=changed)
    files = {name: entry("link" if name in weights(env.data) else "copy", folder / name, env.data, name)
             for name in required(env.data)}
    before = tree_state(folder)
    with audited(folder):
        result = env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": 1 << 30})
    assert result["differs"] == [{"name": SHARDS[1], "source": str(folder / SHARDS[1])}]
    assert result["missing"] == [SHARDS[1]] and not result["complete"]
    assert not (env.model / SHARDS[1]).exists() and SHARDS[1] not in journal(env.model)
    assert not (env.state / "model.json").exists()
    after = tree_state(folder)
    assert after[SHARDS[1]] == before[SHARDS[1]]
    assert set(changes(before, after)) == {name for name in weights(env.data) if name != SHARDS[1]}
    info = os.lstat(folder / SHARDS[1])
    record = json.loads((Path(place.RECORDS) / f"{info.st_dev}-{info.st_ino}.json").read_text())
    assert record["sha256"] == sha256(changed[SHARDS[1]]) and record["identity"] == [info.st_dev, info.st_ino]
    assert record["stats"][:4] == [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
    assert record["seen_as"] == str(folder / SHARDS[1])


def test_non_weight_files_are_copied_and_weight_files_linked(env):
    folder = plain_folder(env.tmp / "copy", env.data)
    files = {name: entry("link" if name in weights(env.data) else "copy", folder / name, env.data, name)
             for name in required(env.data)}
    result = env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": 0})
    assert result["complete"]
    for name in required(env.data):
        placed, source = os.lstat(env.model / name), os.lstat(folder / name)
        if name in weights(env.data):
            assert (placed.st_dev, placed.st_ino) == (source.st_dev, source.st_ino) and placed.st_nlink == 2
        else:
            assert placed.st_ino != source.st_ino and placed.st_nlink == 1 and source.st_nlink == 1
            assert placed.st_uid == os.geteuid() and not placed.st_mode & 0o022
        assert (env.model / name).read_bytes() == env.data[name]
    directory = os.open(env.model, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert sorted(place.listing(directory)[0]) == required(env.data)
    finally:
        os.close(directory)
    assert not any(state_directory(env.model).joinpath("receive").rglob("*.part"))


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM])
def test_exdev_and_eperm_fall_back_to_copy_within_tolerance_and_stop_beyond_it(env, monkeypatch, code):
    folder = plain_folder(env.tmp / "other-mount", env.data)
    files = {name: entry("link" if name in weights(env.data) else "copy", folder / name, env.data, name)
             for name in required(env.data)}
    link = os.link

    def refuse(source, target, *args, **kwargs):
        # Links from the folder fail as they do across mounts (EXDEV) or for immutable files (EPERM).
        if str(source).startswith("/proc/self/fd/") and os.readlink(source).startswith(str(folder) + os.sep):
            raise OSError(code, os.strerror(code))
        return link(source, target, *args, **kwargs)
    monkeypatch.setattr(os, "link", refuse)
    first = sorted(weights(env.data), key=lambda name: len(env.data[name]))[0]
    tolerance = len(env.data[first])
    before = tree_state(folder)
    result = env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": tolerance})
    assert changes(before, tree_state(folder)) == {}
    reason = errno.errorcode[code]
    copied = [name for name in weights(env.data) if name in result["copied"]]
    refused = sorted(item["name"] for item in result["link_failures"])
    assert sum(len(env.data[name]) for name in copied) <= tolerance and copied
    assert sorted(copied + refused) == weights(env.data) and result["linked"] == []
    assert all(item["reason"] == reason and item["source"] == str(folder / item["name"]) for item in result["link_failures"])
    assert result["missing"] == refused and not result["complete"]
    for name in copied:
        assert os.lstat(env.model / name).st_ino != os.lstat(folder / name).st_ino
        assert journal(env.model)[name]["origin"] == "copy"
    assert result["bytes_written"] == sum(len(env.data[name]) for name in required(env.data)
                                          if name not in refused)
    # Within a large enough tolerance every refused link becomes a copy.
    result = env.call("model-adopt", {"files": {name: files[name] for name in refused}, "receipts": [],
                                      "tolerance_bytes": 1 << 40})
    assert result["complete"] and sorted(result["copied"]) == refused and result["link_failures"] == []
    assert changes(before, tree_state(folder)) == {}


def adopt_weights(env, folder):
    """Adopt only the weight files of ``folder`` by hard links."""
    files = {name: entry("link", folder / name, env.data, name) for name in weights(env.data)}
    result = env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": 0})
    assert sorted(result["linked"]) == weights(env.data) and not result["complete"]
    return result


def test_fetch_writes_only_in_staging_and_places_verified_files(env, docker, audited):
    user = plain_folder(env.tmp / "home/code/models/qwen", env.data, weights(env.data))
    adopt_weights(env, user)
    missing = sorted(set(required(env.data)) - set(weights(env.data)))
    before = tree_state(user)
    wrong = "tokenizer.json"
    docker.serve(wrong, env.data[wrong].upper())
    with audited(user):
        with pytest.raises(ValueError, match="huggingface.co served different bytes for tokenizer.json"):
            env.call("model-fetch", {"names": missing})
    assert not (env.model / wrong).exists() and wrong not in journal(env.model)
    assert not (state_directory(env.model) / "fetch").exists()
    assert all((env.model / name).read_bytes() == env.data[name] for name in missing if name != wrong)
    docker.serve(wrong, env.data[wrong])
    with audited(user):
        result = env.call("model-fetch", {"names": [wrong]})
    assert result["complete"] and result["fetched"] == [wrong]
    runs = docker.runs()
    assert len(runs) == 2 and runs[1][-1] == wrong
    for argv in runs:
        assert argv[argv.index("--pull") + 1] == "never" and argv[argv.index("--runtime") + 1] == "runc"
        assert argv[argv.index("--user") + 1] == "0:0" and "-v" not in argv and "--volume" not in argv
        assert [argv[i + 1] for i, value in enumerate(argv) if value == "--mount"] == [
            f"type=bind,src={state_directory(env.model)}/fetch,dst=/fetch"]
        assert argv[argv.index("--name") + 1] == host.fetch_container(env.model)
        assert argv[argv.index(IMAGE) + 3:argv.index(IMAGE) + 5] == [REPOSITORY, REVISION]
    assert changes(before, tree_state(user)) == {}
    for name in weights(env.data):
        assert os.path.samefile(env.model / name, user / name)
    assert {journal(env.model)[name]["origin"] for name in missing} == {"hub"}
    assert json.loads((env.state / "model.json").read_text())["fetched"] == missing
    assert not (state_directory(env.model) / "fetch").exists()


def test_fetch_refuses_unpinned_present_or_concurrent_names(env, docker):
    user = plain_folder(env.tmp / "copy", env.data, weights(env.data))
    adopt_weights(env, user)
    for names, message in (([OPTIONAL[1]], "only required files of the pinned revision, not README.md"),
                           (["../config.json"], "only required files"),
                           ([SHARDS[0]], "already holds " + SHARDS[0]),
                           (["config.json", "config.json"], "distinct file names")):
        with pytest.raises(ValueError, match=message):
            env.call("model-fetch", {"names": names})
    docker.config["containers"] = [host.fetch_container(env.model)]
    docker.save()
    with pytest.raises(ValueError, match="still in progress in container " + host.fetch_container(env.model)):
        env.call("model-fetch", {"names": ["config.json"]})
    assert docker.runs() == [] and not (env.model / "config.json").exists()
    with pytest.raises(ValueError, match="expects"):
        env.call("model-fetch", {"names": ["config.json"], "extra": True})


def in_place_env(tmp_path, monkeypatch, folder):
    return environment(tmp_path, monkeypatch, reuse=True, model=folder)


def test_in_place_rows_are_verified_never_written_and_need_the_exact_name_set(tmp_path, monkeypatch, audited):
    folder = local_dir(tmp_path / "usb/qwen", contents())
    env = in_place_env(tmp_path, monkeypatch, folder)
    before = tree_state(folder)
    with audited(folder):
        assert env.call("model") == {"ok": True}
        assert env.call("model-check") == {"ok": True}
        assert env.call("model-settled") == {"ok": True}
    assert changes(before, tree_state(folder)) == {}
    receipt = json.loads((env.state / "model.json").read_text())
    assert receipt["origin"] == "in-place-verified-copy" and sorted(receipt["files"]) == required(env.data)
    assert not state_directory(folder).exists() and Path(env.row["cache"]).is_dir()
    # A file the serving engine would load from the same folder makes the copy inexact.
    write(folder / "added_tokens.json", b"{}")
    before = tree_state(folder)
    with audited(folder), pytest.raises(ValueError, match=r"differs from its recorded files \(added_tokens.json\); "
                                                         "SparkRing does not change it"):
        env.call("model-check")
    (env.state / "model.json").unlink()
    with audited(folder), pytest.raises(ValueError, match="also holds added_tokens.json, which the serving engine"):
        env.call("model")
    assert changes(before, tree_state(folder)) == {} and not (env.state / "model.json").exists()


def test_in_place_folder_without_download_metadata_gets_no_cache_directory(tmp_path, monkeypatch, audited):
    folder = plain_folder(tmp_path / "data/qwen", contents(), required(contents()))
    env = in_place_env(tmp_path, monkeypatch, folder)
    with audited(folder):
        assert env.call("model") == {"ok": True}
    assert not (folder / ".cache").exists() and sorted(os.listdir(folder)) == sorted(
        {name.split("/")[0] for name in required(env.data)})
    link = folder / "extra-link"
    link.symlink_to(folder / "config.json")
    with audited(folder), pytest.raises(ValueError, match="extra-link"):
        env.call("model-check")


def test_transfer_manifest_uses_required_names_for_in_place_donors(tmp_path, monkeypatch, audited):
    folder = local_dir(tmp_path / "usb/qwen", contents())
    env = in_place_env(tmp_path, monkeypatch, folder)
    with audited(folder):
        env.call("model")
        manifest = env.call("model-transfer-manifest")
    assert sorted(manifest["files"]) == sorted(manifest["sizes"]) == required(env.data)
    assert manifest["files"] == {name: sha256(env.data[name]) for name in required(env.data)}
    assert manifest["sizes"] == {name: len(env.data[name]) for name in required(env.data)}
    assert (manifest["repository"], manifest["revision"]) == (REPOSITORY, REVISION)


def test_model_settled_detects_changes_during_loading(env):
    folder = plain_folder(env.tmp / "var/tmp/models/qwen", env.data)
    files = {name: entry("link" if name in weights(env.data) else "copy", folder / name, env.data, name)
             for name in required(env.data)}
    assert env.call("model-adopt", {"files": files, "receipts": [], "tolerance_bytes": 0})["complete"]
    assert env.call("model") == {"ok": True}
    assert env.call("model-settled") == {"ok": True}
    # Another tool rewrites a linked weight file in place while the model loads; the
    # pause lets the kernel's coarse clock give the write a new modification time.
    time.sleep(0.05)
    with open(folder / SHARDS[0], "r+b") as stream:
        stream.write(b"F")
    with pytest.raises(ValueError, match=rf"Checkpoint files changed while the model was loading on Node 0 \({SHARDS[0]}\)"):
        env.call("model-settled")


def test_owned_rows_without_receipt_adopt_and_fetch_through_a_plain_runner(env, docker, monkeypatch):
    user = plain_folder(env.tmp / "copy", env.data, weights(env.data))
    adopt_weights(env, user)
    current = object.__new__(installer_runner.Runner)
    current.lock = env.lock
    commands = []

    def ssh(target, argv, *, data=None, timeout=7200):
        commands.append(argv)
        monkeypatch.setattr(sys, "stdin", io.StringIO(data.decode() if data else ""))
        return json.dumps(host.perform(argv[-2], env.lock, int(argv[-1])))
    monkeypatch.setattr(installer_runner, "ssh", ssh)
    for operation in ("model", "model-check", "model-settled"):
        result = current._call(env.row["host"], ["installer", operation, "0"], 60)
        assert result["returncode"] == 0, result["stderr"]
    missing = sorted(set(required(env.data)) - set(weights(env.data)))
    assert [argv[argv.index(IMAGE) + 5:] for argv in docker.runs()] == [missing]
    receipt = json.loads((env.state / "model.json").read_text())
    assert receipt["origin"] == "adopted-local-copy" and receipt["fetched"] == missing
    assert [argv[-2] for argv in commands] == ["model", "model-check", "model-settled"]
    assert all(argv[0] == "python3" for argv in commands)


FORBIDDEN_CALLS = {"chmod", "lchmod", "fchmod", "chown", "lchown", "fchown", "utime", "truncate", "ftruncate",
                   "copystat", "copymode", "copy2", "copyfile", "copytree", "move", "rmtree", "write_text",
                   "write_bytes"}


def forbidden_calls(path):
    """Calls in ``path`` that change file metadata, copy with metadata, or open files for writing by path."""
    found = []
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
            found.append(f"{path}:{node.lineno} {node.func.attr}")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            modes = [arg for arg in node.args[1:2] + [k.value for k in node.keywords if k.arg == "mode"]]
            if any(isinstance(mode, ast.Constant) and set(str(mode.value)) & set("wax+") for mode in modes):
                found.append(f"{path}:{node.lineno} open")
    return found


def test_model_code_never_changes_file_metadata(env, docker, audited, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    # The rank operations, the placement primitives, the fabric receiver and sender, the code that builds the
    # rsync argv, and the release of checkpoint directories.
    for module in ("scripts/installer_host.py", "runtime/host/checkpoint_place.py", "runtime/host/fabric_stream.py",
                   "runtime/host/install_assets.py", "runtime/host/checkpoints.py"):
        assert forbidden_calls(root / module) == [], module
    roots, files = layouts(env)
    before = {kind: tree_state(path) for kind, path in roots.items()}
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / ".installer-owner.json").write_text(json.dumps({"deployment": "e" * 64}))
    with audited(*roots.values()):
        partial = {name: item for name, item in files.items() if name != "config.json"}
        result = env.call("model-adopt", {"files": partial, "receipts": [], "tolerance_bytes": 0})
        assert result["missing"] == ["config.json"]
        assert env.call("model-fetch", {"names": ["config.json"]})["complete"]
        for operation in ("model", "model-check", "model-settled", "model-transfer-manifest"):
            env.call(operation)
        manifest = env.call("model-transfer-manifest")
        assert env.call("model-transfer-prepare", manifest)["needed"] == []
        assert env.call("model-transfer-complete", manifest)["complete"]
        assert env.call("model-reuse-receipt", {"workspace": str(previous), "deployment": "e" * 64}) == {"reused": False}
    for kind, path in roots.items():
        allowed = {os.path.relpath(files[name]["source"], path) for name in weights(env.data)
                   if files[name]["source"].startswith(str(path) + os.sep)}
        assert only_links_changed(before[kind], tree_state(path), allowed), (kind, changes(before[kind], tree_state(path)))


def rewrite_keeping_mtime(path):
    """What an in-place writer that keeps modification times does: same size, other bytes, the old mtime."""
    info = os.stat(path)
    time.sleep(0.05)  # the kernel's coarse clock then gives the write a new change time
    with open(path, "r+b") as stream:
        stream.write(b"X")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


def adoption_document(env, folder):
    files = {name: entry("link" if name in weights(env.data) else "copy", folder / name, env.data, name)
             for name in required(env.data)}
    return {"files": files, "receipts": [], "tolerance_bytes": 0}


def test_a_source_rewritten_after_hashing_is_not_linked_or_trusted(env, monkeypatch):
    folder = plain_folder(env.tmp / "var/tmp/models/qwen", env.data)
    name = SHARDS[0]
    inode = os.stat(folder / name).st_ino
    real, done = place.hash_descriptor, []

    def hash_then_rewrite(fd, size):
        digest = real(fd, size)
        if os.fstat(fd).st_ino == inode and not done:
            done.append(True)
            rewrite_keeping_mtime(folder / name)
        return digest
    monkeypatch.setattr(place, "hash_descriptor", hash_then_rewrite)
    result = env.call("model-adopt", adoption_document(env, folder))
    monkeypatch.setattr(place, "hash_descriptor", real)
    # The second read of the descriptor found other bytes, so the file was neither linked nor recorded as verified.
    assert name not in result["linked"] and [item["name"] for item in result["unavailable"]] == [name]
    assert "changed after SparkRing opened it" in result["unavailable"][0]["error"]
    assert not result["complete"] and not (env.model / name).exists() and name not in journal(env.model)
    assert not (env.state / "model.json").exists()


def test_other_receipts_keep_a_change_made_after_the_link(env, monkeypatch):
    folder = plain_folder(env.tmp / "var/tmp/models/qwen", env.data)
    names = required(env.data)
    workspace = env.tmp / "srv/sparkring/tp2/in-place"
    (workspace / "installer").mkdir(parents=True)
    (workspace / ".installer-owner.json").write_text(json.dumps({"deployment": "id-in-place"}))
    other = workspace / "installer/model.json"
    other.write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "path": str(folder),
        "files": {n: sha256(env.data[n]) for n in names},
        "file_stats": {n: place.stats(os.lstat(folder / n)) for n in names},
        "origin": "in-place-verified-copy", "scope": "required-files"}))
    name = SHARDS[0]
    real = place.place_link

    def link_then_rewrite(dir_fd, fd, placed, records, **kwargs):
        stats = real(dir_fd, fd, placed, records, **kwargs)
        if placed == name:
            # The owner rewrites the linked file in place after SparkRing linked it, keeping its mtime.
            rewrite_keeping_mtime(folder / name)
        return stats
    monkeypatch.setattr(place, "place_link", link_then_rewrite)
    document = adoption_document(env, folder)
    document["receipts"] = [{"path": str(other), "deployment": "id-in-place"}]
    time.sleep(0.05)
    result = env.call("model-adopt", document)
    recorded = json.loads(other.read_text())["file_stats"]
    # The other shard kept the stats measured right after SparkRing's link, so its entry is refreshed; the
    # rewritten one no longer has them, so its entry stays as it was and the next start hashes it again.
    assert result["refreshed"] == [str(other)]
    assert recorded[SHARDS[1]] == place.stats(os.lstat(folder / SHARDS[1]))
    assert recorded[name] != place.stats(os.lstat(folder / name))
    row = {**env.row, "model": str(folder), "reuse_verified_model": True}
    with pytest.raises(ValueError, match=f"differs from its recorded files \\({name}\\)"):
        host.verify_model(env.lock, row, other)


def shared_parent(tmp_path):
    """A checkpoint directory path whose parent every account can write, as a site file may name."""
    shared = tmp_path / "data/models"
    shared.mkdir(parents=True)
    os.chmod(shared, 0o777)
    return shared / REVISION


@pytest.mark.parametrize("operation", ["model-fetch", "model-transfer-prepare", "model-adopt"])
def test_a_checkpoint_directory_that_other_accounts_could_redirect_is_refused(tmp_path, monkeypatch, operation):
    env = environment(tmp_path, monkeypatch, model=shared_parent(tmp_path))
    fake = Docker(tmp_path, monkeypatch, env.data)
    document = {"model-fetch": {"names": ["config.json"]},
                "model-transfer-prepare": {"repository": REPOSITORY, "revision": REVISION,
                                           "files": {"config.json": sha256(env.data["config.json"])},
                                           "sizes": {"config.json": len(env.data["config.json"])}},
                "model-adopt": {"files": {}, "receipts": [], "tolerance_bytes": 0}}[operation]
    with pytest.raises(ValueError, match="can be changed by other accounts"):
        env.call(operation, document)
    # Nothing was created below the shared directory and no container ran.
    assert os.listdir(env.model.parent) == [] and fake.runs() == []


@pytest.mark.parametrize("moment", ["before", "during"])
def test_a_replaced_fetch_directory_stops_the_download(env, docker, monkeypatch, moment):
    state = state_directory(env.model)
    elsewhere = env.tmp / "another-account"
    (elsewhere / "fetch").mkdir(parents=True)

    def replace():
        # What an account that could rename SparkRing's state directory would do; claim() keeps other
        # accounts from it, and the check detects it.
        os.rename(state, str(state) + ".moved")
        os.symlink(elsewhere, state)
    if moment == "before":
        require = host._require_image
        monkeypatch.setattr(host, "_require_image", lambda card: (require(card), replace()))
        with pytest.raises(ValueError, match="is no longer SparkRing's staging directory; nothing was downloaded"):
            env.call("model-fetch", {"names": ["config.json"]})
        assert docker.runs() == []
    else:
        real = host.run

        def run(argv, **kwargs):
            if list(argv[:2]) == ["docker", "run"]:
                replace()
            return real(argv, **kwargs)
        monkeypatch.setattr(host, "run", run)
        with pytest.raises(ValueError, match="was replaced while checkpoint files were downloaded; nothing was placed"):
            env.call("model-fetch", {"names": ["config.json"]})
    assert not (env.model / "config.json").exists()


def test_rsync_staging_is_checked_by_path_before_its_files_are_placed(env, monkeypatch):
    names = ["config.json"]
    manifest = {"repository": REPOSITORY, "revision": REVISION,
                "files": {n: sha256(env.data[n]) for n in names}, "sizes": {n: len(env.data[n]) for n in names}}
    prepared = env.call("model-transfer-prepare", manifest)
    staged = Path(prepared["rsync"])
    assert staged == state_directory(env.model) / "receive/rsync" and staged.is_dir()
    write(staged / "config.json", env.data["config.json"])
    # rsync wrote by path; a path that no longer resolves to the staging directory places nothing.
    other = env.tmp / "elsewhere"
    other.mkdir()
    monkeypatch.setattr(host, "_rsync_staging", lambda claimed: str(other))
    with pytest.raises(ValueError, match="was replaced while rsync wrote to it; nothing was placed from it"):
        env.call("model-transfer-complete", manifest)
    assert not (env.model / "config.json").exists()


def test_audit_hook_refuses_writes_and_forbidden_argv_under_protected_trees(env, audited):
    folder = plain_folder(env.tmp / "protected", env.data, ["config.json"])
    with pytest.raises(AssertionError):
        with audited(folder):
            with pytest.raises(PermissionError):
                os.utime(folder / "config.json")
            with pytest.raises(PermissionError):
                fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.open("config.json", os.O_WRONLY, dir_fd=fd)
                finally:
                    os.close(fd)
    assert _forbidden_argv(["rsync", "-rlt", "a", "b"]) == "-rlt"
    assert _forbidden_argv(["rsync", "-r", "--no-perms", "--inplace"]) == "--inplace"
    assert _forbidden_argv(["docker", "run", "--mount", "type=bind,src=/home/code/models,dst=/model"]) is not None
    assert _forbidden_argv(["docker", "run", "--mount", "type=bind,src=/srv/x/.r.sparkring/fetch,dst=/fetch"]) is None
    assert _forbidden_argv(["rsync", "-r", "--no-perms", "--no-owner", "--no-group", "--protect-args"]) is None
