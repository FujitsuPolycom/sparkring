"""Tests for the read-only checkpoint survey that Node A runs on every Spark.

Every test builds a fixture host tree under ``tmp_path`` (``options["root"]``)
with a synthetic ``/proc/self/mountinfo`` and ``/etc/passwd``, and a synthetic
checkpoint pinned in the manifest format ``sparkring-checkpoint-pins/v1``:
three weight files larger than ``small_file_bytes`` (so they are never hashed
unless a test raises the limit), an index, and small configuration files,
one of them nested. Docker is replaced by a fake command in every test.
"""
import ast
import builtins
import dis
import hashlib
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from runtime.host import checkpoint_search as search

POSIX = pytest.mark.skipif(sys.platform != "linux", reason=(
    "Spark hosts run Linux; this fixture needs POSIX symlinks, hard links, change times or statvfs"))

REPOSITORY = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REVISION = "629bc3218833a38b475b719f34aa571666f4a03e"
MAIN = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
INDEX = "model.safetensors.index.json"
WEIGHTS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
           "model-00003-of-00003.safetensors"]
SLUG = "local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
REPO = "models--" + SLUG
OWNED = f"/srv/sparkring/tp2/checkpoints/{SLUG}/{REVISION}"


def pattern(seed, size):
    block = hashlib.sha256(str(seed).encode()).digest()
    return (block * (size // len(block) + 1))[:size]


CONTENT = {
    "config.json": b'{"architectures": ["Qwen3_8ForCausalLM"], "model_type": "qwen3_8_qad"}\n',
    "tokenizer.json": b'{"model": {"vocab": {"a": 1}}}\n',
    "processor/preprocessor_config.json": b'{"size": 224}\n',
    WEIGHTS[0]: pattern(1, 3000),
    WEIGHTS[1]: pattern(2, 3000),
    WEIGHTS[2]: pattern(3, 2600),
    "README.md": b"# fixture\n",
}
CONTENT[INDEX] = json.dumps({"metadata": {}, "weight_map": {f"layers.{n}": name for n, name in enumerate(WEIGHTS)}}).encode()
LFS = {INDEX, *WEIGHTS}
REQUIRED = sorted(set(CONTENT) - {"README.md"})
SMALL = [name for name in REQUIRED if name not in WEIGHTS]
MAIN_CONFIG = b'{"architectures": ["Qwen3_8ForCausalLM"], "model_type": "qwen3_8_main_branch"}\n'
OTHER = pattern(9, 3000)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def xet(data):
    return sha256(b"xet:" + data)


def key(name, data):
    return sha256(data) if name in LFS else git_blob(data)


PINS = {"schema": "sparkring-checkpoint-pins/v1", "repository": REPOSITORY, "revision": REVISION, "index": INDEX,
        "weights": sorted(WEIGHTS), "optional": ["README.md"],
        "files": {name: {"size": len(data), "sha256": sha256(data), "git_blob": git_blob(data),
                         **({"lfs": True, "xet_hash": xet(data)} if name in LFS else {})}
                  for name, data in CONTENT.items()}}


class Host:
    """A fixture host tree: mount table, accounts and files below ``root``."""

    def __init__(self, base):
        self.root = base / "host"
        self.mounts = [(21, 1, "259:2", "/", "/", "ext4", "/dev/nvme0n1p2")]
        self.users = [("root", 0, "/root"), ("code", 1000, "/home/code")]
        for directory in ("/root", "/home/code", "/etc", "/proc/self", "/var/tmp", "/srv"):
            self.path(directory).mkdir(parents=True, exist_ok=True)
        self.save()

    def path(self, name):
        return self.root / name.lstrip("/")

    def save(self):
        rows = [f"{number} {parent} {device} {root} {point.replace(' ', chr(92) + '040')} rw,relatime shared:1 - "
                f"{kind} {source} rw" for number, parent, device, root, point, kind, source in self.mounts]
        self.write("/proc/self/mountinfo", "\n".join(rows) + "\n")
        self.write("/etc/passwd", "".join(f"{name}:x:{uid}:{uid}::{home}:/bin/bash\n"
                                          for name, uid, home in self.users))

    def mount(self, point, kind, *, number, device="259:2", source="/dev/nvme0n1p2", root="/"):
        self.mounts.append((number, 21, device, root, point, kind, source))
        self.save()

    def user(self, name, uid, home, create=True):
        self.users.append((name, uid, home))
        if create:
            self.path(home).mkdir(parents=True, exist_ok=True)
        self.save()

    def write(self, name, data, mtime=None):
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode())
        if mtime is not None:
            os.utime(target, (mtime, mtime))
        return target

    def symlink(self, name, target):
        link = self.path(name)
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)

    def copy(self, folder, *, skip=(), replace=None):
        """A plain folder holding the pinned files, the optional README included."""
        for name, data in CONTENT.items():
            if name not in skip:
                self.write(folder + "/" + name, (replace or {}).get(name, data))
        return folder

    def cache(self, hub, repository, commit, contents, *, refs=None, store=False, symlinks=True):
        """A Hugging Face cache repository folder with one snapshot."""
        folder = hub + "/" + repository
        for name, data in contents.items():
            blob = key(name, data)
            if not symlinks:
                self.write(f"{folder}/snapshots/{commit}/{name}", data)
                continue
            if store and name in LFS:
                shared = xet(data)
                self.write(f"{hub}/blobs/{shared[:2]}/{shared}", data)
                self.symlink(f"{folder}/blobs/{blob}", f"../../blobs/{shared[:2]}/{shared}")
            else:
                self.write(f"{folder}/blobs/{blob}", data)
            self.symlink(f"{folder}/snapshots/{commit}/{name}", "../" * (name.count("/") + 2) + "blobs/" + blob)
        for branch, value in (refs if refs is not None else {"main": commit}).items():
            self.write(f"{folder}/refs/{branch}", value)
        return folder

    def local_dir(self, folder, *, etags=None, written=None, legacy=False, contents=None, commit=REVISION):
        """An ``hf download --local-dir`` folder with its download records."""
        base = ".huggingface/download" if legacy else ".cache/huggingface/download"
        for name, data in (contents or CONTENT).items():
            target = self.write(folder + "/" + name, data)
            stamp = (written or {}).get(name, target.stat().st_mtime + 1)
            etag = (etags or {}).get(name, key(name, data))
            self.write(f"{folder}/{base}/{name}.metadata", f"{commit}\n{etag}\n{stamp}\n")
        return folder


class Docker:
    """A fake ``docker`` command answering the survey's five calls."""

    def __init__(self, containers=(), volumes=(), info=None):
        self.containers, self.volumes = list(containers), list(volumes)
        self.info = info if info is not None else {"Driver": "overlay2", "DockerRootDir": "/var/lib/docker",
                                                   "SecurityOptions": ["name=seccomp,profile=builtin"]}
        self.calls, self.failures, self.delays, self.clock = [], {}, {}, None

    def run(self, argv, **kwargs):
        assert argv[:3] == ["docker", "--context", "default"]
        call = " ".join(argv[3:5]) if argv[3] == "volume" else argv[3]
        self.calls.append((call, kwargs))
        if self.clock is not None:
            self.clock[0] += self.delays.get(call, 0)
        failure = self.failures.get(call)
        if isinstance(failure, BaseException):
            raise failure
        if failure is not None:
            return SimpleNamespace(returncode=failure[0], stdout="", stderr=failure[1])
        if call == "ps":
            output = "".join(item["Id"] + "\n" for item in self.containers)
        elif call == "inspect":
            output = json.dumps([item for item in self.containers if item["Id"] in argv[4:]])
        elif call == "volume ls":
            output = "".join(item["Name"] + "\n" for item in self.volumes)
        elif call == "volume inspect":
            output = json.dumps([item for item in self.volumes if item["Name"] in argv[5:]])
        else:
            output = json.dumps(self.info)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")


@pytest.fixture(autouse=True)
def docker(monkeypatch):
    fake = Docker()
    monkeypatch.setattr(search, "subprocess", SimpleNamespace(
        run=fake.run, TimeoutExpired=subprocess.TimeoutExpired, DEVNULL=subprocess.DEVNULL))
    return fake


@pytest.fixture
def clock(monkeypatch):
    """A fake clock for the survey's budgets; tests advance ``clock[0]``."""
    value = [1000.0]
    monkeypatch.setattr(search, "time", SimpleNamespace(monotonic=lambda: value[0]))
    return value


def settings(host, **changes):
    return search.options(**{"owned": OWNED, "operator": "code", "root": str(host.root), "small_file_bytes": 1024,
                             "hash_bytes": 1 << 20, **changes})


def run(host, **changes):
    return search.survey(json.loads(search.compact(PINS)), settings(host, **changes))


def context(host, **changes):
    return {"root": str(host.root), "mounts": search.mount_table(str(host.root)),
            "options": settings(host, **changes), "pins": json.loads(search.compact(PINS))}


def candidate(result, path):
    found = [item for item in result["candidates"] if item["path"] == path]
    assert found, sorted(item["path"] for item in result["candidates"])
    return found[0]


def paths(result):
    return {item["path"] for item in result["candidates"]}


def states(item):
    return {name: (value["state"], value["evidence"]) for name, value in item["files"].items()}


def record_access(monkeypatch, calls=("scandir", "stat", "lstat", "readlink", "open", "statvfs", "listdir")):
    """Record every path the survey passes to the named ``os`` calls and to ``open``."""
    seen = []

    def recording(function):
        def call(path, *args, **kwargs):
            seen.append(os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path))
            return function(path, *args, **kwargs)
        return call

    for name in calls:
        if hasattr(os, name):
            monkeypatch.setattr(os, name, recording(getattr(os, name)))
    monkeypatch.setattr(builtins, "open", recording(builtins.open))
    return seen


def host_paths(host, seen):
    prefix = str(host.root)
    return [path[len(prefix):].replace("\\", "/") for path in seen if path.startswith(prefix)]


def test_probe_source_runs_in_a_fresh_interpreter_with_an_empty_environment(tmp_path):
    host = Host(tmp_path)
    host.copy("/var/tmp/models/qwen copy")
    source = search.probe_source(PINS, settings(host, docker_seconds=0))
    environment = {} if sys.platform != "win32" else {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
    done = subprocess.run([sys.executable, "-I", "-B", "-"], input=source, env=environment, cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["schema"] == "sparkring-checkpoint-survey/v1"
    found = candidate(result, "/var/tmp/models/qwen copy")
    assert found["counts"]["match"] == len(SMALL) and found["counts"]["size-only"] == len(WEIGHTS)
    assert any(error.startswith("docker: unavailable") for error in result["search"]["errors"])

    header = {"hashlib", "json", "os", "stat", "subprocess", "sys", "time"}
    allowed = set(dir(builtins)) | header | {function.__name__ for function in search.SHIPPED}

    def codes(code):
        yield code
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                yield from codes(constant)

    for function in search.SHIPPED:
        for code in codes(function.__code__):
            for instruction in dis.get_instructions(code):
                if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"):
                    assert instruction.argval in allowed, (function.__name__, instruction.argval)
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = {alias.name for node in imports for alias in node.names}
    assert len(imports) == 1 and isinstance(imports[0], ast.Import) and modules == header
    assert modules <= sys.stdlib_module_names
    assert "/usr/bin/sparkring" not in source


@POSIX
def test_hf_cache_is_identified_by_blob_names_without_reading_weights(tmp_path, monkeypatch):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    host.cache(hub, REPO, REVISION, {name: CONTENT[name] for name in REQUIRED})
    opened = []
    real_open = os.open

    def opening(path, *args, **kwargs):
        opened.append(os.fsdecode(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", opening)
    result = run(host)
    found = candidate(result, hub + "/" + REPO)
    assert found["layout"] == "hf-cache" and found["commit"] == REVISION and found["branches"] == ["main"]
    assert states(found) == {name: ("match", "hub-named") for name in REQUIRED}
    assert found["files"][WEIGHTS[0]]["source"] == f"{hub}/{REPO}/blobs/{sha256(CONTENT[WEIGHTS[0]])}"
    assert found["files"][WEIGHTS[0]]["kind"] == "blob" and found["counts"]["match"] == len(REQUIRED)
    assert found["home"] == {"account": "code", "kind": "operator"} and "hub" in found["found_by"]
    # Blob names identify content: no blob or snapshot file is opened, only refs.
    assert not [path for path in host_paths(host, opened) if "/blobs/" in path or "/snapshots/" in path]
    assert [path for path in host_paths(host, opened) if path.endswith("/refs/main")]


@POSIX
def test_shared_blob_store_two_hop_links_resolve_inside_the_hub(tmp_path):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    host.cache(hub, REPO, REVISION, {name: CONTENT[name] for name in REQUIRED}, store=True)
    # A second hub whose store holds a pinned blob that no repository folder links to.
    other = "/root/.cache/huggingface/hub"
    shared = xet(CONTENT[WEIGHTS[2]])
    host.write(f"{other}/blobs/{shared[:2]}/{shared}", CONTENT[WEIGHTS[2]])
    host.path(other + "/models--someone--unrelated").mkdir(parents=True)
    result = run(host)
    found = candidate(result, hub + "/" + REPO)
    assert states(found) == {name: ("match", "hub-named") for name in REQUIRED}
    store = xet(CONTENT[WEIGHTS[0]])
    assert found["files"][WEIGHTS[0]]["source"] == f"{hub}/blobs/{store[:2]}/{store}"
    assert found["files"][WEIGHTS[0]]["kind"] == "shared-blob"
    assert hub + "/blobs" not in paths(result)
    alone = candidate(result, other + "/blobs")
    assert alone["layout"] == "hf-blob-store" and states(alone) == {WEIGHTS[2]: ("match", "hub-named")}
    # The snapshot link resolves in two hops: snapshot -> repository blob -> shared store.
    ctx = context(host)
    state, path, info = search.safe_resolve(ctx, f"{hub}/{REPO}/snapshots/{REVISION}/{WEIGHTS[0]}", inside=hub)
    assert (state, path, info.st_size) == ("ok", f"{hub}/blobs/{store[:2]}/{store}", len(CONTENT[WEIGHTS[0]]))


def test_hf_cache_without_symlinks_is_classified_from_snapshot_files(tmp_path):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    host.cache(hub, REPO, REVISION, {name: CONTENT[name] for name in REQUIRED}, symlinks=False,
               refs={"main": REVISION})
    result = run(host)
    assert hub + "/" + REPO not in paths(result)
    found = candidate(result, f"{hub}/{REPO}/snapshots/{REVISION}")
    assert found["layout"] == "hf-snapshot" and found["commit"] == REVISION and found["branches"] == ["main"]
    assert states(found) == {**{name: ("match", "hashed") for name in SMALL},
                             **{name: ("size-only", "size") for name in WEIGHTS}}


@POSIX
def test_blob_links_leaving_the_hub_or_looping_are_rejected_quickly(tmp_path, monkeypatch):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    contents = {name: CONTENT[name] for name in REQUIRED if name not in WEIGHTS}
    folder = host.cache(hub, REPO, REVISION, contents)
    host.write("/var/lib/outside/blob-data", CONTENT[WEIGHTS[0]])
    host.symlink(f"{folder}/blobs/{sha256(CONTENT[WEIGHTS[0]])}", "/var/lib/outside/blob-data")
    looping = sha256(CONTENT[WEIGHTS[1]])
    host.symlink(f"{folder}/blobs/{looping}", looping)
    host.symlink(f"{folder}/blobs/{sha256(CONTENT[WEIGHTS[2]])}", "partner")
    host.symlink(f"{folder}/blobs/partner", sha256(CONTENT[WEIGHTS[2]]))
    readlinks, lstats = [], []
    real_readlink, real_lstat = os.readlink, os.lstat

    def readlink(path, *args, **kwargs):
        readlinks.append(os.fsdecode(path))
        return real_readlink(path, *args, **kwargs)

    def lstat(path, *args, **kwargs):
        lstats.append(os.fsdecode(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", readlink)
    monkeypatch.setattr(os, "lstat", lstat)
    result = run(host)
    found = candidate(result, folder)
    assert {name: found["files"][name]["state"] for name in WEIGHTS} == dict.fromkeys(WEIGHTS, "outside")
    assert all(found["files"][name]["state"] == "match" for name in SMALL)
    # Each looping blob costs at most nine links before it is given up.
    assert len([path for path in readlinks if "/blobs/" in path]) <= 1 + 9 + 9
    assert not [path for path in host_paths(host, lstats) if path.startswith("/var/lib/outside")]


@POSIX
def test_every_repository_folder_is_checked_by_content(tmp_path):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    renamed = host.cache(hub, REPO + "-4p89", "b184bb5650367c3e934c7849407be9da3671e7f5",
                         {name: CONTENT[name] for name in REQUIRED})
    fork = host.cache(hub, "models--huginnfork--Qwen3.8-Flash-Next-NVFP4-Abliterated", MAIN,
                      {**{name: CONTENT[name] for name in REQUIRED}, WEIGHTS[1]: OTHER})
    result = run(host)
    assert states(candidate(result, renamed)) == {name: ("match", "hub-named") for name in REQUIRED}
    forked = candidate(result, fork)
    assert states(forked)[WEIGHTS[1]] == ("differs", "hub-named")
    assert all(states(forked)[name] == ("match", "hub-named") for name in REQUIRED if name != WEIGHTS[1])
    assert forked["counts"]["differs"] == 1


@pytest.mark.parametrize("case,name,expected", [
    ("up-to-date", WEIGHTS[0], ("match", "hub-metadata")),
    ("stale", WEIGHTS[0], ("size-only", "size")),
    ("stale", "config.json", ("match", "hashed")),
    ("wrong-etag", WEIGHTS[0], ("differs", "hub-metadata")),
    ("git-blob-etag", "config.json", ("match", "hub-metadata")),
    ("git-blob-etag", WEIGHTS[0], ("match", "hub-metadata")),
    ("legacy-location", WEIGHTS[0], ("match", "hub-metadata")),
    ("corrupt", WEIGHTS[0], ("size-only", "size")),
    ("corrupt", "config.json", ("match", "hashed")),
])
def test_local_dir_metadata_rules(tmp_path, case, name, expected):
    host = Host(tmp_path)
    folder = "/var/tmp/models/Qwen3.8-Flash-Next-NVFP4"
    etags = {"wrong-etag": {name: "0" * 64}, "git-blob-etag": {name: git_blob(CONTENT[name])}}.get(case)
    host.local_dir(folder, etags=etags, legacy=case == "legacy-location")
    record = f"{folder}/.cache/huggingface/download/{name}.metadata"
    if case == "stale":
        stamp = host.path(folder + "/" + name).stat().st_mtime - 100
        host.write(record, f"{REVISION}\n{key(name, CONTENT[name])}\n{stamp}\n")
    if case == "corrupt":
        host.write(record, "not a download record")
    found = candidate(run(host), folder)
    assert found["layout"] == "local-dir" and found["commit"] == REVISION
    assert states(found)[name] == expected


def test_arbitrarily_named_folders_are_found_and_small_files_hashed_within_budget(tmp_path):
    host = Host(tmp_path)
    first = host.copy("/var/tmp/models/My Qwen (copy)")
    second = host.copy("/data/archive/2026/qwen-b")
    result = run(host, hash_bytes=sum(len(CONTENT[name]) for name in SMALL))
    assert {name: states(candidate(result, first))[name] for name in SMALL} == dict.fromkeys(SMALL, ("match", "hashed"))
    assert {name: states(candidate(result, second))[name] for name in SMALL} == dict.fromkeys(SMALL, ("size-only", "size"))
    for folder in (first, second):
        assert {name: states(candidate(result, folder))[name] for name in WEIGHTS} == \
            dict.fromkeys(WEIGHTS, ("size-only", "size"))
        assert candidate(result, folder)["layout"] == "folder" and candidate(result, folder)["found_by"] == ["folder"]


@POSIX
def test_main_revision_cache_supplies_everything_but_config(tmp_path):
    host = Host(tmp_path)
    hub = "/home/code/.cache/huggingface/hub"
    folder = host.cache(hub, REPO, MAIN, {**{name: CONTENT[name] for name in REQUIRED}, "config.json": MAIN_CONFIG},
                        refs={"main": MAIN, "pr/4": MAIN})
    found = candidate(run(host), folder)
    assert found["commit"] == MAIN and found["branches"] == ["main", "pr/4"]
    assert states(found)["config.json"] == ("differs", "hub-named")
    assert all(states(found)[name] == ("match", "hub-named") for name in REQUIRED if name != "config.json")
    assert found["counts"]["match"] == len(REQUIRED) - 1


@POSIX
def test_partial_downloads_are_partial_seeds(tmp_path):
    host = Host(tmp_path)
    pointer = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + sha256(CONTENT[INDEX]).encode()
               + b"\nsize %d\n" % len(CONTENT[INDEX]))
    plain = host.copy("/var/tmp/models/partial", replace={WEIGHTS[1]: CONTENT[WEIGHTS[1]][:1000], INDEX: pointer})
    host.write(plain + "/" + WEIGHTS[0] + ".aria2", b"control")
    local = host.local_dir("/var/tmp/models/interrupted",
                           contents={name: CONTENT[name] for name in REQUIRED if name != WEIGHTS[1]})
    host.write(f"{local}/.cache/huggingface/download/abc.{sha256(CONTENT[WEIGHTS[1]])}.incomplete", b"partial")
    hub = "/home/code/.cache/huggingface/hub"
    cache = host.cache(hub, REPO, REVISION, {name: CONTENT[name] for name in REQUIRED})
    os.unlink(host.path(f"{cache}/blobs/{sha256(CONTENT[WEIGHTS[2]])}"))
    # A git clone whose working tree holds LFS pointers; git-lfs fetched two objects.
    clone = host.copy("/var/tmp/models/git-clone", replace={
        name: b"version https://git-lfs.github.com/spec/v1\noid sha256:" + sha256(CONTENT[name]).encode()
        + b"\nsize %d\n" % len(CONTENT[name]) for name in LFS})
    for name in WEIGHTS[:2]:
        digest = sha256(CONTENT[name])
        host.write(f"{clone}/.git/lfs/objects/{digest[:2]}/{digest[2:4]}/{digest}", CONTENT[name])
    result = run(host)
    cloned = candidate(result, clone)
    assert cloned["layout"] == "git-lfs" and states(cloned)[WEIGHTS[0]] == ("match", "hub-named")
    assert cloned["files"][WEIGHTS[0]]["kind"] == "lfs-object" and "/.git/lfs/objects/" in cloned["files"][WEIGHTS[0]]["source"]
    assert states(cloned)[WEIGHTS[2]] == ("differs", "size") and states(cloned)[INDEX] == ("differs", "size")
    found = states(candidate(result, plain))
    assert found[WEIGHTS[0]][0] == "incomplete" and found[WEIGHTS[2]] == ("size-only", "size")
    assert found[WEIGHTS[1]] == ("differs", "size") and found[INDEX] == ("differs", "size")
    interrupted = candidate(result, local)
    assert WEIGHTS[1] not in interrupted["files"] and interrupted["counts"]["missing"] == 1
    cached = candidate(result, cache)
    assert cached["files"][WEIGHTS[2]]["state"] == "missing" and cached["counts"]["missing"] == 1
    assert cached["counts"]["match"] == len(REQUIRED) - 1


def test_other_checkpoints_are_not_candidates_and_only_near_misses_are_reported(tmp_path):
    host = Host(tmp_path)
    step = "/var/tmp/models/step5500/60215d26cf5e42c2db6128774032d57fc62678da"
    host.write(step + "/config.json", b'{"model_type": "qwen3_8_step5500"}\n')
    host.write(step + "/" + INDEX, b'{"weight_map": {"a": "model-00001-of-00041.safetensors"}}')
    host.write(step + "/model-00001-of-00041.safetensors", CONTENT[WEIGHTS[0]])
    earlier = "/var/tmp/models/pre-qad/b184bb5650367c3e934c7849407be9da3671e7f5"
    host.write(earlier + "/config.json", CONTENT["config.json"])
    host.write(earlier + "/" + INDEX, b'{"weight_map": {"a": "model-00001-of-00034.safetensors"}}')
    host.write(earlier + "/model-00001-of-00034.safetensors", OTHER)
    named = "/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-old"
    host.write(named + "/" + INDEX, b'{"weight_map": {}}')
    unrelated = "/var/tmp/models/llama"
    host.write(unrelated + "/config.json", b'{"model_type": "llama"}\n')
    host.write(unrelated + "/" + INDEX, b'{"weight_map": {"a": "model-00001-of-00002.safetensors"}}')
    result = run(host)
    assert result["candidates"] == []
    reason = "another checkpoint: its index differs from the pinned revision"
    assert [(item["path"], item["reason"]) for item in result["not_used"]] == [(named, reason), (earlier, reason)]


def test_same_names_sizes_and_index_give_only_size_evidence(tmp_path):
    host = Host(tmp_path)
    folder = host.copy("/var/tmp/models/lookalike", replace={WEIGHTS[0]: OTHER})
    found = candidate(run(host), folder)
    assert {name: states(found)[name] for name in WEIGHTS} == dict.fromkeys(WEIGHTS, ("size-only", "size"))
    assert states(found)[INDEX] == ("match", "hashed")
    hashed = candidate(run(host, small_file_bytes=4096), folder)
    assert states(hashed)[WEIGHTS[0]] == ("differs", "hashed") and states(hashed)[WEIGHTS[1]] == ("match", "hashed")


def test_unreadable_directories_are_counted_and_skipped(tmp_path, monkeypatch):
    host = Host(tmp_path)
    host.copy("/data/locked/qwen")
    readable = host.copy("/var/tmp/models/qwen")
    real_scandir = os.scandir

    def scandir(path):
        if os.fsdecode(path).replace("\\", "/").endswith("/data/locked"):
            raise PermissionError(13, "Permission denied", path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    result = run(host)
    assert paths(result) == {readable}
    assert result["search"]["unreadable"] >= 1 and result["search"]["complete"]


@POSIX
@pytest.mark.parametrize("case", ["home-symlink", "docker", "declared", "home", "record", "named"])
def test_network_and_automount_paths_are_never_touched(tmp_path, monkeypatch, docker, case):
    host = Host(tmp_path)
    # Modeled on TP4 rank 3: an NFS mount stacked over its autofs trigger, and an autofs CIFS share.
    host.mount("/srv/spark-share", "autofs", number=40, device="0:50", source="systemd-1")
    host.mount("/srv/spark-share", "nfs4", number=41, device="0:51", source="10.0.0.5:/share")
    host.mount("/mnt/synologytwo", "autofs", number=42, device="0:52", source="systemd-1")
    for share in ("/srv/spark-share", "/mnt/synologytwo"):
        host.copy(share + "/qwen")
        host.cache(share + "/hub", REPO, REVISION, {name: CONTENT[name] for name in REQUIRED})
    named = []
    if case == "home-symlink":
        host.symlink("/home/code/models", "/srv/spark-share/qwen")
        host.symlink("/home/code/.cache/huggingface/hub", "/srv/spark-share/hub")
        host.symlink("/home/code/share", "/mnt/synologytwo")
    elif case == "docker":
        docker.containers.append({"Id": "c1", "Config": {"Env": ["HF_HOME=/data/hf"]}, "Args": [],
                                  "Mounts": [{"Type": "bind", "Source": "/srv/spark-share/qwen",
                                              "Destination": "/models/target"},
                                             {"Type": "bind", "Source": "/mnt/synologytwo",
                                              "Destination": "/data/hf"}]})
    elif case == "declared":
        host.write("/etc/environment", "HF_HUB_CACHE=/srv/spark-share/hub\nHF_HOME=/mnt/synologytwo\n")
    elif case == "home":
        host.user("bob", 1001, "/srv/spark-share/home/bob")
        host.copy("/srv/spark-share/home/bob/models/qwen")
    elif case == "record":
        host.write("/var/lib/sparkring/checkpoints/" + "a" * 64 + ".json",
                   json.dumps({"repository": REPOSITORY, "revision": REVISION, "path": "/srv/spark-share/qwen",
                               "files": {}, "file_stats": {}}))
    else:
        named = ["/srv/spark-share/qwen"]
    seen = record_access(monkeypatch)
    result = run(host, named=named)
    shares = ("/srv/spark-share", "/mnt/synologytwo")
    assert [path for path in host_paths(host, seen) if path.startswith(shares)] == []
    assert not [path for path in paths(result) if path.startswith(shares)]
    skipped = {item["path"]: item for item in result["search"]["skipped_mounts"]}
    assert skipped["/srv/spark-share"] == {"path": "/srv/spark-share", "type": "nfs4", "network": True}
    assert skipped["/mnt/synologytwo"] == {"path": "/mnt/synologytwo", "type": "autofs", "network": True}
    if case == "named":
        assert result["named"] == [{"path": "/srv/spark-share/qwen", "resolved": None, "state": "network"}]


@pytest.mark.parametrize("budget", ["time", "entries"])
def test_budgets_return_partial_results_and_retry_once_with_twice_the_budget(tmp_path, monkeypatch, clock, budget):
    host = Host(tmp_path)
    shallow = host.copy("/var/tmp/models/qwen")
    for number in range(6):
        for item in range(300):
            host.write(f"/data/bulk/b{number}/f{item:03}", b"")
    deep = host.copy("/data/zz/qwen-deep")
    real_scandir = os.scandir

    def scandir(path):
        # Listing a bulk directory takes one second of the fake clock.
        if "/data/bulk/b" in os.fsdecode(path).replace("\\", "/"):
            clock[0] += 1
        return real_scandir(path)

    if budget == "time":
        monkeypatch.setattr(os, "scandir", scandir)
        limits = {"walk_seconds": 3.5}
    else:
        limits = {"walk_entries": 1000}
    partial = run(host, retry=False, **limits)
    first = partial["search"]
    assert (first["complete"], first["stopped"], first["passes"]) == (False, budget, 1)
    assert shallow in paths(partial) and deep not in paths(partial)
    assert first["unvisited"] == {"/data": {"count": 3, "paths": ["/data/bulk/b4", "/data/bulk/b5", deep]}}
    retried = run(host, **limits)
    assert (retried["search"]["complete"], retried["search"]["stopped"], retried["search"]["passes"]) == \
        (True, None, 2)
    assert {shallow, deep} <= paths(retried)


def test_the_whole_survey_has_a_deadline_and_lists_what_it_left(tmp_path, monkeypatch, clock):
    host = Host(tmp_path)
    contents = {name: CONTENT[name] for name in REQUIRED}
    host.cache("/root/.cache/huggingface/hub", REPO, REVISION, contents, symlinks=False)
    host.cache("/home/code/.cache/huggingface/hub", REPO, REVISION, contents, symlinks=False)
    folder = host.copy("/var/tmp/models/qwen")
    real_scandir = os.scandir

    def scandir(path):
        # Listing root's hub takes 80 s of the fake clock, past the survey's deadline of 3 * 20 s + 15 s.
        if os.fsdecode(path).replace("\\", "/").endswith("/root/.cache/huggingface/hub"):
            clock[0] += 80
        return real_scandir(path)
    monkeypatch.setattr(os, "scandir", scandir)
    result = run(host)
    search = result["search"]
    assert (search["complete"], search["stopped"], search["passes"]) == (False, "time", 1)
    # Nothing is examined after the deadline: neither the hubs' repository folders nor the walk's roots.
    assert paths(result) == set() and folder not in paths(result)
    assert "/home/code/.cache/huggingface/hub" in search["unvisited"]["/home"]["paths"]
    assert search["unvisited"]["/var/tmp/models"] == {"count": 1, "paths": ["/var/tmp/models"]}
    # Paths in homes other than the operator's are counted, not shown.
    assert search["unvisited"]["/root"]["count"] >= 2 and "paths" not in search["unvisited"]["/root"]
    # Without the delay the same survey finds both caches and the folder.
    monkeypatch.setattr(os, "scandir", real_scandir)
    assert {folder} < paths(run(host)) and run(host)["search"]["complete"]


def test_large_flat_directories_are_probed_by_pinned_names(tmp_path):
    host = Host(tmp_path)
    folder = "/var/tmp/models/flat"
    for item in range(60):
        host.write(f"{folder}/aaa-{item:02}.bin", b"")
    host.copy(folder)
    result = run(host, directory_entries=20)
    found = candidate(result, folder)
    assert found["counts"]["match"] == len(SMALL) and found["counts"]["size-only"] == len(WEIGHTS)
    assert result["search"]["large_directories"] == 1


def test_docker_failures_and_timeouts_share_one_allowance_and_do_not_abort(tmp_path, monkeypatch, clock):
    host = Host(tmp_path)
    folder = host.copy("/var/tmp/models/qwen")

    def survey_with(fake):
        fake.clock = clock
        monkeypatch.setattr(search, "subprocess", SimpleNamespace(
            run=fake.run, TimeoutExpired=subprocess.TimeoutExpired, DEVNULL=subprocess.DEVNULL))
        result = run(host)
        assert folder in paths(result)
        return result, [call for call, _ in fake.calls]

    slow = Docker(containers=[{"Id": "c1"}])
    slow.delays["ps"] = 10
    slow.failures["inspect"] = subprocess.TimeoutExpired("docker inspect", 5)
    result, calls = survey_with(slow)
    assert calls == ["ps", "inspect"]
    assert slow.calls[0][1]["timeout"] == pytest.approx(15) and slow.calls[1][1]["timeout"] == pytest.approx(5)
    assert [error for error in result["search"]["errors"] if error.startswith("docker: unavailable")]

    failing = Docker()
    failing.failures["ps"] = (1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.\n")
    result, calls = survey_with(failing)
    assert calls == ["ps", "volume ls", "info"] and result["docker"]["driver"] == "overlay2"
    assert "docker: ps -aq failed: Cannot connect to the Docker daemon at unix:///var/run/docker.sock." in \
        result["search"]["errors"]

    missing = Docker()
    missing.failures["ps"] = FileNotFoundError(2, "No such file or directory")
    result, calls = survey_with(missing)
    assert calls == ["ps"] and result["docker"] == {"userns": False, "driver": None, "root": None, "device": None}
    assert "docker: unavailable: No such file or directory" in result["search"]["errors"]


def test_ignore_local_copies_asks_docker_only_for_its_settings(tmp_path, docker):
    host = Host(tmp_path)
    docker.containers.append({"Id": "c1", "Config": {"Env": []}, "Mounts": [
        {"Type": "bind", "Source": "/var/lib/models-a", "Destination": "/models/target"}]})
    docker.volumes.append({"Name": "hf", "Mountpoint": "/var/lib/docker/volumes/hf/_data"})
    docker.info["SecurityOptions"].append("name=userns")
    result = run(host, ignore_local=True)
    # The search ends after SparkRing's own directories and named paths, so containers and volumes are not
    # listed; user-namespace remapping still decides which files can be linked.
    assert [call for call, _ in docker.calls] == ["info"]
    assert result["docker"]["userns"] is True and result["docker"]["driver"] == "overlay2"


@POSIX
def test_container_hub_paths_map_back_to_the_host(tmp_path, docker):
    host = Host(tmp_path)
    contents = {name: CONTENT[name] for name in REQUIRED}
    home_cache = host.cache("/srv/containers/hfcache/hub", REPO, REVISION, contents)
    hf_home = host.cache("/var/lib/hfdata/hub", REPO, REVISION, contents)
    download = host.cache("/var/lib/dl", REPO, REVISION, contents)
    upper = "/var/lib/docker/overlay2/abc/diff"
    layer = host.cache(upper + "/root/.cache/huggingface/hub", REPO, REVISION, contents)
    docker.containers += [
        {"Id": "c1", "Config": {"Env": ["PATH=/usr/bin"]},
         "Mounts": [{"Type": "bind", "Source": "/srv/containers/hfcache", "Destination": "/root/.cache/huggingface"}]},
        {"Id": "c2", "Config": {"Env": ["HF_HOME=/data/hf"]},
         "Mounts": [{"Type": "bind", "Source": "/var/lib/hfdata", "Destination": "/data/hf"}]},
        {"Id": "c3", "Config": {"Env": ["HF_HUB_CACHE=/elsewhere"]},
         "Args": ["serve", REPOSITORY, "--download-dir", "/models/dl"],
         "Mounts": [{"Type": "bind", "Source": "/var/lib/dl", "Destination": "/models/dl"}]},
        {"Id": "c4", "Config": {"Env": []}, "Mounts": [],
         "GraphDriver": {"Name": "overlay2", "Data": {"UpperDir": upper}}},
    ]
    docker.info["SecurityOptions"].append("name=userns")
    host.path("/var/lib/docker").mkdir(parents=True, exist_ok=True)
    sources = search.docker_sources(context(host))
    # c3's HF_HUB_CACHE is covered by no mount and c3 has no writable layer, so only its download dir maps.
    assert sources["hubs"] == [("/srv/containers/hfcache/hub", "docker-mount"), ("/var/lib/hfdata/hub", "docker-mount"),
                               ("/var/lib/dl", "docker-mount"), (upper + "/root/.cache/huggingface/hub", "docker-layer")]
    assert ("/var/lib/hfdata", 1, "docker-mount") in sources["roots"]
    result = run(host)
    # Bind sources are walk roots as well, so the walk can reach the same hub roots.
    for folder in (home_cache, hf_home, download):
        assert "docker-mount" in candidate(result, folder)["found_by"]
    assert candidate(result, layer)["found_by"] == ["docker-layer"]
    assert result["docker"]["userns"] is True and result["docker"]["driver"] == "overlay2"
    assert result["docker"]["device"] == os.lstat(host.path("/var/lib/docker")).st_dev


def test_declared_cache_variables_are_parsed_not_executed(tmp_path):
    host = Host(tmp_path)
    host.write("/etc/environment", 'HF_HUB_CACHE="/var/lib/hf-a"\nHF_HOME=$HOME/hf-home\n')
    host.write("/etc/profile.d/hf.sh", "BASE=/var/lib/hf-b\nexport HF_HOME=${BASE}/home\n"
               "export HF_HUB_CACHE=$(touch /tmp/pwned)\nexport TRANSFORMERS_CACHE=`touch /tmp/pwned`\n"
               "export XDG_CACHE_HOME=$UNKNOWN/x\nexport HUGGINGFACE_HUB_CACHE='$HOME/literal'\n")
    host.write("/home/code/.config/fish/config.fish", "set -gx HUGGINGFACE_HUB_CACHE /var/lib/hf-c\n")
    host.write("/etc/systemd/system/vllm.service",
               '[Service]\nEnvironment=HF_HOME=/var/lib/hf-d "XDG_CACHE_HOME=/var/lib/hf-e"\n'
               "Environment=HF_HUB_CACHE=$HOME/ignored\n")
    host.write("/etc/systemd/system/vllm.service.d/override.conf", '[Service]\nEnvironment="HF_HUB_CACHE=/var/lib/hf-f"\n')
    host.write("/home/code/.bashrc", "export TRANSFORMERS_CACHE=~/tc  # models\n")
    ctx = context(host)
    roots = search.declared_hub_roots(ctx, search.accounts(ctx))
    assert roots == sorted(["/var/lib/hf-a", "/root/hf-home/hub", "/home/code/hf-home/hub", "/var/lib/hf-b/home/hub",
                            "/var/lib/hf-c", "/var/lib/hf-d/hub", "/var/lib/hf-e/huggingface/hub", "/var/lib/hf-f",
                            "/home/code/tc"])
    assert not host.path("/tmp/pwned").exists()
    folder = host.cache("/var/lib/hf-b/home/hub", REPO, REVISION, {name: CONTENT[name] for name in REQUIRED},
                        symlinks=False)
    assert candidate(run(host), folder + "/snapshots/" + REVISION)["found_by"] == ["declared"]


def test_other_accounts_dotfiles_are_not_read_and_their_homes_are_listed_not_used(tmp_path, monkeypatch):
    host = Host(tmp_path)
    host.user("cody", 1001, "/home/cody")
    host.user("vllm", 998, "/var/lib/vllm")
    host.write("/home/cody/.bashrc", "export HF_HOME=/opt/a/b/c/hf\n")
    contents = {name: CONTENT[name] for name in REQUIRED}
    hidden = host.cache("/opt/a/b/c/hf/hub", REPO, REVISION, contents, symlinks=False)
    private_cache = host.cache("/home/cody/.cache/huggingface/hub", REPO, REVISION, contents, symlinks=False)
    private_copy = host.copy("/home/cody/models/qwen")
    service = host.copy("/var/lib/vllm/models/qwen")
    host.write("/home/code/.bashrc", "# the operator's start-up file\n")
    opened = record_access(monkeypatch, calls=("open",))
    result = run(host)
    # The operator's start-up files are read; another account's are not, and
    # nothing in another account's home is read beyond cache metadata.
    assert "/home/code/.bashrc" in host_paths(host, opened)
    assert [path for path in host_paths(host, opened) if path.startswith("/home/cody")] == \
        [f"/home/cody/.cache/huggingface/hub/{REPO}/refs/main"]
    assert not [path for path in paths(result) if path.startswith(("/home/cody", hidden))]
    reasons = {item["path"]: (item["reason"], item["home"]) for item in result["not_used"]}
    private = ("in cody's home, another account", {"account": "cody", "kind": "private"})
    assert reasons[private_copy] == private and reasons[private_cache + "/snapshots/" + REVISION] == private
    assert candidate(result, service)["home"] == {"account": "vllm", "kind": "service"}
    chosen = run(host, named=[private_copy])
    used = candidate(chosen, private_copy)
    assert "named" in used["found_by"] and used["home"] == {"account": "cody", "kind": "private"}
    assert used["exact"] is True
    assert chosen["named"] == [{"path": private_copy, "resolved": private_copy, "extra": [], "symlinks": [],
                                "state": "exact"}]


@POSIX
def test_ignore_marker_excludes_a_folder_in_every_group(tmp_path, docker):
    host = Host(tmp_path)
    contents = {name: CONTENT[name] for name in REQUIRED}
    for folder in ("/var/tmp/models/ignored", "/var/lib/models-a", "/var/lib/recorded", "/var/lib/named"):
        host.copy(folder)
        host.write(folder + "/.sparkring-ignore", b"")
    host.copy("/data/skip/qwen")
    host.write("/data/skip/.sparkring-ignore", b"")
    host.cache("/home/code/.cache/huggingface/hub", REPO, REVISION, contents)
    host.write("/home/code/.cache/huggingface/.sparkring-ignore", b"")
    hub = "/root/.cache/huggingface/hub"
    host.cache(hub, REPO, REVISION, contents)
    host.write(f"{hub}/{REPO}/.sparkring-ignore", b"")
    fork = host.cache(hub, "models--someone--Qwen3.8-Flash-Next-NVFP4-copy", REVISION, contents)
    docker.containers.append({"Id": "c1", "Config": {"Env": []}, "Mounts": [
        {"Type": "bind", "Source": "/var/lib/models-a", "Destination": "/models/target"}]})
    host.write("/var/lib/sparkring/checkpoints/" + "b" * 64 + ".json",
               json.dumps({"repository": REPOSITORY, "revision": REVISION, "path": "/var/lib/recorded",
                           "files": {}, "file_stats": {}}))
    result = run(host, named=["/var/lib/named"])
    assert paths(result) == {fork}
    assert result["named"] == [{"path": "/var/lib/named", "resolved": "/var/lib/named", "state": "ignored"}]


@POSIX
def test_sparkring_directories_are_recognized_by_owner_records_not_paths(tmp_path):
    host = Host(tmp_path)

    def stats(path):
        info = os.lstat(host.path(path))
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

    other = host.copy(f"/srv/sparkring/tp4/checkpoints/{SLUG}/{REVISION}", skip=("README.md",))
    state = f"/srv/sparkring/tp4/checkpoints/{SLUG}/.{REVISION}.sparkring"
    directory = os.lstat(host.path(other))
    host.write(state + "/owner.json", json.dumps({
        "schema": "sparkring-checkpoint-owner/v1", "repository": REPOSITORY, "revision": REVISION, "path": other,
        "directory": [directory.st_dev, directory.st_ino]}))
    host.write(state + "/journal.json", json.dumps({"schema": "sparkring-checkpoint-journal/v1", "files": {
        name: {"state": "placed", "sha256": sha256(CONTENT[name]), "stats": stats(other + "/" + name)}
        for name in REQUIRED}}))
    unowned = host.copy(f"/srv/sparkring/lab/checkpoints/{SLUG}/{REVISION}")
    workspace = "/srv/sparkring/tp2/qwen38-flash-next-tp2-i1"
    legacy = host.copy(f"{workspace}/models/{REVISION}")
    host.write(workspace + "/.installer-owner.json", '{"deployment": "i1"}')
    host.write(workspace + "/installer/model.json", json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "path": legacy, "origin": "pinned-hub-download",
        "files": {name: sha256(CONTENT[name]) for name in REQUIRED},
        "file_stats": {name: stats(legacy + "/" + name) for name in REQUIRED}}))
    workspace = "/srv/sparkring/tp2/qwen38-flash-next-tp2-i2"
    declared = host.copy(f"{workspace}/models/{REVISION}")
    host.write(workspace + "/.installer-owner.json", '{"deployment": "i2"}')
    host.write(workspace + "/installer/model.json", json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "path": declared, "origin": "operator-declared-verified-copy",
        "files": {}, "file_stats": {}}))
    lookalike = host.copy(f"/data/srv/sparkring/tp2/checkpoints/{SLUG}/{REVISION}")
    result = run(host)
    owned = candidate(result, other)
    assert owned["sparkring"] and owned["found_by"] == ["sparkring"] and owned["layout"] == "sparkring"
    assert states(owned) == {name: ("match", "recorded") for name in REQUIRED}
    assert candidate(result, legacy)["sparkring"]
    assert states(candidate(result, legacy)) == {name: ("match", "recorded") for name in REQUIRED}
    for path in (unowned, declared, lookalike):
        assert not candidate(result, path)["sparkring"]
    assert result["owned"]["state"] == "absent"


@POSIX
def test_records_count_only_while_all_five_stats_match(tmp_path):
    host = Host(tmp_path)
    folder = host.copy("/var/tmp/models/recorded")

    def stats(name):
        info = os.lstat(host.path(folder + "/" + name))
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

    hashes = {name: sha256(CONTENT[name]) for name in REQUIRED}
    hashes[WEIGHTS[2]] = "f" * 64
    host.write("/var/lib/sparkring/checkpoints/" + sha256(folder.encode()) + ".json", json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "path": folder, "origin": "operator-declared-verified-copy",
        "files": hashes, "file_stats": {name: stats(name) for name in REQUIRED}}))
    found = candidate(run(host), folder)
    assert "record" in found["found_by"]
    assert states(found) == {**{name: ("match", "recorded") for name in REQUIRED}, WEIGHTS[2]: ("differs", "recorded")}
    later = os.lstat(host.path(folder + "/" + WEIGHTS[0])).st_mtime + 5
    os.utime(host.path(folder + "/" + WEIGHTS[0]), (later, later))
    # A hard link elsewhere changes only the change time, as adoption does.
    os.link(host.path(folder + "/" + WEIGHTS[1]), host.path("/var/tmp/extra-link"))
    os.link(host.path(folder + "/config.json"), host.path("/var/tmp/extra-config"))
    changed = states(candidate(run(host), folder))
    assert changed[WEIGHTS[0]] == ("size-only", "size") and changed[WEIGHTS[1]] == ("size-only", "size")
    assert changed["config.json"] == ("match", "hashed") and changed["tokenizer.json"] == ("match", "recorded")


@POSIX
def test_owned_path_reports_free_space_mount_and_filesystem_type(tmp_path):
    host = Host(tmp_path)
    host.path("/srv/sparkring/tp2").mkdir(parents=True)
    owned = run(host)["owned"]
    device = os.lstat(host.path("/srv/sparkring/tp2")).st_dev
    space = os.statvfs(host.path("/srv/sparkring/tp2"))
    assert (owned["path"], owned["probe_path"], owned["mount_point"], owned["mount_id"], owned["fstype"],
            owned["device"], owned["state"]) == (OWNED, "/srv/sparkring/tp2", "/", 21, "ext4", device, "absent")
    assert abs(owned["free_bytes"] - space.f_bavail * space.f_frsize) < 1 << 30

    host.mount("/srv/sparkring", "xfs", number=40, device="259:3")
    host.path(OWNED).mkdir(parents=True)
    owned = run(host)["owned"]
    assert (owned["probe_path"], owned["mount_point"], owned["mount_id"], owned["fstype"], owned["state"]) == \
        (OWNED, "/srv/sparkring", 40, "xfs", "empty")
    host.write(OWNED + "/notes.txt", "user file")
    assert run(host)["owned"]["state"] == "foreign"
    os.unlink(host.path(OWNED + "/notes.txt"))
    host.copy(OWNED, skip=("README.md",))
    directory = os.lstat(host.path(OWNED))
    state = f"/srv/sparkring/tp2/checkpoints/{SLUG}/.{REVISION}.sparkring"
    owner = {"schema": "sparkring-checkpoint-owner/v1", "repository": REPOSITORY, "revision": REVISION,
             "path": OWNED, "directory": [directory.st_dev, directory.st_ino + 1]}
    host.write(state + "/owner.json", json.dumps(owner))
    result = run(host)
    assert result["owned"]["state"] == "replaced" and OWNED not in paths(result)
    host.write(state + "/owner.json", json.dumps({**owner, "directory": [directory.st_dev, directory.st_ino]}))
    result = run(host)
    assert result["owned"]["state"] == "owned" and OWNED not in paths(result)
    assert sorted(result["owned"]["files"]) == REQUIRED

    network = Host(tmp_path / "network")
    network.mount("/srv", "nfs4", number=40, device="0:60", source="nas:/srv")
    owned = run(network)["owned"]
    assert (owned["fstype"], owned["mount_point"], owned["probe_path"], owned["free_bytes"]) == \
        ("nfs4", "/srv", None, None)


@POSIX
def test_a_symlinked_checkpoint_directory_is_measured_at_its_parent(tmp_path, monkeypatch):
    host = Host(tmp_path)
    host.mount("/mnt/synologytwo", "autofs", number=42, device="0:52", source="systemd-1")
    parent = f"/srv/sparkring/tp2/checkpoints/{SLUG}"
    host.path(parent).mkdir(parents=True)
    host.symlink(OWNED, "/mnt/synologytwo/elsewhere")
    measured = record_access(monkeypatch, calls=("statvfs",))
    owned = run(host)["owned"]
    assert owned["state"] == "symlink" and owned["free_bytes"] is not None
    # statvfs follows a symlink, so it is made on the resolved parent, never on the link.
    assert host_paths(host, measured) == [parent]


@POSIX
def test_the_deployment_cache_decides_where_the_cache_allowance_counts(tmp_path):
    host = Host(tmp_path)
    host.path("/srv/sparkring/tp2").mkdir(parents=True)
    host.path("/mnt/fast/cache").mkdir(parents=True)
    owned = run(host)["owned"]
    assert owned["cache_path"] == "/srv/sparkring/tp2/cache"
    assert owned["cache_device"] == os.lstat(host.path("/srv/sparkring/tp2")).st_dev
    named = run(host, cache="/mnt/fast/cache")["owned"]
    assert named["cache_path"] == "/mnt/fast/cache"
    assert named["cache_device"] == os.lstat(host.path("/mnt/fast/cache")).st_dev
    # A cache that does not exist yet is measured at its nearest existing parent.
    assert run(host, cache="/mnt/fast/new/cache")["owned"]["cache_device"] == os.lstat(host.path("/mnt/fast")).st_dev
    with pytest.raises(ValueError, match="options: malformed values"):
        run(host, cache="relative/cache")


def test_nested_checkpoints_are_found_below_another_checkpoint(tmp_path):
    host = Host(tmp_path)
    outer = "/var/tmp/models/old-release"
    host.write(outer + "/" + INDEX, b'{"weight_map": {"a": "model-00001-of-00003.safetensors"}}')
    host.write(outer + "/" + WEIGHTS[0], b"older weights")
    inner = host.copy(outer + "/qad")
    result = run(host)
    assert inner in paths(result) and outer not in paths(result)
    assert [item["path"] for item in result["not_used"]] == [outer]


def test_linkability_uses_mount_identity_and_device(tmp_path):
    host = Host(tmp_path)
    # A bind mount of part of the root filesystem: the same device, another mount.
    host.mount("/data/models", "ext4", number=50, root="/srv/pool")
    same = host.copy("/var/tmp/models/qwen")
    bound = host.copy("/data/models/qwen")
    result = run(host)
    owned = result["owned"]
    assert owned["mount_id"] == 21
    assert candidate(result, same)["mount_id"] == 21 and candidate(result, bound)["mount_id"] == 50
    assert candidate(result, same)["device"] == candidate(result, bound)["device"] == owned["device"]
    assert {value["mount_id"] for value in candidate(result, bound)["files"].values()} == {50}
    assert {value["mount_id"] for value in candidate(result, same)["files"].values()} == {21}


@POSIX
def test_symlinked_fixed_locations_are_resolved_with_the_guard(tmp_path, monkeypatch):
    host = Host(tmp_path)
    host.mount("/mnt/synologytwo", "autofs", number=42, device="0:52", source="systemd-1")
    real = host.copy("/srv/spark-models/qwen")
    host.symlink("/models", "/srv/spark-models")
    host.copy("/var/lib/elsewhere/qwen")
    host.symlink("/srv/spark-models/linked", "/var/lib/elsewhere")
    host.copy("/mnt/synologytwo/models/qwen")
    host.symlink("/data/models", "/mnt/synologytwo/models")
    seen = record_access(monkeypatch)
    result = run(host)
    assert paths(result) == {real}
    assert "folder" in candidate(result, real)["found_by"]
    touched = [path for path in host_paths(host, seen) if path.startswith(("/mnt/synologytwo", "/var/lib/elsewhere"))]
    assert touched == []
    assert [item["path"] for item in result["search"]["skipped_mounts"]] == ["/mnt/synologytwo"]
