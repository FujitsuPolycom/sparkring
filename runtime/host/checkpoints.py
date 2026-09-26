"""List and release SparkRing's checkpoint directories on every Spark of the cluster.

A SparkRing checkpoint directory ``D`` holds the required files of one pinned
checkpoint revision for one cluster, by default
``/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>``. Its sibling
state directory ``.<name of D>.sparkring`` holds ``owner.json`` (the directory's
repository, revision, path and identity) and ``journal.json`` (every name
SparkRing placed in ``D``, with the placed inode, its SHA-256, its stats and the
path it was linked or copied from); ``runtime/host/checkpoint_place.py``
describes both. Weight files are usually hard links to a copy the operator
already had, so deleting that copy frees no space while ``D`` exists.

Two commands use this module:

- ``sudo sparkring checkpoints`` on Node A (``main``) lists, for every Spark,
  each SparkRing checkpoint directory with its revision and size, the
  deployments whose locks name it, the user copies whose inodes it shares, and
  the bytes that deleting those copies would not free.
- ``sudo sparkring checkpoints --release PATH`` removes one checkpoint directory
  and its state directory on every Spark that holds it. It refuses the directory
  that the active deployment, the rollback target recorded in
  ``transaction.json`` or a candidate whose model switch has not finished uses,
  whether as SparkRing's directory or served in place, and it refuses while the
  Sparks' package revisions differ. It names the other retained deployments
  that use the directory, which need ``sudo sparkring install`` again, and asks
  before it starts. Each Spark also refuses while a running container mounts
  the directory.

Each Spark runs its own part as ``sudo sparkring node checkpoints [--release
PATH]`` (``node``), reading a JSON request on stdin: ``list_local`` and
``release_local``. A release removes only names whose inode the journal
records and directories the journal records, never follows a symlink, and never
writes, moves or deletes a file SparkRing did not create. Removing a hard link
changes the change time of the inode it shared, so the release then refreshes
the recorded stats of other deployments' receipts and of the path records in
``/var/lib/sparkring/checkpoints`` for exactly those inodes
(``refresh_receipts``).

Directories are found in ``/srv/sparkring`` (``<cluster>/checkpoints/*/``,
``<cluster>/<workspace>/models/`` and ``<cluster>/models/``) and at the model
paths that Node A's retained deployments name for that Spark. Deployments
initialized from a site file outside ``/var/lib/sparkring/controller`` are not
known to Node A, so the listing cannot name them as users of a directory.
"""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import socket
import stat
import subprocess
import sys

try:
    import fcntl
except ImportError:  # Non-POSIX development hosts import the module; listing and release need Linux.
    fcntl = None

from runtime.host import checkpoint_place
from runtime.host.install_errors import NeedsInput

ROOT = Path(__file__).resolve().parents[2]
LOCAL_SCHEMA = "sparkring-checkpoints-local/v1"
SCHEMA = "sparkring-checkpoints/v1"
# Path records written by the rank operations; per-inode records live below them in ``files/``.
CHECKPOINTS = "/var/lib/sparkring/checkpoints"
# Where SparkRing creates checkpoint directories, relative to the search root.
SEARCH = (("srv", "sparkring", "*", "checkpoints", "*", ".*.sparkring"),
          ("srv", "sparkring", "*", "*", "models", ".*.sparkring"),
          ("srv", "sparkring", "*", "models", ".*.sparkring"))
STATE_SUFFIX = ".sparkring"
STATE_FILES = frozenset({"owner.json", "journal.json", "lock"})
_TEMPORARY = re.compile(r"(?:owner|journal)\.json\.[0-9]+\.[0-9]+\.writing")
# Model-switch states in which a candidate deployment may hold containers or still be started.
SETTLED = frozenset({"complete", "failed", "preparation-failed", "failed-recovered"})
JSON_LIMIT = 16 << 20
GIB = 1024 ** 3

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | _CLOEXEC
_READ = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | _CLOEXEC


# Paths and small readers ---------------------------------------------------

def _absolute(path):
    if (not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or path == "/"
            or "\0" in path or posixpath.normpath(path) != path):
        raise ValueError("Give the checkpoint directory as an absolute path, as sudo sparkring checkpoints "
                         "lists it: " + repr(path)[:200])
    return path


def _state_of(path):
    parent, name = posixpath.split(path)
    return posixpath.join(parent, "." + name + STATE_SUFFIX)


def _lstat(path):
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _open_directory(path):
    """Descriptor of absolute directory ``path``, opened one component at a time without following symlinks."""
    fd = os.open("/", _DIRECTORY)
    try:
        for part in [part for part in path.split("/") if part]:
            child = os.open(part, _DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_json_at(dir_fd, name, limit=JSON_LIMIT):
    """Parsed JSON of regular file ``name`` in ``dir_fd``; ``None`` when absent, not regular or not JSON."""
    try:
        fd = os.open(name, _READ, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return None
        data = b""
        while block := os.read(fd, 1 << 20):
            data += block
            if len(data) > limit:
                return None
    finally:
        os.close(fd)
    try:
        return json.loads(data)
    except ValueError:
        return None


def _read_json(directory, name):
    try:
        fd = os.open(directory, _DIRECTORY)
    except OSError:
        return None
    try:
        return _read_json_at(fd, name)
    finally:
        os.close(fd)


def _ints(value, count):
    return isinstance(value, list) and len(value) == count and all(type(item) is int and item >= 0 for item in value)


def _owner(state, path):
    """``owner.json`` of state directory ``state`` when it describes ``path``, else ``None``."""
    owner = _read_json(state, "owner.json")
    if (not isinstance(owner, dict) or owner.get("schema") != checkpoint_place.OWNER_SCHEMA
            or owner.get("path") != path or not _ints(owner.get("directory"), 2)
            or not isinstance(owner.get("repository"), str) or not isinstance(owner.get("revision"), str)):
        return None
    return owner


def _safe(name):
    try:
        return checkpoint_place.safe_name(name) == name
    except ValueError:
        return False


def _journal_files(state):
    journal = _read_json(state, "journal.json")
    files = journal.get("files") if isinstance(journal, dict) else None
    if not isinstance(files, dict):
        return {}
    return {name: entry for name, entry in files.items()
            if _safe(name) and isinstance(entry, dict) and _ints(entry.get("identity"), 2)}


def _copy_folder(source, name):
    """The folder a linked source belongs to: the checkpoint folder, or the repository folder of an HF blob."""
    if source.endswith("/" + name):
        return source[:-len(name) - 1]
    parent = posixpath.dirname(source)
    if posixpath.basename(parent) == "blobs":
        return posixpath.dirname(parent)
    return parent


def _staged_bytes(path):
    """Bytes of the regular files in a staging directory, without its marker."""
    total = 0
    for directory, directories, files in os.walk(path):
        for name in files:
            if directory == path and name == checkpoint_place.STAGING_MARKER:
                continue
            info = _lstat(posixpath.join(directory, name))
            if info is not None and stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def package_revision():
    """Source revision of this host's installed SparkRing package, or ``None`` for a checkout."""
    from runtime.common import distribution
    record = distribution.installed(ROOT, verify=False)
    return record["revision"] if record else None


# Listing on one Spark -------------------------------------------------------

def _matches(base, parts):
    if not parts:
        yield base
        return
    try:
        with os.scandir(base) as entries:
            names = sorted(entry.name for entry in entries
                           if entry.is_dir(follow_symlinks=False) and fnmatch.fnmatchcase(entry.name, parts[0]))
    except OSError:
        return
    for name in names:
        yield from _matches(posixpath.join(base, name), parts[1:])


def _state_directories(root, paths):
    found = set()
    for pattern in SEARCH:
        found.update(_matches(root, pattern))
    for path in paths:
        try:
            found.add(_state_of(_absolute(path)))
        except ValueError:
            continue
    return sorted(found)


def _describe(state):
    """One SparkRing checkpoint directory as ``list_local`` reports it, or ``None``."""
    parent, base = posixpath.split(state)
    if not (base.startswith(".") and base.endswith(STATE_SUFFIX)) or len(base) <= len(STATE_SUFFIX) + 1:
        return None
    path = posixpath.join(parent, base[1:-len(STATE_SUFFIX)])
    held = _lstat(state)
    if held is None or not stat.S_ISDIR(held.st_mode):
        return None
    owner = _owner(state, path)
    if owner is None:
        return None
    entry = {"path": path, "repository": owner["repository"], "revision": owner["revision"], "state": "ok",
             "files": 0, "bytes": 0, "frees_bytes": 0, "shared_bytes": 0, "shared": [],
             "staging_bytes": sum(_staged_bytes(posixpath.join(state, purpose))
                                  for purpose in checkpoint_place.STAGING_PURPOSES)}
    info = _lstat(path)
    if info is None:
        entry["state"] = "missing"
        return entry
    if not stat.S_ISDIR(info.st_mode) or checkpoint_place.identity(info) != owner["directory"]:
        entry["state"] = "replaced"
        return entry
    copies = {}
    for name, record in sorted(_journal_files(state).items()):
        current = _lstat(posixpath.join(path, name))
        if current is None or not stat.S_ISREG(current.st_mode) or checkpoint_place.identity(current) != record["identity"]:
            continue
        entry["files"] += 1
        entry["bytes"] += current.st_size
        if current.st_nlink <= 1:
            entry["frees_bytes"] += current.st_size
            continue
        entry["shared_bytes"] += current.st_size
        source = record.get("source")
        if isinstance(source, str) and source.startswith("/") and source != posixpath.join(path, name):
            linked = _lstat(source)
            if linked is not None and checkpoint_place.identity(linked) == record["identity"]:
                copy = copies.setdefault(_copy_folder(source, name), {"files": 0, "bytes": 0})
                copy["files"] += 1
                copy["bytes"] += current.st_size
    entry["shared"] = [{"path": folder, **copies[folder]} for folder in sorted(copies)]
    return entry


def list_local(paths=(), *, root="/"):
    """SparkRing checkpoint directories on this host (``sparkring-checkpoints-local/v1``).

    ``root`` relocates the search of ``/srv/sparkring``; ``paths`` adds model
    paths that deployments name on this host. A directory is listed only when its
    state directory holds an ``owner.json`` naming it. For each directory:

    - ``state``: ``ok``, ``missing`` (only the state directory remains, as after
      an interrupted release) or ``replaced`` (another directory now has its path);
    - ``files`` and ``bytes``: the journaled names that still hold their placed inode;
    - ``frees_bytes``: the bytes of those inodes with no other link, which a
      release frees; ``staging_bytes``: files in its staging directories;
    - ``shared_bytes``: the bytes of inodes with another link, and ``shared``, the
      copies they were linked from whose files are still those inodes
      (``{"path", "files", "bytes"}``). Deleting such a copy does not free them.
    """
    directories = [entry for entry in map(_describe, _state_directories(root, paths)) if entry is not None]
    return {"schema": LOCAL_SCHEMA, "hostname": socket.gethostname(), "package_revision": package_revision(),
            "directories": directories}


# Releasing on one Spark -----------------------------------------------------

def _lock(state_fd, path):
    if fcntl is None:
        raise RuntimeError("Releasing a checkpoint directory needs POSIX file locks")
    fd = os.open("lock", os.O_RDWR | os.O_CREAT | _NOFOLLOW | _CLOEXEC, 0o600, dir_fd=state_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"The lock of {path} is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ValueError(f"Another SparkRing operation is changing {path}; wait until it finishes, then repeat "
                         "the release") from None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _check_state(state_fd, state, path):
    """Refuse, before any change, a state directory holding anything SparkRing does not create there."""
    info = os.fstat(state_fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ValueError(f"{state} is not a private directory of this account; SparkRing does not remove it")
    unknown = []
    for name in sorted(os.listdir(state_fd)):
        entry = os.lstat(name, dir_fd=state_fd)
        if name in checkpoint_place.STAGING_PURPOSES and stat.S_ISDIR(entry.st_mode):
            fd = os.open(name, _DIRECTORY, dir_fd=state_fd)
            try:
                marker = _read_json_at(fd, checkpoint_place.STAGING_MARKER)
                if (marker is None and os.listdir(fd)) or (marker is not None and marker != {"purpose": name, "path": path}):
                    unknown.append(name)
            finally:
                os.close(fd)
        elif not stat.S_ISREG(entry.st_mode) or not (name in STATE_FILES or _TEMPORARY.fullmatch(name)):
            unknown.append(name)
    if unknown:
        raise ValueError(f"{state} holds entries SparkRing did not create there ({', '.join(unknown)}); SparkRing "
                         f"keeps {path}. Move them away, then repeat the release.")


def _remove_state(claimed, parent_fd):
    """Remove the state directory of ``claimed`` after ``D`` is gone; ``_check_state`` has approved its entries."""
    for purpose in checkpoint_place.STAGING_PURPOSES:
        checkpoint_place.remove_staging(claimed, purpose)
    names = sorted(os.listdir(claimed.state_fd), key=lambda name: (name == "owner.json", name == "lock"))
    for name in names:
        os.unlink(name, dir_fd=claimed.state_fd)
    os.fsync(claimed.state_fd)
    os.rmdir(posixpath.basename(claimed.state), dir_fd=parent_fd)
    os.fsync(parent_fd)


def _rmdir_below(dir_fd, name):
    parts = name.split("/")
    fd = dir_fd
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY, dir_fd=fd)
            if fd != dir_fd:
                os.close(fd)
            fd = child
        os.rmdir(parts[-1], dir_fd=fd)
    finally:
        if fd != dir_fd:
            os.close(fd)


def _remove_records(path):
    """Remove SparkRing's path record of ``path`` and its per-inode records of names inside ``path``.

    Records are evidence only; a record that stays behind stops counting once
    its inode's stats change, so a failure here is ignored.
    """
    name = hashlib.sha256(path.encode()).hexdigest() + ".json"
    saved = _read_json(CHECKPOINTS, name)
    if isinstance(saved, dict) and saved.get("path") == path:
        try:
            os.unlink(posixpath.join(CHECKPOINTS, name))
        except OSError:
            pass
    try:
        with os.scandir(checkpoint_place.RECORDS) as entries:
            names = [entry.name for entry in entries if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False)]
    except OSError:
        return
    for name in names:
        saved = _read_json(checkpoint_place.RECORDS, name)
        seen = saved.get("seen_as") if isinstance(saved, dict) else None
        if isinstance(seen, str) and seen.startswith(path + "/"):
            try:
                os.unlink(posixpath.join(checkpoint_place.RECORDS, name))
            except OSError:
                pass


def receipt_paths(receipts):
    """Receipts SparkRing may refresh: owned deployment receipts on this host, then its path records.

    A deployment receipt is ``<workspace>/installer/model.json`` whose workspace
    holds ``.installer-owner.json`` naming that deployment, reached without a
    symlink component. A path record is a regular ``*.json`` file directly in
    ``CHECKPOINTS``.
    """
    found = []
    for item in receipts or ():
        try:
            workspace = _absolute(item["workspace"])
            deployment = item["deployment"]
            fd = _open_directory(workspace)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        try:
            if _read_json_at(fd, ".installer-owner.json") != {"deployment": deployment}:
                continue
            installer = os.open("installer", _DIRECTORY, dir_fd=fd)
            try:
                info = os.lstat("model.json", dir_fd=installer)
            finally:
                os.close(installer)
        except OSError:
            continue
        finally:
            os.close(fd)
        if stat.S_ISREG(info.st_mode):
            found.append(posixpath.join(workspace, "installer", "model.json"))
    try:
        with os.scandir(CHECKPOINTS) as entries:
            found.extend(sorted(entry.path for entry in entries
                                if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False)))
    except OSError:
        pass
    return list(dict.fromkeys(found))


def refresh_receipts(changed, receipts, *, content_verified=False):
    """Refresh recorded stats of inodes whose change time SparkRing's link or unlink changed.

    ``changed`` maps ``(st_dev, st_ino)`` to ``{"sha256", "before", "after"}``:
    the verified SHA-256 of that inode and its five stats ``[st_dev, st_ino,
    st_size, st_mtime_ns, st_ctime_ns]`` measured before and right after
    SparkRing changed its links. ``receipts`` are the files from
    ``receipt_paths``. A receipt entry ``n`` is refreshed only when its recorded
    hash equals the verified one, its recorded stats equal ``before`` (all five
    when the content was not read again, as in a release, which must not revive
    an entry that was already out of date; the first four with
    ``content_verified``, when SparkRing hashed the inode's complete content
    just before linking it), and ``lstat(<receipt path>/n)`` still shows exactly
    ``after``, so a change made after SparkRing's own link or unlink stays
    visible. Without ``after``, the ``lstat`` must show the recorded device,
    inode, size and modification time. The entry's stats become that ``lstat``;
    nothing else in the receipt changes. Returns the refreshed receipt paths.
    """
    from scripts import deploy_engine
    refreshed = []
    for receipt in receipts:
        document = _read_json(posixpath.dirname(receipt), posixpath.basename(receipt))
        if not isinstance(document, dict) or not isinstance(document.get("path"), str):
            continue
        files, recorded = document.get("files"), document.get("file_stats")
        if not isinstance(files, dict) or not isinstance(recorded, dict):
            continue
        updated = False
        for name, values in recorded.items():
            if not isinstance(name, str) or not _ints(values, 5):
                continue
            change = changed.get((values[0], values[1]))
            compared = 4 if content_verified else 5
            if (change is None or values[:compared] != list(change["before"])[:compared]
                    or files.get(name) != change["sha256"]):
                continue
            current = _lstat(posixpath.join(document["path"], name))
            if current is None or not stat.S_ISREG(current.st_mode):
                continue
            measured = checkpoint_place.stats(current)
            expected = list(change["after"]) if change.get("after") is not None else None
            if measured[:4] != values[:4] or (expected is not None and measured != expected):
                continue
            if measured != values:
                recorded[name] = measured
                updated = True
        if updated:
            deploy_engine.save_receipt(Path(receipt), document)
            refreshed.append(receipt)
    return refreshed


def running_containers(path, *, run=None):
    """Names of the running containers on this host that mount ``path`` or a path below it.

    Node A knows the deployments it installed, not containers started by hand
    or by other tools, so each Spark checks its own before a release. A host
    without the ``docker`` command runs no containers; any other failure to
    list them refuses the release.
    """
    run = run or subprocess.run
    docker = ["docker", "--context", "default"]

    def call(arguments):
        try:
            done = run([*docker, *arguments], capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            return None
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"SparkRing could not list this host's running containers ({error}); nothing was "
                             "released") from None
        if done.returncode and not done.stdout.strip():
            lines = [line.strip() for line in (done.stderr or "").splitlines() if line.strip()]
            raise ValueError("SparkRing could not list this host's running containers ("
                             + (lines[-1] if lines else f"docker exited with status {done.returncode}")
                             + "); nothing was released")
        return done.stdout

    identifiers = (call(["ps", "--quiet", "--no-trunc"]) or "").split()
    if not identifiers:
        return []
    try:
        containers = json.loads(call(["inspect", *identifiers]) or "[]")
    except ValueError:
        containers = None
    if not isinstance(containers, list):
        raise ValueError("SparkRing could not read this host's running containers; nothing was released")
    names = set()
    for container in containers:
        for mount in (container.get("Mounts") or []) if isinstance(container, dict) else []:
            source = mount.get("Source") if isinstance(mount, dict) else None
            if isinstance(source, str) and (source.rstrip("/") == path or source.startswith(path + "/")):
                names.add(str(container.get("Name") or container.get("Id", "")[:12]).lstrip("/"))
    return sorted(names)


def _release_state_only(path, state):
    """Remove the state directory that remains when ``D`` is gone, as after an interrupted release."""
    parent, name = posixpath.split(state)
    parent_fd = _open_directory(parent)
    try:
        state_fd = os.open(name, _DIRECTORY, dir_fd=parent_fd)
        lock_fd = None
        try:
            lock_fd = _lock(state_fd, path)
            owner = _owner(state, path)
            if owner is None or _lstat(path) is not None:
                raise ValueError(f"{path} changed while SparkRing prepared its release; repeat the release")
            _check_state(state_fd, state, path)
            claimed = checkpoint_place.Claim(path, None, None, state_fd, lock_fd, owner["directory"], "verified")
            _remove_state(claimed, parent_fd)
        finally:
            for fd in (state_fd, lock_fd):
                if fd is not None:
                    os.close(fd)
    finally:
        os.close(parent_fd)
    _remove_records(path)
    return {"path": path, "state": "released", "unlinked": 0, "freed_bytes": 0, "kept_bytes": 0, "refreshed": []}


def release_local(path, receipts=(), *, table=None):
    """Remove SparkRing checkpoint directory ``path`` and its state directory from this host.

    ``receipts`` lists ``{"workspace", "deployment"}`` of Node A's retained
    deployments with a row on this host; their receipts and this host's path
    records are refreshed for the inodes whose links the release changed
    (``refresh_receipts``). Before anything changes, the release refuses when
    ``path`` is not a SparkRing checkpoint directory, was replaced after
    SparkRing created it, holds an entry SparkRing did not place (a name the
    journal does not record with its current inode, or a directory that the
    journal neither records nor needs for a journaled name), a running
    container on this host mounts it, or its state directory holds entries
    SparkRing did not create. It then records in the journal the directories
    that hold journaled names, unlinks each journaled name whose inode the
    journal records, removes the recorded directories, ``D`` itself, the
    staging directories and the state directory, and SparkRing's records of
    ``D`` and of names inside it. A repeated release after an interruption
    continues. ``table`` replaces the mount table.

    Returns ``{"path", "state": "released" | "absent", "unlinked",
    "freed_bytes", "kept_bytes", "refreshed"}``; ``kept_bytes`` counts inodes
    that stay on disk through another link, such as the copy they were linked
    from.
    """
    path = _absolute(path)
    state = _state_of(path)
    held = _lstat(state)
    owner = _owner(state, path) if held is not None and stat.S_ISDIR(held.st_mode) else None
    if owner is None:
        if _lstat(path) is not None:
            raise ValueError(f"{path} is not a SparkRing checkpoint directory (no owner record in {state}); "
                             "SparkRing does not remove it")
        return {"path": path, "state": "absent", "unlinked": 0, "freed_bytes": 0, "kept_bytes": 0, "refreshed": []}
    if _lstat(path) is None:
        return _release_state_only(path, state)
    users = running_containers(path)
    if users:
        raise ValueError(f"{path} is mounted by running containers on this Spark ({', '.join(users)}); SparkRing "
                         "does not release a checkpoint directory that a container uses. Stop them first. Nothing "
                         "was released.")
    result = {"path": path, "state": "released", "unlinked": 0, "freed_bytes": 0, "kept_bytes": 0, "refreshed": []}
    with checkpoint_place.claim(path, owner["repository"], owner["revision"], table=table) as claimed:
        journal = checkpoint_place.journal_load(claimed)
        checkpoint_place.journal_recover(claimed.dir_fd, journal)
        files, directories = checkpoint_place.listing(claimed.dir_fd)
        # A recorded directory is SparkRing's own; it is left empty when a release stopped after unlinking
        # its files. Any file below it that the journal does not record is still reported.
        unexpected = [name for name in checkpoint_place.unexpected_names(claimed.dir_fd, journal, set(journal.files))
                      if name not in journal.directories]
        if unexpected:
            shown = ", ".join(unexpected[:10]) + (f" and {len(unexpected) - 10} more" if len(unexpected) > 10 else "")
            raise ValueError(f"{path} holds files SparkRing did not place: {shown}. SparkRing removes only the names "
                             "it placed, so it keeps the directory. Move those files away, then repeat the release.")
        _check_state(claimed.state_fd, claimed.state, path)
        # The directories that hold journaled names are recorded before any name goes, so a release that
        # stops after unlinking the files still removes them when repeated.
        checkpoint_place.journal_directories(journal, list(journal.files))
        changed = {}
        for name in sorted(journal.files):
            entry = journal.files[name]
            before = _lstat(posixpath.join(path, name))
            if before is None or checkpoint_place.identity(before) != entry["identity"]:
                before = None
            watched = None
            if before is not None and before.st_nlink > 1:
                # A descriptor held across the unlink gives the inode's stats right after it.
                try:
                    watched, _ = checkpoint_place.open_source(name, entry["identity"], before.st_size,
                                                              dir_fd=claimed.dir_fd)
                except (OSError, ValueError):
                    watched = None
            try:
                unlinked = checkpoint_place.unlink_placed(claimed.dir_fd, name, journal)
                after = os.fstat(watched) if watched is not None else None
            finally:
                if watched is not None:
                    os.close(watched)
            if unlinked and before is not None:
                result["unlinked"] += 1
                if before.st_nlink > 1:
                    result["kept_bytes"] += before.st_size
                    if after is not None:
                        changed[tuple(entry["identity"])] = {"sha256": entry["sha256"],
                                                             "before": checkpoint_place.stats(before),
                                                             "after": checkpoint_place.stats(after)}
                else:
                    result["freed_bytes"] += before.st_size
        for name in sorted(directories, key=lambda item: item.count("/"), reverse=True):
            if name in journal.directories:
                _rmdir_below(claimed.dir_fd, name)
        result["refreshed"] = refresh_receipts(changed, receipt_paths(receipts))
        claimed.check()
        os.rmdir(claimed.name, dir_fd=claimed.parent_fd)
        os.fsync(claimed.parent_fd)
        _remove_state(claimed, claimed.parent_fd)
    _remove_records(path)
    return result


def node(release=None, stream=None):
    """``sudo sparkring node checkpoints [--release PATH]``: this host's part, request read as JSON from ``stream``.

    Listing reads ``{"paths": [...]}`` (model paths deployments name on this
    host); a release reads ``{"receipts": [{"workspace", "deployment"}, ...]}``.
    An empty or interactive ``stream`` is an empty request.
    """
    stream = sys.stdin if stream is None else stream
    text = "" if stream is None or stream.isatty() else stream.read()
    request = json.loads(text) if text.strip() else {}
    if not isinstance(request, dict):
        raise ValueError("The checkpoints request must be a JSON object")
    if release is not None:
        return release_local(release, request.get("receipts") or ())
    return list_local(request.get("paths") or ())


# Node A ----------------------------------------------------------------------

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deployments(state_root):
    """Retained deployments of the controller: ``[{"name", "directory", "id", "workspace", "rows"}]``.

    ``rows`` holds each rank's ``host``, ``model`` and whether that model is a
    named copy served in place (``reuse``). A deployment whose lock cannot be
    read is skipped.
    """
    found = []
    base = Path(state_root) / "deployments"
    try:
        directories = sorted(base.iterdir())
    except OSError:
        return found
    for directory in directories:
        try:
            lock = _read(directory / "deployment.lock.json")
            site = lock["site"]
            rows = [{"host": row["host"], "model": row["model"], "reuse": bool(row.get("reuse_verified_model"))}
                    for row in site["ranks"]]
            found.append({"name": directory.name, "directory": os.path.realpath(directory), "id": lock["id"],
                          "workspace": site["workspace"], "rows": rows})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def roles(state_root):
    """``{deployment directory: role}`` for ``active``, ``rollback`` and ``switching`` deployments.

    The active deployment is ``active.json``'s. ``transaction.json`` names the
    rollback target (``previous``) and, while a model switch has not settled,
    the candidate being switched to.
    """
    state_root = Path(state_root)
    result = {}
    try:
        transaction = _read(state_root / "transaction.json")
    except (OSError, ValueError):
        transaction = {}
    if isinstance(transaction, dict):
        if transaction.get("candidate") and transaction.get("state") not in SETTLED:
            result[os.path.realpath(transaction["candidate"])] = "switching"
        if transaction.get("previous"):
            result[os.path.realpath(transaction["previous"])] = "rollback"
    try:
        result[os.path.realpath(_read(state_root / "active.json")["path"])] = "active"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def _users(retained, host, path):
    """Retained deployments whose row for ``host`` names ``path``, as SparkRing's directory or served in place."""
    return [item for item in retained if any(row["host"] == host and row["model"] == path for row in item["rows"])]


def _survey(hosts, retained, invoke):
    """This package's listing of every Spark, in rank order; an unreachable Spark carries ``error``."""
    def one(rank):
        host = hosts[rank]
        paths = sorted({row["model"] for item in retained for row in item["rows"]
                        if row["host"] == host and not row["reuse"]})
        try:
            listing = json.loads(invoke(host, ["sudo", "-n", "/usr/bin/sparkring", "node", "checkpoints"],
                                        data=json.dumps({"paths": paths}), timeout=120))
            if not isinstance(listing, dict) or listing.get("schema") != LOCAL_SCHEMA:
                raise ValueError("unexpected answer from sparkring node checkpoints")
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            return {"rank": rank, "host": host, "error": str(error).strip().splitlines()[-1][:300] if str(error).strip() else type(error).__name__}
        return {"rank": rank, "host": host, **listing}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        return list(pool.map(one, range(len(hosts))))


def _annotate(nodes, retained, role):
    for node_entry in nodes:
        for entry in node_entry.get("directories", ()):
            entry["deployments"] = [{"name": item["name"], **({"role": role[item["directory"]]}
                                                              if item["directory"] in role else {})}
                                    for item in _users(retained, node_entry["host"], entry["path"])]
    return nodes


def _cluster(state_root):
    try:
        return _read(Path(state_root) / "cluster.json")
    except FileNotFoundError:
        return None


def _hosts(cluster):
    return [row["host"] for row in cluster["plan"]["spec"]["hosts"]]


def list_cluster(state_root, invoke):
    """``sparkring checkpoints``: every Spark's listing with the deployments that use each directory."""
    retained, role = deployments(state_root), roles(state_root)
    cluster = _cluster(state_root)
    if cluster is None:
        nodes = [{"rank": None, "host": None, **list_local([row["model"] for item in retained for row in item["rows"]
                                                              if not row["reuse"]])}]
    else:
        nodes = _survey(_hosts(cluster), retained, invoke)
    return {"schema": SCHEMA, "state": "listed", "nodes": _annotate(nodes, retained, role)}


def release_cluster(path, state_root, invoke, *, yes, interactive, write=print):
    """``sparkring checkpoints --release PATH`` on Node A; see the module description for its rules."""
    from runtime.common import process_lock
    from runtime.host import controller
    path = _absolute(path)
    with process_lock.hold(Path(state_root) / "install.lock"):
        cluster = _cluster(state_root)
        if cluster is None:
            raise ValueError("Release a checkpoint directory from Node A of a configured cluster, which records "
                             "the deployments that use it. Nothing was released.")
        retained, role = deployments(state_root), roles(state_root)
        # A deployment that serves the directory in place (sparkring up --model-path, or a site row with
        # reuse_verified_model) uses it as much as one whose lock names it as SparkRing's directory.
        users = [item for item in retained if any(row["model"] == path for row in item["rows"])]
        reasons = {"active": "the active deployment", "rollback": "the rollback target recorded in transaction.json",
                   "switching": "the deployment of an unfinished model switch"}
        for item in users:
            if item["directory"] in role:
                in_place = all(row["reuse"] for row in item["rows"] if row["model"] == path)
                use = "the copy served in place by" if in_place else "the checkpoint directory of"
                raise ValueError(f"{path} is {use} {reasons[role[item['directory']]]} {item['name']}; SparkRing does "
                                 "not release it. Nothing was released.")
        hosts = _hosts(cluster)
        nodes = _survey(hosts, retained, invoke)
        failed = [item for item in nodes if "error" in item]
        if failed:
            raise ValueError("; ".join(f"Node {item['rank']} {item['host']} could not be listed: {item['error']}"
                                       for item in failed) + ". Nothing was released.")
        revisions = {item.get("package_revision") for item in nodes}
        if len(revisions) > 1:
            shown = ", ".join(f"Node {item['rank']} {str(item.get('package_revision'))[:12]}" for item in nodes)
            raise ValueError(f"The Sparks run different SparkRing package revisions ({shown}). Install one package "
                             "revision on every Spark, for example with sudo sparkring install, then repeat the "
                             "release. Nothing was released.")
        holders = [(item, entry) for item in nodes for entry in item["directories"] if entry["path"] == path]
        if not holders:
            raise ValueError(f"{path} is not a SparkRing checkpoint directory on any Spark. "
                             "sudo sparkring checkpoints lists them.")
        for item, entry in holders:
            if entry["state"] == "replaced":
                raise ValueError(f"Node {item['rank']} {item['hostname']}: {path} was replaced after SparkRing "
                                 "created it; SparkRing does not remove it. Nothing was released.")
        write(f"Release {path}:")
        for item, entry in holders:
            write("    " + _release_line(item, entry))
        if users:
            write("Retained deployments that use it need sudo sparkring install again: "
                  + ", ".join(item["name"] for item in users))
        write("SparkRing removes only the names it placed there; it never writes, moves or deletes the copies they "
              "were linked from.")
        if not yes:
            if not interactive:
                raise NeedsInput("Review the release above, then repeat with --yes to apply it. Nothing was released.",
                                 field="approval", details={"path": path, "nodes": [item["rank"] for item, _ in holders]})
            controller.confirm("Release this checkpoint directory?")

        def one(pair):
            item, _ = pair
            receipts = [{"workspace": deployment["workspace"], "deployment": deployment["id"]}
                        for deployment in retained if any(row["host"] == item["host"] for row in deployment["rows"])]
            try:
                answer = json.loads(invoke(item["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "checkpoints",
                                                          "--release", path],
                                           data=json.dumps({"receipts": receipts}), timeout=300))
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                return {"rank": item["rank"], "host": item["host"], "hostname": item["hostname"], "error": str(error).strip()}
            return {"rank": item["rank"], "host": item["host"], "hostname": item["hostname"], **answer}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(holders)) as pool:
            results = list(pool.map(one, holders))
    for item in results:
        if "error" not in item:
            write(f"Node {item['rank']} {item['hostname']}: released; freed {_size(item['freed_bytes'])}"
                  + (f", refreshed {_plural(len(item['refreshed']), 'receipt')}" if item["refreshed"] else ""))
    failed = [item for item in results if "error" in item]
    if failed:
        raise RuntimeError("The release did not finish. " + " ".join(
            f"Node {item['rank']} {item['hostname']}: {item['error']}" for item in failed)
            + " Repeating the release continues it.")
    return {"schema": SCHEMA, "state": "released", "path": path, "nodes": results,
            "deployments": [item["name"] for item in users]}


# Text -------------------------------------------------------------------------

def _size(value):
    """Sizes as the checkpoint plan prints them: GiB with one decimal from 1 GiB, else MB or KB."""
    if value >= GIB:
        return f"{value / GIB:.1f} GiB"
    for unit, scale in (("MB", 10 ** 6), ("KB", 10 ** 3)):
        if value >= scale:
            number = value / scale
            text = f"{number:.0f}" if number >= 99.95 else f"{number:.1f}" if number >= 9.995 else f"{number:.2f}"
            return f"{text} {unit}"
    return f"{value} bytes"


def _plural(number, noun):
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _node_name(item):
    name = item.get("hostname") or item.get("host")
    return "This Spark " + name if item.get("rank") is None else f"Node {item['rank']} {name}"


def _release_line(item, entry):
    text = f"Node {item['rank']} {item['hostname']}: removes {_plural(entry['files'], 'file')}; frees " \
           f"{_size(entry['frees_bytes'] + entry['staging_bytes'])}"
    kept = entry["shared_bytes"]
    if entry["shared"]:
        text += f"; {_size(kept)} stays on disk in " + ", ".join(copy["path"] for copy in entry["shared"])
    elif kept:
        text += f"; {_size(kept)} stays on disk through other hard links"
    return text


def describe(result):
    """Printed lines of a ``list_cluster`` result."""
    lines = []
    for item in result["nodes"]:
        if "error" in item:
            lines.append(f"{_node_name(item)}: not reachable: {item['error']}")
            continue
        lines.append(_node_name(item))
        if not item["directories"]:
            lines.append("    no SparkRing checkpoint directories")
        for entry in item["directories"]:
            lines.append("    " + entry["path"])
            if entry["state"] == "missing":
                lines.append("        only SparkRing's state directory remains; --release removes it")
                continue
            if entry["state"] == "replaced":
                lines.append("        replaced after SparkRing created it; SparkRing neither uses nor removes it")
                continue
            lines.append(f"        {entry['repository']} at {entry['revision'][:12]}: "
                         f"{_plural(entry['files'], 'file')}, {_size(entry['bytes'])}")
            users = [deployment["name"] + (f" ({deployment['role']})" if deployment.get("role") else "")
                     for deployment in entry.get("deployments", ())]
            lines.append("        used by " + (", ".join(users) if users else "no retained deployment"))
            for copy in entry["shared"]:
                size = _size(copy["bytes"])
                lines.append(f"        shares {_plural(copy['files'], 'file')} ({size}) with {copy['path']}; deleting "
                             f"that copy does not free these {size} while this directory exists")
            other = entry["shared_bytes"] - sum(copy["bytes"] for copy in entry["shared"])
            if other:
                lines.append(f"        {_size(other)} is also linked from other paths")
            lines.append(f"        releasing it frees {_size(entry['frees_bytes'] + entry['staging_bytes'])}")
    lines.append("Release a directory with sudo sparkring checkpoints --release PATH. SparkRing refuses the active "
                 "deployment's directory and the rollback target's.")
    return lines


def _administrator():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def main(argv=None, *, state_root=None, invoke=None):
    parser = argparse.ArgumentParser(
        prog="sparkring checkpoints",
        description="List SparkRing's checkpoint directories on every Spark, or release one.")
    parser.add_argument("--release", metavar="PATH",
                        help="remove this SparkRing checkpoint directory from every Spark that holds it; the copies "
                             "its files were linked from are never written, moved or deleted")
    parser.add_argument("--yes", action="store_true", help="approve the release without a prompt")
    parser.add_argument("--json", action="store_true", help="emit one JSON result on stdout")
    args = parser.parse_args(argv)
    if state_root is None or invoke is None:
        from runtime.host import controller, discovery
        state_root = controller.STATE if state_root is None else state_root
        invoke = discovery.ssh if invoke is None else invoke
    output = sys.stdout
    text = sys.stderr if args.json else sys.stdout

    def write(line):
        print(line, file=text, flush=True)

    code = 0
    try:
        if not _administrator():
            raise ValueError("Run sudo sparkring checkpoints")
        if args.release:
            result = release_cluster(args.release, state_root, invoke, yes=args.yes,
                                     interactive=not args.json and sys.stdin.isatty(), write=write)
        else:
            result = list_cluster(state_root, invoke)
            if not args.json:
                for line in describe(result):
                    write(line)
            if any("error" in item for item in result["nodes"]):
                code = 2
    except NeedsInput as error:
        result, code = {"schema": SCHEMA, **error.document()}, 3
        write(str(error))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result, code = {"schema": SCHEMA, "state": "failed", "message": str(error)}, 2
        print("SparkRing: " + str(error), file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps(result, indent=2), file=output)
    return code

