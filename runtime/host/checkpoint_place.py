"""Placement and verification primitives for SparkRing checkpoint directories.

A SparkRing checkpoint directory ``D`` holds exactly the required files of one
pinned checkpoint revision. Its sibling state directory ``.<name of D>.sparkring``
(mode 0700, on the same mount) holds:

- ``owner.json``: repository, revision, path and ``(st_dev, st_ino)`` of the
  directory that SparkRing created or claimed while it was empty;
- ``journal.json``: every name SparkRing placed in ``D``, with the placed
  inode, its SHA-256 and its stats, and every directory SparkRing created in
  ``D`` for nested names;
- ``lock``: the ``flock`` target held by every operation that changes ``D``;
- ``fetch/`` and ``receive/``: staging directories, each marked with
  ``.sparkring-staging.json``.

Invariants enforced here:

- A name enters ``D`` only through ``link(2)`` onto a name that does not exist,
  from the descriptor whose complete content the caller hashed. The journal
  records the name as ``placing`` before it is created and as ``placed`` after
  the linked inode was confirmed unchanged since hashing: device, inode, size
  and modification time equal, and a change time that moved is followed by a
  second read of the same descriptor.
- A name is removed from ``D`` only when the journal records that name's inode,
  and a directory only when the journal records the directory.
- Files SparkRing did not create are opened read-only with ``O_NOFOLLOW``,
  ``O_NONBLOCK`` and ``O_NOATIME``. Their content, mode, owner and times are
  never changed; linking adds a name, which changes their link count and change
  time.
- Writes inside ``D`` and its state directory use directory descriptors opened
  with ``O_DIRECTORY | O_NOFOLLOW``, after ``D``'s identity was compared with
  ``owner.json``. Docker's bind mount of ``fetch/`` and rsync's destination in
  ``receive/`` are resolved by path, so ``claim`` accepts ``D`` only below
  directories that other accounts cannot rename or replace, and only when an
  existing ``D`` is empty and writable by this account alone.

The module is shipped to Sparks as source text (``inspect.getsource`` of the
module) and runs there as ``python3 -I`` under root, so it imports only the
standard library and refers to no other SparkRing module.
"""
import errno
import hashlib
import json
import os
import posixpath
import re
import stat
import threading
import time

try:
    import fcntl
except ImportError:  # Non-POSIX development hosts import the module; placement itself needs Linux.
    fcntl = None

MOUNTINFO = "/proc/self/mountinfo"
RECORDS = "/var/lib/sparkring/checkpoints/files"
LOCAL_TYPES = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs"})
OWNER_SCHEMA = "sparkring-checkpoint-owner/v1"
JOURNAL_SCHEMA = "sparkring-checkpoint-journal/v1"
STAGING_MARKER = ".sparkring-staging.json"
STAGING_PURPOSES = ("fetch", "receive")
ORIGINS = ("link", "copy", "fabric", "rsync", "hub")
BLOCK = 16 << 20
JSON_LIMIT = 16 << 20

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
READ_FLAGS = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOATIME", 0) | _CLOEXEC
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | _CLOEXEC
_SHA256 = re.compile(r"[0-9a-f]{64}")
NOT_EMPTY = ("{} is not empty and was not created by SparkRing; SparkRing does not adopt a directory it did "
             "not create. Move it away or choose another path.")
REPLACED = "{} was replaced after SparkRing created it; SparkRing does not write into it."


def stats(value):
    """``[st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns]`` of a stat result.

    Recorded hashes stay valid only while all five values are unchanged.
    """
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def identity(value):
    """``[st_dev, st_ino]`` of a stat result."""
    return [value.st_dev, value.st_ino]


def _content_key(value):
    # Linking a file changes its change time, so the checks around a link compare the other four values.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_ints(value, count):
    return (isinstance(value, list) and len(value) == count
            and all(type(item) is int and item >= 0 for item in value))


def safe_name(name):
    """Return ``name`` when it is a normalized relative POSIX path that may be placed in ``D``."""
    if (not isinstance(name, str) or not name or name.startswith("/") or posixpath.normpath(name) != name
            or any(character in name for character in "\0\r\n\\")
            or any(part in ("", ".", "..", ".cache", ".git") for part in name.split("/"))):
        raise ValueError("Unsafe checkpoint file name: " + repr(name)[:200])
    return name


def _checked_path(path):
    if (not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or path == "/"
            or "\0" in path or posixpath.normpath(path) != path):
        raise ValueError("Installer paths must be absolute and normalized: " + repr(path)[:200])
    return path


# Mount table ---------------------------------------------------------------

def _unescape(field):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), field)


def _makedev(major, minor):
    """Linux ``st_dev`` value of a device number (the glibc ``makedev`` encoding)."""
    return (minor & 0xFF) | ((major & 0xFFF) << 8) | ((minor & ~0xFF) << 12) | ((major & ~0xFFF) << 32)


def parse_mounts(text):
    """Entries of ``/proc/self/mountinfo`` text, in file order; malformed lines are skipped.

    Each entry has ``id`` and ``parent`` (mount IDs), ``major``, ``minor`` and
    ``device`` (the ``st_dev`` of files on it), ``root``, ``point`` (the mount
    point, octal escapes decoded), ``type`` and ``source``.
    """
    entries = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        fields, tail = before.split(), after.split()
        if not separator or len(fields) < 6 or not tail:
            continue
        try:
            major, minor = (int(value) for value in fields[2].split(":"))
            mount_id, parent = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        entries.append({"id": mount_id, "parent": parent, "major": major, "minor": minor,
                        "device": _makedev(major, minor), "root": _unescape(fields[3]),
                        "point": _unescape(fields[4]), "type": tail[0],
                        "source": _unescape(tail[1]) if len(tail) > 1 else ""})
    return entries


def mounts(path=None):
    """Entries of this process's mount table (``MOUNTINFO`` unless ``path`` is given)."""
    with open(MOUNTINFO if path is None else path, encoding="utf-8", errors="surrogateescape") as stream:
        return parse_mounts(stream.read())


def mount_of(path, table=None):
    """The mount entry covering ``path`` lexically, or ``None``.

    The deepest mount point wins. Among entries on the same mount point, the one
    listed last is mounted over the others and is the one visible at ``path``.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Mount lookups need an absolute path: " + repr(path)[:200])
    table = mounts() if table is None else table
    path = posixpath.normpath(path)
    found, depth = None, -1
    for entry in table:
        point = posixpath.normpath(entry["point"])
        if point == "/" or path == point or path.startswith(point + "/"):
            level = 0 if point == "/" else point.count("/")
            if level >= depth:
                found, depth = entry, level
    return found


def _nearest_existing(path):
    """The deepest existing prefix of ``path``; every existing component must not be a symlink."""
    current, found = "", "/"
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except NotADirectoryError:
            raise ValueError(f"{path} has a file where a directory belongs ({found})") from None
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Installer paths must be absolute without symlink components: " + path)
        found = current
    return found


def local_filesystem(path, table=None):
    """Mount entry of checkpoint directory ``path`` after checking where it lives.

    The deepest mount covering ``path``'s nearest existing ancestor must be one
    of ``LOCAL_TYPES``: network storage is shared or disappears, and tmpfs and
    ramfs lose a checkpoint at reboot. ``path`` itself must not be a mount point,
    so that its state directory shares its mount. Existing components must not
    be symlinks, because the lookup is lexical.
    """
    path = _checked_path(path)
    ancestor = _nearest_existing(path)
    info = os.lstat(ancestor)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{ancestor} is not a directory")
    table = mounts() if table is None else table
    entry = mount_of(ancestor, table)
    if entry is None:
        raise ValueError(f"The checkpoint directory's filesystem at {path} is not in the mount table, "
                         "so it is not a known local filesystem")
    if entry["type"] not in LOCAL_TYPES:
        raise ValueError(f"The checkpoint directory's filesystem at {path} is {entry['type']}, not a local "
                         "filesystem. SparkRing keeps checkpoints only on ext2, ext3, ext4, xfs, btrfs, f2fs or zfs.")
    point = any(posixpath.normpath(other["point"]) == path for other in table)
    if not point and ancestor == path:
        point = os.lstat(posixpath.dirname(path)).st_dev != info.st_dev
    if point:
        raise ValueError(f"{path} is a mount point; a SparkRing checkpoint directory must be a plain directory "
                         "on its parent's filesystem, so that its state directory shares its mount")
    return entry


# Descriptor helpers --------------------------------------------------------

def _open_directory(dir_fd, name):
    try:
        return os.open(name, DIRECTORY_FLAGS, dir_fd=dir_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(f"{name} is a symlink or a file where SparkRing expects a directory") from None
        raise


def _open_tree(path, create):
    """Descriptor of absolute directory ``path``, opened one component at a time without following symlinks."""
    fd = os.open("/", DIRECTORY_FLAGS)
    try:
        for part in [part for part in path.split("/") if part]:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            child = _open_directory(fd, part)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _parent(dir_fd, name, create=False):
    """``(descriptor, last component)`` for relative ``name`` below ``dir_fd``.

    Intermediate directories are opened without following symlinks and, with
    ``create``, made with mode 0755. The caller closes the descriptor when it
    differs from ``dir_fd``.
    """
    parts = safe_name(name).split("/")
    fd = dir_fd
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            child = _open_directory(fd, part)
            if fd != dir_fd:
                os.close(fd)
            fd = child
    except BaseException:
        if fd != dir_fd:
            os.close(fd)
        raise
    return fd, parts[-1]


def _lstat_at(dir_fd, name):
    """``lstat`` of relative ``name`` below ``dir_fd``, or ``None`` when it or a parent is absent."""
    try:
        parent, base = _parent(dir_fd, name)
    except FileNotFoundError:
        return None
    try:
        return os.lstat(base, dir_fd=parent)
    except FileNotFoundError:
        return None
    finally:
        if parent != dir_fd:
            os.close(parent)


def _pread(fd, count, offset):
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, count)


def _read_json_at(dir_fd, name, limit=JSON_LIMIT):
    """Parsed JSON of regular file ``name`` in ``dir_fd``, or ``None`` when absent."""
    try:
        fd = os.open(name, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | _CLOEXEC, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{name} is a symlink, not a SparkRing record") from None
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{name} is not a SparkRing record")
        data, offset = [], 0
        while block := _pread(fd, BLOCK, offset):
            data.append(block)
            offset += len(block)
            if offset > limit:
                raise ValueError(f"{name} is not a SparkRing record")
    finally:
        os.close(fd)
    try:
        return json.loads(b"".join(data))
    except ValueError:
        raise ValueError(f"{name} is not valid JSON") from None


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _write_json_at(dir_fd, name, value):
    """Replace ``name`` in ``dir_fd`` atomically and durably with ``value`` (mode 0600)."""
    temporary = f"{name}.{os.getpid()}.{threading.get_ident()}.writing"
    try:
        os.unlink(temporary, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC, 0o600, dir_fd=dir_fd)
    try:
        _write_all(fd, (json.dumps(value, indent=2) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    os.fsync(dir_fd)


def _empty(fd, keep=()):
    """Remove every entry below directory ``fd`` except the top-level names in ``keep``; symlinks are not followed."""
    for name in os.listdir(fd):
        if name in keep:
            continue
        info = os.lstat(name, dir_fd=fd)
        if stat.S_ISDIR(info.st_mode):
            child = _open_directory(fd, name)
            try:
                _empty(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)


# Claiming ------------------------------------------------------------------

class Claim:
    """A SparkRing checkpoint directory held under its lock.

    ``dir_fd``, ``state_fd`` and ``parent_fd`` are directory descriptors of
    ``D``, its state directory and its parent. ``identity`` is ``D``'s
    ``[st_dev, st_ino]`` from ``owner.json``. ``action`` is ``created``,
    ``claimed`` (an empty directory that existed) or ``verified`` (an earlier
    claim). Closing the claim releases the lock.
    """

    def __init__(self, path, parent_fd, dir_fd, state_fd, lock_fd, directory, action):
        self.path = path
        self.name = posixpath.basename(path)
        self.state = posixpath.join(posixpath.dirname(path), "." + self.name + ".sparkring")
        self.parent_fd, self.dir_fd, self.state_fd, self.lock_fd = parent_fd, dir_fd, state_fd, lock_fd
        self.identity = list(directory)
        self.action = action

    def check(self):
        """Refuse when ``D``, by path, through its parent or as held, is no longer the recorded directory."""
        try:
            views = [os.lstat(self.name, dir_fd=self.parent_fd), os.lstat(self.path), os.fstat(self.dir_fd)]
        except (FileNotFoundError, NotADirectoryError):
            views = [None]
        if any(view is None or not stat.S_ISDIR(view.st_mode) or identity(view) != self.identity for view in views):
            raise ValueError(REPLACED.format(self.path))

    def close(self):
        for fd in (self.dir_fd, self.state_fd, self.parent_fd, self.lock_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.dir_fd = self.state_fd = self.parent_fd = self.lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def _open_state(parent_fd, name, path, create):
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        fd = _open_directory(parent_fd, name)
    except FileNotFoundError:
        if create:
            raise
        return None
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        os.close(fd)
        raise ValueError(f"{path} is not a private directory of this account; SparkRing uses only a state "
                         "directory it created")
    return fd


def _lock(state_fd, path):
    if fcntl is None:
        raise RuntimeError("SparkRing checkpoint directories need POSIX file locks")
    fd = os.open("lock", os.O_RDWR | os.O_CREAT | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | _CLOEXEC, 0o600,
                 dir_fd=state_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"The lock of {path} is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ValueError(f"Another SparkRing operation is changing {path}; wait until it finishes, "
                         "then repeat the command") from None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _private_ancestors(path):
    """Refuse ``path`` when an existing directory above it can be changed by other accounts.

    Docker resolves the bind-mount source of ``fetch/`` and rsync resolves its
    destination in ``receive/`` by path, after SparkRing's last check. An
    account that can write a directory above ``D`` could rename the state
    directory and put a symlink in its place, so every existing ancestor must
    be writable by its owner alone or carry the sticky bit, which keeps other
    accounts from renaming entries they do not own. A directory owned by
    another single account stays that account's to change; checkpoint paths
    belong below directories that only root or the operator can change.
    """
    parts = [part for part in posixpath.dirname(path).split("/") if part]
    for depth in range(len(parts) + 1):
        current = "/" + "/".join(parts[:depth])
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise ValueError(f"{current} can be changed by other accounts (mode {stat.S_IMODE(info.st_mode):04o}); "
                             "SparkRing keeps a checkpoint directory only below directories that other accounts "
                             f"cannot rename or replace. Remove group and other write permission from {current}, "
                             "or choose another path.")


def _claimable(parent_fd, name, path):
    """Whether ``D`` exists as an empty directory (``True``) or is absent (``False``); anything else is refused.

    An existing ``D`` must also belong to this account and be writable by it
    alone, so that no other account can add, remove or replace names in it.
    """
    try:
        info = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{path} is a symlink; SparkRing does not adopt a directory it did not create. "
                         "Move it away or choose another path.")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(NOT_EMPTY.format(path))
    fd = _open_directory(parent_fd, name)
    try:
        if os.listdir(fd):
            raise ValueError(NOT_EMPTY.format(path))
        held = os.fstat(fd)
        if held.st_uid != os.geteuid() or held.st_mode & 0o022:
            raise ValueError(f"{path} belongs to another account or other accounts can write to it; SparkRing "
                             "claims only an empty directory that it owns and that only it can change. Remove the "
                             "directory, or choose another path.")
    finally:
        os.close(fd)
    return True


def _verify_owner(owner, parent_fd, name, path, state, repository, revision):
    if (not isinstance(owner, dict) or owner.get("schema") != OWNER_SCHEMA or owner.get("path") != path
            or not _is_ints(owner.get("directory"), 2)):
        raise ValueError(f"{state}/owner.json does not describe {path}; SparkRing does not write into {path}")
    if (owner.get("repository"), owner.get("revision")) != (repository, revision):
        raise ValueError(f"{path} holds {owner.get('repository')} at {owner.get('revision')}, "
                         f"not {repository} at {revision}")
    try:
        info = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        info = None
    if info is None or not stat.S_ISDIR(info.st_mode) or identity(info) != owner["directory"]:
        raise ValueError(REPLACED.format(path) + f" Move it away and remove {state} to let SparkRing create it again.")
    return owner["directory"]


def claim(path, repository, revision, *, table=None):
    """Claim or re-verify checkpoint directory ``path`` and hold its lock; returns a ``Claim``.

    - No ``owner.json`` and ``D`` absent: create the state directory, ``D``
      (mode 0755), an empty journal and ``owner.json`` with ``D``'s identity.
    - No ``owner.json`` and ``D`` an empty, non-symlink directory that is not a
      mount point, owned by this account and writable by it alone: claim it the
      same way.
    - No ``owner.json`` and ``D`` anything else: refuse before creating anything.
    - ``owner.json`` present: ``D`` must still be the recorded directory.

    Every existing directory above ``D`` must be writable by its owner alone
    or carry the sticky bit (``_private_ancestors``). Missing parents of ``D``
    are created with mode 0755. ``table`` replaces the mount table for the
    local-filesystem check.
    """
    path = _checked_path(path)
    local_filesystem(path, table)
    _private_ancestors(path)
    parent_path, name = posixpath.split(path)
    state_name = "." + name + ".sparkring"
    state = posixpath.join(parent_path, state_name)
    parent_fd = _open_tree(parent_path, create=True)
    state_fd = lock_fd = dir_fd = None
    try:
        state_fd = _open_state(parent_fd, state_name, state, create=False)
        if state_fd is None or _read_json_at(state_fd, "owner.json") is None:
            _claimable(parent_fd, name, path)
            if state_fd is None:
                state_fd = _open_state(parent_fd, state_name, state, create=True)
        lock_fd = _lock(state_fd, path)
        owner = _read_json_at(state_fd, "owner.json")
        if owner is not None:
            directory = _verify_owner(owner, parent_fd, name, path, state, repository, revision)
            dir_fd = _open_directory(parent_fd, name)
            action = "verified"
        else:
            existed = _claimable(parent_fd, name, path)
            if not existed:
                os.mkdir(name, 0o755, dir_fd=parent_fd)
            dir_fd = _open_directory(parent_fd, name)
            held = os.fstat(dir_fd)
            if os.listdir(dir_fd):
                raise ValueError(NOT_EMPTY.format(path))
            if held.st_dev != os.fstat(parent_fd).st_dev:
                raise ValueError(f"{path} is a mount point; a SparkRing checkpoint directory must be a plain "
                                 "directory on its parent's filesystem")
            directory = identity(held)
            _write_json_at(state_fd, "journal.json", {"schema": JOURNAL_SCHEMA, "files": {}, "directories": []})
            _write_json_at(state_fd, "owner.json", {"schema": OWNER_SCHEMA, "repository": repository,
                                                    "revision": revision, "path": path, "directory": directory})
            action = "claimed" if existed else "created"
        claimed = Claim(path, parent_fd, dir_fd, state_fd, lock_fd, directory, action)
        claimed.check()
    except BaseException:
        for fd in (dir_fd, state_fd, parent_fd, lock_fd):
            if fd is not None:
                os.close(fd)
        raise
    return claimed


# Staging -------------------------------------------------------------------

def staging(claimed, purpose, *, empty=False):
    """Descriptor of the marked staging directory ``<state>/<purpose>``, created when absent.

    An existing directory is used only when it is a direct, non-symlink child of
    the state directory whose marker names this purpose and ``D``; an unmarked
    one is used only while empty. With ``empty``, every entry except the marker
    is removed. The caller closes the descriptor.
    """
    if purpose not in STAGING_PURPOSES:
        raise ValueError("Unknown staging purpose: " + repr(purpose)[:100])
    marker = {"purpose": purpose, "path": claimed.path}
    where = posixpath.join(claimed.state, purpose)
    try:
        os.mkdir(purpose, 0o700, dir_fd=claimed.state_fd)
    except FileExistsError:
        pass
    fd = _open_directory(claimed.state_fd, purpose)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise ValueError(f"{where} is not a private directory of this account")
        found = _read_json_at(fd, STAGING_MARKER)
        if found is None:
            if os.listdir(fd):
                raise ValueError(f"{where} holds files without SparkRing's staging marker; SparkRing neither uses "
                                 "nor removes it. Move it away and repeat the command.")
            _write_json_at(fd, STAGING_MARKER, marker)
        elif found != marker:
            raise ValueError(f"{where} is marked for another use ({found!r:.200}); SparkRing neither uses nor removes it")
        if empty:
            _empty(fd, keep=(STAGING_MARKER,))
    except BaseException:
        os.close(fd)
        raise
    return fd


def remove_staging(claimed, purpose):
    """Remove the staging directory ``<state>/<purpose>`` when its marker matches; ``False`` when absent."""
    if purpose not in STAGING_PURPOSES:
        raise ValueError("Unknown staging purpose: " + repr(purpose)[:100])
    where = posixpath.join(claimed.state, purpose)
    try:
        fd = _open_directory(claimed.state_fd, purpose)
    except FileNotFoundError:
        return False
    try:
        found = _read_json_at(fd, STAGING_MARKER)
        if found is None and os.listdir(fd):
            raise ValueError(f"{where} holds files without SparkRing's staging marker; SparkRing does not remove it")
        if found is not None and found != {"purpose": purpose, "path": claimed.path}:
            raise ValueError(f"{where} is marked for another use; SparkRing does not remove it")
        _empty(fd, keep=(STAGING_MARKER,))
        if found is not None:
            os.unlink(STAGING_MARKER, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(purpose, dir_fd=claimed.state_fd)
    return True


def create_part(staging_fd, part):
    """Descriptor of a new, empty staging file ``part`` (``O_EXCL``, mode 0644); a stale one is unlinked first."""
    parent, base = _parent(staging_fd, part, create=True)
    try:
        try:
            os.unlink(base, dir_fd=parent)
        except FileNotFoundError:
            pass
        return os.open(base, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC, 0o644, dir_fd=parent)
    finally:
        if parent != staging_fd:
            os.close(parent)


def discard_part(staging_fd, part, fd=None):
    """Close ``fd`` when given and unlink staging file ``part`` when present."""
    if fd is not None:
        os.close(fd)
    parent, base = _parent(staging_fd, part)
    try:
        os.unlink(base, dir_fd=parent)
    except FileNotFoundError:
        pass
    finally:
        if parent != staging_fd:
            os.close(parent)


def copy_part(source_fd, size, staging_fd, part):
    """Copy exactly ``size`` bytes of ``source_fd`` into a new staging file while hashing them.

    Returns ``(descriptor of the part, sha256)``; the part is synced to disk and
    stays open for ``place_staged``. A source that ends early or holds more
    bytes is an error, and the part is removed.
    """
    output = create_part(staging_fd, part)
    digest = hashlib.sha256()
    try:
        offset = 0
        while offset < size:
            block = _pread(source_fd, min(BLOCK, size - offset), offset)
            if not block:
                raise ValueError(f"The source ended after {offset} of its pinned {size} bytes")
            digest.update(block)
            _write_all(output, block)
            offset += len(block)
        if _pread(source_fd, 1, size):
            raise ValueError(f"The source holds more than its pinned {size} bytes")
        os.fsync(output)
    except BaseException:
        discard_part(staging_fd, part, output)
        raise
    return output, digest.hexdigest()


# Journal -------------------------------------------------------------------

class Journal:
    """The placement journal of one claimed directory, persisted on every change.

    ``files`` maps each name to ``{"state", "identity", "sha256", "stats",
    "origin", "source"}``. ``directories`` is the set of directories inside
    ``D`` that SparkRing created for nested names, recorded before each is
    made. Methods of this module serialize changes, so threads may place
    different names concurrently.
    """

    def __init__(self, claimed, files, directories=()):
        self.claim = claimed
        self.files = files
        self.directories = set(directories)
        self.guard = threading.RLock()


def _valid_entry(entry):
    return (isinstance(entry, dict) and entry.get("state") in ("placing", "placed")
            and _is_ints(entry.get("identity"), 2) and isinstance(entry.get("sha256"), str)
            and bool(_SHA256.fullmatch(entry["sha256"])) and _is_ints(entry.get("stats"), 5)
            and entry.get("origin") in ORIGINS
            and (entry.get("source") is None or isinstance(entry.get("source"), str)))


def journal_load(claimed):
    """The journal of ``claimed``; an absent file is an empty journal, a damaged one is refused.

    A journal without ``directories`` records no directory.
    """
    document = _read_json_at(claimed.state_fd, "journal.json")
    files, directories = {}, []
    if document is not None:
        if (not isinstance(document, dict) or document.get("schema") != JOURNAL_SCHEMA
                or not isinstance(document.get("files"), dict)
                or not isinstance(document.get("directories", []), list)):
            raise ValueError(f"{claimed.state}/journal.json is not a SparkRing checkpoint journal")
        for name, entry in document["files"].items():
            safe_name(name)
            if not _valid_entry(entry):
                raise ValueError(f"{claimed.state}/journal.json has an invalid entry for {name}")
            files[name] = entry
        directories = [safe_name(name) for name in document.get("directories", [])]
    return Journal(claimed, files, directories)


def _journal_save(journal):
    _write_json_at(journal.claim.state_fd, "journal.json", {"schema": JOURNAL_SCHEMA, "files": journal.files,
                                                            "directories": sorted(journal.directories)})


def journal_directories(journal, names):
    """Durably record the directories that hold ``names`` (relative file names) and their parents.

    Called before those directories are created, so that every directory
    SparkRing makes inside ``D`` is recorded first and a release removes only
    recorded directories.
    """
    wanted = set()
    for name in names:
        parent = posixpath.dirname(safe_name(name))
        while parent:
            wanted.add(parent)
            parent = posixpath.dirname(parent)
    with journal.guard:
        if wanted - journal.directories:
            journal.directories |= wanted
            _journal_save(journal)


def journal_begin(journal, name, *, inode, sha256, measured, origin, source=None):
    """Durably record ``name`` as ``placing`` for ``inode`` before the name is created.

    ``measured`` is the five-value stats of the verified file before it was
    placed. A name already recorded for another inode is refused.
    """
    safe_name(name)
    inode, measured = list(inode), list(measured)
    if (not _is_ints(inode, 2) or not _is_ints(measured, 5) or measured[:2] != inode
            or not isinstance(sha256, str) or not _SHA256.fullmatch(sha256) or origin not in ORIGINS
            or not (source is None or isinstance(source, str))):
        raise ValueError("Invalid checkpoint journal entry for " + name)
    with journal.guard:
        entry = journal.files.get(name)
        if entry is not None and entry["identity"] != inode:
            raise ValueError(f"{journal.claim.path}/{name} is already recorded for another file; SparkRing "
                             "removes its own name first")
        journal.files[name] = {"state": "placing", "identity": inode, "sha256": sha256, "stats": measured,
                               "origin": origin, "source": source}
        _journal_save(journal)


def journal_place(journal, name, measured):
    """Durably record ``name`` as ``placed`` with the five-value stats of the placed inode."""
    measured = list(measured)
    with journal.guard:
        entry = journal.files.get(name)
        if entry is None or not _is_ints(measured, 5) or measured[:2] != entry["identity"]:
            raise ValueError(f"{journal.claim.path}/{name} is not recorded for this file")
        entry["state"], entry["stats"] = "placed", measured
        _journal_save(journal)


def journal_drop(journal, name):
    """Durably forget ``name``; returns whether it was recorded."""
    with journal.guard:
        if journal.files.pop(name, None) is None:
            return False
        _journal_save(journal)
        return True


def journal_recover(dir_fd, journal):
    """Settle the journal after an interrupted operation.

    - ``placing`` with the recorded inode at the name: completed as ``placed``.
      Its stats become the current ones when size and modification time equal
      the recorded values; otherwise the recorded stats stay, so the next stat
      comparison hashes it again.
    - ``placing`` or ``placed`` with the name absent: dropped.
    - Any entry whose name holds another inode: reported as ``foreign`` and kept.

    Returns ``{"completed", "dropped", "missing", "foreign"}`` name lists.
    """
    result = {"completed": [], "dropped": [], "missing": [], "foreign": []}
    with journal.guard:
        for name in sorted(journal.files):
            entry = journal.files[name]
            current = _lstat_at(dir_fd, name)
            if current is None:
                journal_drop(journal, name)
                result["dropped" if entry["state"] == "placing" else "missing"].append(name)
            elif identity(current) != entry["identity"] or not stat.S_ISREG(current.st_mode):
                result["foreign"].append(name)
            elif entry["state"] == "placing":
                same = [current.st_size, current.st_mtime_ns] == entry["stats"][2:4]
                journal_place(journal, name, stats(current) if same else entry["stats"])
                result["completed"].append(name)
    return result


def unlink_placed(dir_fd, name, journal):
    """Unlink SparkRing's own name ``name`` from ``D``; only a name whose inode the journal records is removed.

    Returns ``False`` when the name was already absent (its entry is dropped).
    """
    with journal.guard:
        entry = journal.files.get(name)
        if entry is None:
            raise ValueError(f"{journal.claim.path}/{name} is not in SparkRing's journal; SparkRing removes only "
                             "names it placed")
        journal.claim.check()
        parent, base = _parent(dir_fd, name)
        try:
            try:
                current = os.lstat(base, dir_fd=parent)
            except FileNotFoundError:
                journal_drop(journal, name)
                return False
            if identity(current) != entry["identity"]:
                raise ValueError(f"{journal.claim.path}/{name} is not the file SparkRing placed; SparkRing does not "
                                 "remove it")
            os.unlink(base, dir_fd=parent)
            os.fsync(parent)
        finally:
            if parent != dir_fd:
                os.close(parent)
        journal_drop(journal, name)
        return True


# Reading and hashing sources ----------------------------------------------

def open_source(path, inode, size, *, dir_fd=None):
    """Open a file for reading and check that it is the planned regular file.

    Opens with ``O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_NOATIME | O_CLOEXEC``, so
    a symlink, FIFO or device is refused without blocking and access times stay
    unchanged. ``fstat`` must show a regular file of ``size`` bytes and, when
    ``inode`` is given, that ``[st_dev, st_ino]``. With ``dir_fd``, ``path`` is
    a relative name reached without following symlinks.

    Returns ``(descriptor, fstat result)``; the caller closes the descriptor.
    """
    if dir_fd is None:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("Checkpoint sources need an absolute path: " + repr(path)[:200])
        parent, base = None, path
    else:
        parent, base = _parent(dir_fd, path)
    try:
        info = os.lstat(base, dir_fd=parent)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path} is not a regular file; SparkRing reads only regular files")
        try:
            fd = os.open(base, READ_FLAGS, dir_fd=parent)
        except OSError as error:
            raise ValueError(f"{path} could not be opened for reading: {error.strerror}") from None
    finally:
        if parent is not None and parent != dir_fd:
            os.close(parent)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{path} is not a regular file; SparkRing reads only regular files")
        if inode is not None and identity(opened) != list(inode):
            raise ValueError(f"{path} is no longer the file the plan identified")
        if opened.st_size != size:
            raise ValueError(f"{path} is {opened.st_size} bytes, not its pinned {size}")
    except BaseException:
        os.close(fd)
        raise
    return fd, opened


def hash_descriptor(fd, size):
    """SHA-256 of exactly ``size`` bytes read through ``fd`` from offset 0.

    A file that ends early or holds more bytes is an error.
    """
    digest, offset = hashlib.sha256(), 0
    while offset < size:
        block = _pread(fd, min(BLOCK, size - offset), offset)
        if not block:
            raise ValueError(f"The file ended after {offset} of its pinned {size} bytes")
        digest.update(block)
        offset += len(block)
    if _pread(fd, 1, size):
        raise ValueError(f"The file holds more than its pinned {size} bytes")
    return digest.hexdigest()


# Placement -----------------------------------------------------------------

def _place(dir_fd, fd, name, journal, sha256, before, origin, source, staged=None):
    claimed = journal.claim
    claimed.check()
    if identity(os.fstat(dir_fd)) != claimed.identity:
        raise ValueError("Placement needs the descriptor of the claimed checkpoint directory " + claimed.path)
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError("Placement needs the SHA-256 computed through the descriptor")
    current = os.fstat(fd)
    label = source or name
    if not stat.S_ISREG(current.st_mode) or _content_key(current) != _content_key(before):
        raise ValueError(f"{label} changed after SparkRing opened it; it was not placed")
    if stats(current) != stats(before) and (hash_descriptor(fd, current.st_size) != sha256
                                            or stats(os.fstat(fd)) != stats(current)):
        # Only the change time moved since hashing: a rename, another link, or a
        # write that restored the modification time. The second read decides.
        raise ValueError(f"{label} changed after SparkRing opened it; it was not placed")
    inode = identity(current)
    journal_directories(journal, [name])
    parent, base = _parent(dir_fd, name, create=True)
    try:
        journal_begin(journal, name, inode=inode, sha256=sha256, measured=stats(before), origin=origin,
                      source=source)
        try:
            os.link(f"/proc/self/fd/{fd}", base, dst_dir_fd=parent)
        except FileExistsError:
            if identity(os.lstat(base, dir_fd=parent)) != inode:
                journal_drop(journal, name)
                raise ValueError(f"{claimed.path} already holds {name} as another file; SparkRing never replaces "
                                 "a name") from None
        except OSError:
            journal_drop(journal, name)
            raise
        os.fsync(parent)
        placed, after = os.lstat(base, dir_fd=parent), os.fstat(fd)
        if identity(placed) != inode:
            raise ValueError(f"{claimed.path}/{name} changed while SparkRing placed it")
        if _content_key(after) != _content_key(before):
            os.unlink(base, dir_fd=parent)
            journal_drop(journal, name)
            raise ValueError(f"{label} changed while SparkRing verified it; it was not placed")
        if staged is not None:
            staged_fd, staged_name = staged
            try:
                if identity(os.lstat(staged_name, dir_fd=staged_fd)) == inode:
                    os.unlink(staged_name, dir_fd=staged_fd)
            except FileNotFoundError:
                pass
            # Removing the staging name changes the link count and so the change time.
            after = os.fstat(fd)
            if _content_key(after) != _content_key(before):
                raise ValueError(f"{claimed.path}/{name} changed while SparkRing placed it")
        journal_place(journal, name, stats(after))
    finally:
        if parent != dir_fd:
            os.close(parent)
    return after


def _record(value, sha256, seen_as):
    # Per-inode records are evidence only; a failed write costs a later re-hash, never correctness.
    try:
        record_inode(value, sha256, seen_as)
    except (OSError, ValueError):
        pass


def place_link(dir_fd, fd, name, journal, *, sha256, before, source=None):
    """Hard-link the verified inode open at ``fd`` into ``D`` as ``name``.

    ``sha256`` is the digest ``hash_descriptor`` computed through ``fd`` and
    must already equal the pin; ``before`` is the ``fstat`` taken before
    hashing (from ``open_source``). Before the link, device, inode, size and
    modification time must still equal ``before``; when only the change time
    moved, the descriptor is hashed again and must still give ``sha256``. The
    journal records ``placing`` first; the link is made from ``/proc/self/fd``
    onto a name that does not exist, so the placed inode is the hashed one even
    if the source path was swapped. ``EEXIST`` with the same inode is accepted;
    another inode is refused. ``EXDEV`` or ``EPERM`` drops the journal entry
    and is re-raised, so the caller can copy instead. Afterwards device, inode,
    size and modification time must equal ``before``; the entry becomes
    ``placed`` with the post-link stats and a per-inode record is written.
    Returns those stats.
    """
    after = _place(dir_fd, fd, name, journal, sha256, before, "link", source)
    _record(after, sha256, source)
    return stats(after)


def place_staged(dir_fd, staging_fd, part, name, journal, *, fd, sha256, origin, source=None, before=None):
    """Place staging file ``part``, open and verified at ``fd``, as ``name`` in ``D``.

    ``origin`` is ``copy``, ``fabric``, ``rsync`` or ``hub``. The part must be a
    regular file owned by this account, not writable by group or others, whose
    only other link, if any, is ``name`` itself from an interrupted placement.
    The journal records ``placing``, the inode at ``fd`` is linked to ``name``
    (never replacing), the part is unlinked and the entry becomes ``placed``.
    ``before`` defaults to the current ``fstat`` of ``fd``. Returns the placed
    stats.
    """
    if origin not in ("copy", "fabric", "rsync", "hub"):
        raise ValueError("Unknown staged origin: " + repr(origin)[:100])
    current = os.fstat(fd)
    before = current if before is None else before
    staged_fd, staged_name = _parent(staging_fd, part)
    try:
        staged = os.lstat(staged_name, dir_fd=staged_fd)
        final = _lstat_at(dir_fd, name)
        links = 2 if final is not None and identity(final) == identity(current) else 1
        if (identity(staged) != identity(current) or not stat.S_ISREG(current.st_mode)
                or current.st_nlink != links or current.st_uid != os.geteuid() or current.st_mode & 0o022):
            raise ValueError(f"The staged file {part} is not a private regular file linked only by SparkRing's "
                             "staging directory; it was not placed")
        after = _place(dir_fd, fd, name, journal, sha256, before, origin, source,
                       staged=(staged_fd, staged_name))
    finally:
        if staged_fd != staging_fd:
            os.close(staged_fd)
    _record(after, sha256, posixpath.join(journal.claim.path, name))
    return stats(after)


def check_placed(dir_fd, name, journal, size, pin):
    """Verify journaled name ``name`` of ``D`` against its pinned size and SHA-256.

    Returns ``verified`` when all five stats equal the journal, ``rehashed``
    when the file was hashed again through a descriptor, matched, and kept all
    five stats while it was read (the journal's stats are updated), ``removed``
    when its content or size differed or it changed while it was read and
    SparkRing unlinked its own name, and ``missing`` when the name is absent
    (its entry is dropped). A name holding another inode is refused.
    """
    entry = journal.files.get(name)
    if entry is None or entry["state"] != "placed":
        raise ValueError(f"{journal.claim.path}/{name} is not a placed name in SparkRing's journal")
    current = _lstat_at(dir_fd, name)
    if current is None:
        journal_drop(journal, name)
        return "missing"
    if identity(current) != entry["identity"] or not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{journal.claim.path} holds files SparkRing did not place: {name}")
    if stats(current) == entry["stats"] and entry["sha256"] == pin:
        return "verified"
    digest, after, before = None, None, current
    try:
        fd, before = open_source(name, entry["identity"], size, dir_fd=dir_fd)
    except ValueError:
        fd = None
    if fd is not None:
        try:
            digest = hash_descriptor(fd, size)
            after = os.fstat(fd)
        except ValueError:
            digest = None
        finally:
            os.close(fd)
    if digest == pin and after is not None and stats(after) == stats(before):
        with journal.guard:
            entry["sha256"] = pin
            journal_place(journal, name, stats(after))
        return "rehashed"
    unlink_placed(dir_fd, name, journal)
    return "removed"


# Records and listings ------------------------------------------------------

def record_inode(value, sha256, seen_as=None, *, directory=None, now=time.time):
    """Write the per-inode record ``<RECORDS>/<st_dev>-<st_ino>.json`` and return it.

    ``value`` is the ``fstat`` of the descriptor that was hashed, so the record
    describes the inode whose content produced ``sha256``. ``seen_as`` is kept
    only while that path still names the same inode. Records are evidence for
    planning and never replace the read that placement requires.
    """
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError("Per-inode records need a SHA-256")
    record = {"identity": identity(value), "stats": stats(value), "sha256": sha256}
    if seen_as:
        try:
            current = os.lstat(seen_as)
        except OSError:
            current = None
        if current is not None and stat.S_ISREG(current.st_mode) and identity(current) == record["identity"]:
            record["seen_as"] = seen_as
    record["observed_at"] = now()
    root = RECORDS if directory is None else directory
    os.makedirs(root, mode=0o700, exist_ok=True)
    fd = os.open(root, DIRECTORY_FLAGS)
    try:
        _write_json_at(fd, f"{value.st_dev}-{value.st_ino}.json", record)
    finally:
        os.close(fd)
    return record


def listing(dir_fd):
    """``(files, directories)`` below ``dir_fd``, without following symlinks.

    ``files`` maps the relative name of every entry that is not a directory
    (symlinks and special files included) to its ``lstat``; ``directories`` is
    the set of relative directory names.
    """
    files, directories = {}, set()

    def walk(fd, prefix):
        for name in sorted(os.listdir(fd)):
            info = os.lstat(name, dir_fd=fd)
            relative = prefix + name
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                child = _open_directory(fd, name)
                try:
                    walk(child, relative + "/")
                finally:
                    os.close(child)
            else:
                files[relative] = info

    walk(dir_fd, "")
    return files, directories


def unexpected_names(dir_fd, journal, required):
    """Entries of ``D`` that SparkRing did not place for a required name, sorted.

    A file is expected when it is a required name, a regular file, and journaled
    with its current inode. A directory is expected when it is a parent of a
    required name.
    """
    files, directories = listing(dir_fd)
    required = set(required)
    parents = {posixpath.dirname(name) for name in required}
    for name in list(parents):
        while name:
            parents.add(name)
            name = posixpath.dirname(name)
    wrong = []
    for name, info in files.items():
        entry = journal.files.get(name)
        if (name not in required or entry is None or not stat.S_ISREG(info.st_mode)
                or entry["identity"] != identity(info)):
            wrong.append(name)
    wrong.extend(name for name in directories if name not in parents)
    return sorted(wrong)
