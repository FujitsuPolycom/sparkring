"""Report SparkRing's disk use on every Spark and release data that no deployment uses.

Two commands use this module:

- ``sudo sparkring storage`` on Node A (``main``) reports, for every Spark of
  the cluster, the filesystems that hold ``/srv/sparkring``, Docker's data root
  and ``/`` (size, used and free bytes), every item in ``/srv/sparkring`` and
  every Docker image with its size and class, and the releases it proposes.
- ``sudo sparkring storage --release PATH`` removes one cache directory or
  deployment workspace of class ``unreferenced`` from every Spark that holds
  it, after the operator approves (``--yes`` in scripts). Node A lists and
  classifies every Spark again while it holds the installation lock, and
  refuses while a Spark cannot be listed or the Sparks run different package
  revisions. Each Spark then checks again that no installed deployment and no
  running container uses the path and that it holds no checkpoint directory,
  model files or mount point (``release_local``). It never removes a
  checkpoint directory, which only ``sudo sparkring checkpoints --release
  PATH`` releases, a Docker image, or anything SparkRing's installer did not
  create.

Each Spark runs its own part as ``sudo sparkring node storage [--release
PATH]`` (``node``), reading a JSON request on stdin: ``list_local`` and
``release_local``.

Items
-----
``kind`` says what an item is:

- ``checkpoint``: a SparkRing checkpoint directory, as
  ``runtime/host/checkpoints.py`` lists it, with the sizes it reports.
- ``cache``: a directory directly in a cache root whose name has the form
  installer containers give their caches, ``<family>-<image-12>-<revision-12>``
  or ``<family>-cuda<version>-<revision-12>`` (``cache_names``). Cache
  roots are the cluster cache ``/srv/sparkring/<cluster>/cache`` and the cache
  paths of Node A's retained deployments. Containers mount a cache root at
  ``/cache`` and name their directory in ``XDG_CACHE_HOME`` and related
  variables.
- ``workspace``: a deployment workspace, which the installer creates holding an
  ``.installer-owner.json`` record: a directory directly in
  ``/srv/sparkring/<cluster>`` that holds one, or a workspace that a retained
  deployment's lock names.
- ``releasing``: the remainder of an interrupted release. A release first
  renames ``PATH`` to ``.<name>.sparkring-releasing`` in the same directory,
  then removes it; a repeated ``--release PATH`` removes the remainder.
- ``other``: anything else in ``/srv/sparkring``: entries other than the
  cluster's directory, entries of the cluster directory and of cache roots that
  are none of the above, and files below the cluster's ``checkpoints`` outside
  SparkRing's checkpoint directories.

``class`` says what references an item:

- ``installed``: a deployment that SparkRing keeps installed uses it: the active
  deployment (``active.json``), the rollback target recorded in
  ``transaction.json`` or the candidate of an unfinished model switch. Node A
  reads each deployment's lock and container specifications
  (``deployment.lock.json`` and ``rank<N>/container.json`` in its directory
  below ``/var/lib/sparkring/controller/deployments``): the workspace, model, source
  and container paths of the Spark's row, every bind source, and every host
  path that an environment value or argument names through a bind. An item is
  used when one of those paths is the item or lies inside it. A deployment
  whose container specifications cannot be read uses every entry of its
  cache root.
- ``profile``: an installer profile of the installed package references it:
  a checkpoint revision the profile lists, the image of the image lock that the
  installer selects for it (``installer_image.for_profile``), or a cache name
  that they produce. A later installation of that profile reuses it.
- ``unreferenced``: neither. Other retained deployments that use the item are
  named; after a release they need ``sudo sparkring install`` again. An item
  that a running container on the Spark reaches (``_users``) is named with
  that container and not proposed for release.
- ``unmanaged``: every ``other`` item. SparkRing's installer did not create
  it, and SparkRing never proposes or performs its release.

Sizes
-----
An item's ``bytes`` are the allocated bytes of the distinct inodes below it, as
``du -x`` counts them: a file with several hard links in the item counts once,
the walk stays on the item's filesystem (other filesystems are listed in
``mounts``) and does not descend into other items. ``frees_bytes`` counts the
inodes whose every hard link lies in the item, which removing it frees; an
inode also linked from elsewhere, such as a checkpoint file linked from an
operator's copy, stays on disk. Checkpoint directories carry the byte counts of
their journal instead. All walks of one Spark share a time budget
(``BUDGET_SECONDS``); an item whose walk stopped at it has ``complete``
false and its sizes are lower bounds.

Docker images are listed with ``docker image inspect``'s size, which counts
layers shared with other images in each of them. SparkRing does not remove
images.
"""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import posixpath
import re
import socket
import stat
import subprocess
import sys
import time

from runtime.host import checkpoint_place, checkpoints
from runtime.host.install_errors import NeedsInput

LOCAL_SCHEMA = "sparkring-storage-local/v1"
SCHEMA = "sparkring-storage/v1"
OWNER = ".installer-owner.json"
RELEASING = ".sparkring-releasing"
_RELEASING = re.compile(r"\.(.+)\.sparkring-releasing")
# Directory names that installer containers give their caches in a cache root.
CACHE_NAME = re.compile(r"[a-z0-9][a-z0-9.-]*-(?:[0-9a-f]{12}|cuda[0-9]+(?:\.[0-9]+)*)-[0-9a-f]{12}")
CLUSTER_NAME = re.compile(r"[a-z][a-z0-9-]{0,39}")
# Seconds that one Spark spends walking its items; Node A waits longer for its answer.
BUDGET_SECONDS = 60
RELEASED = frozenset({"cache", "workspace", "releasing"})
TIB = 1024 ** 4
# Absolute paths inside an environment value, argument or JSON option.
_PATH = re.compile(r"/[^\s\"'=:,;{}\[\]()]*")

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


class _Expired(Exception):
    """The walk's time budget ran out."""


# Paths ------------------------------------------------------------------------

def _absolute(path):
    if (not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or path == "/"
            or "\0" in path or posixpath.normpath(path) != path):
        raise ValueError("Give the path as an absolute path, as sudo sparkring storage lists it: " + repr(path)[:200])
    return path


def _valid(path):
    try:
        return _absolute(path)
    except ValueError:
        return None


def _inside(path, parent):
    """Whether ``path`` is ``parent`` or lies below it."""
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _lstat(path):
    try:
        return os.lstat(path)
    except OSError:
        return None


def _is_directory(info):
    return info is not None and stat.S_ISDIR(info.st_mode)


def _entries(directory):
    """``(name, lstat)`` of the entries of ``directory`` by name; empty when it cannot be read."""
    found = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    found.append((entry.name, entry.stat(follow_symlinks=False)))
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(found, key=lambda pair: pair[0])


def _owner(directory):
    """The deployment ID that ``directory``'s ``.installer-owner.json`` names, or ``None``."""
    try:
        fd = os.open(directory, _DIRECTORY)
    except OSError:
        return None
    try:
        record = checkpoints._read_json_at(fd, OWNER)
    finally:
        os.close(fd)
    if isinstance(record, dict) and set(record) == {"deployment"} and isinstance(record["deployment"], str):
        return record["deployment"]
    return None


def _state_of(path):
    parent, name = posixpath.split(path)
    return posixpath.join(parent, "." + name + checkpoints.STATE_SUFFIX)


# Items on one Spark --------------------------------------------------------------

def _items(root, request):
    """This host's items without sizes, and the checkpoint directories and state directories among them."""
    srv = posixpath.join(root, "srv", "sparkring")
    cluster = request.get("cluster")
    cluster_dir = posixpath.join(srv, cluster) if isinstance(cluster, str) and CLUSTER_NAME.fullmatch(cluster) else None
    listing = checkpoints.list_local([path for path in request.get("models") or () if _valid(path)], root=root)
    items, taken = [], set()

    def add(path, kind, location=None, **values):
        location = location or path
        if location in taken:
            return
        taken.add(location)
        items.append({"path": path, "kind": kind, **({"location": location} if location != path else {}), **values})

    held = set()
    for entry in listing["directories"]:
        add(entry["path"], "checkpoint", repository=entry["repository"], revision=entry["revision"],
            state=entry["state"], bytes=entry["bytes"], frees_bytes=entry["frees_bytes"] + entry["staging_bytes"],
            complete=True)
        held.update((entry["path"], _state_of(entry["path"])))

    def remainders(directory):
        for name, info in _entries(directory):
            released = _RELEASING.fullmatch(name)
            if released and _is_directory(info):
                add(posixpath.join(directory, released[1]), "releasing", posixpath.join(directory, name))

    roots = [posixpath.join(cluster_dir, "cache")] if cluster_dir else []
    roots += [path for path in request.get("caches") or () if _valid(path)]
    roots = [path for path in dict.fromkeys(roots) if _is_directory(_lstat(path))]
    for cache_root in roots:
        remainders(cache_root)
        for name, info in _entries(cache_root):
            path = posixpath.join(cache_root, name)
            if stat.S_ISLNK(info.st_mode) or path in taken:
                continue
            if _is_directory(info) and CACHE_NAME.fullmatch(name):
                add(path, "cache", cache_root=cache_root)
            else:
                add(path, "other")

    workspaces = [posixpath.join(cluster_dir, name) for name, info in _entries(cluster_dir)
                  if _is_directory(info) and not _RELEASING.fullmatch(name)] if cluster_dir else []
    workspaces += [path for path in request.get("workspaces") or () if _valid(path)]
    for path in dict.fromkeys(workspaces):
        deployment = _owner(path) if _is_directory(_lstat(path)) else None
        if deployment is not None:
            models = posixpath.join(path, "models")
            add(path, "workspace", deployment=deployment,
                holds_models=_is_directory(_lstat(models)) and bool(_entries(models)))

    for directory in dict.fromkeys([srv, *([cluster_dir] if cluster_dir else []),
                                    *(posixpath.dirname(item["path"]) for item in items if item["kind"] == "workspace")]):
        remainders(directory)
    if cluster_dir and cluster_dir not in taken:
        for name, info in _entries(cluster_dir):
            path = posixpath.join(cluster_dir, name)
            if stat.S_ISLNK(info.st_mode) or path in taken or path in roots:
                continue
            add(path, "other", **({"remainder": True} if name == "checkpoints" and _is_directory(info) else {}))
    for name, info in _entries(srv):
        path = posixpath.join(srv, name)
        if not stat.S_ISLNK(info.st_mode) and path != cluster_dir and path not in taken:
            add(path, "other")
    return items, held


def _allocated(info):
    blocks = getattr(info, "st_blocks", None)
    return blocks * 512 if blocks is not None else info.st_size


def disk_use(path, *, deadline, exclude=frozenset()):
    """``{"bytes", "frees_bytes", "files", "complete", "mounts"}`` of ``path``; see the module's Sizes section.

    ``exclude`` holds paths whose subtrees are not walked. ``deadline`` is a
    ``time.monotonic()`` value.
    """
    result = {"bytes": 0, "frees_bytes": 0, "files": 0, "complete": True, "mounts": []}
    top = _lstat(path)
    if top is None:
        return result
    size = _allocated(top)
    if not stat.S_ISDIR(top.st_mode):
        result.update(bytes=size, frees_bytes=size if top.st_nlink <= 1 else 0, files=1)
        return result
    total = frees = size
    linked = {}
    pending = [path]
    try:
        while pending:
            if time.monotonic() >= deadline:
                raise _Expired
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for count, entry in enumerate(entries, 1):
                        if not count % 4096 and time.monotonic() >= deadline:
                            raise _Expired
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except OSError:
                            result["complete"] = False
                            continue
                        size = _allocated(info)
                        if stat.S_ISDIR(info.st_mode):
                            if entry.path in exclude:
                                continue
                            if info.st_dev != top.st_dev:
                                result["mounts"].append(entry.path)
                                continue
                            total += size
                            frees += size
                            pending.append(entry.path)
                            continue
                        result["files"] += 1
                        if info.st_nlink <= 1:
                            total += size
                            frees += size
                            continue
                        seen = linked.get(info.st_ino)
                        if seen is None:
                            linked[info.st_ino] = [size, info.st_nlink, 1]
                            total += size
                        else:
                            seen[2] += 1
            except OSError:
                result["complete"] = False
    except _Expired:
        result["complete"] = False
    frees += sum(size for size, links, seen in linked.values() if seen >= links)
    result.update(bytes=total, frees_bytes=frees)
    result["mounts"].sort()
    return result


def _measure(items, held, deadline, measured=None):
    """Add sizes to ``items``, walking only the paths in ``measured`` when it is given."""
    locations = {item.get("location", item["path"]) for item in items}
    exclude = frozenset(held | locations)
    checkpoint_paths = [item["path"] for item in items if item["kind"] == "checkpoint"]

    def one(item):
        location = item.get("location", item["path"])
        if item["kind"] == "workspace":
            item["checkpoints"] = [path for path in checkpoint_paths if _inside(path, location)]
        if item["kind"] == "checkpoint":
            return
        if measured is not None and item["path"] not in measured:
            item.update(bytes=None, frees_bytes=None, complete=None, mounts=[])
            return
        item.update(disk_use(location, deadline=deadline, exclude=exclude - {location}))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, items))
    # The cluster's checkpoints directory is listed only for what lies outside SparkRing's checkpoint directories.
    items[:] = [item for item in items if not item.pop("remainder", False) or item.get("files")]
    return items


def filesystems(paths):
    """The filesystems holding ``paths``, ``(role, path)`` pairs, one entry per device in first-seen order.

    Each entry has the ``mount_point`` and ``fstype`` from the mount table,
    ``size_bytes``, ``used_bytes``, ``free_bytes`` (the space ``df`` shows as
    available, which leaves out the blocks reserved for root) and ``paths``.
    """
    try:
        table = checkpoint_place.mounts()
    except OSError:
        table = []
    found = {}
    for role, path in paths:
        if not isinstance(path, str) or not path.startswith("/"):
            continue
        existing = path
        while existing != "/" and _lstat(existing) is None:
            existing = posixpath.dirname(existing)
        try:
            info, space = os.stat(existing), os.statvfs(existing)
        except (OSError, AttributeError):
            continue
        entry = found.get(info.st_dev)
        if entry is None:
            mount = checkpoint_place.mount_of(existing, table) if table else None
            entry = found[info.st_dev] = {
                "mount_point": mount["point"] if mount else None, "fstype": mount["type"] if mount else None,
                "size_bytes": space.f_blocks * space.f_frsize,
                "used_bytes": (space.f_blocks - space.f_bfree) * space.f_frsize,
                "free_bytes": space.f_bavail * space.f_frsize, "paths": []}
        entry["paths"].append({"role": role, "path": path})
    return list(found.values())


def _docker(arguments, run):
    done = run(["docker", "--context", "default", *arguments], capture_output=True, text=True, timeout=120)
    if done.returncode:
        lines = [line.strip() for line in (done.stderr or "").splitlines() if line.strip()]
        raise ValueError(lines[-1] if lines else f"docker exited with status {done.returncode}")
    return done.stdout


def _name(container):
    return str(container.get("Name") or container.get("Id", "")[:12]).lstrip("/")


def _docker_state(run):
    """``(docker_state(), the inspected containers)``; the containers are empty when Docker could not be read."""
    try:
        root = _docker(["info", "--format", "{{.DockerRootDir}}"], run).strip() or None
        identifiers = list(dict.fromkeys(_docker(["image", "ls", "--no-trunc", "--quiet"], run).split()))
        images = json.loads(_docker(["image", "inspect", *identifiers], run)) if identifiers else []
        containers = _docker(["ps", "--all", "--no-trunc", "--quiet"], run).split()
        containers = json.loads(_docker(["container", "inspect", *containers], run)) if containers else []
        if not isinstance(images, list) or not isinstance(containers, list):
            raise ValueError("unexpected answer from docker inspect")
    except FileNotFoundError:
        return None, []
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"root": None, "images": [], "error": str(error)[:300]}, []
    containers = [container for container in containers if isinstance(container, dict)]
    users = {}
    for container in containers:
        users.setdefault(container.get("Image"), []).append(
            {"name": _name(container), "running": bool((container.get("State") or {}).get("Running"))})
    listed = []
    for image in images:
        if isinstance(image, dict) and isinstance(image.get("Id"), str):
            listed.append({"id": image["Id"],
                           "tags": [tag for tag in image.get("RepoTags") or () if isinstance(tag, str)],
                           "digests": [value for value in image.get("RepoDigests") or () if isinstance(value, str)],
                           "bytes": image["Size"] if type(image.get("Size")) is int else 0,
                           "containers": sorted(users.get(image["Id"], []), key=lambda user: user["name"])})
    state = {"root": root, "images": sorted(listed, key=lambda image: (-image["bytes"], image["id"])), "error": None}
    return state, containers


def docker_state(*, run=None):
    """This host's Docker data root and images with the containers that use them; ``None`` without Docker.

    Returns ``{"root", "images": [{"id", "tags", "digests", "bytes",
    "containers": [{"name", "running"}]}], "error"}``; ``error`` names why
    Docker could not be read.
    """
    return _docker_state(run or subprocess.run)[0]


def _mentions(word, path, destination):
    """Whether ``word`` names ``path``, a path below it, or a directory from ``destination`` down to ``path``.

    A container whose cache variable names the mount destination itself may
    create any directory below it, so that counts as naming ``path`` as well.
    """
    for token in _PATH.findall(str(word)):
        token = posixpath.normpath(token)
        if _inside(token, path) or (_inside(path, token) and _inside(token, destination)):
            return True
    return False


def _users(containers, path):
    """Names of the running ``containers`` (``docker container inspect`` documents) that reach ``path``.

    A container reaches ``path`` when it mounts ``path`` or a directory below
    it, or mounts a directory above it and names the corresponding container
    path in its environment, command or working directory (``_mentions``).
    """
    names = set()
    for container in containers:
        if not isinstance(container, dict) or not (container.get("State") or {}).get("Running"):
            continue
        config = container.get("Config") or {}
        words = [*(config.get("Env") or ()), *(config.get("Cmd") or ()), *(config.get("Entrypoint") or ()),
                 *(container.get("Args") or ()), container.get("Path") or "", config.get("WorkingDir") or ""]
        for mount in container.get("Mounts") or ():
            source = mount.get("Source") if isinstance(mount, dict) else None
            destination = mount.get("Destination") if isinstance(mount, dict) else None
            if not isinstance(source, str) or not source.startswith("/"):
                continue
            source = posixpath.normpath(source)
            if _inside(source, path):
                names.add(_name(container))
                break
            if _inside(path, source) and isinstance(destination, str) and destination.startswith("/"):
                destination = posixpath.normpath(destination)
                inner = posixpath.normpath(destination + path[len(source.rstrip("/")):])
                if any(_mentions(word, inner, destination) for word in words):
                    names.add(_name(container))
                    break
    return sorted(names)


def list_local(request=None, *, root="/", run=None):
    """This host's storage report (``sparkring-storage-local/v1``).

    ``request`` holds ``cluster`` (the cluster name, whose directory in
    ``/srv/sparkring`` SparkRing's installer creates), ``workspaces``,
    ``caches`` and ``models`` (paths that Node A's retained deployments name on
    this host), ``budget_seconds`` and ``measure``, a list of item paths that
    limits the walks to those items. Every item except ``other`` and
    ``releasing`` also carries ``containers``, the running containers that
    reach it (``_users``). ``root`` relocates the search of ``/srv/sparkring``;
    ``run`` replaces ``subprocess.run`` for Docker.
    """
    request = request if isinstance(request, dict) else {}
    started = time.monotonic()
    budget = request.get("budget_seconds", BUDGET_SECONDS)
    if type(budget) not in (int, float) or budget <= 0:
        budget = BUDGET_SECONDS
    measured = request.get("measure")
    measured = {path for path in measured if isinstance(path, str)} if isinstance(measured, list) else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        docker = pool.submit(_docker_state, run or subprocess.run)
        items, held = _items(root, request)
        _measure(items, held, started + budget, measured)
        docker, containers = docker.result()
    for item in items:
        if item["kind"] not in ("other", "releasing"):
            item["containers"] = _users(containers, item["path"])
    srv = posixpath.join(root, "srv", "sparkring")
    return {"schema": LOCAL_SCHEMA, "hostname": socket.gethostname(),
            "package_revision": checkpoints.package_revision(),
            "filesystems": filesystems([("sparkring", srv), ("docker", (docker or {}).get("root")), ("root", "/")]),
            "items": items, "docker": docker,
            "measurement": {"budget_seconds": budget, "seconds": round(time.monotonic() - started, 1),
                            "complete": all(item.get("complete") is not False for item in items)}}


# Releasing on one Spark ------------------------------------------------------------

def containers_using(path, *, run=None):
    """Names of the running containers on this host that reach ``path`` (``_users``).

    A host without the ``docker`` command runs no containers; any other
    failure to list them refuses the release.
    """
    run = run or subprocess.run
    try:
        identifiers = _docker(["ps", "--quiet", "--no-trunc"], run).split()
        containers = json.loads(_docker(["container", "inspect", *identifiers], run)) if identifiers else []
    except FileNotFoundError:
        return []
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError(f"SparkRing could not list this host's running containers ({error}); nothing was "
                         "released") from None
    if not isinstance(containers, list):
        raise ValueError("SparkRing could not read this host's running containers; nothing was released")
    return _users(containers, path)


def _remove_tree(parent_fd, name, device):
    """Remove directory ``name`` of ``parent_fd`` and everything below it; returns the bytes freed.

    Entries are opened relative to their directory without following
    symlinks, and a directory on another device than ``device`` stops the
    removal, so it never leaves the tree or its filesystem. An inode is freed
    when its last link goes.
    """
    fd = os.open(name, _DIRECTORY, dir_fd=parent_fd)
    try:
        info = os.fstat(fd)
        if info.st_dev != device:
            raise ValueError(f"{name} is on another filesystem; SparkRing stopped the release there")
        freed = 0
        for entry in sorted(os.listdir(fd)):
            current = os.lstat(entry, dir_fd=fd)
            if stat.S_ISDIR(current.st_mode):
                freed += _remove_tree(fd, entry, device)
                continue
            os.unlink(entry, dir_fd=fd)
            if current.st_nlink <= 1:
                freed += _allocated(current)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)
    return freed + _allocated(info)


def _remove(location):
    parent, name = posixpath.split(location)
    parent_fd = checkpoints._open_directory(parent)
    try:
        info = os.lstat(name, dir_fd=parent_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{location} is not a directory; SparkRing does not remove it")
        freed = _remove_tree(parent_fd, name, info.st_dev)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return freed


def _rename(path):
    """Rename ``path`` to its ``releasing`` name beside it, so that nothing finds it while it is removed."""
    parent, name = posixpath.split(path)
    hidden = "." + name + RELEASING
    parent_fd = checkpoints._open_directory(parent)
    try:
        os.rename(name, hidden, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return posixpath.join(parent, hidden)


def _unmounted(path, table):
    """Refuse ``path`` when a mount point, a bind mount of the same filesystem included, is at or below it."""
    mounted = sorted({entry["point"] for entry in table if _inside(posixpath.normpath(entry["point"]), path)})
    if mounted:
        raise ValueError(f"{path} holds mount points ({', '.join(mounted[:3])}); SparkRing does not release it. "
                         "Nothing was released.")


def _check(item, items, request, run, table):
    """Refuse, before anything changes, a release of ``item`` that this Spark does not allow."""
    path = item["path"]
    if item["kind"] == "checkpoint":
        raise ValueError(f"{path} is a SparkRing checkpoint directory; sudo sparkring checkpoints --release {path} "
                         "releases it. Nothing was released.")
    if item["kind"] not in RELEASED:
        raise ValueError(f"{path} was not created by SparkRing's installer; SparkRing does not remove it. Nothing was "
                         "released.")
    used = sorted({value for value in request.get("in_use") or () if isinstance(value, str) and _inside(value, path)}
                  | {value for value in request.get("opaque") or () if isinstance(value, str) and _inside(path, value)})
    if used:
        raise ValueError(f"The installed deployment uses {path} ({', '.join(used[:3])}); SparkRing does not release "
                         "it. Nothing was released.")
    inner = [other["path"] for other in items if other["kind"] == "checkpoint" and _inside(other["path"], path)]
    if inner or item.get("holds_models"):
        shown = inner[0] if inner else posixpath.join(path, "models")
        raise ValueError(f"{path} holds model files ({shown}); SparkRing releases checkpoint directories only with sudo "
                         "sparkring checkpoints --release PATH. Nothing was released.")
    _unmounted(path, table)
    users = containers_using(path, run=run)
    if users:
        raise ValueError(f"Running containers on this Spark use {path} ({', '.join(users)}); SparkRing does not "
                         "release it. Stop them first. Nothing was released.")


def release_local(path, request=None, *, root="/", run=None, table=None):
    """Remove cache directory or deployment workspace ``path`` from this host.

    ``request`` is the listing request (``list_local``) plus ``in_use``, the
    paths the installed deployments name on this host, and ``opaque``, the cache
    roots whose entries they may all use. The item is found as the listing
    finds it. Before anything changes, the release refuses a path that is no
    ``cache``, ``workspace`` or ``releasing`` item, that is or holds a path in
    ``in_use`` or lies in an ``opaque`` root, that holds a SparkRing checkpoint
    directory or a non-empty ``models`` directory, that holds a mount point
    (``table`` replaces the mount table), or that a running container on this
    host uses (``containers_using``). It then renames ``path`` to its
    ``releasing`` name and removes that directory; a remainder of an earlier
    interrupted release of ``path`` is removed as well.

    Returns ``{"path", "state": "released" | "absent", "kind", "freed_bytes"}``.
    """
    path = _absolute(path)
    request = request if isinstance(request, dict) else {}
    items, _ = _items(root, request)
    matches = [item for item in items if item["path"] == path]
    current = next((item for item in matches if item["kind"] != "releasing"), None)
    remainders = [item for item in matches if item["kind"] == "releasing"]
    if current is None and not remainders:
        if _lstat(path) is None:
            return {"path": path, "state": "absent", "kind": None, "freed_bytes": 0}
        raise ValueError(f"{path} is not a SparkRing cache directory or deployment workspace on this Spark; "
                         "SparkRing does not remove it. Nothing was released.")
    table = checkpoint_place.mounts() if table is None else table
    if current is not None:
        _check(current, items, request, run, table)
    for item in remainders:
        _unmounted(item["location"], table)
    freed = sum(_remove(item["location"]) for item in remainders)
    if current is not None:
        freed += _remove(_rename(path))
    return {"path": path, "state": "released", "kind": (current or remainders[0])["kind"], "freed_bytes": freed}


def node(release=None, stream=None):
    """``sudo sparkring node storage [--release PATH]``: this host's part, request read as JSON from ``stream``.

    An empty or interactive ``stream`` is an empty request.
    """
    stream = sys.stdin if stream is None else stream
    text = "" if stream is None or stream.isatty() else stream.read()
    request = json.loads(text) if text.strip() else {}
    if not isinstance(request, dict):
        raise ValueError("The storage request must be a JSON object")
    if release is not None:
        return release_local(release, request)
    return list_local(request)


# Node A ------------------------------------------------------------------------------

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spec_paths(spec):
    """Host paths that a container specification (``ContainerSpec.document()``) names.

    These are the bind sources, every path that an environment value, the
    command or the entrypoint names below a bind target, translated to the
    host, and the host paths in security options.
    """
    binds = []
    for mount in spec.get("mounts") or ():
        source = mount.get("source") if isinstance(mount, dict) else None
        target = mount.get("target") if isinstance(mount, dict) else None
        if (isinstance(source, str) and source.startswith("/") and isinstance(target, str)
                and target.startswith("/") and posixpath.normpath(target) != "/"):
            binds.append((posixpath.normpath(source), posixpath.normpath(target)))
    found = {source for source, _ in binds}
    environment = spec.get("environment")
    words = [*(environment.values() if isinstance(environment, dict) else ()), *(spec.get("command") or ()),
             *(spec.get("entrypoint") or ())]
    for word in words:
        for token in _PATH.findall(str(word)):
            token = posixpath.normpath(token)
            for source, target in binds:
                if _inside(token, target):
                    found.add(posixpath.normpath(source + token[len(target):]))
    for option in spec.get("security_opt") or ():
        found.update(posixpath.normpath(token) for token in _PATH.findall(str(option)))
    return found


def deployments(state_root):
    """Retained deployments of the controller with the host paths they use.

    Each is ``{"name", "directory", "id", "workspace", "image_id", "rows",
    "references"}``; ``rows`` holds each rank's ``host``, ``model``, ``cache``
    and whether the model is served in place (``reuse``), and ``references``
    maps a host to ``{"paths", "opaque"}``: the paths its row and container
    specification name, and its cache root when the specification cannot be
    read. A deployment whose lock cannot be read is skipped.
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
            rows = [{"rank": row.get("rank", number), "host": row["host"], "model": row["model"],
                     "cache": row.get("cache"), "repository": row.get("repository"),
                     "deployment_root": row.get("deployment_root"), "reuse": bool(row.get("reuse_verified_model"))}
                    for number, row in enumerate(site["ranks"])]
            record = {"name": directory.name, "directory": os.path.realpath(directory), "id": lock["id"],
                      "workspace": site["workspace"], "image_id": (lock.get("selection") or {}).get("image_id"),
                      "rows": rows}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        references = {}
        for row in rows:
            entry = references.setdefault(row["host"], {"paths": set(), "opaque": set()})
            entry["paths"].update(value for value in (record["workspace"], row["model"], row["repository"],
                                                       row["deployment_root"]) if _valid(value))
            try:
                spec = _read(directory / f"rank{row['rank']}" / "container.json")
            except (OSError, ValueError):
                spec = None
            if isinstance(spec, dict):
                entry["paths"].update(spec_paths(spec))
            elif _valid(row["cache"]):
                entry["opaque"].add(row["cache"])
        record["references"] = {host: {key: sorted(values) for key, values in entry.items()}
                                for host, entry in references.items()}
        found.append(record)
    return found


def profile_references():
    """What the installer profiles of this package reference, for the ``profile`` class.

    Returns ``{"checkpoints": {"<repository>@<revision>": [profile, ...]},
    "caches": {name: [...]}, "images": {image_id: [...]}, "locks": {image_id:
    [lock name, ...]}}``. A profile references the checkpoints it lists, the
    image of the lock ``installer_image.for_profile`` selects and the caches
    those produce. ``locks`` names every installer image lock in
    ``runtime/releases``, including development locks that only ``--image-lock``
    selects.
    """
    from runtime.common import installer_image, profiles, qwen_flash_next
    tables = {"checkpoints": {}, "caches": {}, "images": {}, "locks": {}}

    def note(table, key, value):
        tables[table].setdefault(key, set()).add(value)

    for path in sorted((profiles.ROOT / "runtime" / "releases").glob("*/installer-image.json")):
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
            note("locks", lock["image_id"], lock["name"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for profile in installer_image.SUPPORTED:
        try:
            image = installer_image.for_profile(profile)["image_id"]
            metadata, _ = profiles.load(profile)
            configuration = qwen_flash_next.read(profiles.ROOT / metadata["configuration"]["path"])
            _, names = qwen_flash_next.checkpoint_names(configuration)
            selections = [qwen_flash_next.checkpoint_settings(configuration, name) for name in names] or [configuration]
            toolchain = qwen_flash_next.image_policy(configuration)["kind"] == "toolchain"
        except (OSError, ValueError, KeyError, TypeError):
            continue
        note("images", image, profile)
        for selection in selections:
            note("checkpoints", selection["model"]["repository"] + "@" + selection["model"]["revision"], profile)
            for name in cache_names(selection, image, toolchain=toolchain):
                note("caches", name, profile)
    return {name: {key: sorted(values) for key, values in table.items()} for name, table in tables.items()}


def cache_names(selection, image, *, toolchain):
    """Directory names that an installer container of ``selection`` on ``image`` uses in its cache root.

    ``selection`` is a profile configuration with its checkpoint applied. The
    names are those ``qwen_flash_next.container_spec`` and
    ``installer_image.adapt`` give: compile and tuning caches in
    ``<family>-<image-12>-<revision-12>`` and, on the shared toolchain images,
    B12X kernels in ``<family>-cuda<version>-<revision-12>``, where the family
    is the profile's ``cache_namespace``. ``runtime/host/test_storage.py``
    compares them with the rendered installer containers of every profile.
    """
    from runtime.common import installer_image
    family, revision = selection.get("cache_namespace", "qwen-flash-next"), selection["model"]["revision"][:12]
    names = [f"{family}-{image[7:19]}-{revision}"]
    if toolchain:
        names.append(f"{family}-cuda{installer_image.CUDA_VERSION}-{revision}")
    return names


def _on(deployment, host):
    """Paths and opaque cache roots that ``deployment`` uses on ``host`` (every host for a local listing)."""
    entries = [deployment["references"][host]] if host in deployment["references"] else [] if host is not None \
        else list(deployment["references"].values())
    return ([path for entry in entries for path in entry["paths"]],
            [path for entry in entries for path in entry["opaque"]])


def _uses(deployment, host, path):
    paths, opaque = _on(deployment, host)
    return any(_inside(value, path) for value in paths) or any(_inside(path, value) for value in opaque)


def _release_command(item):
    if item["class"] != "unreferenced" or item.get("containers"):
        return None
    if item["kind"] == "checkpoint":
        # The checkpoints command refuses a directory that replaced SparkRing's after it was created.
        return None if item.get("state") == "replaced" else "sudo sparkring checkpoints --release " + item["path"]
    if item["kind"] in RELEASED and not item.get("checkpoints") and not item.get("holds_models"):
        return "sudo sparkring storage --release " + item["path"]
    return None


def classify(nodes, retained, role, references):
    """Add ``class``, ``deployments``, ``profiles`` and ``release`` to every item and image, and each Spark's proposal."""
    for node_entry in nodes:
        host = node_entry.get("host")
        installed = {deployment["image_id"] for deployment in retained if deployment["directory"] in role
                     and (host is None or any(row["host"] == host for row in deployment["rows"]))}
        for item in node_entry.get("items", ()):
            users = [] if item["kind"] == "releasing" else [deployment for deployment in retained
                                                            if _uses(deployment, host, item["path"])]
            item["deployments"] = [{"name": deployment["name"], **({"role": role[deployment["directory"]]}
                                                                    if deployment["directory"] in role else {})}
                                   for deployment in users]
            item["profiles"] = []
            if item["kind"] == "other":
                item["class"] = "unmanaged"
            elif any(deployment["directory"] in role for deployment in users):
                item["class"] = "installed"
            else:
                if item["kind"] == "cache":
                    item["profiles"] = references["caches"].get(posixpath.basename(item["path"]), [])
                elif item["kind"] == "checkpoint":
                    item["profiles"] = references["checkpoints"].get(f"{item['repository']}@{item['revision']}", [])
                item["class"] = "profile" if item["profiles"] else "unreferenced"
            item["release"] = _release_command(item)
        node_entry["items"] = sorted(node_entry.get("items", ()), key=lambda item: (-(item.get("bytes") or 0),
                                                                                   item["path"]))
        for image in (node_entry.get("docker") or {}).get("images", ()):
            image["profiles"] = references["images"].get(image["id"], [])
            image["locks"] = references["locks"].get(image["id"], [])
            image["class"] = ("installed" if image["id"] in installed else "profile" if image["profiles"]
                              else "unreferenced")
        releasable = [item for item in node_entry.get("items", ()) if item.get("release")]
        node_entry["proposed"] = {"frees_bytes": sum(item.get("frees_bytes") or 0 for item in releasable),
                                  "commands": [item["release"] for item in releasable]}
    return nodes


def _request(host, cluster, retained, **values):
    """The listing request for ``host`` (every host for a local listing)."""
    rows = [(deployment, row) for deployment in retained for row in deployment["rows"]
            if host is None or row["host"] == host]
    return {"cluster": cluster, "workspaces": sorted({deployment["workspace"] for deployment, _ in rows}),
            "caches": sorted({row["cache"] for _, row in rows if isinstance(row["cache"], str)}),
            "models": sorted({row["model"] for _, row in rows if not row["reuse"]}),
            "budget_seconds": BUDGET_SECONDS, **values}


def _survey(hosts, cluster, retained, invoke, **values):
    """Every Spark's listing in rank order; an unreachable Spark carries ``error``."""
    def one(rank):
        host = hosts[rank]
        try:
            listing = json.loads(invoke(host, ["sudo", "-n", "/usr/bin/sparkring", "node", "storage"],
                                        data=json.dumps(_request(host, cluster, retained, **values)),
                                        timeout=BUDGET_SECONDS + 120))
            if not isinstance(listing, dict) or listing.get("schema") != LOCAL_SCHEMA:
                raise ValueError("unexpected answer from sparkring node storage")
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            text = str(error).strip()
            return {"rank": rank, "host": host, "error": text.splitlines()[-1][:300] if text else type(error).__name__}
        return {"rank": rank, "host": host, **listing}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        return list(pool.map(one, range(len(hosts))))


def list_cluster(state_root, invoke):
    """``sparkring storage``: every Spark's report with each item's and image's class."""
    retained, role = deployments(state_root), checkpoints.roles(state_root)
    cluster = checkpoints._cluster(state_root)
    if cluster is None:
        nodes = [{"rank": None, "host": None, **list_local(_request(None, None, retained))}]
    else:
        nodes = _survey(checkpoints._hosts(cluster), cluster["name"], retained, invoke)
    return {"schema": SCHEMA, "state": "listed", "nodes": classify(nodes, retained, role, profile_references())}


def _refusal(item, node_entry):
    """Why Node A does not release ``item`` on this Spark, or ``None``."""
    path, where = item["path"], f"Node {node_entry['rank']} {node_entry['hostname']}"
    if item["kind"] == "checkpoint":
        return (f"{path} is a SparkRing checkpoint directory; sudo sparkring checkpoints --release {path} "
                "releases it")
    if item["kind"] == "other":
        return f"{where}: {path} was not created by SparkRing's installer; SparkRing does not remove it"
    if item["class"] == "installed":
        users = ", ".join(entry["name"] + (f" ({entry['role']})" if entry.get("role") else "")
                          for entry in item["deployments"] if entry.get("role"))
        return f"{where}: the installed deployment {users} uses {path}; SparkRing does not release it"
    if item["class"] == "profile":
        return (f"{path} is referenced by the installer profiles {', '.join(item['profiles'])} of the installed "
                "package, whose next installation reuses it; SparkRing keeps it")
    if item.get("checkpoints") or item.get("holds_models"):
        shown = item["checkpoints"][0] if item.get("checkpoints") else posixpath.join(path, "models")
        return (f"{where}: {path} holds model files ({shown}); release checkpoint directories first with sudo "
                "sparkring checkpoints --release PATH")
    if item.get("containers"):
        return (f"{where}: running containers use {path} ({', '.join(item['containers'])}); stop them first, "
                "then repeat the release")
    return None


def release_cluster(path, state_root, invoke, *, yes, interactive, write=print):
    """``sparkring storage --release PATH`` on Node A; see the module description for its rules."""
    from runtime.common import process_lock
    from runtime.host import controller
    path = _absolute(path)
    with process_lock.hold(Path(state_root) / "install.lock"):
        cluster = checkpoints._cluster(state_root)
        if cluster is None:
            raise ValueError("Release SparkRing data from Node A of a configured cluster, which records the "
                             "deployments that use it. Nothing was released.")
        retained, role = deployments(state_root), checkpoints.roles(state_root)
        nodes = _survey(checkpoints._hosts(cluster), cluster["name"], retained, invoke, measure=[path])
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
        classify(nodes, retained, role, profile_references())
        holders = [(item, entry) for item in nodes for entry in item["items"] if entry["path"] == path]
        if not holders:
            raise ValueError(f"{path} is not a SparkRing cache directory or deployment workspace on any Spark. "
                             "sudo sparkring storage lists them.")
        for item, entry in holders:
            reason = _refusal(entry, item)
            if reason:
                raise ValueError(reason + ". Nothing was released.")
        users = sorted({user["name"] for _, entry in holders for user in entry["deployments"]})
        write(f"Release {path}:")
        for item, entry in holders:
            write(f"    Node {item['rank']} {item['hostname']}: removes the {_KIND[entry['kind']]}; frees "
                  + _measured(entry["frees_bytes"], entry["complete"]))
        if users:
            write("Retained deployments that use it need sudo sparkring install again: " + ", ".join(users))
        if not yes:
            if not interactive:
                raise NeedsInput("Review the release above, then repeat with --yes to apply it. Nothing was released.",
                                 field="approval", details={"path": path, "nodes": [item["rank"] for item, _ in holders]})
            controller.confirm("Release this directory?")

        def one(pair):
            item, _ = pair
            protected = [deployment for deployment in retained if deployment["directory"] in role]
            in_use = sorted({value for deployment in protected for value in _on(deployment, item["host"])[0]})
            opaque = sorted({value for deployment in protected for value in _on(deployment, item["host"])[1]})
            request = _request(item["host"], cluster["name"], retained, in_use=in_use, opaque=opaque)
            try:
                answer = json.loads(invoke(item["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "storage",
                                                          "--release", path],
                                           data=json.dumps(request), timeout=3600))
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                return {"rank": item["rank"], "host": item["host"], "hostname": item["hostname"], "error": str(error).strip()}
            return {"rank": item["rank"], "host": item["host"], "hostname": item["hostname"], **answer}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(holders)) as pool:
            results = list(pool.map(one, holders))
    for item in results:
        if "error" not in item:
            write(f"Node {item['rank']} {item['hostname']}: released; freed {_size(item['freed_bytes'])}")
    failed = [item for item in results if "error" in item]
    if failed:
        raise RuntimeError("The release did not finish. " + " ".join(
            f"Node {item['rank']} {item['hostname']}: {item['error']}" for item in failed)
            + " Repeating the release continues it.")
    return {"schema": SCHEMA, "state": "released", "path": path, "nodes": results, "deployments": users}


# Text ----------------------------------------------------------------------------------

_KIND = {"checkpoint": "checkpoint directory", "cache": "cache directory", "workspace": "deployment workspace",
         "releasing": "remainder of an interrupted release", "other": "directory"}
LEGEND = ("Classes: installed = used by the installed deployment (active, rollback target or unfinished model "
          "switch); profile = referenced by an installer profile of the installed package, kept for its next "
          "installation; unreferenced = neither, proposed for release unless a running container uses it or it "
          "holds model files; unmanaged = not created by SparkRing's installer, never removed. A size ending in + "
          "was still being measured at the time limit.")


def _size(value):
    return f"{value / TIB:.1f} TiB" if value >= TIB else checkpoints._size(value)


def _measured(value, complete):
    if value is None:
        return "-"
    return _size(value) + ("+" if complete is False else "")


def _filesystem_line(entry):
    used, free = entry["used_bytes"], entry["free_bytes"]
    percent = math.ceil(100 * used / (used + free)) if used + free else 0
    held = [("Docker's data root " if path["role"] == "docker" else "") + path["path"]
            for path in entry["paths"] if path["role"] != "root"]
    return (f"{entry['mount_point'] or 'filesystem'} ({entry['fstype'] or 'unknown type'}): "
            f"{_size(entry['size_bytes'])}, {_size(free)} free, {percent}% used"
            + ("; holds " + ", ".join(held) if held else ""))


def _details(item):
    lines = []
    if item["kind"] == "checkpoint":
        lines.append(f"{item['repository']} at {item['revision'][:12]}"
                     + ("" if item["state"] == "ok" else f" ({item['state']})"))
    if item["kind"] == "releasing":
        lines.append("remainder of an interrupted release; releasing the path again removes it")
    if item.get("deployments"):
        lines.append("used by " + ", ".join(f"{entry['name']} ({entry.get('role') or 'retained'})"
                                            for entry in item["deployments"]))
    if item.get("containers"):
        lines.append("used by running containers " + ", ".join(item["containers"]))
    if item.get("profiles"):
        lines.append("referenced by installer profiles " + ", ".join(item["profiles"]))
    for path in item.get("checkpoints") or ():
        lines.append(f"holds checkpoint directory {path}; release it first with sudo sparkring checkpoints "
                     f"--release {path}")
    if item.get("holds_models") and not item.get("checkpoints"):
        lines.append("holds files in models/; SparkRing does not release it")
    if item.get("mounts"):
        lines.append("other filesystems below it are not counted: " + ", ".join(item["mounts"][:3]))
    if item["class"] == "unmanaged" and item.get("bytes") and item.get("frees_bytes") is not None \
            and item["frees_bytes"] < item["bytes"]:
        lines.append(f"deleting it would free {_size(item['frees_bytes'])}; the rest is hard-linked from elsewhere")
    return lines


def _image_name(image):
    names = image["tags"] or [digest.split("@")[0] + "@..." for digest in image["digests"]] or ["<untagged>"]
    return image["id"][7:19] + " " + names[0]


def describe(result):
    """Printed lines of a ``list_cluster`` result."""
    lines = []
    for node_entry in result["nodes"]:
        name = checkpoints._node_name(node_entry)
        if "error" in node_entry:
            lines.append(f"{name}: not reachable: {node_entry['error']}")
            continue
        lines.append(name)
        for entry in node_entry.get("filesystems", ()):
            lines.append("    " + _filesystem_line(entry))
        docker = node_entry.get("docker")
        images = (docker.get("images") or []) if docker else []
        rows = [(item, _measured(item.get("bytes"), item.get("complete"))) for item in node_entry["items"]]
        width = max([len(size) for _, size in rows] + [len(_size(image["bytes"])) for image in images] + [4])
        lines.append(f"    {'SIZE':>{width}}  {'CLASS':<12}  {'KIND':<10}  PATH")
        for item, size in rows:
            lines.append(f"    {size:>{width}}  {item['class']:<12}  {item['kind']:<10}  {item['path']}")
            lines.extend(" " * (width + 32) + detail for detail in _details(item))
        if docker is None:
            lines.append("    Docker: not installed")
        elif docker.get("error"):
            lines.append(f"    Docker: could not be read: {docker['error']}")
        else:
            lines.append(f"    Docker images in {docker['root']}:")
            for image in images:
                notes = []
                if image["locks"]:
                    notes.append("installer image lock " + ", ".join(image["locks"]))
                running = [user["name"] for user in image["containers"] if user["running"]]
                stopped = [user["name"] for user in image["containers"] if not user["running"]]
                if running:
                    notes.append("running in " + ", ".join(running))
                if stopped:
                    notes.append("stopped container " + ", ".join(stopped))
                lines.append(f"    {_size(image['bytes']):>{width}}  {image['class']:<12}  {'image':<10}  "
                             f"{_image_name(image)}" + (f" ({'; '.join(notes)})" if notes else ""))
        proposed = node_entry.get("proposed") or {}
        if proposed.get("commands"):
            lines.append(f"    Proposed releases on this Spark free {_size(proposed['frees_bytes'])}:")
            lines.extend("        " + command for command in proposed["commands"])
        else:
            lines.append("    No release is proposed on this Spark.")
    lines.append(LEGEND)
    lines.append("sudo sparkring storage --release PATH removes one unreferenced cache directory or deployment "
                 "workspace from every Spark that holds it; checkpoint directories are released with sudo sparkring "
                 "checkpoints --release PATH. SparkRing does not remove Docker images: docker image rm ID on that "
                 "Spark removes one that no container uses.")
    return lines


def _administrator():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def main(argv=None, *, state_root=None, invoke=None):
    parser = argparse.ArgumentParser(
        prog="sparkring storage",
        description="Report SparkRing's disk use on every Spark with the data no deployment uses, or release one "
                    "unreferenced cache directory or deployment workspace.")
    parser.add_argument("--release", metavar="PATH",
                        help="remove this unreferenced cache directory or deployment workspace from every Spark that "
                             "holds it")
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
            raise ValueError("Run sudo sparkring storage")
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
