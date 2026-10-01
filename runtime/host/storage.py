"""Report SparkRing's disk use on every Spark and release data that no deployment uses.

These commands use this module:

- ``sudo sparkring storage`` on Node A (``main``) reports, for every Spark of
  the cluster, the filesystems that hold ``/srv/sparkring``, Docker's data root
  and ``/`` (size, used and free bytes), every item in ``/srv/sparkring`` and
  every Docker image with its size and class, and the releases it proposes.
  On a configured cluster it also lists the deployments that automatic release
  keeps, why, and what it would release (``runtime/host/retention.py``).
- ``sudo sparkring storage --retain-deployments N|off`` saves how many recent
  deployments of each profile automatic release keeps, or turns it off.
- ``sudo sparkring storage --release PATH`` removes one cache directory or
  deployment workspace of class ``unreferenced`` from every Spark that holds
  it, after the operator approves (``--yes`` in scripts). Node A lists and
  classifies every Spark again while it holds the installation lock, and
  refuses while a Spark cannot be listed or the Sparks run different package
  revisions. Each Spark then checks again that no installed deployment, no
  SparkRing mesh installed on that Spark and no running container uses the
  path and that it holds no checkpoint directory, model files or mount point
  (``release_local``). It never removes a
  checkpoint directory, which only ``sudo sparkring checkpoints --release
  PATH`` releases, a Docker image, or anything SparkRing's installer did not
  create.

Each Spark runs its own part as ``sudo sparkring node storage [--release
PATH | --release-batch]`` (``node``), reading a JSON request on stdin:
``list_local``, ``release_local`` and ``release_batch_local``. The batch is
automatic release's share of one Spark: the stopped model containers of the
deployments it releases (``remove_container``), then each workspace and cache
directory through ``release_local``. The listing names every installer
deployment's model container with its state and writable-layer size
(``model_containers``).

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
  cache root. A cache directory or workspace is also ``installed`` while a
  SparkRing mesh installed on that Spark uses it (``installed_meshes``): when
  a path that the mesh's ``site.json`` names is the item or lies inside it, or
  when that site cannot be read. On a four-Spark ring, the deployment that
  creates the mesh places the mesh's host marker binary and bundle root in its
  own workspace, and later deployments reuse that mesh, so the workspace stays
  in use after its deployment is neither active nor the rollback target. The
  item's ``meshes`` name each such mesh by its unit and site file.
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

from runtime.common import managed_deployment
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
# Items that an installed mesh can keep in use (``installed_meshes``).
MESH_KINDS = frozenset({"cache", "workspace"})
# Parent of every named managed mesh layout's configuration directory (``managed_deployment.layout(name)``).
MESH_NAMED_ROOT = "/etc/sparkring/deployments"
# Largest mesh site or service file that ``installed_meshes`` reads; a larger one counts as unreadable.
MESH_FILE_LIMIT = 1024 * 1024
TIB = 1024 ** 4
# Absolute paths inside an environment value, argument or JSON option.
_PATH = re.compile(r"/[^\s\"'=:,;{}\[\]()]*")
# Docker's full container IDs, and the lock IDs (SHA-256) that label a Compose deployment's model containers.
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
DEPLOYMENT_ID = re.compile(r"[0-9a-f]{64}")
# Labels of an installer deployment's model containers (installer.container_labels).
DEPLOYMENT_LABEL = "io.sparkring.deployment"
RANK_LABEL = "io.sparkring.rank"
# Names of the model containers that automatic release may remove (installer.container_name).
MODEL_CONTAINER = re.compile(r"sr-[a-z][a-z0-9-]{0,39}-r[0-9]")
# Docker states in which a container runs no process and ``docker container rm`` removes it without --force.
STOPPED_STATES = frozenset({"created", "exited"})
# Seconds that the listing waits for the writable-layer sizes of stopped model containers, during its walk.
SIZE_SECONDS = 60

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


def model_containers(containers, *, run, sizes):
    """The model containers of installer deployments among ``containers`` (``docker container inspect`` documents).

    A model container carries the ``io.sparkring.deployment`` label with its
    deployment's lock ID. Returns ``[{"id", "name", "deployment", "rank",
    "running", "status", "bytes"}]`` sorted by name; ``bytes`` is the size of
    the container's writable layer (Docker's ``SizeRw``), which removing it
    frees, measured for stopped containers when ``sizes`` is true and ``None``
    otherwise or when Docker does not report it.
    """
    found = []
    for item in containers:
        labels = (item.get("Config") or {}).get("Labels") or {}
        deployment = labels.get(DEPLOYMENT_LABEL)
        if not isinstance(deployment, str) or not isinstance(item.get("Id"), str):
            continue
        state = item.get("State") or {}
        found.append({"id": item["Id"], "name": _name(item), "deployment": deployment,
                      "rank": labels.get(RANK_LABEL), "running": bool(state.get("Running")),
                      "status": state.get("Status"), "bytes": None})
    stopped = [entry for entry in found if not entry["running"]]
    if sizes and stopped:
        try:
            # Docker walks each writable layer for its size; a slow answer leaves the sizes unknown.
            done = run(["docker", "--context", "default", "container", "inspect", "--size",
                        *(entry["id"] for entry in stopped)], capture_output=True, text=True, timeout=SIZE_SECONDS)
            measured = json.loads(done.stdout) if not done.returncode else []
            by_id = {item.get("Id"): item.get("SizeRw") for item in measured if isinstance(item, dict)}
        except (OSError, ValueError, subprocess.SubprocessError):
            by_id = {}
        for entry in stopped:
            value = by_id.get(entry["id"])
            entry["bytes"] = value if type(value) is int and value >= 0 else None
    return sorted(found, key=lambda entry: entry["name"])


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


def _read_mesh_file(path):
    """The JSON object in mesh configuration file ``path``, or ``None`` when it cannot be read as one."""
    try:
        with open(path, "rb") as stream:
            data = stream.read(MESH_FILE_LIMIT + 1)
        value = json.loads(data) if len(data) <= MESH_FILE_LIMIT else None
    except (OSError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _site_paths(value):
    """Every absolute path among the string values of a site document, at any depth, normalized."""
    if isinstance(value, dict):
        return {path for item in value.values() for path in _site_paths(item)}
    if isinstance(value, list):
        return {path for item in value for path in _site_paths(item)}
    if isinstance(value, str) and value.startswith("/"):
        path = _valid(posixpath.normpath(value))
        return {path} if path else set()
    return set()


def installed_meshes(root="/"):
    """The SparkRing meshes installed on this host, with the host paths that their sites name.

    A mesh is installed while its configuration directory holds its
    ``service.json`` or ``site.json``: the default layout's
    ``/etc/sparkring/managed-mesh`` or a named layout's
    ``/etc/sparkring/deployments/<name>`` (``managed_deployment.layout``), the
    directories that ``native_mesh.inspect_local`` reads. The state of its
    systemd unit does not matter. A stopped or disabled mesh keeps its units
    and site, ``native_mesh.serve_ring`` enables and starts the mesh that a
    deployment's fabric reference names whatever state its unit is in, and the
    started mesh checks and runs the marker binary that its site names. A mesh
    whose files ``native_mesh.set_aside`` moved to
    ``/var/lib/sparkring/replaced-meshes`` is not installed.

    A mesh's sites are the ``site_path`` that its ``service.json`` names and
    the ``site.json`` in its configuration directory. Returns ``[{"unit",
    "config", "site", "paths", "readable", "container"}]``: the mesh unit
    (``None`` for a directory whose name is no layout name), the configuration
    directory, the site file, every absolute path among the sites' values
    (``_site_paths``), whether ``service.json`` and every site could be read,
    and the ID of the model container that ``service.json`` names (``None``
    without one). The mesh's model unit starts that container, which the
    deployment that created the mesh created. ``root`` relocates ``/etc``.
    """
    def local(path):
        return posixpath.join(root, path.lstrip("/"))

    default = managed_deployment.layout()
    configs = [(default["config_dir"], default["mesh_unit"])]
    for name, info in _entries(local(MESH_NAMED_ROOT)):
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            continue
        try:
            unit = managed_deployment.layout(name)["mesh_unit"]
        except ValueError:
            unit = None
        configs.append((posixpath.join(MESH_NAMED_ROOT, name), unit))
    meshes = []
    for config, unit in configs:
        service, site = posixpath.join(config, "service.json"), posixpath.join(config, "site.json")
        present = [path for path in (service, site) if os.path.lexists(local(path))]
        if not present:
            continue
        sites, readable, container = [], True, None
        if service in present:
            document = _read_mesh_file(local(service))
            named = _valid(document.get("site_path")) if document else None
            if named is None:
                readable = False
            else:
                sites.append(named)
            value = document.get("container_id") if document else None
            container = value if isinstance(value, str) and CONTAINER_ID.fullmatch(value) else None
        if site in present and site not in sites:
            sites.append(site)
        paths = set()
        for path in sites:
            document = _read_mesh_file(local(path))
            if document is None:
                readable = False
            else:
                paths |= _site_paths(document)
        meshes.append({"unit": unit, "config": config, "site": sites[0] if sites else site,
                       "paths": sorted(paths), "readable": readable, "container": container})
    return meshes


def _mesh_users(meshes, path):
    """The ``installed_meshes`` that use ``path``, as ``[{"unit", "site", "paths"}]``.

    A mesh uses ``path`` when a path that its site names is ``path`` or lies
    inside it (``paths`` lists those), and when its site cannot be read
    (``readable`` false). A site path above ``path``, such as the cluster
    cache that ``cache_roots`` names, does not make the mesh use ``path``: the
    deployments' container specifications name the cache directories that
    their containers use.
    """
    found = []
    for mesh in meshes:
        named = [value for value in mesh["paths"] if _inside(value, path)]
        if named or not mesh["readable"]:
            found.append({"unit": mesh["unit"], "site": mesh["site"], "paths": named,
                          **({} if mesh["readable"] else {"readable": False})})
    return found


def _mesh_phrase(mesh):
    """``the installed mesh UNIT, whose site SITE names PATHS`` for one entry of an item's ``meshes``."""
    named = ("cannot be read" if mesh.get("readable") is False
             else "names " + ", ".join(mesh["paths"][:3]) + (", ..." if len(mesh["paths"]) > 3 else ""))
    return f"the installed mesh{' ' + mesh['unit'] if mesh.get('unit') else ''}, whose site {mesh['site']} {named}"


def _mesh_refusal(meshes, path):
    """The clause saying that ``meshes`` (an item's ``meshes``) keep ``path``."""
    several = len(meshes) > 1
    return (" and ".join(_mesh_phrase(mesh) for mesh in meshes) + (", use " if several else ", uses ") + path
            + "; SparkRing does not release it while " + ("those meshes are" if several else "that mesh is")
            + " installed")


def list_local(request=None, *, root="/", run=None):
    """This host's storage report (``sparkring-storage-local/v1``).

    ``request`` holds ``cluster`` (the cluster name, whose directory in
    ``/srv/sparkring`` SparkRing's installer creates), ``workspaces``,
    ``caches`` and ``models`` (paths that Node A's retained deployments name on
    this host), ``budget_seconds`` and ``measure``, a list of item paths that
    limits the walks to those items. Every item except ``other`` and
    ``releasing`` also carries ``containers``, the running containers that
    reach it (``_users``), and every cache directory and workspace carries
    ``meshes``, the meshes installed on this host that use it
    (``_mesh_users``); ``meshes`` of the report lists every installed mesh
    (``installed_meshes``), and ``model_containers`` every installer
    deployment's model container on this host, running or stopped
    (``model_containers``), with writable-layer sizes unless ``measure``
    limits the walks. ``root`` relocates the search of ``/srv/sparkring`` and
    of the mesh configurations in ``/etc``; ``run`` replaces
    ``subprocess.run`` for Docker.
    """
    request = request if isinstance(request, dict) else {}
    started = time.monotonic()
    budget = request.get("budget_seconds", BUDGET_SECONDS)
    if type(budget) not in (int, float) or budget <= 0:
        budget = BUDGET_SECONDS
    measured = request.get("measure")
    measured = {path for path in measured if isinstance(path, str)} if isinstance(measured, list) else None
    run = run or subprocess.run

    def inspect_docker():
        state, found = _docker_state(run)
        return state, found, model_containers(found, run=run, sizes=measured is None)

    # Docker is read, with the model containers' sizes, while the items are walked.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        docker = pool.submit(inspect_docker)
        items, held = _items(root, request)
        _measure(items, held, started + budget, measured)
        docker, containers, models = docker.result()
    meshes = installed_meshes(root)
    for item in items:
        if item["kind"] not in ("other", "releasing"):
            item["containers"] = _users(containers, item["path"])
        if item["kind"] in MESH_KINDS:
            item["meshes"] = _mesh_users(meshes, item["path"])
    srv = posixpath.join(root, "srv", "sparkring")
    return {"schema": LOCAL_SCHEMA, "hostname": socket.gethostname(),
            "package_revision": checkpoints.package_revision(),
            "filesystems": filesystems([("sparkring", srv), ("docker", (docker or {}).get("root")), ("root", "/")]),
            "items": items, "meshes": meshes, "docker": docker, "model_containers": models,
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


def _check(item, items, request, run, table, meshes):
    """Refuse, before anything changes, a release of ``item`` that this Spark does not allow.

    ``meshes`` are the meshes installed on this Spark (``installed_meshes``).
    """
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
    users = _mesh_users(meshes, path)
    if users:
        reason = _mesh_refusal(users, path)
        raise ValueError(reason[0].upper() + reason[1:] + ". Nothing was released.")
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


def release_local(path, request=None, *, root="/", run=None, table=None, remainders_only=False):
    """Remove cache directory or deployment workspace ``path`` from this host.

    With ``remainders_only``, only the remainders of earlier interrupted
    releases of ``path`` are removed, never an item now at ``path``.

    ``request`` is the listing request (``list_local``) plus ``in_use``, the
    paths the installed deployments name on this host, and ``opaque``, the cache
    roots whose entries they may all use. The item is found as the listing
    finds it. Before anything changes, the release refuses a path that is no
    ``cache``, ``workspace`` or ``releasing`` item, that is or holds a path in
    ``in_use`` or lies in an ``opaque`` root, that a mesh installed on this
    host uses (``_mesh_users`` of ``installed_meshes``, read below ``root``
    whatever the request says), that holds a SparkRing checkpoint directory or
    a non-empty ``models`` directory, that holds a mount point (``table``
    replaces the mount table), or that a running container on this host uses
    (``containers_using``). It then renames ``path`` to its
    ``releasing`` name and removes that directory; a remainder of an earlier
    interrupted release of ``path`` is removed as well.

    Returns ``{"path", "state": "released" | "absent", "kind", "freed_bytes"}``.
    """
    path = _absolute(path)
    request = request if isinstance(request, dict) else {}
    items, _ = _items(root, request)
    matches = [item for item in items if item["path"] == path]
    current = None if remainders_only else next((item for item in matches if item["kind"] != "releasing"), None)
    remainders = [item for item in matches if item["kind"] == "releasing"]
    if remainders_only and not remainders:
        return {"path": path, "state": "absent", "kind": None, "freed_bytes": 0}
    if current is None and not remainders:
        if _lstat(path) is None:
            return {"path": path, "state": "absent", "kind": None, "freed_bytes": 0}
        raise ValueError(f"{path} is not a SparkRing cache directory or deployment workspace on this Spark; "
                         "SparkRing does not remove it. Nothing was released.")
    table = checkpoint_place.mounts() if table is None else table
    if current is not None:
        _check(current, items, request, run, table, installed_meshes(root))
    for item in remainders:
        _unmounted(item["location"], table)
    freed = sum(_remove(item["location"]) for item in remainders)
    if current is not None:
        freed += _remove(_rename(path))
    return {"path": path, "state": "released", "kind": (current or remainders[0])["kind"], "freed_bytes": freed}


def remove_container(entry, *, run, meshes):
    """Remove one stopped model container that Node A's automatic release names; refuse any other.

    ``entry`` is ``{"deployment", "name"}``: the deployment's lock ID and the
    container's name (``sr-<site>-r<rank>``). The container is removed only
    when it carries that deployment's ``io.sparkring.deployment`` label, runs
    no process (Docker state ``created`` or ``exited``) and is not the model
    container that a mesh installed on this host starts (``meshes``, from
    ``installed_meshes``). ``docker container rm`` without ``--force`` also
    refuses a container that started meanwhile; ``--volumes`` removes its
    anonymous volumes, never named ones.

    Returns ``{"deployment", "name", "id", "state": "removed" | "absent",
    "freed_bytes"}``; ``freed_bytes`` is the writable layer's size.
    """
    deployment, name = entry.get("deployment"), entry.get("name")
    if not (isinstance(deployment, str) and DEPLOYMENT_ID.fullmatch(deployment) and isinstance(name, str)
            and MODEL_CONTAINER.fullmatch(name)):
        raise ValueError(f"{str(name)[:80]} is not a SparkRing model container; nothing was removed")
    result = {"deployment": deployment, "name": name, "id": None, "state": "absent", "freed_bytes": 0}
    try:
        found = json.loads(_docker(["container", "inspect", "--size", name], run))
    except ValueError as error:
        if re.search(r"No such (?:container|object)", str(error), re.IGNORECASE):
            return result
        raise
    info = found[0] if isinstance(found, list) and len(found) == 1 and isinstance(found[0], dict) else {}
    labels = (info.get("Config") or {}).get("Labels") or {}
    state = info.get("State") or {}
    if _name(info) != name or not CONTAINER_ID.fullmatch(str(info.get("Id"))):
        raise ValueError(f"Docker answered for another container than {name}; nothing was removed")
    if labels.get(DEPLOYMENT_LABEL) != deployment:
        raise ValueError(f"{name} does not carry deployment {deployment[:12]}'s label; SparkRing does not remove it")
    if state.get("Running") or state.get("Paused") or state.get("Restarting") or state.get("Status") not in STOPPED_STATES:
        raise ValueError(f"{name} is {state.get('Status') or 'not stopped'}; SparkRing removes only stopped model "
                         "containers")
    users = [mesh for mesh in meshes if mesh.get("container") == info["Id"]]
    if users:
        unit = users[0].get("unit") or users[0]["config"]
        raise ValueError(f"The installed mesh {unit} starts {name}; SparkRing does not remove it while that mesh is "
                         "installed")
    _docker(["container", "rm", "--volumes", info["Id"]], run)
    size = info.get("SizeRw")
    return {**result, "id": info["Id"], "state": "removed", "freed_bytes": size if type(size) is int and size > 0 else 0}


def release_batch_local(request=None, *, root="/", run=None, table=None):
    """Remove the stopped model containers, workspaces and cache directories that Node A's automatic release names.

    ``request`` is the release request of ``release_local`` (the listing
    request plus ``in_use`` and ``opaque``, here the paths that the
    deployments the policy keeps name on this host) with ``containers``, a
    list of ``{"deployment", "name"}``, ``paths``, the workspaces and cache
    directories to remove, and ``remainders``, the paths whose remainders of
    interrupted releases to remove. Containers go first
    (``remove_container``), because they bind-mount files of their
    workspace. A workspace whose deployment's container on this host was not
    removed stays. Every path then goes through ``release_local``, which lists
    and checks it again as ``sudo sparkring storage --release`` does; a
    remainder's path is released with ``remainders_only``, so an item that
    exists again at that path stays. A refusal or failure of one entry is
    reported with it and does not stop the others.

    Returns ``{"containers": [...], "paths": [...]}``: each entry the result
    of ``remove_container`` or ``release_local``, or the entry with ``error``.
    """
    request = request if isinstance(request, dict) else {}
    run = run or subprocess.run
    meshes = installed_meshes(root)
    containers, kept = [], set()
    for entry in request.get("containers") or ():
        entry = entry if isinstance(entry, dict) else {}
        try:
            containers.append(remove_container(entry, run=run, meshes=meshes))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            kept.add(entry.get("deployment"))
            containers.append({"deployment": entry.get("deployment"), "name": entry.get("name"),
                               "error": str(error)[:500]})
    items, _ = _items(root, request)
    owners = {item["path"]: item["deployment"] for item in items if item["kind"] == "workspace"}
    table = checkpoint_place.mounts() if table is None else table
    released = []
    for path in request.get("paths") or ():
        if isinstance(path, str) and path in owners and owners[path] in kept:
            released.append({"path": path, "error": f"{path} stays because its deployment's model container on this "
                                                    "Spark was not removed"})
            continue
        try:
            released.append(release_local(path, request, root=root, run=run, table=table))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            released.append({"path": path, "error": str(error)[:500]})
    for path in request.get("remainders") or ():
        try:
            released.append(release_local(path, request, root=root, run=run, table=table, remainders_only=True))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            released.append({"path": path, "error": str(error)[:500]})
    return {"containers": containers, "paths": released}


def node(release=None, stream=None, *, batch=False):
    """``sudo sparkring node storage [--release PATH | --release-batch]``: this host's part, request read as JSON from ``stream``.

    An empty or interactive ``stream`` is an empty request.
    """
    stream = sys.stdin if stream is None else stream
    text = "" if stream is None or stream.isatty() else stream.read()
    request = json.loads(text) if text.strip() else {}
    if not isinstance(request, dict):
        raise ValueError("The storage request must be a JSON object")
    if batch:
        if release is not None:
            raise ValueError("Give --release PATH or --release-batch, not both")
        return release_batch_local(request)
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
            elif item.get("meshes") or any(deployment["directory"] in role for deployment in users):
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
    """``sparkring storage``: every Spark's report with each item's and image's class.

    On Node A of a configured cluster the result also carries ``retention``:
    the deployments that automatic release keeps and why, and what it would
    release (``retention.view``).
    """
    from runtime.host import retention
    retained, role = retention.deployments(state_root), checkpoints.roles(state_root)
    cluster = checkpoints._cluster(state_root)
    if cluster is None:
        nodes = [{"rank": None, "host": None, **list_local(_request(None, None, retained))}]
    else:
        nodes = _survey(checkpoints._hosts(cluster), cluster["name"], retained, invoke)
    result = {"schema": SCHEMA, "state": "listed", "nodes": classify(nodes, retained, role, profile_references())}
    if cluster is not None:
        result["retention"] = retention.view(state_root, retained, role, nodes)
    return result


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
        if not users:
            return f"{where}: {_mesh_refusal(item['meshes'], path)}"
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
          "switch) or by a mesh installed on that Spark; profile = referenced by an installer profile of the "
          "installed package, kept for its next installation; unreferenced = neither, proposed for release unless a "
          "running container uses it or it holds model files; unmanaged = not created by SparkRing's installer, "
          "never removed. A size ending in + was still being measured at the time limit.")


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
    lines.extend("used by " + _mesh_phrase(mesh) for mesh in item.get("meshes") or ())
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
    if result.get("retention"):
        from runtime.host import retention
        lines.extend(retention.describe(result["retention"]))
    lines.append(LEGEND)
    lines.append("sudo sparkring storage --release PATH removes one unreferenced cache directory or deployment "
                 "workspace from every Spark that holds it; checkpoint directories are released with sudo sparkring "
                 "checkpoints --release PATH. SparkRing does not remove Docker images: docker image rm ID on that "
                 "Spark removes one that no container uses. An image's size counts the layers it shares with other "
                 "images, and removing it frees only the layers no other image uses; each installer image adds a "
                 "few megabytes to the one it was built on.")
    return lines


def _administrator():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def main(argv=None, *, state_root=None, invoke=None):
    parser = argparse.ArgumentParser(
        prog="sparkring storage",
        description="Report SparkRing's disk use on every Spark with the data no deployment uses, release one "
                    "unreferenced cache directory or deployment workspace, or set how many deployments automatic "
                    "release keeps.")
    parser.add_argument("--release", metavar="PATH",
                        help="remove this unreferenced cache directory or deployment workspace from every Spark that "
                             "holds it")
    parser.add_argument("--yes", action="store_true", help="approve the release without a prompt")
    parser.add_argument("--json", action="store_true", help="emit one JSON result on stdout")
    parser.add_argument("--retain-deployments", metavar="N|off",
                        help="after each install and up, keep the N most recent deployments of each profile and "
                             "release what older ones hold on the Sparks (default 2); off turns this off")
    args = parser.parse_args(argv)
    if args.retain_deployments is not None and args.release:
        parser.error("--retain-deployments and --release are separate commands")
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
        if args.retain_deployments is not None:
            from runtime.common import process_lock
            from runtime.host import retention
            with process_lock.hold(Path(state_root) / "install.lock"):
                retain = retention.save_setting(state_root, args.retain_deployments)
            result = {"schema": SCHEMA, "state": "saved", "retain_deployments": "off" if retain is None else retain}
            write(retention.setting_text(retain))
        elif args.release:
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
