"""Free space that the serving image and the compile cache need on a Spark during ``sparkring install``.

The checkpoint plan (``runtime.host.checkpoint_plan``) adds these needs to the
checkpoint files a Spark writes when Docker's data root or the compile cache is
on the checkpoint directory's filesystem. Before the image is downloaded, the
image distribution (``runtime.host.install_assets``) checks the image needs
again with the registry's layer list; before checkpoint files are written, the
rank operations (``scripts/installer_host.py``) check the compile cache need
again. The figures come from the image locks' recorded sizes and from each
Spark's inspection; a lock without sizes and a compile cache not yet filled
get the allowances of ``profiles/storage-planning.json``.

Serving image
-------------
``lineage`` lists the lock's image and the images it derives from. A derived
image extends its parent's layers, so a Spark that holds an ancestor holds
those layers and lacks only the layers added after it. Docker
reports an image's unpacked size as the sum of its layers' sizes, so the
missing layers unpack to the target's ``image_bytes`` minus the ancestor's.
A lock's ``download_bytes`` is the registry's figure or, for a lock that
publication did not update, an upper bound, so the difference of two locks'
download sizes can even be negative. The missing layers' download is bounded
by the larger of that difference and their unpacked size: gzip adds only a few
bytes per block to a layer's tar stream, and the tar header of each file falls
within the load margin below.

``image_need`` gives each Spark one ``basis``:

- ``present``: the Spark holds the image; nothing is reserved.
- ``layers``: the Spark holds an ancestor and Docker uses its graph-driver
  image store. The installer then loads only the missing layers
  (``install_assets.Assets.load_layers``), whose own check reserves
  ``LOAD_FACTOR`` times their download size plus ``LOAD_MARGIN_BYTES`` because
  the received blobs, the loader's extracted copy and the unpacked layers
  coexist. The plan reserves the same figure with the download bound above,
  which also covers the unpacked size.
- ``whole``: no ancestor is present; Docker uses the containerd image store, into
  which the installer loads no single layers (``install_assets.layer_prefix``);
  or the Spark's images could not be listed. A pull holds the compressed layers
  beside the unpacked image: ``image_bytes`` plus ``download_bytes`` plus
  ``PULL_MARGIN_BYTES`` for Docker's metadata and filesystem allocation.
- ``unsized``: the lock records no sizes (``sparkring-installer-image/v1``);
  the allowance ``image_allowance_gib``.

Node A's registry relay keeps one compressed copy of every layer a Spark
downloads until the image is distributed: the largest ``relay_bytes`` of the
Sparks' needs (``relay_need``).

Compile cache
-------------
Installer containers keep their compile and tuning caches (Triton,
TorchInductor, vLLM's torch.compile cache, FlashInfer, CuTe DSL, TileLang, TVM
FFI) in a directory of the cluster cache named by model family, image and
checkpoint revision, and B12X kernels in one named by family, CUDA version and
revision (``runtime.host.storage.cache_names``). When every directory a
deployment uses exists and holds files, an earlier start compiled them and
nothing is reserved; otherwise the allowance ``compile_cache_allowance_gib``
applies (``cache_need``).
"""
import inspect
import json
from pathlib import Path
import re

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"[a-z][a-z0-9-]{0,63}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
# A pull: Docker's metadata and filesystem allocation beside the compressed and unpacked layers.
PULL_MARGIN_BYTES = 8 * GIB
# A layer load: the received blobs, the loader's extracted copy and the unpacked
# layers, bounded by this multiple of the layers' download size plus the margin.
LOAD_FACTOR = 4
LOAD_MARGIN_BYTES = 4 * GIB
# The relay's pull check for an image lock without sizes: the compressed layers
# beside the image allowance, which stands for the unpacked image.
UNSIZED_DOWNLOAD_BYTES = 32 * GIB
# The inspection runs as root and reads its source on stdin, as the checkpoint survey does.
PROBE_COMMAND = ["sudo", "-n", "python3", "-I", "-B", "-"]
PROBE_TIMEOUT = 120


# Serving image

def _sizes(value):
    """``(image_bytes, download_bytes)`` of a ``sparkring-installer-image/v2`` lock, else ``(None, None)``."""
    sizes = value.get("image_bytes"), value.get("download_bytes")
    if value.get("schema") != "sparkring-installer-image/v2" or not all(
            type(size) is int and size > 0 for size in sizes):
        return None, None
    return sizes


def _release_lock(directory, image_id):
    """The lock in ``directory/installer-image.json`` when it records ``image_id``, else None."""
    try:
        value = json.loads((directory / "installer-image.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("image_id") == image_id else None


def lineage(value, root=None):
    """The image of installer image lock ``value`` followed by the images it derives from, nearest first.

    Each entry is ``{"name", "image_id", "image_bytes", "download_bytes"}``;
    the sizes are None for a lock that does not record them (v1). The first
    entry is ``value`` itself. The release directory
    ``runtime/releases/<name>/`` whose ``installer-image.json`` records an
    entry's image ID names, in its ``publication.json``,
    ``derivation.parent_release`` and ``derivation.parent_image_id``; the
    parent release's lock must record that image ID. The chain ends at a
    release without a derivation, at a parent whose lock is missing,
    unreadable or records another image, and at an image already listed.
    """
    releases = Path(root if root is not None else ROOT) / "runtime" / "releases"
    image_bytes, download_bytes = _sizes(value)
    chain = [{"name": value.get("name"), "image_id": value.get("image_id"), "image_bytes": image_bytes,
              "download_bytes": download_bytes}]
    name = value.get("name")
    directory = releases / name if isinstance(name, str) and NAME.fullmatch(name) else None
    if directory is None or _release_lock(directory, value.get("image_id")) is None:
        directory = next((path.parent for path in sorted(releases.glob("*/installer-image.json"))
                          if _release_lock(path.parent, value.get("image_id")) is not None), None)
    while directory is not None and len(chain) < 64:
        try:
            derivation = json.loads((directory / "publication.json").read_text(encoding="utf-8"))["derivation"]
            parent, image_id = derivation["parent_release"], derivation["parent_image_id"]
        except (OSError, ValueError, KeyError, TypeError):
            break
        if (not isinstance(parent, str) or not NAME.fullmatch(parent) or not isinstance(image_id, str)
                or not IMAGE_ID.fullmatch(image_id) or any(entry["image_id"] == image_id for entry in chain)):
            break
        directory = releases / parent
        lock = _release_lock(directory, image_id)
        if lock is None:
            break
        image_bytes, download_bytes = _sizes(lock)
        chain.append({"name": parent, "image_id": image_id, "image_bytes": image_bytes,
                      "download_bytes": download_bytes})
    return chain


def sized(entry):
    """Whether a lineage entry or an installer card records both image sizes."""
    return all(type(entry.get(key)) is int and entry[key] > 0 for key in ("image_bytes", "download_bytes"))


def whole_bytes(entry):
    """Free bytes in Docker's data root that pulling the whole image of a sized lock needs."""
    return entry["image_bytes"] + entry["download_bytes"] + PULL_MARGIN_BYTES


def pull_bytes(card, allowance):
    """Free bytes in Docker's data root that pulling the whole image of ``card`` needs.

    A card without recorded sizes needs the image allowance ``allowance`` for
    the unpacked image and ``UNSIZED_DOWNLOAD_BYTES`` for its compressed layers.
    """
    return whole_bytes(card) if sized(card) else allowance + UNSIZED_DOWNLOAD_BYTES


def load_bytes(download):
    """Free bytes in Docker's data root that loading layers of ``download`` compressed bytes needs."""
    return LOAD_FACTOR * download + LOAD_MARGIN_BYTES


def image_need(lineage, found, allowance):
    """The free space the serving image needs in one Spark's Docker data root.

    ``lineage`` is ``lineage`` of the lock. ``found`` is the
    Spark's ``inspect_spark`` result, ``{"error": text}`` when that failed, or
    None when the image is taken to be present. ``allowance`` is the image
    allowance in bytes. Returns ``{"basis", "bytes", "relay_bytes"}``; a
    ``layers`` need adds ``ancestor`` (release name), ``unpacked_bytes`` and
    ``download_bytes`` of the missing layers, and a ``whole`` need ``reason``:
    ``none`` (no ancestor present), ``containerd`` (with ``ancestor``) or
    ``unchecked`` (with ``error``).
    """
    target = lineage[0]
    images = None if found is None or found.get("error") else found.get("images")
    if found is None or (isinstance(images, list) and target["image_id"] in images):
        return {"basis": "present", "bytes": 0, "relay_bytes": 0}
    if not sized(target):
        # The relay check of such an image has its own allowance; the plan counts the image allowance only.
        return {"basis": "unsized", "bytes": allowance, "relay_bytes": 0}
    whole = {"basis": "whole", "bytes": whole_bytes(target), "relay_bytes": target["download_bytes"]}
    if not isinstance(images, list):
        return {**whole, "reason": "unchecked", "error": str((found or {}).get("error") or "no image list")[:300]}
    ancestor = next((entry for entry in lineage[1:] if entry["image_id"] in images), None)
    if ancestor is None or not sized(ancestor):
        return {**whole, "reason": "none"}
    if found.get("containerd") is not False:
        return {**whole, "reason": "containerd" if found.get("containerd") else "unchecked",
                "ancestor": ancestor["name"], "error": "Docker's image store is unknown"}
    unpacked = target["image_bytes"] - ancestor["image_bytes"]
    if unpacked <= 0:
        return {**whole, "reason": "none"}
    download = max(unpacked, target["download_bytes"] - ancestor["download_bytes"])
    return {"basis": "layers", "bytes": load_bytes(download), "relay_bytes": download,
            "ancestor": ancestor["name"], "unpacked_bytes": unpacked, "download_bytes": download}


def relay_need(needs):
    """Bytes Node A's relay holds: one copy of the layers the Spark lacking the most needs."""
    return max((need.get("relay_bytes") or 0 for need in needs if need), default=0)


def relay_check(card, missing, held, layers, allowance):
    """Free bytes each Spark's Docker data root needs before the relay distributes the image.

    ``missing`` lists the ranks without the image and ``held`` maps each to the
    number of leading layers it holds (``install_assets.layer_prefix``);
    ``layers`` is the registry's ``[(diff_id, digest, size)]``, or None when
    the relay could not read it, and ``allowance`` the image allowance in
    bytes. A rank holding leading layers loads the rest; any other rank pulls
    the whole image. Node A also keeps the relay's copy of every layer that
    some rank lacks. Returns ``{rank: bytes}`` for Node A and each missing rank.
    """
    whole = pull_bytes(card, allowance)
    relay = card["download_bytes"] if sized(card) else allowance
    required = {0: 0}
    for rank in missing:
        count = held.get(rank, 0) if layers else 0
        required[rank] = load_bytes(sum(size for _, _, size in layers[count:])) if count else whole
    if layers and missing:
        relay = sum(size for _, _, size in layers[min(held.get(rank, 0) for rank in missing):])
    required[0] += relay if missing else 0
    return required


# Compile cache

def cache_names(card, image_id):
    """Directory names the deployment's containers use in the cache root, for the card's checkpoint."""
    from runtime.common import profiles, qwen_flash_next
    from runtime.host import storage
    configuration = qwen_flash_next.read(profiles.ROOT / card["configuration"])
    if configuration.get("checkpoints"):
        configuration = qwen_flash_next.checkpoint_settings(configuration, card["target_variant"])
    toolchain = qwen_flash_next.image_policy(configuration)["kind"] == "toolchain"
    return storage.cache_names(configuration, image_id, toolchain=toolchain)


def cache_need(states, allowance):
    """The compile cache need from ``directory_use`` of each of the deployment's cache directories.

    ``states`` maps each directory to its use, or None when it is absent; an
    empty mapping means the directories are unknown. Returns ``{"basis":
    "built" | "allowance", "bytes", "present_bytes"}``.
    """
    built = bool(states) and all(isinstance(state, dict) and state.get("files", 0) > 0 for state in states.values())
    present = sum((state or {}).get("bytes", 0) for state in states.values()) if states else 0
    return {"basis": "built" if built else "allowance", "bytes": 0 if built else allowance, "present_bytes": present}


# Inspection on each Spark; these functions are shipped as source and import only the standard library.

def directory_use(path, entries=200000, seconds=10.0):
    """``{"files", "bytes", "complete"}`` of directory ``path``, or None when it is absent or not a directory.

    Symlinks are not followed and other filesystems are not entered; ``bytes``
    are allocated bytes. The walk stops after ``entries`` entries or
    ``seconds`` seconds and is then reported incomplete.
    """
    import os
    import stat
    import time
    try:
        top = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(top.st_mode):
        return None
    deadline, seen = time.monotonic() + seconds, 0
    result = {"files": 0, "bytes": 0, "complete": True}
    pending = [path]
    while pending:
        if seen >= entries or time.monotonic() >= deadline:
            result["complete"] = False
            break
        try:
            with os.scandir(pending.pop()) as listing:
                for entry in listing:
                    seen += 1
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        result["complete"] = False
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        # A directory entry's cached stat may lack the device; lstat reports it.
                        try:
                            if os.lstat(entry.path).st_dev == top.st_dev:
                                pending.append(entry.path)
                        except OSError:
                            result["complete"] = False
                    elif stat.S_ISREG(info.st_mode):
                        result["files"] += 1
                        result["bytes"] += getattr(info, "st_blocks", 0) * 512 or info.st_size
        except OSError:
            result["complete"] = False
    return result


def inspect_spark(image_ids, cache_paths):
    """The nearest present image of ``image_ids``, Docker's image store and each cache directory's use.

    ``image_ids`` lists the lock's image and its ancestors, nearest first; the
    first one ``docker image inspect`` finds under its own ID ends the search.
    ``containerd`` is whether Docker uses the containerd image store, as
    ``install_assets.layer_prefix`` decides it, or None when ``docker info``
    failed. Returns ``{"images", "containerd", "caches"}``, with ``error`` when
    Docker could not be asked.
    """
    import subprocess
    docker = ["docker", "--context", "default"]
    result = {"images": None, "containerd": None, "caches": {path: directory_use(path) for path in cache_paths}}
    try:
        status = subprocess.run([*docker, "info", "--format", "{{json .DriverStatus}}"], capture_output=True,
                                text=True, timeout=30, stdin=subprocess.DEVNULL)
        if status.returncode == 0:
            result["containerd"] = "io.containerd.snapshotter" in status.stdout
        images = []
        for image in image_ids:
            found = subprocess.run([*docker, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True,
                                   text=True, timeout=30, stdin=subprocess.DEVNULL)
            if found.returncode == 0 and found.stdout.strip() == image:
                images.append(image)
                break
        result["images"] = images
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)[:300] or type(error).__name__
    return result


def probe_source(image_ids, cache_paths):
    """Self-contained Python source that prints ``inspect_spark(image_ids, cache_paths)`` as JSON."""
    body = "\n\n".join(inspect.getsource(function) for function in (directory_use, inspect_spark))
    call = (f"print(json.dumps(inspect_spark(json.loads({json.dumps(list(image_ids))!r}), "
            f"json.loads({json.dumps(list(cache_paths))!r}))))\n")
    return "import json\n\n\n" + body + "\n\n\n" + call
