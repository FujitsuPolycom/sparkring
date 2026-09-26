"""Rank-local installer operations, invoked only by the approved controller plan.

Checkpoint rows have two modes:

- A row whose ``model`` is a SparkRing checkpoint directory (the default) is
  *owned*. SparkRing claims that directory, places every pinned file in it by a
  hard link or through a staging directory only after hashing the file's
  complete content, journals each name before creating it, and removes only
  names whose inode the journal records (``runtime.host.checkpoint_place``).
- A row with ``reuse_verified_model`` names a copy that SparkRing did not
  create. It is served in place only while it holds exactly the pinned files;
  SparkRing reads it and never creates, links, downloads or writes anything
  under it.
"""
from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request

from runtime.common import compose, glm_native_candidate, installer, native_candidate, profiles, qwen_flash_next, setup
from runtime.common.container_spec import expected_inspection
from runtime.host import checkpoint_place as place
from scripts import deploy_engine

POSIX_STATS = os.name == "posix"
GIB = 1024 ** 3
# A copy served in place may also hold its download client's and git's own
# metadata; nothing below these directories is served.
IN_PLACE_IGNORED = frozenset({".cache/huggingface", ".huggingface", ".git"})
MODEL_OPERATIONS = frozenset({"model", "model-check", "model-adopt", "model-fetch", "model-settled",
                              "model-reuse-receipt", "model-transfer-manifest", "model-transfer-prepare",
                              "model-transfer-complete"})
# Files SparkRing did not create are read without following a final symlink,
# without blocking on a FIFO or device, and without changing access times.
_READ = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
         | getattr(os, "O_BINARY", 0))
_NOATIME = getattr(os, "O_NOATIME", 0)
FETCH_CODE = ("import sys; from huggingface_hub import hf_hub_download; repo, rev, *names = sys.argv[1:]; "
              "[hf_hub_download(repo_id=repo, revision=rev, filename=n, local_dir='/fetch') for n in names]")
# How long one checkpoint download container may run.
FETCH_SECONDS = 7200


class CommandError(subprocess.CalledProcessError):
    """A failed command whose message names the program and carries the tail of its own error output.

    The message leaves out the argument list, which for a checkpoint download
    holds the mount, the image, the program and every file name.
    """

    def lines(self, count=3):
        detail = self.stderr or self.output or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        return [line.strip() for line in detail.strip().splitlines() if line.strip()][-count:]

    def __str__(self):
        words = [str(word) for word in self.cmd] if isinstance(self.cmd, (list, tuple)) else str(self.cmd).split()
        if words[:3] == ["docker", "--context", "default"]:
            words = words[:1] + words[3:]
        name = " ".join(words[:1] + [word for word in words[1:2] if not word.startswith("-")])
        lines = self.lines()
        return f"{name} exited with status {self.returncode}" + (": " + " | ".join(lines) if lines else "")


def run(argv, **kwargs):
    if argv[0] == "docker" and argv[1:3] != ["--context", "default"]:
        argv = ["docker", "--context", "default", *argv[1:]]
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("timeout", 7200)
    kwargs.setdefault("text", True)
    try:
        return subprocess.run(argv, **kwargs)
    except subprocess.CalledProcessError as error:
        raise CommandError(error.returncode, error.cmd, error.output, error.stderr) from None


def plain(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Installer paths must be absolute without symlink components")
    return path


def image_info(lock):
    return json.loads(run(["docker", "image", "inspect", lock["selection"]["image_id"]]).stdout)[0]


ADMISSIONS = Path("/var/lib/sparkring/admissions")


def admit_image(lock):
    card = lock["selection"]
    if "image_runtime" in lock:
        from runtime.common import installer_image, loader_policy
        # Admission starts three isolated containers. Its result depends on the
        # content-addressed image, the lock, the profile and this host's kernel,
        # Docker and loader policy, so it is reused while all of them match.
        current = json.loads(run(["docker", "image", "inspect", card["image_id"]]).stdout)[0]["Id"]
        key = hashlib.sha256(json.dumps({
            "lock": lock["image_runtime"], "profile": card["profile"], "nodes": card["nodes"],
            "kernel": os.uname().release if hasattr(os, "uname") else "",
            "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]).stdout.strip(),
            "policy": hashlib.sha256(loader_policy.PROFILE.read_bytes()).hexdigest(),
            # Qwen admission depends on the profile's HC mode and feature selection.
            "recipe": repr(installer_image.qwen_recipe(installer_image.profile_environment(card["profile"])))
            if card["profile"] in installer_image.QWEN else None,
        }, sort_keys=True).encode()).hexdigest()
        record = ADMISSIONS / (key + ".json")
        if record.is_file() and not record.is_symlink():
            saved = profiles.read_json(record)
            if saved.get("image_id") == current:
                return saved["receipt"]
        receipt = installer_image.admit(lock["image_runtime"], run=run, profile=card["profile"], nodes=card["nodes"])
        loader_policy.check(card["image_id"], run=run)
        try:
            ADMISSIONS.mkdir(parents=True, exist_ok=True, mode=0o700)
            deploy_engine.save_receipt(record, {"image_id": current, "receipt": receipt})
        except OSError:
            pass
        return receipt
    def native_run(argv, **kwargs):
        # Native admission deliberately reads the installed receipt as bytes.
        kwargs.setdefault("text", False)
        return run(argv, **kwargs)
    if card["profile"].startswith("glm53-"):
        return glm_native_candidate.observe(card["image_id"], card["release"], run=native_run)
    return native_candidate.verify_image(card["image_id"], card["release"], run=native_run)


# Checkpoint pins and file inventories ------------------------------------------

def _names(names, limit=12):
    names = list(names)
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def checkpoint_files(card):
    """``(required, optional)`` from the pin manifest of the card's revision, or ``None`` without one.

    ``required`` maps each required file name (the manifest's ``files`` minus
    ``optional``) to ``{"size", "sha256"}``. ``optional`` lists documentation and
    repository metadata that no serving component reads; a copy served in place
    may hold them. A manifest that exists but fails validation is an error.
    """
    try:
        pins = installer.checkpoint_pins(card)
    except ValueError:
        manifest = (installer.ROOT / "profiles/checkpoints" / str(card["model_repository"]).replace("/", "--")
                    / (str(card["model_revision"]) + ".json"))
        if manifest.exists():
            raise
        return None
    optional = set(pins["optional"])
    required = {name: {"size": entry["size"], "sha256": entry["sha256"]}
                for name, entry in sorted(pins["files"].items()) if name not in optional}
    return required, sorted(optional)


def _pinned(card):
    found = checkpoint_files(card)
    if found is None:
        raise ValueError(f"{card['model_repository']} at {card['model_revision']} has no pin manifest; SparkRing "
                         "assembles its own checkpoint directory only for pinned revisions. Name a complete copy "
                         "with --model-path to serve it in place.")
    return found


class NameSetError(ValueError):
    """A checkpoint folder whose file names differ from the pinned set."""

    def __init__(self, path, *, missing=(), extra=()):
        self.path, self.missing, self.extra = str(path), sorted(missing), sorted(extra)
        parts = (["lacks " + _names(self.missing)] if self.missing else []) + \
                (["also holds " + _names(self.extra)] if self.extra else [])
        super().__init__(f"Checkpoint at {self.path} " + " and ".join(parts))


def _entries(root, ignored=()):
    """``{relative name: lstat}`` of every entry below ``root`` that is not a directory.

    Directories are listed with ``os.scandir`` and never entered through a
    symlink. Entries at or below a relative directory named in ``ignored`` are
    skipped.
    """
    found, pending = {}, [""]
    while pending:
        prefix = pending.pop()
        with os.scandir(root / prefix if prefix else root) as listing:
            for entry in listing:
                name = prefix + entry.name
                if name in ignored:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(name + "/")
                else:
                    found[name] = os.lstat(root / name)
    return found


def model_file_stats(path, expected=None, *, in_place=False, optional=()):
    """``[st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns]`` of each checkpoint file.

    With ``expected`` (the required names) the folder must hold every expected
    name as a regular file, and ``NameSetError`` names what differs:

    - a SparkRing checkpoint directory holds nothing else;
    - a copy served in place (``in_place``) may also hold the ``optional`` names
      and anything below ``.cache/huggingface``, ``.huggingface`` and ``.git``,
      and nothing else: no symlink, no other file type, no extra tokenizer or
      processor file that the serving engine would load from the same folder.

    Stats are returned for the expected names. Without ``expected`` every
    regular file outside the top-level ``.cache`` and ``.git`` directories is
    described and a symlink is refused.
    """
    root = plain(path)
    if expected is None:
        result = {}
        for name, value in sorted(_entries(root, (".cache", ".git")).items()):
            if stat.S_ISLNK(value.st_mode):
                raise ValueError("Checkpoint verification cannot follow symlinks")
            if stat.S_ISREG(value.st_mode):
                result[name] = place.stats(value)
        return result
    expected = set(expected)
    entries = _entries(root, IN_PLACE_IGNORED if in_place else ())
    allowed = expected | set(optional) if in_place else expected
    extra = [name for name, value in entries.items() if name not in allowed or not stat.S_ISREG(value.st_mode)]
    missing = expected - entries.keys()
    if extra or missing:
        raise NameSetError(root, missing=missing, extra=extra)
    return {name: place.stats(entries[name]) for name in sorted(expected)}


def _hash_file(root, name, recorded):
    path = os.path.join(root, *name.split("/"))
    try:
        fd = os.open(path, _READ | _NOATIME)
    except PermissionError:
        # O_NOATIME needs the file's owner or root; other accounts read normally.
        fd = os.open(path, _READ)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or place.stats(info)[:4] != list(recorded[:4]):
            raise ValueError("Checkpoint changed during checksum verification: " + name)
        return place.hash_descriptor(fd, recorded[2])
    finally:
        os.close(fd)


def _hash_files(root, recorded):
    """SHA-256 of each named file, read through descriptors whose identity equals ``recorded``.

    Files are hashed concurrently: hashlib releases the GIL for large updates,
    so threads spread the work across cores until storage bandwidth is the
    limit.
    """
    names = sorted(recorded)
    if not names:
        return {}
    workers = max(1, min(16, os.cpu_count() or 4, len(names)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        digests = pool.map(lambda name: _hash_file(root, name, recorded[name]), names)
        return dict(zip(names, digests))


def model_files(path, expected=None, *, in_place=False, optional=()):
    """SHA-256 of every checkpoint file, after the name checks of ``model_file_stats``.

    Files are opened with ``O_NOFOLLOW`` and ``O_NOATIME`` and must keep their
    identity, size and modification time while they are read. Without
    ``expected``, ``config.json``, the weight index and every indexed shard must
    be present.
    """
    root = plain(path)
    before = model_file_stats(root, expected, in_place=in_place, optional=optional)
    result = _hash_files(root, before)
    if expected is None:
        if "config.json" not in result or "model.safetensors.index.json" not in result:
            raise ValueError("Checkpoint configuration/index are absent")
        index = profiles.read_json(root / "model.safetensors.index.json")
        shards = set(index["weight_map"].values())
        if not shards or not shards <= result.keys():
            raise ValueError("Checkpoint is missing indexed weight shards")
    return result


def checksum_manifest(profile, revision=None):
    """Per-file SHA-256 pins for a profile's checkpoint revision, if recorded.

    Each profile keeps its own file for its default checkpoint, because
    profiles that share a model repository can pin different revisions of it.
    Another checkpoint of the profile's table (``revision`` differs) has no
    such file; its pin manifest lists every required file instead.
    """
    own = profiles.ROOT / "profiles" / profile / "SHA256SUMS"
    if not own.is_file():
        return None
    if revision is not None and revision != profiles.resolve(profile)["model"]["revision"]:
        return None
    return own


def pinned_differences(profile, files, revision=None):
    """Names whose recorded pin differs from, or is absent in, a measured tree."""
    sums = checksum_manifest(profile, revision)
    if sums is None:
        return []
    differences = []
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if files.get(name.lstrip("*")) != digest:
            differences.append(name.lstrip("*"))
    return differences


def _check_pins(card, files, required=None):
    """Refuse hashes that differ from the profile's checkpoint contract, its SHA256SUMS or the pins."""
    model = installer.checkpoint_contract(card)
    for filename, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        if files.get(filename) != model[key]:
            raise ValueError("Checkpoint metadata differs from the selected profile: " + filename)
    sums = checksum_manifest(card["profile"], card["model_revision"])
    if sums is not None:
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if files.get(name.lstrip("*")) != digest:
                raise ValueError("Checkpoint shard checksum differs: " + name)
    for name, pin in (required or {}).items():
        if files.get(name) != pin["sha256"]:
            raise ValueError("Checkpoint file differs from the pinned revision: " + name)


# Receipts and records -------------------------------------------------------

CHECKPOINTS = Path("/var/lib/sparkring/checkpoints")


def _checkpoint_record(model):
    return CHECKPOINTS / (hashlib.sha256(str(model).encode()).hexdigest() + ".json")


def remembered_checkpoint(card, model, stats):
    """File hashes from an earlier verification of this exact tree on this host.

    Returns the recorded hash of every file in ``stats`` when the path record
    names the same repository, revision and path and records equal five-value
    stats for each of them (Linux only); otherwise ``None``.
    """
    record = _checkpoint_record(model)
    if not POSIX_STATS or not record.is_file() or record.is_symlink():
        return None
    saved = profiles.read_json(record)
    if (saved.get("repository"), saved.get("revision"), saved.get("path")) != (card["model_repository"], card["model_revision"], str(model)):
        return None
    recorded, hashes = saved.get("file_stats") or {}, saved.get("files") or {}
    if any(recorded.get(name) != value or name not in hashes for name, value in stats.items()):
        return None
    return {name: hashes[name] for name in stats}


def remember_checkpoint(receipt):
    """Best-effort: the record only saves re-hashing, so failing to write it is harmless."""
    if not POSIX_STATS:
        return
    try:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True, mode=0o700)
        deploy_engine.save_receipt(_checkpoint_record(receipt["path"]), receipt)
    except OSError:
        pass


def _record_inode(value, digest, source):
    """Best-effort per-inode record of a hashed source; records are evidence only."""
    try:
        place.record_inode(value, digest, source)
    except (OSError, ValueError):
        pass


def _changed_message(row, names):
    path = row["model"]
    if row.get("reuse_verified_model"):
        return (f"Checkpoint at {path} differs from its recorded files ({_names(names)}); SparkRing does not change "
                "it. Restore the files, or run sudo sparkring install without naming it.")
    return (f"Checkpoint at {path} differs from its recorded files ({_names(names)}). Repeat sudo sparkring install; "
            "it replaces files only in SparkRing's own checkpoint directory.")


def verify_model(lock, row, receipt_path, *, expected=None, refresh=False, receipt=None):
    """Check the checkpoint at ``row["model"]`` against its receipt and pins; return the receipt.

    ``expected`` maps each required name to ``{"size", "sha256"}`` (default: the
    lock's pin manifest; a revision without one keeps the receipt's whole-tree
    meaning). Only files whose five stat values differ from the receipt are
    hashed again; the receipt's hashes stand for the others, and unknown receipt
    keys and names outside ``expected`` are ignored. A copy served in place must
    also hold exactly the pinned names.

    When files were hashed again and matched, the returned receipt carries
    their re-measured stats. With ``refresh``, which only callers passing the
    rank's own ``state/model.json`` use, that receipt and its path record are
    saved, so later verifications need no hashing. A receipt read from another
    deployment is never rewritten here.
    """
    card = lock["selection"]
    receipt = profiles.read_json(receipt_path) if receipt is None else receipt
    optional = ()
    if expected is None:
        found = checkpoint_files(card)
        if found is not None:
            expected, optional = found
    in_place = bool(row.get("reuse_verified_model"))
    if ((receipt.get("repository"), receipt.get("revision"), receipt.get("path"))
            != (card["model_repository"], card["model_revision"], row["model"])):
        raise ValueError("Checkpoint differs from its recorded revision/files")
    names = sorted(expected) if expected is not None else None
    try:
        current = model_file_stats(row["model"], names, in_place=in_place, optional=optional)
    except NameSetError as error:
        raise ValueError(_changed_message(row, error.missing + error.extra)) from None
    recorded, hashes = receipt.get("file_stats") or {}, receipt.get("files") or {}
    if names is None and set(hashes) != set(current):
        raise ValueError(_changed_message(row, sorted(set(hashes) ^ set(current))))
    changed = {name: value for name, value in current.items()
               if not POSIX_STATS or recorded.get(name) != value or name not in hashes}
    measured = {name: hashes[name] for name in current if name not in changed}
    if changed:
        measured.update(_hash_files(plain(row["model"]), changed))
        after = model_file_stats(row["model"], names, in_place=in_place, optional=optional)
        if any(after[name] != value for name, value in changed.items()):
            raise ValueError("Checkpoint changed during checksum verification")
    differing = sorted(name for name in current if hashes.get(name) != measured[name])
    if differing:
        raise ValueError(_changed_message(row, differing))
    _check_pins(card, measured, expected)
    if changed and POSIX_STATS:
        receipt = {**receipt, "file_stats": {**recorded, **changed}}
        if refresh:
            deploy_engine.save_receipt(Path(receipt_path), receipt)
            remember_checkpoint(receipt)
    return receipt


def _receipt_location(path, deployment, kind):
    """``path`` when it is a receipt ``refresh_receipts`` may rewrite, else ``None``."""
    if not isinstance(path, str) or not path.startswith("/") or posixpath.normpath(path) != path:
        return None
    location = plain(path)
    info = os.lstat(location)
    if not stat.S_ISREG(info.st_mode):
        return None
    if kind == "record":
        return location if location.parent == CHECKPOINTS else None
    if location.name != "model.json" or location.parent.name != "installer":
        return None
    owner = profiles.read_json(location.parent.parent / ".installer-owner.json")
    if (not isinstance(owner, dict) or set(owner) != {"deployment"} or not isinstance(owner["deployment"], str)
            or (deployment is not None and owner["deployment"] != deployment)):
        return None
    return location


def refresh_receipts(verified, paths, *, exclude=()):
    """Refresh other SparkRing receipts that record inodes SparkRing just verified and linked.

    Linking a file changes its change time, which would make every receipt that
    records that inode hash the file again at each later verification.
    ``verified`` lists ``{"identity", "sha256", "before", "after"}`` per linked
    inode, where ``before`` are the five stats measured before hashing.
    ``paths`` are the other deployments' ``<workspace>/installer/model.json``
    files on this host (a path, or ``{"path", "deployment"}``); each must be a
    regular non-symlink file in a workspace whose ``.installer-owner.json``
    names that deployment, or any deployment for a plain path. The path records
    in ``CHECKPOINTS`` are examined too.

    An entry is updated only when its recorded device and inode are a verified
    inode's, its hash equals that inode's verified SHA-256, its recorded size
    and modification time equal the values measured before hashing, and the
    receipt's file still names that inode with exactly the five stats
    ``after`` measured right after SparkRing's own link; a change made after
    the link, even one that kept the modification time, leaves the entry as it
    was. Its stats become the current ``lstat``; nothing else changes. Returns
    the paths of the receipts that were rewritten.
    """
    inodes = {tuple(item["identity"]): item for item in verified}
    if not inodes or not POSIX_STATS:
        return []
    excluded = {os.path.normpath(str(item)) for item in exclude}
    candidates = []
    for item in paths:
        if isinstance(item, dict):
            candidates.append((item.get("path"), item.get("deployment"), "receipt"))
        else:
            candidates.append((item, None, "receipt"))
    if CHECKPOINTS.is_dir() and not CHECKPOINTS.is_symlink():
        candidates.extend((str(record), None, "record") for record in sorted(CHECKPOINTS.glob("*.json")))
    refreshed, seen = [], set()
    for path, deployment, kind in candidates:
        if not isinstance(path, str) or os.path.normpath(path) in excluded or path in seen:
            continue
        seen.add(path)
        try:
            location = _receipt_location(path, deployment, kind)
            if location is None:
                continue
            receipt = profiles.read_json(location)
            if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
                continue
            stats, hashes = receipt.get("file_stats"), receipt.get("files")
            if not isinstance(stats, dict) or not isinstance(hashes, dict):
                continue
            updated = dict(stats)
            for name, recorded in stats.items():
                if not (isinstance(recorded, list) and len(recorded) == 5
                        and all(type(value) is int for value in recorded)):
                    continue
                item = inodes.get((recorded[0], recorded[1]))
                if item is None or hashes.get(name) != item["sha256"] or recorded[2:4] != item["before"][2:4]:
                    continue
                place.safe_name(name)
                current = os.lstat(os.path.join(receipt["path"], *name.split("/")))
                if stat.S_ISREG(current.st_mode) and place.stats(current) == list(item["after"]):
                    updated[name] = place.stats(current)
            if updated != stats:
                deploy_engine.save_receipt(location, {**receipt, "file_stats": updated})
                refreshed.append(str(location))
        except (OSError, ValueError):
            continue
    return refreshed


# SparkRing checkpoint directories ------------------------------------------------

def _owned(row, number):
    """The owned checkpoint directory of ``row``; a copy served in place is never written."""
    if row.get("reuse_verified_model"):
        raise ValueError(f"Node {number} serves {row['model']} in place; SparkRing never writes into a copy "
                         "it did not create")
    return plain(row["model"])


def _claim(card, model):
    return place.claim(str(model), card["model_repository"], card["model_revision"])


def _settle(claimed, journal, required):
    """Finish interrupted placements, refuse names SparkRing did not place, and check every placed name.

    A placed name whose five stats equal the journal is verified; any other is
    hashed through a descriptor, and on a mismatch SparkRing unlinks its own
    name. Returns ``{name: outcome}`` for the names that were checked.
    """
    place.journal_recover(claimed.dir_fd, journal)
    unexpected = place.unexpected_names(claimed.dir_fd, journal, required)
    if unexpected:
        raise ValueError(f"{claimed.path} holds files SparkRing did not place: {_names(unexpected)}. SparkRing "
                         "does not remove them; move them away, then repeat the command.")
    outcomes = {}
    for name in sorted(journal.files):
        if journal.files[name]["state"] == "placed" and name in required:
            pin = required[name]
            outcomes[name] = place.check_placed(claimed.dir_fd, name, journal, pin["size"], pin["sha256"])
    return outcomes


def _placed(claimed, journal, required):
    """``{name: stats}`` of required names that the journal records with their current stats and pin."""
    files, _ = place.listing(claimed.dir_fd)
    result = {}
    for name, pin in required.items():
        entry, info = journal.files.get(name), files.get(name)
        if (entry is not None and info is not None and entry["state"] == "placed" and stat.S_ISREG(info.st_mode)
                and place.stats(info) == entry["stats"] and entry["sha256"] == pin["sha256"]):
            result[name] = entry["stats"]
    return result


def _existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def _require_space(model, row, sizes, *, number):
    """Refuse to write files of ``sizes`` bytes into ``model`` unless its filesystem has room.

    The formula is the plan's (``checkpoint_plan.required_space``): the bytes written, the largest
    file once more for its staging copy, and the cache allowance when the cluster cache shares the
    filesystem. Linked and present files write nothing.
    """
    sizes = list(sizes)
    if not sizes:
        return
    from runtime.host import checkpoint_plan
    policy = checkpoint_plan.storage_policy(installer.ROOT)
    target = _existing(model)
    shared = os.stat(_existing(row["cache"])).st_dev == os.stat(target).st_dev
    cache = policy["cache_bytes"] if shared else 0
    need = checkpoint_plan.required_space(sizes, cache_bytes=cache)
    free = shutil.disk_usage(target).free
    if free < need:
        parts = [f"{sum(sizes) / GIB:.1f} GiB", f"{max(sizes) / GIB:.1f} GiB for the largest file"]
        if cache:
            parts.append(f"{cache / GIB:.0f} GiB for the compile cache")
        raise ValueError(f"Node {number} needs {need / GIB:.1f} GiB free on the filesystem of {model} to write "
                         f"checkpoint files ({', '.join(parts)}); {free / GIB:.1f} GiB is free. Free space, then "
                         "repeat sudo sparkring install. The running model has not been stopped.")


def _finish(lock, row, state, claimed, journal, required):
    """Write the receipt and path record when the directory holds exactly the verified required files."""
    card = lock["selection"]
    placed = _placed(claimed, journal, required)
    if set(placed) != set(required) or place.unexpected_names(claimed.dir_fd, journal, required):
        return None
    entries = {name: journal.files[name] for name in sorted(required)}
    origins = {entry["origin"] for entry in entries.values()}
    origin = ("pinned-hub-download" if origins == {"hub"} else
              "verified-fabric-copy" if origins <= {"fabric", "rsync"} else "adopted-local-copy")
    files = {name: entry["sha256"] for name, entry in entries.items()}
    folders = {}
    for entry in entries.values():
        if entry["origin"] in ("link", "copy") and entry.get("source"):
            folder = posixpath.dirname(entry["source"])
            folders[folder] = folders.get(folder, 0) + 1
    receipt = {"repository": card["model_repository"], "revision": card["model_revision"], "path": row["model"],
               "files": files, "file_stats": placed, "origin": origin, "scope": "required-files",
               "sources": [{"path": path, "files": count} for path, count in sorted(folders.items())],
               "fetched": sorted(name for name, entry in entries.items() if entry["origin"] == "hub")}
    _check_pins(card, files, required)
    claimed.check()
    deploy_engine.save_receipt(state / "model.json", receipt)
    remember_checkpoint(receipt)
    if not lock["backend"].startswith("glm-"):
        plain(row["cache"]).mkdir(parents=True, exist_ok=True)
    return receipt


def _adoption(document, required):
    """Validated ``model-adopt`` input: ``(files, receipts, tolerance_bytes)``."""
    if (not isinstance(document, dict) or not isinstance(document.get("files"), dict)
            or set(document) - {"files", "receipts", "tolerance_bytes"}):
        raise ValueError("model-adopt expects {files, receipts, tolerance_bytes}")
    files = {}
    for name, item in document["files"].items():
        if name not in required:
            raise ValueError(f"model-adopt names {str(name)[:200]!r}, which is not a required file of the pinned revision")
        if not isinstance(item, dict) or item.get("action") not in ("present", "link", "copy"):
            raise ValueError(f"model-adopt has no valid action for {name}")
        if item.get("size", required[name]["size"]) != required[name]["size"]:
            raise ValueError(f"model-adopt gives {name} another size than its pin")
        source, inode = item.get("source"), item.get("identity")
        if item["action"] != "present":
            if (not isinstance(source, str) or not source.startswith("/") or "\0" in source
                    or posixpath.normpath(source) != source):
                raise ValueError(f"model-adopt needs an absolute, normalized source for {name}")
            if inode is not None and not (isinstance(inode, list) and len(inode) == 2
                                          and all(type(value) is int and value >= 0 for value in inode)):
                raise ValueError(f"model-adopt has an invalid source identity for {name}")
        files[name] = {"action": item["action"], "source": source, "identity": inode}
    receipts = document.get("receipts") or []
    tolerance = document.get("tolerance_bytes", GIB)
    if not isinstance(receipts, list) or type(tolerance) is not int or tolerance < 0:
        raise ValueError("model-adopt expects a receipt list and a non-negative tolerance")
    return files, receipts, tolerance


def _source_path(source):
    """``source`` after checking, from the root down, that none of its components is a symlink."""
    current = "/"
    for part in source.strip("/").split("/"):
        current = posixpath.join(current, part)
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError(f"{source} is no longer the file the plan identified (a component is a symlink)")
    return source


def _copy_into(claimed, journal, receive, fd, name, pin, source):
    """Copy the open source ``fd`` into staging while hashing it; place it when it matches the pin."""
    part = name + ".part"
    output, digest = place.copy_part(fd, pin["size"], receive, part)
    try:
        if digest != pin["sha256"]:
            place.discard_part(receive, part, output)
            output = None
            return digest, None
        placed = place.place_staged(claimed.dir_fd, receive, part, name, journal, fd=output, sha256=digest,
                                    origin="copy", source=source)
        return digest, placed
    finally:
        if output is not None:
            os.close(output)


def _open_and_hash(item, pin):
    """``(fd, fstat before hashing, sha256)`` of a planned source."""
    fd, before = place.open_source(_source_path(item["source"]), item["identity"], pin["size"])
    try:
        return fd, before, place.hash_descriptor(fd, pin["size"])
    except BaseException:
        os.close(fd)
        raise


def _hash_sources(names, files, required):
    """Open and hash planned sources concurrently: ``({name: (fd, before, sha256)}, {name: error})``.

    A source that cannot be opened as the planned regular file, or whose length
    differs from its pin, is reported with its error. The caller closes every
    returned descriptor.
    """
    opened, failed, fatal = {}, {}, None
    if not names:
        return opened, failed
    workers = max(1, min(16, os.cpu_count() or 4, len(names)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(_open_and_hash, files[name], required[name]) for name in names}
    for name, future in futures.items():
        error = future.exception()
        if error is None:
            opened[name] = future.result()
        elif isinstance(error, (OSError, ValueError)):
            failed[name] = error
        else:
            fatal = fatal or error
    if fatal is not None:
        for fd, _, _ in opened.values():
            os.close(fd)
        raise fatal
    return opened, failed


def adopt_model(lock, row, state, document, *, number):
    """``model-adopt``: place this Spark's planned local copies into SparkRing's checkpoint directory.

    ``document`` is the Spark's plan entry, ``{"files": {name: {"action":
    "present" | "link" | "copy", "source", "identity", "size"}}, "receipts":
    [...], "tolerance_bytes": N}``. Existing names are settled first. Weight
    sources are then opened, hashed through their descriptors and hard-linked
    from those descriptors in sorted order; a source whose content differs is
    never linked and gets a per-inode record. ``EXDEV`` or ``EPERM`` turns a
    link into a copy while the unplanned bytes stay within ``tolerance_bytes``;
    beyond that the file stays missing and is listed in ``link_failures``.
    Other files are copied through the ``receive`` staging directory, verified
    and linked into place. Afterwards the receipts that record a linked inode
    are refreshed, and when the directory is complete the receipt is written.
    """
    card = lock["selection"]
    required, _ = _pinned(card)
    files, receipts, tolerance = _adoption(document, required)
    model = _owned(row, number)
    result = {"complete": False, "verified": {}, "missing": [], "differs": [], "linked": [], "copied": [],
              "bytes_written": 0, "refreshed": [], "link_failures": [], "unavailable": []}
    with _claim(card, model) as claimed:
        journal = place.journal_load(claimed)
        receive = place.staging(claimed, "receive", empty=True)
        try:
            _settle(claimed, journal, required)
            present = set(_placed(claimed, journal, required))
            links = sorted(n for n, item in files.items() if item["action"] == "link" and n not in present)
            copies = sorted(n for n, item in files.items() if item["action"] == "copy" and n not in present)
            planned = {name: required[name]["size"] for name in copies}
            linked, unplanned = [], 0

            def unavailable(name, error):
                claimed.check()
                result["unavailable"].append({"name": name, "source": files[name]["source"], "error": str(error)[:500]})

            # Sources are hashed concurrently and linked one by one in sorted order.
            opened, failed = _hash_sources(links, files, required)
            try:
                for name in links:
                    if name in failed:
                        unavailable(name, failed[name])
                        continue
                    fd, before, digest = opened[name]
                    pin, source = required[name], files[name]["source"]
                    if digest != pin["sha256"]:
                        _record_inode(before, digest, source)
                        result["differs"].append({"name": name, "source": source})
                        continue
                    try:
                        after = place.place_link(claimed.dir_fd, fd, name, journal, sha256=digest, before=before,
                                                 source=source)
                    except OSError as error:
                        if error.errno not in (errno.EXDEV, errno.EPERM):
                            raise
                        reason = errno.errorcode[error.errno]
                        if unplanned + pin["size"] > tolerance:
                            result["link_failures"].append({"name": name, "reason": reason, "source": source})
                            continue
                        _require_space(model, row, [pin["size"], *planned.values()], number=number)
                        copied, placed = _copy_into(claimed, journal, receive, fd, name, pin, source)
                        if placed is None:
                            # The source changed after it was hashed; its current content is recorded.
                            _record_inode(before, copied, source)
                            result["differs"].append({"name": name, "source": source})
                            continue
                        unplanned += pin["size"]
                        result["bytes_written"] += pin["size"]
                        result["copied"].append(name)
                        continue
                    except ValueError as error:
                        unavailable(name, error)
                        continue
                    linked.append({"identity": place.identity(before), "sha256": digest,
                                   "before": place.stats(before), "after": after})
                    result["linked"].append(name)
            finally:
                for fd, _, _ in opened.values():
                    os.close(fd)
            _require_space(model, row, planned.values(), number=number)
            for name in copies:
                pin, source = required[name], files[name]["source"]
                try:
                    fd, before = place.open_source(_source_path(source), files[name]["identity"], pin["size"])
                except (OSError, ValueError) as error:
                    unavailable(name, error)
                    planned.pop(name)
                    continue
                try:
                    digest, placed = _copy_into(claimed, journal, receive, fd, name, pin, source)
                except (OSError, ValueError) as error:
                    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                        raise
                    unavailable(name, error)
                    planned.pop(name)
                    continue
                finally:
                    os.close(fd)
                planned.pop(name)
                if placed is None:
                    _record_inode(before, digest, source)
                    result["differs"].append({"name": name, "source": source})
                    continue
                result["bytes_written"] += pin["size"]
                result["copied"].append(name)
        finally:
            os.close(receive)
        # Every copy is placed or discarded, so the staging directory holds only its marker.
        place.remove_staging(claimed, "receive")
        own = [state / "model.json", _checkpoint_record(model)]
        result["refreshed"] = refresh_receipts(linked, receipts, exclude=own)
        result["complete"] = _finish(lock, row, state, claimed, journal, required) is not None
        result["verified"] = _placed(claimed, journal, required)
        result["missing"] = sorted(set(required) - set(result["verified"]))
    return result


def fetch_container(model):
    """Name of the container that downloads checkpoint files for directory ``model``."""
    return "sparkring-fetch-" + hashlib.sha256(str(model).encode()).hexdigest()[:16]


def _same_directory(path, fd):
    """Whether ``path``, resolved as Docker and rsync resolve it, is the directory open at ``fd``.

    Docker's bind mount and rsync's destination name a staging directory by
    path; ``claim`` keeps other accounts from replacing it, and this check
    detects a replacement that happened anyway.
    """
    try:
        info = os.stat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and place.identity(info) == place.identity(os.fstat(fd))


def _place_fetched(claimed, journal, fetch, name, pin):
    """Place downloaded file ``fetch/<name>`` when it is a regular file with the pinned content.

    Returns ``placed``, ``absent`` or ``differs``; a differing file is left in
    staging and never placed.
    """
    try:
        fd, before = place.open_source(name, None, pin["size"], dir_fd=fetch)
    except FileNotFoundError:
        return "absent"
    except ValueError:
        return "differs"
    try:
        if place.hash_descriptor(fd, pin["size"]) != pin["sha256"]:
            return "differs"
        place.place_staged(claimed.dir_fd, fetch, name, name, journal, fd=fd, sha256=pin["sha256"], origin="hub",
                           before=before)
        return "placed"
    finally:
        os.close(fd)


def _drop_links(fetch, required):
    """Unlink staged names that another name links, so the download client never writes through a link."""
    files, _ = place.listing(fetch)
    for name in required:
        info = files.get(name)
        if info is not None and info.st_nlink > 1:
            place.discard_part(fetch, name)


def _require_image(card):
    if run(["docker", "image", "inspect", card["image_id"]], check=False).returncode:
        raise ValueError(f"The pinned image {card['image_id']} is not on this Spark. SparkRing downloads checkpoint "
                         "files with that image's Hugging Face client; prepare the image first (sudo sparkring "
                         "install distributes it), then repeat the command.")


def fetch_model(lock, row, state, names, *, number):
    """``model-fetch``: download pinned files that the checkpoint directory lacks, then place them.

    Every name must be a required file that is absent from the directory. The
    serving image's Hugging Face client writes only into the marked ``fetch``
    staging directory, which is the container's only mount and never holds a
    link. Docker names that directory by path, so its identity is compared with
    the staging descriptor just before and after the container runs. Each
    downloaded file is hashed and linked into place, never replacing a name; a
    file whose content differs is never placed. Files that an interrupted run
    completed are placed without downloading them again. A failed download is
    reported with the client's last error line.
    """
    card = lock["selection"]
    required, _ = _pinned(card)
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("model-fetch expects a list of distinct file names")
    unpinned = sorted(name for name in names if name not in required)
    if unpinned:
        raise ValueError("model-fetch downloads only required files of the pinned revision, not " + _names(unpinned))
    model = _owned(row, number)
    result = {"fetched": [], "resumed": [], "bytes_written": 0, "complete": False}
    with _claim(card, model) as claimed:
        journal = place.journal_load(claimed)
        _settle(claimed, journal, required)
        present = sorted(set(names) & set(place.listing(claimed.dir_fd)[0]))
        if present:
            raise ValueError(f"{model} already holds {_names(present)}; model-fetch downloads only missing files")
        container = fetch_container(model)
        if names and run(["docker", "container", "inspect", container], check=False).returncode == 0:
            raise ValueError(f"A checkpoint download into {model} is still in progress in container {container}. "
                             f"Wait until sudo docker ps no longer lists it, or stop it with sudo docker rm --force "
                             f"{container}, then repeat sudo sparkring install.")
        if names:
            fetch = place.staging(claimed, "fetch")
            try:
                _drop_links(fetch, required)
                pending, stale = [], False
                for name in sorted(names):
                    outcome = _place_fetched(claimed, journal, fetch, name, required[name])
                    if outcome == "placed":
                        result["resumed"].append(name)
                    else:
                        pending.append(name)
                        stale |= outcome == "differs"
                if stale:
                    # A differing file would be kept by the client's own completion
                    # records, so the whole staging directory starts again.
                    os.close(fetch)
                    fetch = place.staging(claimed, "fetch", empty=True)
                if pending:
                    _require_space(model, row, [required[name]["size"] for name in pending], number=number)
                    _require_image(card)
                    where = f"{claimed.state}/fetch"
                    what = _names(pending) if len(pending) <= 3 else f"{len(pending)} files"
                    claimed.check()
                    if not _same_directory(where, fetch):
                        raise ValueError(f"{where} is no longer SparkRing's staging directory; nothing was "
                                         "downloaded. Check who can change the directories above " + str(model))
                    try:
                        run(["docker", "run", "--rm", "--name", container, "--pull", "never", "--runtime", "runc",
                             "--user", "0:0", "--env", "HF_HOME=/tmp/huggingface", "--env",
                             "HF_HUB_DISABLE_TELEMETRY=1", "--mount", f"type=bind,src={where},dst=/fetch",
                             "--entrypoint", "python3" if "image_runtime" in lock else "/opt/venv/bin/python",
                             card["image_id"], "-c", FETCH_CODE, card["model_repository"], card["model_revision"],
                             *pending], timeout=FETCH_SECONDS)
                    except CommandError as error:
                        cause = (error.lines(1) or [f"docker run exited with status {error.returncode}"])[0]
                        raise ValueError(f"Downloading {what} from huggingface.co failed: {cause[:300]}. "
                                         "Nothing was placed for those files; files the download completed are "
                                         f"placed by the next run. Check that Node {number} reaches huggingface.co, "
                                         "then repeat the command.") from None
                    except subprocess.TimeoutExpired:
                        raise ValueError(f"Downloading {what} from huggingface.co did not finish within "
                                         f"{FETCH_SECONDS // 3600} hours. Nothing was placed for those files; files the "
                                         "download completed are placed by the next run. Wait until sudo docker ps "
                                         f"no longer lists {container}, then repeat the command.") from None
                    if not _same_directory(where, fetch):
                        raise ValueError(f"{where} was replaced while checkpoint files were downloaded; nothing was "
                                         "placed. Check who can change the directories above " + str(model))
                    wrong = []
                    for name in pending:
                        if _place_fetched(claimed, journal, fetch, name, required[name]) == "placed":
                            result["fetched"].append(name)
                            result["bytes_written"] += required[name]["size"]
                        else:
                            wrong.append(name)
                    if wrong:
                        os.close(fetch)
                        fetch = None
                        place.remove_staging(claimed, "fetch")
                        raise ValueError(f"huggingface.co served different bytes for {_names(wrong)}; nothing was "
                                         "placed for them. Repeat sudo sparkring install to download them again.")
            finally:
                if fetch is not None:
                    os.close(fetch)
            place.remove_staging(claimed, "fetch")
        result["complete"] = _finish(lock, row, state, claimed, journal, required) is not None
    return result


def prepare_owned(lock, row, state, *, number):
    """Complete an owned checkpoint directory on this Spark: settle what it holds, then download the rest.

    The lower-level ``sparkring up`` flows use this; ``sparkring install``
    orchestrates adoption, peers and downloads across Sparks instead.
    """
    result = adopt_model(lock, row, state, {"files": {}, "receipts": [], "tolerance_bytes": 0}, number=number)
    if not result["complete"]:
        fetch_model(lock, row, state, result["missing"], number=number)


def _in_place_message(model, *, differs=(), missing=(), extra=()):
    parts = []
    if differs:
        parts.append("differs from the pinned revision in " + _names(differs))
    if missing:
        parts.append("lacks " + _names(missing))
    if extra:
        parts.append("also holds " + _names(extra) + ", which the serving engine would load")
    return (f"Checkpoint at {model} " + " and ".join(parts) + "; SparkRing does not change it. A named folder is "
            "served in place only when it holds exactly the pinned files: install without naming it so that "
            "SparkRing assembles its own copy, or change the folder yourself.")


def verify_in_place(lock, row, state, *, number):
    """Verify a named copy for serving in place and write its receipt; nothing under it is written."""
    card = lock["selection"]
    model = plain(row["model"])
    if not model.exists():
        raise ValueError(f"The copy named for Node {number} ({model}) does not exist; SparkRing does not create it")
    if not model.is_dir():
        raise ValueError(f"The copy named for Node {number} ({model}) is not a directory")
    found = checkpoint_files(card)
    required, optional = found if found is not None else (None, ())
    names = sorted(required) if required is not None else None
    try:
        current = model_file_stats(model, names, in_place=True, optional=optional)
    except NameSetError as error:
        raise ValueError(_in_place_message(model, missing=error.missing, extra=error.extra)) from None
    # A path record whose stats all still match stands for an earlier full read of these files.
    known = remembered_checkpoint(card, model, current)
    if known is not None:
        hashes = known
    elif names is None:
        hashes = model_files(model)
    else:
        hashes = _hash_files(model, current)
    if model_file_stats(model, names, in_place=True, optional=optional) != current:
        raise ValueError("Checkpoint changed while preparing its checksum receipt")
    differs = (sorted(n for n in names if hashes[n] != required[n]["sha256"]) if names is not None
               else pinned_differences(card["profile"], hashes, card["model_revision"]))
    if differs:
        raise ValueError(_in_place_message(model, differs=differs))
    receipt = {"repository": card["model_repository"], "revision": card["model_revision"], "path": row["model"],
               "files": hashes, "file_stats": current, "origin": "in-place-verified-copy"}
    if names is not None:
        receipt["scope"] = "required-files"
    _check_pins(card, hashes, required)
    deploy_engine.save_receipt(state / "model.json", receipt)
    remember_checkpoint(receipt)
    if not lock["backend"].startswith("glm-"):
        plain(row["cache"]).mkdir(parents=True, exist_ok=True)
    return receipt


def model_settled(lock, row, receipt_path, *, number):
    """``model-settled``: refuse when checkpoint files changed after ``start`` verified them.

    Runs after the smoke request, when every rank has finished loading. The
    five stat values of every required file must still equal the receipt; a
    difference fails the switch, so the previous deployment is restored.
    """
    card = lock["selection"]
    receipt = profiles.read_json(receipt_path)
    found = checkpoint_files(card)
    required, optional = found if found is not None else (None, ())
    names = sorted(required) if required is not None else None
    recorded = receipt.get("file_stats") or {}
    try:
        current = model_file_stats(row["model"], names, in_place=bool(row.get("reuse_verified_model")),
                                   optional=optional)
    except NameSetError as error:
        changed = error.missing + error.extra
    else:
        changed = sorted(name for name in set(current) | (set(recorded) if names is None else set())
                         if recorded.get(name) != current.get(name))
    if changed:
        raise ValueError(f"Checkpoint files changed while the model was loading on Node {number} ({_names(changed)})")
    return {"ok": True}


# Transfers between Sparks -----------------------------------------------------------

def _transfer_manifest(document, card, required):
    """A validated transfer manifest whose names are required files with their pinned hashes and sizes."""
    if (not isinstance(document, dict) or set(document) != {"repository", "revision", "files", "sizes"}
            or document["repository"] != card["model_repository"] or document["revision"] != card["model_revision"]
            or not isinstance(document["files"], dict) or not isinstance(document["sizes"], dict)
            or set(document["files"]) != set(document["sizes"])):
        raise ValueError("Checkpoint transfer identity differs")
    for name, digest in document["files"].items():
        try:
            place.safe_name(name)
        except ValueError:
            raise ValueError("Unsafe checkpoint transfer manifest") from None
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(document["sizes"][name]) is not int or document["sizes"][name] < 0):
            raise ValueError("Unsafe checkpoint transfer manifest")
        if name not in required or (digest, document["sizes"][name]) != (required[name]["sha256"], required[name]["size"]):
            raise ValueError(f"Checkpoint transfer names {name}, which is not a required file with its pinned hash "
                             "and size")
    return document


def _rsync_staging(claimed):
    return posixpath.join(claimed.state, "receive", "rsync")


def _place_rsync(claimed, journal, receive, manifest, required):
    """Place files that rsync delivered into ``receive/rsync``; returns names whose content differed.

    rsync names its destination by path, so that path must still resolve to
    the staging directory opened here; otherwise nothing is placed from it.
    """
    try:
        staged = os.open("rsync", place.DIRECTORY_FLAGS, dir_fd=receive)
    except FileNotFoundError:
        return []
    wrong = []
    try:
        if not _same_directory(_rsync_staging(claimed), staged):
            raise ValueError(f"{_rsync_staging(claimed)} was replaced while rsync wrote to it; nothing was placed "
                             f"from it. Check who can change the directories above {claimed.path}")
        present = place.listing(staged)[0]
        placed = _placed(claimed, journal, required)
        for name in sorted(manifest["files"]):
            if name not in present or name in placed:
                continue
            pin = required[name]
            try:
                fd, before = place.open_source(name, None, pin["size"], dir_fd=staged)
            except (OSError, ValueError):
                wrong.append(name)
                continue
            try:
                digest = place.hash_descriptor(fd, pin["size"])
                if digest != pin["sha256"]:
                    wrong.append(name)
                    continue
                place.place_staged(claimed.dir_fd, staged, name, name, journal, fd=fd, sha256=digest,
                                   origin="rsync", before=before)
            finally:
                os.close(fd)
    finally:
        os.close(staged)
    return wrong


def transfer_model(operation, lock, row, state):
    """Checkpoint operations between Sparks and between deployments on one Spark.

    - ``model-transfer-manifest`` (donor): the verified hashes and sizes of the
      required names only, also for a copy served in place.
    - ``model-transfer-prepare`` (receiver): claim the owned directory, settle
      its names, empty the ``receive`` staging directory with an empty
      ``receive/rsync`` inside it, and check free space for the names still
      needed. Returns ``needed`` and the ``rsync`` destination path.
    - ``model-transfer-complete`` (receiver): place what rsync delivered while
      its destination path still resolves to the staging directory, check
      every placed name from the journal, and write the receipt when the
      directory is complete.
    - ``model-reuse-receipt``: copy an earlier deployment's verified receipt
      for the same path.
    """
    number = row.get("rank", "?")
    receipt_path = plain(state / "model.json")
    card = lock["selection"]
    if operation == "model-transfer-manifest":
        found = checkpoint_files(card)
        receipt = verify_model(lock, row, receipt_path, refresh=True)
        names = sorted(found[0]) if found is not None else sorted(receipt["files"])
        return {"repository": receipt["repository"], "revision": receipt["revision"],
                "files": {name: receipt["files"][name] for name in names},
                "sizes": {name: receipt["file_stats"][name][2] for name in names}}
    if operation == "model-reuse-receipt":
        previous = json.load(sys.stdin)
        source = plain(previous["workspace"])
        if profiles.read_json(source / ".installer-owner.json") != {"deployment": previous["deployment"]}:
            raise ValueError("Previous checkpoint receipt belongs to another deployment")
        saved = plain(source / "installer/model.json")
        if receipt_path.exists() or not saved.is_file():
            return {"reused": False}
        receipt = profiles.read_json(saved)
        if (receipt.get("repository"), receipt.get("revision"), receipt.get("path")) != (card["model_repository"], card["model_revision"], row["model"]):
            return {"reused": False}
        # The earlier deployment's receipt is read, never rewritten; this
        # deployment saves its own copy with any re-measured stats.
        receipt = verify_model(lock, row, saved)
        state.mkdir(parents=True, exist_ok=True)
        deploy_engine.save_receipt(receipt_path, receipt)
        if not lock["backend"].startswith("glm-"):
            plain(row["cache"]).mkdir(parents=True, exist_ok=True)
        return {"reused": True}
    if operation not in ("model-transfer-prepare", "model-transfer-complete"):
        raise ValueError("Unknown checkpoint transfer operation: " + operation)
    model = _owned(row, number)
    required, _ = _pinned(card)
    manifest = _transfer_manifest(json.load(sys.stdin), card, required)
    with _claim(card, model) as claimed:
        journal = place.journal_load(claimed)
        if operation == "model-transfer-prepare":
            receive = place.staging(claimed, "receive", empty=True)
            try:
                _settle(claimed, journal, required)
                placed = _placed(claimed, journal, required)
                needed = sorted(name for name in manifest["files"] if name not in placed)
                _require_space(model, row, [required[name]["size"] for name in needed], number=number)
                os.mkdir("rsync", 0o700, dir_fd=receive)
                staged = os.open("rsync", place.DIRECTORY_FLAGS, dir_fd=receive)
                try:
                    if not _same_directory(_rsync_staging(claimed), staged):
                        raise ValueError(f"{_rsync_staging(claimed)} does not resolve to SparkRing's staging "
                                         f"directory. Check who can change the directories above {model}")
                finally:
                    os.close(staged)
            finally:
                os.close(receive)
            return {"ok": True, "needed": needed, "rsync": _rsync_staging(claimed)}
        receive = place.staging(claimed, "receive")
        try:
            _settle(claimed, journal, required)
            wrong = _place_rsync(claimed, journal, receive, manifest, required)
        finally:
            os.close(receive)
        place.remove_staging(claimed, "receive")
        placed = _placed(claimed, journal, required)
        absent = sorted(name for name in manifest["files"] if name not in placed)
        if absent:
            detail = f"; rsync delivered different bytes for {_names(wrong)}" if wrong else ""
            raise ValueError(f"Transferred checkpoint differs from its verified source ({_names(absent)} not placed"
                             f"{detail}); repeat sudo sparkring install to resume")
        complete = _finish(lock, row, state, claimed, journal, required) is not None
        return {"ok": True, "complete": complete, "missing": sorted(set(required) - set(placed))}


def container(spec):
    names = run(["docker", "container", "ls", "--all", "--format", "{{.Names}}" ]).stdout.splitlines()
    return json.loads(run(["docker", "inspect", spec.name]).stdout)[0] if spec.name in names else None


def owned(spec, info, image):
    if info is None:
        raise ValueError("Deployment container is absent")
    expected = expected_inspection(spec, image, backend="compose")
    if "io.sparkring.image-lock" in spec.labels:
        from runtime.common import loader_policy
        expected["host_config"]["SecurityOpt"] = [loader_policy.inspection_option() if option.startswith("seccomp=") else option
                                                  for option in spec.security_opt]
    config, host = info["Config"], info["HostConfig"]
    host = dict(host)
    # Docker uses null for absent optional lists. NVIDIA-backed containers can
    # also receive label=disable on hosts where SELinux is not an active policy.
    for field in ("CapAdd", "SecurityOpt"):
        if host.get(field) is None and expected["host_config"][field] == []:
            host[field] = []
    if "label=disable" in host.get("SecurityOpt", []) and "label=disable" not in expected["host_config"]["SecurityOpt"]:
        options = json.loads(run(["docker", "info", "--format", "{{json .SecurityOptions}}"] ).stdout)
        if isinstance(options, list) and not any("selinux" in str(value).lower() for value in options):
            host["SecurityOpt"] = [option for option in host["SecurityOpt"] if option != "label=disable"]
    environment = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    mounts = {item["Destination"]: {k: item[k] for k in ("Source", "Type", "RW")} for item in info["Mounts"]}
    if (info["Image"] != spec.image_id or config.get("Cmd") != expected["cmd"]
            or config.get("Entrypoint") != expected["entrypoint"] or environment != expected["env"]
            or mounts != expected["mounts"] or not deploy_engine.matches(config.get("Labels", {}), expected["labels"])
            or config.get("Healthcheck") != expected["healthcheck"]
            or config.get("WorkingDir", "") != expected["working_dir"] or config.get("User", "") != expected["user"]
            or not deploy_engine.matches(host, expected["host_config"])
            or len(host.get("Devices") or []) != len(spec.devices)
            or len(host.get("DeviceRequests") or []) != 1):
        raise ValueError("Existing container differs from this locked deployment; refusing adoption")
    compose.check_project_containers(spec.name, owned_id=info["Id"], run=run)
    return info


def require_idle():
    if run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
        raise ValueError("GPU has a workload; stop only the intended workload before retrying")


def smoke_request(card):
    """Chat-template settings for the smoke request, owned by the serving profile.

    Profiles whose template always reasons (GLM-5.3) request low effort instead
    of disabling thinking, which would merge reasoning into the answer text.
    """
    source = profiles.load(card["profile"])[0]["configuration"]
    if source["format"] == "serving-profile":
        config = profiles.read_json(profiles.local_path(source["path"]))
        if "smoke" in config:
            return config["smoke"]
    return {"chat_template_kwargs": {"enable_thinking": False}}


def http_json(port, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        return json.loads(payload) if payload.strip() else {}


def runtime_binding(lock, row, info, *, root="/"):
    """Installer assertions for the exact stopped/inspected container, not attestation."""
    from runtime.host import node
    identity = node.observation_identity(root=root)
    node_id = identity["node_id"]
    if node_id is None or row.get("node_id", node_id) != node_id:
        raise ValueError("Runtime binding requires this host's expected persistent node identity")
    labels = info.get("Config", {}).get("Labels", {})
    if (not re.fullmatch(r"[0-9a-f]{64}", info.get("Id", "")) or info.get("Image") != lock["selection"]["image_id"]
            or labels.get(compose.LABEL) != lock["id"] or labels.get("io.sparkring.rank") != str(row["rank"])):
        raise ValueError("Runtime binding requires the exact owned container and image")
    return {"schema": "sparkring-runtime-binding/v1", "deployment_id": lock["id"], "node_id": node_id,
            "container_id": info["Id"], "image_id": info["Image"], "rank": row["rank"]}


def local_binding_path(lock, row, *, root="/"):
    """Reject remote/unknown mounts before inspecting the binding source file."""
    from runtime.common import installer_image
    name = installer_image.binding_path(lock, row)
    path = Path(root) / name.lstrip("/")
    candidates = []
    for line in (Path(root) / "proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or len(before.split()) < 5 or not after.split():
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), before.split()[4]))
        if path.is_relative_to(mount):
            candidates.append((len(mount.parts), after.split()[0]))
    deepest = max((depth for depth, _ in candidates), default=-1)
    kinds = {kind for depth, kind in candidates if depth == deepest}
    if not kinds or not kinds <= {"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "tmpfs", "ramfs"}:
        raise ValueError("Runtime binding must use a verified local filesystem, not a NAS or unknown mount")
    # Check parents first, so a symlink is rejected before following it to a
    # deeper component that could live on a remote filesystem.
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise ValueError("Runtime binding path contains a symlink")
    return path


def read_runtime_binding(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
            raise ValueError("Runtime binding must be a small regular local file")
        raw = stream.read(16385)
        if len(raw) > 16384:
            raise ValueError("Runtime binding grew beyond its size limit")
    return json.loads(raw)


def check_runtime_binding(lock, row, info, *, root="/", finalize=False):
    from runtime.host import node
    expected = runtime_binding(lock, row, info, root=root)
    path = local_binding_path(lock, row, root=root)
    actual = read_runtime_binding(path)
    if actual == expected:
        return expected
    if not finalize or info.get("State", {}).get("Running"):
        raise ValueError("Runtime binding differs; finalize it while the owned container is stopped")
    node.save(path.parent, path.name, expected, mode=0o644)
    return expected


def model_observation(lock, row, info, *, root="/", now=time.time):
    """Allowlisted identities from the inspected container and this host only."""
    from runtime.host import node
    identity = node.observation_identity(root=root)
    expected_node = row.get("node_id")
    state = info.get("State", {}) if info else {}
    return {"schema": "sparkring-model-observation/v1", "source": "installer-docker-inspect", "observed_at": now(),
            "deployment_id": lock["id"], "rank": row["rank"], **identity,
            "expected_node_id": expected_node,
            "node_identity_matches": identity["node_id"] == expected_node if expected_node and identity["node_id"] else None,
            "container_id": info["Id"] if info else None, "container_name": info.get("Name", "").lstrip("/") if info else None,
            "container_started_at": state.get("StartedAt"), "image_id": info["Image"] if info else None,
            "expected_image_id": lock["selection"]["image_id"],
            "present": info is not None, "running": bool(state.get("Running")),
            "health": state.get("Health", {}).get("Status")}


def model_operation(operation, lock, number, row, state):
    """Route one checkpoint operation of rank ``number``; ``state`` is the deployment's ``installer`` directory.

    Owned rows without a receipt settle SparkRing's checkpoint directory and
    download what it lacks; rows served in place are verified only.
    """
    receipt = state / "model.json"
    if operation == "model-reuse-receipt" or operation.startswith("model-transfer-"):
        return transfer_model(operation, lock, row, state)
    if operation == "model-adopt":
        return adopt_model(lock, row, state, json.load(sys.stdin), number=number)
    if operation == "model-fetch":
        document = json.load(sys.stdin)
        if not isinstance(document, dict) or set(document) != {"names"}:
            raise ValueError("model-fetch expects {names}")
        return fetch_model(lock, row, state, document["names"], number=number)
    if operation == "model-settled":
        return model_settled(lock, row, receipt, number=number)
    if operation == "model-check" or receipt.exists():
        verify_model(lock, row, receipt, refresh=True)
    elif row.get("reuse_verified_model"):
        verify_in_place(lock, row, state, number=number)
    else:
        prepare_owned(lock, row, state, number=number)
    return {"ok": True}


def perform(operation, lock, number):
    installer.validate(lock)
    row = lock["site"]["ranks"][number]
    workspace = plain(lock["site"]["workspace"])
    owner = profiles.read_json(workspace / ".installer-owner.json")
    if owner != {"deployment": lock["id"]}:
        raise ValueError("Workspace belongs to another deployment")
    state = plain(workspace / "installer")
    if operation in ("image", "create") or operation in MODEL_OPERATIONS:
        state.mkdir(mode=0o700, exist_ok=True)
    card = lock["selection"]
    image_receipt = state / "image.json"
    model_receipt = state / "model.json"
    if operation in MODEL_OPERATIONS:
        return model_operation(operation, lock, number, row, state)
    if operation == "ring-serve":
        from runtime.host import native_mesh
        return native_mesh.serve_ring(row["fabric"], number, row["hcas"], row["gid"], row["host_ip"])
    if operation == "ring-check":
        from runtime.common import qwen_mesh
        qwen_mesh.check(row["fabric"], number, row["hcas"], row["gid"], row["host_ip"])
        return {"ok": True}
    if operation in ("gid-serve", "gid-check"):
        from runtime.host import roce_gid
        return (roce_gid.serve if operation == "gid-serve" else roce_gid.check)(row["hcas"], row["gid"])
    if operation.startswith("mesh-"):
        from runtime.host import native_mesh
        if operation == "mesh-prepare":
            return native_mesh.prepare_local(lock, number)
        if operation == "mesh-prepared":
            value = lock["site_input"]["native_mesh"]
            owner, _ = native_mesh.modules()
            owner.external_marker_attestation(binary=Path(value["site"]["marker_binary"]))
            return {"ok": True}
        if operation == "mesh-install-local":
            return native_mesh.install_local(lock, number, json.load(sys.stdin))
        return native_mesh.operate_local(lock, number, operation)

    if operation == "image":
        run(["docker", "info"])
        # A config-ID-only docker save/load yields an untagged image which the
        # default image listing can omit. Admission addresses the exact ID.
        existing = run(["docker", "image", "inspect", card["image_id"]], check=False)
        if existing.returncode:
            if card["image_reference"] == card["image_id"]:
                raise ValueError("Pinned local image is absent; preload " + card["image_id"] + " on every rank before up")
            docker_path = run(["docker", "info", "--format", "{{.DockerRootDir}}"]).stdout.strip()
            budget = setup.storage_plan(card, model_path=row["model"], cache_path=row["cache"],
                                        docker_path=docker_path,
                                        reuse_model=row["reuse_verified_model"] or model_receipt.exists())
            if not budget["passed"]:
                raise ValueError("Insufficient space for image/checkpoint/cache: " + json.dumps(budget["filesystems"]))
            run(["docker", "pull", "--platform", "linux/arm64", card["image_reference"]])
        receipt = admit_image(lock)
        deploy_engine.save_receipt(image_receipt, receipt)
        return {"ok": True}
    if operation == "image-check":
        if admit_image(lock) != profiles.read_json(image_receipt):
            raise ValueError("Image verification differs from the saved receipt")
        return {"ok": True}

    if lock["backend"] == "glm-existing-mesh":
        if operation == "receipt":
            return profiles.read_json(image_receipt)
        from runtime.host import glm_existing_mesh
        return glm_existing_mesh.perform(operation, lock, number, state)
    if lock["backend"] == "glm-managed":
        if operation == "receipt":
            return profiles.read_json(image_receipt)
        if operation == "smoke":
            connection = installer.connection(lock)
            name, port = connection["model"], connection["port"]
        else:
            raise ValueError("Managed GLM uses the existing host lifecycle coordinator")
    else:
        spec = installer.specifications(lock, only_rank=number)[0]
        image = image_info(lock)
        info = container(spec)
        if info:
            owned(spec, info, image)
        if operation == "status":
            return model_observation(lock, row, info)
        if operation == "owned":
            if info:
                owned(spec, info, image)
            return {"ok": True}
        if operation in ("stop", "stopped"):
            if operation == "stop" and info and info["State"].get("Running"):
                # Serving containers hold no state. Once every rank stops at
                # once, a worker only waits for its departed peers before its
                # own executor kills it (28 s for GLM TP2); rank 0 exits in
                # about 8 s. A short grace period ends that wait.
                run(["docker", "stop", "--time", "15", info["Id"]])
            elif operation == "stopped" and info and info["State"].get("Running"):
                raise ValueError("Container is still running")
            return {"ok": True}
        if operation == "running":
            if not info or not info["State"].get("Running"):
                raise ValueError("Rank is not running")
            return {"ok": True}
        if operation == "created":
            owned(spec, info, image)
            if "image_runtime" in lock:
                check_runtime_binding(lock, row, info)
            return {"ok": True}
        if operation == "container-record":
            owned(spec, info, image)
            if info["State"].get("Running"):
                raise ValueError("Native installation requires stopped containers")
            return {"Id": info["Id"], "Image": info["Image"], "Name": info["Name"],
                    "State": {"Running": False}, "HostConfig": {"RestartPolicy": info["HostConfig"]["RestartPolicy"]},
                    "Config": {"Env": info["Config"].get("Env", [])}}
        if operation in ("preflight", "create", "start"):
            receipt = admit_image(lock)
            verify_model(lock, row, model_receipt, refresh=True)
            if card["profile"].startswith("glm53-"):
                actual = installer.specifications(lock, receipt=receipt, local=True, only_rank=number)[0]
                if actual != spec:
                    raise ValueError("Observed native-image settings differ from the planned Compose file")
            else:
                metadata, _ = profiles.load(card["profile"])
                profile = profiles.read_json(profiles.local_path(metadata["configuration"]["path"]))
                profile = qwen_flash_next.checkpoint_settings(profile, card.get("target_variant"))
                qwen_flash_next.verify_model_paths(profile, Path(row["model"]), Path(row["cache"]))
            if card["nodes"] == 4 and not (operation == "create" and "native_mesh" in lock["site_input"]):
                from runtime.common import qwen_mesh
                qwen_mesh.check(row["fabric"], number, row["hcas"], row["gid"], row["host_ip"])
            if not (info and info["State"].get("Running")):
                require_idle()
            compose.check_project_containers(spec.name, owned_id=info["Id"] if info else None, run=run)
            # Installer-owned containers name the admitted local image by its
            # configuration ID. A copy received over the fabric has no registry
            # digest, so a digest reference would not resolve with --pull never.
            local_image = card["image_id"] if "image_runtime" in lock else card["image_reference"]
            text = compose.compose_text(spec, local_image)
            compose.check_equivalence(spec, local_image, text, run=run)
            target = plain(Path(row["deployment_root"]) / lock["id"])
            if operation == "create":
                target.mkdir(parents=True, exist_ok=True)
                path = plain(target / "compose.yaml")
                if path.exists() and path.read_text() != text:
                    raise ValueError("Saved Compose file was edited; reinitialize explicitly")
                if not path.exists():
                    installer.write(path, text)
                if not info:
                    if "image_runtime" in lock:
                        binding = local_binding_path(lock, row)
                        if read_runtime_binding(binding) is None:
                            installer.write(binding, {})
                    run(compose.compose_command(spec.name, path) + ["create", "--no-build", "--no-recreate", "--pull", "never", "model"])
                if "image_runtime" in lock:
                    created = owned(spec, container(spec), image)
                    check_runtime_binding(lock, row, created, finalize=True)
            elif operation == "start":
                owned(spec, info, image)
                if "image_runtime" in lock:
                    check_runtime_binding(lock, row, info)
                if not info["State"].get("Running"):
                    run(["docker", "start", info["Id"]])
            return {"ok": True}
        if operation == "ready":
            if number != 0:
                if not info or not info["State"].get("Running"):
                    raise ValueError("Worker is not running")
                return {"ok": True}
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                current = owned(spec, container(spec), image)
                if not current["State"].get("Running"):
                    raise ValueError("API rank exited during startup")
                health = current["State"].get("Health", {}).get("Status")
                if health == "healthy":
                    return {"ok": True}
                if health == "unhealthy":
                    raise ValueError("API health check failed; inspect this rank's logs")
                time.sleep(2)
            raise ValueError("Readiness exceeded 30 minutes; inspect logs before restarting")
        name = spec.command[spec.command.index("--served-model-name") + 1]
        port = int(spec.command[spec.command.index("--port") + 1])
    if operation == "smoke":
        listed = http_json(port, "/v1/models")
        if name not in [entry["id"] for entry in listed["data"]]:
            raise ValueError("API serves a different model")
        response = http_json(port, "/v1/chat/completions", {"model": name,
                             "messages": [{"role": "user", "content": "Reply only READY"}],
                             "max_tokens": 256, "temperature": 0, **smoke_request(card)})
        if not response.get("choices") or not (response["choices"][0]["message"].get("content") or "").strip():
            raise ValueError("Smoke request returned no answer")
        result = {"ok": True, "model": name, "scope": "One short generation; not cache/performance qualification"}
        if "image_runtime" in lock:
            # The shared image carries the runtime-status dashboard; its absence
            # means the status plugin or runtime binding did not load.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/sparkring/status/view", timeout=60) as page:
                if page.status != 200 or page.headers.get_content_type() != "text/html":
                    raise ValueError("Runtime-status dashboard is unavailable")
            result["dashboard"] = "/v1/sparkring/status/view"
        return result
    raise ValueError("Unknown rank operation: " + operation)
