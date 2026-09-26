"""Checkpoint plan for ``sparkring install``: where every Spark gets each pinned file.

The plan combines three inputs:

- the pin manifest of the checkpoint revision (``sparkring-checkpoint-pins/v1``,
  ``profiles/checkpoints/<owner>--<name>/<revision>.json``);
- one survey per Spark (``sparkring-checkpoint-survey/v1``, produced by
  ``runtime.host.checkpoint_search``), or a failure for a Spark whose survey did
  not return;
- the deployment rows, in rank order, whose ``model`` is the cluster's SparkRing
  checkpoint directory or, with ``reuse_verified_model``, a named copy served in
  place.

For every required file on every Spark the plan names one action:

- ``present``: already placed in SparkRing's checkpoint directory;
- ``link``: hard-linked from a local copy on the directory's mount;
- ``copy``: copied from a local copy (non-weight files, other mounts);
- ``pool``: sent to Node A from another Spark while Node A assembles the
  checkpoint, over the fabric from a cable-adjacent Spark or with rsync over the
  administration path from any other;
- ``hub``: downloaded by Node A from huggingface.co;
- ``receive``: sent along the fabric ring from the donor Spark;
- ``in-place``: read from a named exact copy that is served without SparkRing's
  directory.

The plan document (``sparkring-checkpoint-plan/v1``) is printed before approval
(``describe``), carried in the install result (``summary``), saved with the
deployment, bounds a later ``--yes`` run (``envelope``) and bounds what the rank
operations may write (``unplanned``). This module reads no host state except the
storage policy file and writes nothing.

Plan document keys: ``repository``, ``revision``, ``pins_sha256``,
``created_at``, ``reviewed``, ``approval``, ``operator``, ``named``,
``ignore_local_copies``, ``request`` (``profile``, ``cache_path`` and
``image_lock`` of the deployment request), ``profile``, ``command`` (the
``sudo sparkring install`` command that repeats this request, which messages
suggest), ``required`` (``files``, ``bytes``, ``sizes``), ``hub_files``,
``hub_bytes``, ``distribution`` (``donor``, ``pool``, ``hub``, ``receive``),
``nodes``, ``retained``, ``refreshed_receipts`` and ``problems``.
Each node holds ``rank``, ``host``, ``hostname``, ``mode`` (``owned`` or
``in-place``), ``path``, ``files`` (name to action entry with ``action``,
``size`` and, by action, ``source``, ``identity``, ``evidence``, ``candidate``,
``from`` and ``transport``), ``bytes``, ``write_bytes``, ``required_bytes``,
``free_bytes``, ``storage``, ``search``, ``sources`` and ``not_used``; a
node served in place also holds ``in_place``, whose ``state`` is ``exact``,
``changed`` or ``unchecked`` (the Spark's search failed). ``problems`` lists
requests for input that stop the installation, each ``{"field": "model_path" |
"storage", "rank", "message"}``.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from runtime.host import fabric_stream

SCHEMA = "sparkring-checkpoint-plan/v1"
SURVEY_SCHEMA = "sparkring-checkpoint-survey/v1"
GIB = 1024 ** 3
# Per-Spark growth accepted without another approval: unplanned writes during
# adoption, and a fresh plan compared with a plan reviewed with --plan.
TOLERANCE_BYTES = GIB
# A download from huggingface.co larger than this needs its own approval when
# only setup approved a first installation.
ATTENTION_DOWNLOAD_BYTES = GIB
# The survey's first walk budget and Docker allowance; a second pass doubles the walk budget.
SEARCH_SECONDS = 20
DOCKER_SECONDS = 15
# The command that messages suggest when the plan does not name its request.
COMMAND = "sudo sparkring install"
LOCAL_FILESYSTEMS = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs"})
NETWORK_TYPES = frozenset({"autofs", "nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "fuse.rclone",
                           "9p", "ceph", "glusterfs", "lustre"})
USABLE_STATES = frozenset({"match", "hashed", "size-only"})
ABSENT_STATES = frozenset({"absent", "missing", "not-present", "not present"})
# The survey's classification of a named path on one Spark.
NAMED_STATES = frozenset({"exact", "partial", "differs", "network", "absent", "ignored", "unsupported"})
NAMED_REASONS = {"ignored": "named, but a .sparkring-ignore file excludes it",
                 "unsupported": "named, but it is on a filesystem type SparkRing does not read"}
# Survey states of the checkpoint directory that the rank operations refuse to claim.
OWNED_REFUSALS = {
    "foreign": "{path} is not empty and was not created by SparkRing; SparkRing does not adopt a directory it did "
               "not create. Move it away or choose another path.",
    "replaced": "{path} was replaced after SparkRing created it. Move it away or choose another path.",
    "symlink": "{path} is a symlink; SparkRing writes its checkpoint directory only through plain paths.",
    "mount-point": "{path} is a mount point; SparkRing keeps its checkpoint directory and its state on one mount.",
}
# Evidence tiers of the survey, strongest first. Equal values rank equally.
EVIDENCE_ORDER = {"recorded": 0, "hashed": 0, "hub-named": 1, "hub-metadata": 1, "size": 2, "network": 2}
EVIDENCE_CLASSES = ("recorded", "hashed", "hub-named", "hub-metadata", "size", "network")
EVIDENCE_TEXT = {
    "recorded": ("identified by SparkRing's earlier checksums", "identified by SparkRing's earlier checksums"),
    "hashed": ("checked by SHA-256", "checked by SHA-256"),
    "hub-named": ("identified by their blob names", "identified by its blob name"),
    "hub-metadata": ("identified by their download records", "identified by its download record"),
    "size": ("match by name and size only", "matches by name and size only"),
    "network": ("on network storage, read once when copied", "on network storage, read once when copied"),
}
LAYOUT_NAMES = {
    "local-dir": "Hugging Face download folder",
    "hf-cache": "Hugging Face cache",
    "hf-snapshot": "Hugging Face cache without symlinks",
    "hf-blob-store": "Hugging Face shared blob store",
    "plain": "plain folder",
    "plain-folder": "plain folder",
    "folder": "plain folder",
    "git-lfs": "git clone with LFS",
    "sparkring": "SparkRing checkpoint directory",
    "network": "folder on network storage",
}
LOCAL_ACTIONS = ("present", "link", "copy", "in-place")
WRITE_ACTIONS = ("copy", "pool", "receive", "hub")
BYTE_KEYS = ("present", "link", "copy", "pool", "receive", "hub")


# Manifest and operator inputs

def required_files(pins):
    """``{name: size}`` of the manifest's required files: ``files`` minus ``optional``."""
    optional = set(pins.get("optional") or ())
    return {name: int(entry["size"]) for name, entry in sorted(pins["files"].items()) if name not in optional}


def weight(name):
    """Weight files are the ``.safetensors`` files; only these are linked from user copies."""
    return name.endswith(".safetensors")


def pins_digest(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def named_paths(values, count=None):
    """Normalize ``--model-path`` values to ``[{"rank": N or None, "path": PATH}]``.

    ``PATH`` names a copy for every Spark and ``N=PATH`` one for Node N. Entries
    already in that form and a mapping from rank (or None) to paths are accepted
    too. Paths are absolute POSIX paths; duplicates are dropped.
    """
    if not values:
        return []
    items = []
    if isinstance(values, dict):
        for rank, paths in values.items():
            for path in ([paths] if isinstance(paths, str) else paths):
                items.append({"rank": None if rank in (None, "", "*") else int(rank), "path": path})
    else:
        for value in values:
            if isinstance(value, dict):
                items.append({"rank": value.get("rank"), "path": value.get("path")})
                continue
            head, separator, tail = str(value).partition("=")
            if separator and head.isdigit() and not head.startswith("/"):
                items.append({"rank": int(head), "path": tail})
            else:
                items.append({"rank": None, "path": value})
    result = []
    for item in items:
        path, rank = item["path"], item["rank"]
        if not isinstance(path, str) or not path.startswith("/") or "\0" in path or "\n" in path:
            raise ValueError(f"--model-path needs an absolute path: {path!r}")
        if rank is not None and (type(rank) is not int or rank < 0 or (count is not None and rank >= count)):
            raise ValueError(f"--model-path names Node {rank}, which this cluster does not have")
        entry = {"rank": rank, "path": str(PurePosixPath(path))}
        if entry not in result:
            result.append(entry)
    return result


def option(entry, *, abbreviate=False):
    """The ``--model-path`` option text that names ``entry``."""
    path = "..." if abbreviate else entry["path"]
    return "--model-path " + (f"{entry['rank']}={path}" if entry["rank"] is not None else path)


def install_command(request=None, named=(), *, ignore_local=False):
    """The ``sudo sparkring install`` command that repeats one deployment request.

    ``request`` holds ``profile``, ``checkpoint``, ``cache_path`` and ``image_lock``. With the
    ``--model-path`` entries ``named`` they make the deployment's identity, so
    a command that leaves one out plans another deployment.
    ``--ignore-local-copies`` is kept when set, because it narrows the search
    whose plan the command repeats.
    """
    request = request or {}
    argv = ["sudo", "sparkring", "install"]
    if request.get("profile"):
        argv += ["--profile", str(request["profile"])]
    if request.get("checkpoint"):
        argv += ["--checkpoint", str(request["checkpoint"])]
    for entry in named_paths(named):
        argv += ["--model-path", (f"{entry['rank']}=" if entry["rank"] is not None else "") + entry["path"]]
    if request.get("cache_path"):
        argv += ["--cache-path", str(request["cache_path"])]
    if request.get("image_lock"):
        argv += ["--image-lock", str(request["image_lock"])]
    if ignore_local:
        argv.append("--ignore-local-copies")
    return shlex.join(argv)


def _variant(plan, *, add=None, remove=None, ignore_local=None):
    """The command of ``plan`` with one named path added or removed, or with ``--ignore-local-copies``."""
    named = [entry for entry in named_paths(plan.get("named") or ())
             if not (remove and (entry["rank"], entry["path"]) == (remove["rank"], remove["path"]))]
    if add:
        named.append(add)
    return install_command(plan.get("request"), named, ignore_local=plan.get("ignore_local_copies")
                           if ignore_local is None else ignore_local)


def storage_policy(root=None):
    """Allowances of ``profiles/storage-planning.json`` in bytes."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    value = json.loads((root / "profiles/storage-planning.json").read_text(encoding="utf-8"))
    if value.get("schema") != "sparkring-storage-planning/v1":
        raise ValueError("Unsupported storage planning policy")
    return {"cache_bytes": int(value["cache_and_jit_allowance_gib"]) * GIB,
            "image_bytes": int(value["image_allowance_gib"]) * GIB}


def required_space(written, *, cache_bytes=0, image_bytes=0):
    """Free bytes one filesystem needs before SparkRing writes checkpoint files on it.

    ``written`` holds the size of every file copied, pooled, received or
    downloaded into checkpoint directories on that filesystem; linked and present
    files are not written and count zero. The largest such file is counted twice,
    for its staging copy. ``cache_bytes`` is the cluster cache allowance when the
    cluster cache is on this filesystem. ``image_bytes`` is the image allowance
    when the serving image is absent and Docker's root is on this filesystem; it
    applies only at plan time, because rank operations run after the image is in
    place. The plan and the rank operations call this one function.
    """
    sizes = [int(size) for size in written]
    return sum(sizes) + max(sizes, default=0) + int(cache_bytes) + int(image_bytes)


def announce(pins, count, seconds=SEARCH_SECONDS, docker=DOCKER_SECONDS):
    """The line printed before the survey starts.

    A search that runs out of time walks once more with twice the budget, and
    Docker's queries have their own allowance; the survey's deadline is the sum.
    """
    return (f"Looking for existing copies of {pins['repository']} at {pins['revision'][:12]} "
            f"on {count} Sparks (up to {seconds} s each; up to {3 * seconds + docker} s on a Spark whose search "
            f"needs a second pass).")


# Survey reading

def _failure(survey):
    """None for a usable survey document, else the failure text."""
    if isinstance(survey, dict) and survey.get("schema") == SURVEY_SCHEMA and not survey.get("error"):
        return None
    if isinstance(survey, BaseException):
        return str(survey) or type(survey).__name__
    if isinstance(survey, dict) and survey.get("error"):
        return str(survey["error"])
    if isinstance(survey, str) and survey:
        return survey
    return "invalid survey output"


def _named_entries(survey):
    """The survey's classification of named paths: ``[{"path", "state", "resolved"?}]``."""
    value = survey.get("named") or []
    if isinstance(value, dict):
        value = [{"path": path, **(item if isinstance(item, dict) else {"state": item})} for path, item in value.items()]
    return [item for item in value if isinstance(item, dict) and isinstance(item.get("path"), str)]


def _related(a, b):
    a, b = PurePosixPath(a), PurePosixPath(b)
    return a == b or a in b.parents or b in a.parents


def _candidates(survey, required):
    """The survey's candidates, plus a stand-in for each named folder on network storage.

    The survey never reads network storage, so a named network folder has no file
    entries. Every required name is then a copy source whose content the copy
    itself verifies.
    """
    result = [c for c in survey.get("candidates") or [] if isinstance(c, dict) and isinstance(c.get("path"), str)]
    known = {c["path"] for c in result}
    for entry in _named_entries(survey):
        path = entry.get("resolved") or entry["path"]
        if entry.get("state") == "network" and path not in known:
            result.append({"path": path, "layout": "network", "network": True, "found_by": ["named"],
                           "files": {name: {"state": "network", "evidence": "network", "size": size,
                                            "source": str(PurePosixPath(path) / name)}
                                     for name, size in required.items()}})
            known.add(path)
    return result


def _named_entry(candidate, entries):
    """The named-path entry that selects ``candidate`` on this Spark, or None.

    A named hub root selects every repository folder below it, and a named
    snapshot folder selects its repository folder. A path with symlink components
    also matches through the survey's resolved form (``aliases``). A candidate
    the survey reached through a named path carries that path as ``named_as``.
    """
    for entry in entries:
        if any(_related(candidate["path"], path) for path in (entry["path"], *entry.get("aliases", ()))):
            return entry
    requested = candidate.get("named_as")
    if isinstance(requested, str):
        return next((entry for entry in entries if requested in (entry["path"], *entry.get("aliases", ()))), None)
    return None


def _network(candidate):
    files = candidate.get("files") or {}
    return (bool(candidate.get("network")) or candidate.get("layout") == "network"
            or any(isinstance(e, dict) and e.get("state") == "network" for e in files.values()))


def _device(candidate):
    if candidate.get("device") is not None:
        return candidate["device"]
    for entry in (candidate.get("files") or {}).values():
        if isinstance(entry, dict) and entry.get("identity"):
            return entry["identity"][0]
    return None


def _on_owned_mount(owned, candidate):
    return (owned.get("mount_id") is not None and candidate.get("mount_id") == owned.get("mount_id")
            and _device(candidate) == owned.get("device"))


def _linkable(owned, candidate, entry, userns):
    """``link(2)`` needs the same mount and device; userns remapping also needs world-readable files."""
    mount = entry.get("mount_id", candidate.get("mount_id"))
    identity = entry.get("identity")
    device = identity[0] if identity else _device(candidate)
    if owned.get("mount_id") is None or mount != owned.get("mount_id") or device != owned.get("device"):
        return False
    return not userns or bool(int(entry.get("mode") or 0) & 0o004)


def _blob(entry):
    """Hub blobs, shared-store blobs and git-lfs objects, which their tools never rewrite in place."""
    kind = entry.get("kind")
    if kind:
        return kind in ("blob", "shared-blob", "lfs-object")
    parts = PurePosixPath(entry.get("source") or "").parts
    return "blobs" in parts or any(parts[i:i + 3] == (".git", "lfs", "objects") for i in range(len(parts)))


def _slow(candidate):
    return bool(candidate.get("rotational") or candidate.get("usb") or candidate.get("removable"))


def _matches(candidate):
    counts = candidate.get("counts")
    if isinstance(counts, dict) and "match" in counts:
        return int(counts["match"])
    return sum(1 for e in (candidate.get("files") or {}).values() if isinstance(e, dict) and e.get("state") == "match")


def _mismatch(candidate, required):
    files = candidate.get("files") or {}
    differs = sorted(n for n in required if (files.get(n) or {}).get("state") == "differs")
    missing = sorted(n for n in required
                     if (files.get(n) or {}).get("state") not in USABLE_STATES and n not in differs)
    return {"differs": differs, "missing": missing, "extra": sorted(candidate.get("extra") or [])}


def _exact(survey, candidate, required):
    """Whether a folder holds exactly the required files (and optionally the optional ones).

    The survey's classification decides when present. Without one, every required
    name must be present at its pinned size without negative evidence and no other
    file may be reported. Serving in place hashes every file again before start.
    """
    if _network(candidate):
        return False
    explicit = candidate.get("exact")
    if isinstance(explicit, bool):
        return explicit
    state = None
    for entry in _named_entries(survey):
        if candidate["path"] in (entry["path"], entry.get("resolved")) and entry.get("state") in NAMED_STATES:
            state = entry["state"]
    if state:
        return state == "exact"
    mismatch = _mismatch(candidate, required)
    files = candidate.get("files") or {}
    # Served in place, the folder itself must hold each file under its name,
    # not through a symlink into a cache.
    at_name = all((files.get(n) or {}).get("source") in (None, str(PurePosixPath(candidate["path"]) / n))
                  for n in required)
    return at_name and not any(mismatch.values())


def _hostname(survey, row):
    if isinstance(survey, dict) and survey.get("host"):
        return survey["host"]
    return row.get("hostname") or row["host"]


def _reason(text):
    text = str(text or "not used")
    suffix = " from the pinned revision"
    return text[:-len(suffix)] if text.endswith(suffix) else text


# One Spark

def choose_sources(pins, survey, row, named, *, ignore_local=False, network=None, locked=False, request=None):
    """Choose where one Spark gets each required file from its own disks.

    ``row`` is the deployment row (``rank``, ``host``, ``model`` and optionally
    ``reuse_verified_model``); ``named`` holds ``--model-path`` entries, of which
    those for every Spark and for this rank apply. ``network`` limits named
    network folders to those this Spark reads (None allows every one).
    ``locked`` keeps the row's mode, as for a deployment that already exists.
    ``request`` (``install_command``) names the command that messages suggest.

    Ranking, first key first: named for this Spark; a SparkRing checkpoint
    directory; linkable (weight files); evidence; blob or git-lfs object before
    folder file; the candidate's completeness; outside home directories; for
    copies, not on a rotational or USB-attached device; the smallest path.
    """
    required = required_files(pins)
    rank = row.get("rank", 0)
    failure = _failure(survey)
    resolved = {} if failure else {e["path"]: e["resolved"] for e in _named_entries(survey) if e.get("resolved")}
    entries = [{**e, "aliases": [resolved[e["path"]]] if e["path"] in resolved else []}
               for e in named_paths(named) if e["rank"] in (None, rank)]
    hostname = _hostname(survey, row)
    owned = (survey.get("owned") or {}) if failure is None else {}
    candidates = _candidates(survey, required) if failure is None else []
    node = {"rank": rank, "host": row["host"], "hostname": hostname, "mode": "owned", "path": row["model"],
            "directory": row["model"], "files": {}, "sources": [], "not_used": [], "differs": {},
            "named": [{"rank": e["rank"], "path": e["path"]} for e in entries], "absent_named": [],
            "exact_elsewhere": [], "in_place": None, "problems": []}

    def serve_in_place(path, found, entry, state="exact"):
        node.update(mode="in-place", path=path, directory=owned.get("path") or row["model"])
        found = found or {}
        node["in_place"] = {"named": entry and {"rank": entry["rank"], "path": entry["path"]},
                            "layout": found.get("layout"), "commit": found.get("commit"),
                            "branches": list(found.get("branches") or []), "home": found.get("home") or None,
                            "other_filesystem": bool(found) and not _on_owned_mount(owned, found), "state": state}
        node["files"] = {name: {"action": "in-place", "size": size, "source": str(PurePosixPath(path) / name),
                                "candidate": path} for name, size in required.items()}
        return node

    if row.get("reuse_verified_model"):
        path = row["model"]
        found = next((c for c in candidates if c["path"] == path), None) or next(
            (c for c in candidates if _related(c["path"], path) and _named_entry(c, entries)), None)
        entry = next((e for e in entries if any(_related(p, path) for p in (e["path"], *e["aliases"]))), None)
        if failure is not None:
            return serve_in_place(path, found, entry, "unchecked")
        if found is not None and _exact(survey, found, required):
            return serve_in_place(path, found, entry, "exact")
        serve_in_place(path, found, entry, "changed")
        others = [item for item in named_paths(named)
                  if not (entry and (item["rank"], item["path"]) == (entry["rank"], entry["path"]))]
        command = install_command(request, others, ignore_local=ignore_local)
        node["problems"].append({"field": "model_path", "rank": rank,
                                 "message": _changed_in_place(node, path, found, required, command)})
        return node
    if failure is not None:
        return node
    if not locked:
        exact = sorted((c for c in candidates if _named_entry(c, entries) and not _network(c)
                        and not _on_owned_mount(owned, c) and _exact(survey, c, required)),
                       key=lambda c: c["path"])
        if exact:
            return serve_in_place(exact[0]["path"], exact[0], _named_entry(exact[0], entries))

    userns = bool((survey.get("docker") or {}).get("userns"))
    own_paths = {row["model"], owned.get("path")}
    files, options, matched = node["files"], {name: [] for name in required}, set()
    for name, entry in (owned.get("files") or {}).items():
        if name in required and isinstance(entry, dict):
            if entry.get("state") in USABLE_STATES:
                files[name] = {"action": "present", "size": required[name],
                               "source": entry.get("source") or str(PurePosixPath(row["model"]) / name),
                               "identity": entry.get("identity"), "evidence": entry.get("evidence") or "recorded",
                               "candidate": row["model"], "sparkring": True}
            elif entry.get("state") == "differs":
                node["differs"].setdefault(name, []).append(row["model"])
    for candidate in candidates:
        path = candidate["path"]
        entry_named = _named_entry(candidate, entries)
        if entry_named:
            matched.add(entries.index(entry_named))
        if path in own_paths:
            for name, entry in (candidate.get("files") or {}).items():
                if name in required and name not in files and (entry or {}).get("state") in USABLE_STATES:
                    files[name] = {"action": "present", "size": required[name], "source": entry.get("source"),
                                   "identity": entry.get("identity"), "evidence": entry.get("evidence") or "recorded",
                                   "candidate": path, "sparkring": True}
            continue
        home = candidate.get("home") or None
        remote = _network(candidate)
        if ignore_local and not (candidate.get("sparkring") or entry_named):
            continue
        if remote and (not entry_named or (network is not None and path not in network)):
            continue
        if home and home.get("kind") == "private" and not entry_named:
            node["not_used"].append({"path": path, "reason": f"in {home.get('account')}'s home, another account",
                                     "option": option({"rank": rank, "path": path})})
            continue
        if not remote and not _on_owned_mount(owned, candidate) and not entry_named and _exact(survey, candidate, required):
            node["exact_elsewhere"].append(path)
        for name, entry in sorted((candidate.get("files") or {}).items()):
            if name not in required or not isinstance(entry, dict):
                continue
            state = entry.get("state")
            if state == "differs":
                node["differs"].setdefault(name, []).append(path)
                continue
            if state == "network" and remote:
                evidence = "network"
            elif state in USABLE_STATES:
                evidence = entry.get("evidence") or ("size" if state == "size-only" else "hashed")
            else:
                continue
            if int(entry.get("size", required[name])) != required[name]:
                continue
            linkable = not remote and _linkable(owned, candidate, entry, userns)
            action = "link" if linkable and (weight(name) or candidate.get("sparkring")) else "copy"
            key = (0 if entry_named else 1,
                   0 if candidate.get("sparkring") else 1,
                   (0 if linkable else 1) if weight(name) else 0,
                   EVIDENCE_ORDER.get(evidence, 3),
                   0 if _blob(entry) else 1,
                   -_matches(candidate),
                   0 if home is None else 1,
                   1 if action == "copy" and _slow(candidate) else 0,
                   path, entry.get("source") or "")
            options[name].append((key, {
                "action": action, "size": required[name],
                "source": entry.get("source") or str(PurePosixPath(path) / name),
                "identity": entry.get("identity"), "evidence": evidence, "candidate": path,
                "sparkring": bool(candidate.get("sparkring")), "named": entry_named is not None,
                "slow": _slow(candidate), "owner": entry.get("owner")}))
    for name, choices in options.items():
        if name not in files and choices:
            files[name] = min(choices, key=lambda choice: choice[0])[1]

    by_path = {c["path"]: c for c in candidates}
    used = {}
    for name, entry in files.items():
        if entry["action"] in ("link", "copy"):
            used.setdefault(entry["candidate"], []).append(name)
    for path, names in sorted(used.items(), key=lambda item: (-len(item[1]), item[0])):
        candidate = by_path[path]
        node["sources"].append({
            "path": path, "layout": "sparkring" if candidate.get("sparkring") else candidate.get("layout"),
            "commit": candidate.get("commit"), "branches": list(candidate.get("branches") or []),
            "home": candidate.get("home") or None, "sparkring": bool(candidate.get("sparkring")),
            "named": bool(_named_entry(candidate, entries)),
            "other_filesystem": not _on_owned_mount(owned, candidate),
            "owners": sorted({files[n].get("owner") for n in names if files[n].get("owner")}),
            "files": len(names),
            "differs": sorted(n for n, paths in node["differs"].items() if path in paths),
            "mismatch": _mismatch(candidate, required)})
    seen = {item["path"] for item in node["not_used"]} | set(used)
    for item in survey.get("not_used") or []:
        if not isinstance(item, dict) or not item.get("path") or item["path"] in seen:
            continue
        seen.add(item["path"])
        home = item.get("home") or {}
        node["not_used"].append({"path": item["path"], "reason": _reason(item.get("reason")),
                                 "option": option({"rank": rank, "path": item["path"]})
                                 if home.get("kind") == "private" else None})
    reported = {e["path"]: e for e in _named_entries(survey)}
    for index, entry in enumerate(entries):
        if index in matched:
            continue
        state = (reported.get(entry["path"]) or {}).get("state")
        if state is None or state in ABSENT_STATES:
            node["absent_named"].append({"rank": entry["rank"], "path": entry["path"]})
        elif entry["path"] not in seen:
            node["not_used"].append({"path": entry["path"], "option": None, "reason": NAMED_REASONS.get(
                state, "named, but it holds none of the pinned files")})
    return node


def _changed_in_place(node, path, found, required, command=COMMAND):
    """``NeedsInput`` text for a copy served in place that no longer holds exactly the pinned files.

    ``command`` repeats the request without the named path, so that SparkRing
    assembles its own directory on that Spark.
    """
    rank, host = node["rank"], node["hostname"]
    if found is None:
        what = "is missing or holds none of the pinned files"
    else:
        mismatch = _mismatch(found, required)
        parts = []
        if mismatch["differs"]:
            parts.append("differs from the pinned revision in " + _names(mismatch["differs"]))
        if mismatch["missing"]:
            parts.append("lacks " + _names(mismatch["missing"], noun="of the pinned files"))
        if mismatch["extra"]:
            parts.append(("also holds " if parts else "holds ") + _names(mismatch["extra"], noun="other files")
                         + ", which the serving engine would load")
        what = " and ".join(parts) or "no longer holds exactly the pinned files"
    return (f"Node {rank} {host}: {path} {what}. SparkRing never changes your copy and serves a named folder in "
            f"place only when it holds exactly the pinned files. Change the folder yourself, or install without "
            f"naming it on Node {rank} so that SparkRing assembles its own copy: {command} --plan.")


# Several Sparks

def _distance(count, rank):
    return 1 if count <= 2 else min(rank, count - rank)


def _ring_order(count):
    return sorted(range(1, count), key=lambda rank: (_distance(count, rank), rank))


def _distribute(count, holdings, sizes):
    """Donor, pooling, downloads and ring receives, given the names each Spark holds (``holdings``).

    The lowest-numbered complete Spark is the donor. Without one, Node A pools
    each name it lacks from one other Spark, nearest in ring order first, and
    downloads the names no Spark holds; it is then the donor. The other Sparks
    receive their missing names along ``fabric_stream.tree``.
    """
    names = set(sizes)
    held = [set(h) & names for h in holdings]
    complete = [rank for rank in range(count) if held[rank] == names]
    pool, hub = {}, []
    if complete:
        donor = complete[0]
    else:
        donor = 0
        for name in sorted(names - held[0]):
            source = next((rank for rank in _ring_order(count) if name in held[rank]), None)
            if source is None:
                hub.append(name)
            else:
                pool.setdefault(source, []).append(name)
    receive = []
    for level, edges in enumerate(fabric_stream.tree(count, donor)):
        for source, target in edges:
            need = sorted(names - held[target])
            if need:
                receive.append({"source": source, "target": target, "names": need, "level": level,
                                "transport": "fabric"})
    return {"donor": donor, "complete": complete, "hub": hub, "receive": receive,
            "pool": [{"source": rank, "names": pool[rank], "transport": "fabric" if _distance(count, rank) == 1 else "rsync"}
                     for rank in _ring_order(count) if rank in pool]}


def _writes(count, distribution, sizes):
    """Bytes each Spark writes for the transfers of ``distribution``."""
    result = [0] * count
    for item in distribution["pool"]:
        result[0] += sum(sizes[name] for name in item["names"])
    result[0] += sum(sizes[name] for name in distribution["hub"])
    for item in distribution["receive"]:
        result[item["target"]] += sum(sizes[name] for name in item["names"])
    return result


def _images(value, count):
    if value is None:
        return [True] * count
    if isinstance(value, dict):
        return [bool(value.get(rank, value.get(str(rank), True))) for rank in range(count)]
    if isinstance(value, (set, frozenset)):
        return [rank in value for rank in range(count)]
    return [bool(item) for item in value]


def _retained(value):
    """Normalize retained deployments to ``[(name, [model path of each rank])]``."""
    if not value:
        return []
    if isinstance(value, dict):
        return [(name, list(paths)) for name, paths in value.items()]
    return [(item["name"], list(item.get("paths") or item.get("models") or [])) for item in value]


def plan(pins, surveys, rows, *, named=(), ignore_local=False, operator="root", images_present=None,
         retained=None, locked=False, policy=None, now=None, request=None):
    """Plan the checkpoint for every Spark of one deployment.

    ``surveys`` holds one survey document per rank, or the failure (an exception,
    a string, or ``{"error": ...}``) for a rank whose survey did not return; that
    rank is planned as holding no copy. ``images_present`` tells, per rank, whether
    the serving image is already there (default: everywhere). ``retained`` maps
    each retained deployment's name to the model path of each rank, to name the
    deployments whose receipts adoption refreshes. ``locked`` keeps every row's
    mode, for a deployment that already exists. ``request`` holds the
    ``profile``, ``checkpoint``, ``cache_path`` and ``image_lock`` of the deployment request;
    with ``named`` and ``ignore_local`` it gives the command that the plan's
    messages suggest (``install_command``).
    """
    sizes = required_files(pins)
    count = len(rows)
    if len(surveys) != count:
        raise ValueError("The checkpoint plan needs one survey per Spark")
    entries = named_paths(named, count)
    request = {key: (request or {}).get(key) for key in ("profile", "checkpoint", "cache_path", "image_lock")}
    context = {"request": request, "named": entries, "ignore_local_copies": bool(ignore_local)}
    command = install_command(request, entries, ignore_local=ignore_local)
    policy = policy or storage_policy()
    images = _images(images_present, count)
    failures = [_failure(survey) for survey in surveys]
    rows = [{**row, "rank": row.get("rank", rank)} for rank, row in enumerate(rows)]

    # A named folder on network storage is read on the lowest-numbered Spark that
    # has it; the other Sparks receive its files over the fabric.
    network, claimed = [set() for _ in range(count)], set()
    for rank, survey in enumerate(surveys):
        if failures[rank] is None:
            mine = [e for e in entries if e["rank"] in (None, rank)]
            for candidate in _candidates(survey, sizes):
                if _network(candidate) and _named_entry(candidate, mine) and candidate["path"] not in claimed:
                    claimed.add(candidate["path"])
                    network[rank].add(candidate["path"])
    nodes = [choose_sources(pins, survey, row, entries, ignore_local=ignore_local, network=network[rank], locked=locked,
                            request=request)
             for rank, (survey, row) in enumerate(zip(surveys, rows))]

    # When a Spark holds a complete copy without slow copies, the other Sparks
    # receive over the fabric instead of copying from rotational or USB-attached
    # devices they were not told to use.
    def held(node):
        return {name for name, entry in node["files"].items() if entry["action"] in LOCAL_ACTIONS}

    def slow(node):
        return [name for name, entry in node["files"].items()
                if entry["action"] == "copy" and entry.get("slow") and not entry.get("named")]
    complete = [rank for rank in range(count) if held(nodes[rank]) == set(sizes)]
    keepers = [rank for rank in complete if not slow(nodes[rank])] or complete[:1]
    if keepers:
        for rank, node in enumerate(nodes):
            if rank not in keepers and slow(node):
                dropped = sorted({node["files"][name]["candidate"] for name in slow(node)})
                for name in slow(node):
                    del node["files"][name]
                node["sources"] = [s for s in node["sources"]
                                   if any(e.get("candidate") == s["path"] for e in node["files"].values())]
                node["not_used"][:0] = [{"path": path, "option": None, "reason": (
                    "on a rotational or USB-attached disk; this Spark receives those files over the fabric")}
                    for path in dropped if path not in {s["path"] for s in node["sources"]}]

    distribution = _distribute(count, [held(node) for node in nodes], sizes)
    for item in distribution["pool"]:
        for name in item["names"]:
            nodes[0]["files"][name] = {"action": "pool", "size": sizes[name], "from": item["source"],
                                       "transport": item["transport"]}
    for name in distribution["hub"]:
        nodes[0]["files"][name] = {"action": "hub", "size": sizes[name]}
    for item in distribution["receive"]:
        for name in item["names"]:
            nodes[item["target"]]["files"][name] = {"action": "receive", "size": sizes[name], "from": item["source"],
                                                    "transport": item["transport"]}

    problems = []
    for rank, node in enumerate(nodes):
        survey = surveys[rank] if failures[rank] is None else {}
        owned = survey.get("owned") or {}
        node["files"] = dict(sorted(node["files"].items()))
        node["bytes"] = {key: 0 for key in BYTE_KEYS}
        if node["mode"] == "in-place":
            node["bytes"]["in_place"] = 0
        for entry in node["files"].values():
            node["bytes"][entry["action"].replace("-", "_")] += entry["size"]
        written = [entry["size"] for entry in node["files"].values() if entry["action"] in WRITE_ACTIONS]
        node["write_bytes"] = sum(written)
        # Unknown placement of the cluster cache or Docker's root counts as
        # sharing the checkpoint directory's filesystem.
        cache_on = owned.get("cache_device") in (None, owned.get("device"))
        docker = survey.get("docker") or {}
        docker_on = docker.get("device") in (None, owned.get("device"))
        node["storage"] = {"written_bytes": sum(written), "largest_bytes": max(written, default=0),
                           "cache_bytes": policy["cache_bytes"] if cache_on else 0,
                           "image_bytes": policy["image_bytes"] if not images[rank] and docker_on else 0}
        node["required_bytes"] = required_space(written, cache_bytes=node["storage"]["cache_bytes"],
                                                image_bytes=node["storage"]["image_bytes"])
        node["free_bytes"] = owned.get("free_bytes")
        node["mount_point"] = owned.get("mount_point") or owned.get("probe_path")
        node["probe_path"] = owned.get("probe_path")
        node["fstype"] = owned.get("fstype")
        node["storage"]["passed"] = None if node["free_bytes"] is None else node["free_bytes"] >= node["required_bytes"]
        search = survey.get("search") or {}
        node["search"] = ({"complete": False, "failed": failures[rank], "seconds": None} if failures[rank] else {
            "complete": bool(search.get("complete")), "failed": None, "seconds": search.get("seconds"),
            "entries": search.get("entries"), "passes": search.get("passes"), "stopped": search.get("stopped"),
            "unvisited": search.get("unvisited") or {},
            "network": sorted({m["path"] for m in search.get("skipped_mounts") or []
                               if isinstance(m, dict) and m.get("path")
                               and (m.get("network") or m.get("type") in NETWORK_TYPES)})})
        problems.extend(node.pop("problems"))
    problems.extend(_named_problems(entries, nodes, failures))
    for rank, node in enumerate(nodes):
        state = ((surveys[rank].get("owned") or {}).get("state") if failures[rank] is None else None)
        if node["mode"] == "owned" and state in OWNED_REFUSALS:
            problems.append({"field": "storage", "rank": rank, "message": (
                f"Node {rank} {node['hostname']}: " + OWNED_REFUSALS[state].format(path=node["path"])
                + " Nothing has been changed.")})
        elif node["mode"] == "owned" and node["fstype"] and node["fstype"] not in LOCAL_FILESYSTEMS:
            problems.append({"field": "storage", "rank": node["rank"], "message": (
                f"Node {node['rank']} {node['hostname']}: the checkpoint directory's filesystem at "
                f"{node['probe_path']} is {node['fstype']}, not a local filesystem (ext2, ext3, ext4, xfs, btrfs, "
                f"f2fs or zfs). Nothing has been changed.")})
        elif node["storage"]["passed"] is False:
            problems.append({"field": "storage", "rank": node["rank"],
                             "message": _storage_message(pins, node, context)})

    refreshed = []
    for name, paths in _retained(retained):
        for rank, node in enumerate(nodes):
            linked = {e["candidate"] for e in node["files"].values() if e["action"] == "link" and not e.get("sparkring")}
            if rank < len(paths) and paths[rank] in linked:
                refreshed.append(name)
                break
    created = now or datetime.datetime.now(datetime.timezone.utc)
    return {"schema": SCHEMA, "repository": pins["repository"], "revision": pins["revision"],
            "pins_sha256": pins_digest(pins), "created_at": created.isoformat(timespec="seconds"),
            "reviewed": False, "approval": None, "operator": operator, "named": entries,
            "ignore_local_copies": bool(ignore_local), "request": request, "profile": request["profile"],
            "command": command,
            "required": {"files": len(sizes), "bytes": sum(sizes.values()), "sizes": sizes},
            "hub_files": sorted(distribution["hub"]), "hub_bytes": sum(sizes[n] for n in distribution["hub"]),
            "distribution": {key: distribution[key] for key in ("donor", "pool", "hub", "receive")},
            "nodes": nodes, "retained": refreshed, "refreshed_receipts": len(refreshed),
            "problems": sorted(problems, key=lambda p: (p["field"] != "model_path", p["rank"] is not None, p["rank"] or 0))}


def _named_problems(entries, nodes, failures):
    problems = []
    for entry in entries:
        ranks = [entry["rank"]] if entry["rank"] is not None else list(range(len(nodes)))
        checked = [rank for rank in ranks if failures[rank] is None]
        if not checked or len(checked) != len(ranks):
            continue
        if all(entry in nodes[rank]["absent_named"] for rank in checked):
            if entry["rank"] is None:
                message = f"--model-path {entry['path']} exists on no Spark. Nothing has been changed."
            else:
                node = nodes[entry["rank"]]
                message = (f"{option(entry)} does not exist on Node {node['rank']} {node['hostname']}. "
                           f"Nothing has been changed.")
            problems.append({"field": "model_path", "rank": entry["rank"], "message": message})
    return problems


def _storage_message(pins, node, context):
    """``NeedsInput`` text for a Spark without the free space its plan needs, with what would make room.

    ``context`` holds the plan's ``request``, ``named`` and
    ``ignore_local_copies``, from which the suggested commands are built.
    """
    rank, host, storage = node["rank"], node["hostname"], node["storage"]
    where = node["mount_point"] or node["path"]
    command = _variant(context)
    inexact = next((s for s in node["sources"] if s["named"] and s["other_filesystem"] and s["layout"] != "network"
                    and any(s["mismatch"].values())), None)
    if inexact:
        mismatch, path = inexact["mismatch"], inexact["path"]
        parts = []
        if mismatch["differs"]:
            branch = inexact["branches"][0] if inexact["branches"] else None
            parts.append(f"holds the {branch} branch's {_names(mismatch['differs'])}" if branch else
                         f"holds {_names(mismatch['differs'])} that "
                         f"{'differs' if len(mismatch['differs']) == 1 else 'differ'} from the pinned revision")
        if mismatch["missing"]:
            parts.append("lacks " + _names(mismatch["missing"], noun="of the pinned files"))
        if mismatch["extra"]:
            parts.append("also holds " + _names(mismatch["extra"], noun="other files")
                         + ", which the serving engine would load")
        text = (f"Node {rank} {host}: {path} is on another filesystem and {' and '.join(parts)}, so SparkRing cannot "
                f"serve it in place, and copying it needs {_gib1(node['required_bytes'])} free on {where} "
                f"({_gib1(node['free_bytes'])} free).")
        fetch = mismatch["differs"] + mismatch["missing"]
        sizes = required_files(pins)
        if not mismatch["extra"] and fetch and sum(sizes[n] for n in fetch) < GIB:
            which = fetch[0] if len(fetch) == 1 else "any of these files"
            return (text + f" To make that folder an exact copy yourself: hf download {pins['repository']} "
                    + " ".join(shlex.quote(n) for n in fetch)
                    + f" --revision {pins['revision']} --local-dir {shlex.quote(path)} (this changes your folder; "
                    f"if {which} there is a hard link to another copy, remove it first). Then repeat {command} --plan. "
                    "The running model has not been stopped.")
        return text + " The running model has not been stopped."
    categories = {action: node["bytes"][action] for action in WRITE_ACTIONS}
    verb = {"copy": "copy", "pool": "receive", "receive": "receive", "hub": "download"}[max(categories, key=categories.get)]
    terms = []
    if storage["written_bytes"]:
        terms += [f"{_human(storage['written_bytes'])} of checkpoint files",
                  f"{_human(storage['largest_bytes'])} to stage the largest file"]
    if storage["cache_bytes"]:
        terms.append(f"{storage['cache_bytes'] // GIB} GiB for the compile cache")
    if storage["image_bytes"]:
        terms.append(f"{storage['image_bytes'] // GIB} GiB for the image")
    purpose = f" to {verb} the checkpoint" if storage["written_bytes"] else ""
    text = (f"Node {rank} {host} needs {_gib1(node['required_bytes'])} free on {where}{purpose} "
            f"({', '.join(terms)}); {_gib1(node['free_bytes'])} is free.")
    shortfall = f"Free another {_gib1(node['required_bytes'] - node['free_bytes'])} there"
    if any(weight(name) and entry["action"] in WRITE_ACTIONS for name, entry in node["files"].items()):
        # With a copy on this filesystem every weight file is linked and only the other files are written.
        linked = required_space([size for name, size in required_files(pins).items() if not weight(name)],
                                cache_bytes=storage["cache_bytes"], image_bytes=storage["image_bytes"])
        text += (f" {shortfall}, or put a copy of the pinned checkpoint on that filesystem: SparkRing hard-links "
                 f"its weight files, so it would need {_gib1(linked)}.")
    else:
        text += f" {shortfall}."
    text += f" Then repeat {command} --plan."
    for path in node["exact_elsewhere"][:1]:
        text += (f" An exact copy was found on another filesystem at {path}; to serve it in place instead, review "
                 f"{_variant(context, add={'rank': rank, 'path': path})} --plan.")
    return text + " The running model has not been stopped."


# Approval

def attention(plan):
    """Plan items that setup's approval of a first installation does not cover.

    A download from huggingface.co larger than 1 GiB, a Spark whose search is
    incomplete after its second pass, and a Spark whose search failed each need
    a terminal confirmation or a ``--yes`` typed on the command line.
    """
    items = []
    if plan["hub_bytes"] > ATTENTION_DOWNLOAD_BYTES:
        node = plan["nodes"][0]
        items.append({"kind": "download", "rank": 0, "bytes": plan["hub_bytes"],
                      "text": f"downloads {_human(plan['hub_bytes'])} from huggingface.co on Node 0 {node['hostname']}"})
    for node in plan["nodes"]:
        search, name = node["search"], f"Node {node['rank']} {node['hostname']}"
        if search.get("failed"):
            items.append({"kind": "search-failed", "rank": node["rank"],
                          "text": f"the search on {name} failed ({search['failed']})"})
        elif not search.get("complete"):
            limit = "entry" if search.get("stopped") == "entries" else "time"
            unvisited = _unvisited(search.get("unvisited"))
            items.append({"kind": "search-incomplete", "rank": node["rank"],
                          "text": f"the search on {name} stopped at its {limit} limit"
                                  + (f" (not searched: {unvisited})" if unvisited else "")})
    return items


def attention_message(items, command=COMMAND):
    """``NeedsInput`` text when attention items meet an approval that came only from setup.

    ``command`` is the plan's ``command``, which repeats the deployment request.
    """
    sentence = _series(item["text"] for item in items)
    sentence = "The checkpoint plan " + sentence if items[0]["kind"] == "download" else sentence[0].upper() + sentence[1:]
    return (sentence + f". Setup's approval did not show this plan. Review it with {command} --plan, or approve "
            f"it with {command} --yes. Nothing has been changed.")


def _source_paths(node):
    paths = {source["path"] for source in node["sources"]}
    if node["mode"] == "in-place":
        paths.add(node["path"])
    return paths


def envelope(reviewed, fresh):
    """Differences by which ``fresh`` leaves the plan reviewed with ``--plan``; empty when within it.

    The fresh plan stays within the reviewed one when its downloads are a subset
    of the reviewed downloads, each Spark writes at most 1 GiB more, each Spark
    keeps its mode, and each Spark's source paths are a subset of the reviewed
    ones. Growth on Node A that its extra downloads explain is reported once, as
    the download.
    """
    same = (reviewed.get("repository"), reviewed.get("revision"), reviewed.get("pins_sha256"),
            [n["host"] for n in reviewed["nodes"]]) == (fresh.get("repository"), fresh.get("revision"),
                                                         fresh.get("pins_sha256"), [n["host"] for n in fresh["nodes"]])
    if not same:
        return [{"kind": "checkpoint", "rank": None,
                 "text": "the reviewed plan is for another checkpoint or another set of Sparks"}]
    sizes = fresh["required"]["sizes"]
    items = []
    extra = sorted(set(fresh["hub_files"]) - set(reviewed["hub_files"]))
    extra_bytes = sum(sizes[name] for name in extra)
    if extra:
        node = fresh["nodes"][0]
        text = (f"Node 0 {node['hostname']} would download {_amount(extra_bytes)} from huggingface.co "
                f"({', '.join(extra)})")
        taken = {}
        for approved in reviewed["nodes"]:
            for name in extra:
                entry = approved["files"].get(name) or {}
                if entry.get("action") in ("link", "copy", "present") and entry.get("candidate"):
                    taken.setdefault(entry["candidate"], []).append((approved["rank"], name))
        if taken:
            differs = any(path in fresh["nodes"][rank]["differs"].get(name, ())
                          for path, uses in taken.items() for rank, name in uses)
            text += (f", which the reviewed plan took from {_and(sorted(taken))}, where "
                     + ("that file differs from the pinned revision" if differs else "that file is absent"))
        items.append({"kind": "download", "rank": 0, "names": extra, "bytes": extra_bytes, "text": text})
    for approved, node in zip(reviewed["nodes"], fresh["nodes"]):
        rank, name = node["rank"], f"Node {node['rank']} {node['hostname']}"
        growth = node["write_bytes"] - approved["write_bytes"] - (extra_bytes if rank == 0 else 0)
        if growth > TOLERANCE_BYTES:
            items.append({"kind": "writes", "rank": rank, "bytes": node["write_bytes"],
                          "text": f"{name} would write {_amount(node['write_bytes'])} instead of "
                                  f"{_amount(approved['write_bytes'])}"})
        if node["mode"] != approved["mode"]:
            planned = (f"serve {node['path']} in place" if node["mode"] == "in-place"
                       else "use SparkRing's checkpoint directory")
            reviewed_mode = (f"serving {approved['path']} in place" if approved["mode"] == "in-place"
                             else "SparkRing's checkpoint directory")
            items.append({"kind": "mode", "rank": rank, "text": f"{name} would {planned} instead of {reviewed_mode}"})
        added = sorted(_source_paths(node) - _source_paths(approved))
        if added:
            items.append({"kind": "sources", "rank": rank, "paths": added,
                          "text": f"{name} would read {_and(added)}, which the reviewed plan did not use"})
    return items


def envelope_message(items, command=COMMAND):
    """``NeedsInput`` text for a fresh plan that leaves the reviewed plan; the reviewed plan stays the bound."""
    return ("The checkpoint plan differs from the plan reviewed with --plan: " + "; ".join(i["text"] for i in items)
            + f". Nothing was changed. Review the plan again with {command} --plan, then repeat {command} --yes.")


# Guard

def _parent(path):
    return str(PurePosixPath(path).parent) if path else None


def redistribute(plan, results):
    """Donor, pooling, downloads and receives after each Spark's adoption.

    ``results`` holds, per rank, the ``model-adopt`` result (``verified``,
    ``bytes_written``, ...), or None for a rank that has none; such a rank is
    taken to hold what the plan placed there locally. A Spark served in place
    holds every file. ``writes`` gives each Spark's total bytes written:
    adoption's writes plus the transfers.
    """
    sizes = plan["required"]["sizes"]
    count = len(plan["nodes"])
    holdings, adopted = [], []
    for rank, node in enumerate(plan["nodes"]):
        result = results[rank] if rank < len(results) else None
        if node["mode"] == "in-place":
            holdings.append(set(sizes))
            adopted.append(0)
        elif result is None:
            holdings.append({n for n, e in node["files"].items() if e["action"] in LOCAL_ACTIONS})
            adopted.append(node["bytes"]["copy"])
        else:
            holdings.append(set(result.get("verified") or ()) & set(sizes))
            adopted.append(int(result.get("bytes_written") or 0))
    distribution = _distribute(count, holdings, sizes)
    transfers = _writes(count, distribution, sizes)
    return {**distribution, "adopted": adopted, "transfers": transfers,
            "writes": [a + t for a, t in zip(adopted, transfers)]}


def unplanned(plan, results):
    """What the adoption results require beyond the approved plan; empty when within it.

    Checked before any request to huggingface.co and before any write that the
    plan did not include: the names to download must be among the plan's
    ``hub_files``, and each Spark may write at most 1 GiB more than planned.
    """
    sizes = plan["required"]["sizes"]
    after = redistribute(plan, results)
    items = []
    extra = sorted(set(after["hub"]) - set(plan["hub_files"]))
    extra_bytes = sum(sizes[name] for name in extra)
    if extra:
        items.append(_unplanned_download(plan, results, extra, extra_bytes))
    for rank, node in enumerate(plan["nodes"]):
        explained = extra_bytes if rank == 0 else 0
        if after["writes"][rank] - explained > node["write_bytes"] + TOLERANCE_BYTES:
            result = results[rank] if rank < len(results) else None
            items.append(_unplanned_writes(plan, node, result, after, rank))
    return items


def _unplanned_download(plan, results, extra, extra_bytes):
    causes = {}
    for rank, result in enumerate(results):
        for item in (result or {}).get("differs") or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name in extra:
                entry = plan["nodes"][rank]["files"].get(name) or {}
                folder = entry.get("candidate") or _parent(item.get("source") if isinstance(item, dict) else None)
                causes.setdefault((rank, folder or "a local folder"), []).append(name)
    parts = []
    for (rank, folder), names in sorted(causes.items()):
        node = plan["nodes"][rank]
        verb = "is" if len(names) == 1 else "are"
        parts.append(f"Node {rank} {node['hostname']}: {_plural(len(names), 'file')} in {folder} {verb} not the "
                     f"pinned model's ({', '.join(sorted(names))}); that folder holds another fine-tune or damaged files.")
    subject = ("the pinned file" if len(extra) == 1 else "the pinned files") if causes else _names(extra)
    need = "it needs" if len(extra) == 1 else "they need"
    parts.append(f"No Spark holds {subject}, so {need} {_amount(extra_bytes)} from huggingface.co, which the approved "
                 f"plan did not include.")
    return {"kind": "download", "rank": 0, "names": extra, "bytes": extra_bytes, "text": " ".join(parts)}


def _unplanned_writes(plan, node, result, after, rank):
    sizes = plan["required"]["sizes"]
    failures = [item for item in ((result or {}).get("link_failures") or []) + ((result or {}).get("missing") or [])
                if isinstance(item, dict) and (item.get("reason") or item.get("errno")) in ("EXDEV", "EPERM")]
    kinds = {"pool": 0, "hub": 0, "receive": sum(sizes[n] for item in after["receive"] if item["target"] == rank
                                                  for n in item["names"])}
    if rank == 0:
        kinds["pool"] = sum(sizes[n] for item in after["pool"] for n in item["names"])
        kinds["hub"] = sum(sizes[n] for n in after["hub"])
    how = {"pool": "copied from the other Sparks", "receive": "received over the fabric",
           "hub": "downloaded from huggingface.co"}[max(kinds, key=kinds.get)]
    name = f"Node {rank} {node['hostname']}"
    total = after["writes"][rank]
    immutable = False
    if failures:
        folders = sorted({(node["files"].get(f.get("name")) or {}).get("candidate") or _parent(f.get("source"))
                          or "a local folder" for f in failures})
        reasons = {f.get("reason") or f.get("errno") for f in failures}
        immutable = reasons == {"EPERM"}
        why = ("they are different mounts of one filesystem" if reasons == {"EXDEV"} else
               "the files are immutable or append-only" if immutable else
               "different mounts of one filesystem or immutable files")
        amount = after["transfers"][rank] or total
        text = (f"{name}: hard links from {_and(folders)} into {node['path']} failed ({why}), so {_human(amount)} "
                f"must be {how} instead, which the approved plan did not include.")
        if immutable:
            # The search does not read file attributes, so another plan would try the same links again.
            text += (f" SparkRing does not change those files: remove the attribute yourself (lsattr shows it), "
                     f"name another copy with --model-path {rank}=PATH, or repeat "
                     f"{_variant(plan, ignore_local=True)} --plan.")
    else:
        text = (f"{name}: {_human(total)} must be written instead of the planned {_human(node['write_bytes'])}, "
                f"which the approved plan did not include.")
    return {"kind": "writes", "rank": rank, "bytes": total, "text": text, "immutable": immutable}


def unplanned_message(items, command=COMMAND):
    """``NeedsInput`` text for the guard's items; nothing has been downloaded when it is raised.

    ``command`` is the approved plan's ``command``. Items whose remedy is not a
    new plan, such as immutable source files, carry their own next steps.
    """
    text = " ".join(item["text"] for item in items)
    review = f" Review the resulting plan with {command} --plan."
    if any(item["kind"] == "download" for item in items):
        return text + " Nothing was downloaded and the running model was not changed." + review
    closing = " Nothing more was written; the running model was not changed."
    return text + closing + ("" if all(item.get("immutable") for item in items) else review)


def adoption(plan, rank, receipts=(), tolerance=TOLERANCE_BYTES):
    """The ``model-adopt`` input of one Spark: its local actions, the receipts to refresh and the tolerance."""
    node = plan["nodes"][rank]
    return {"files": {name: {"action": entry["action"], "source": entry.get("source"),
                             "identity": entry.get("identity"), "size": entry["size"]}
                      for name, entry in node["files"].items() if entry["action"] in ("present", "link", "copy")},
            "receipts": list(receipts), "tolerance_bytes": tolerance}


# Output

def summary(plan):
    """The ``checkpoint`` value of ``sparkring install --json``: a summary of the plan.

    The full plan, with each file's action, evidence and source, is
    ``checkpoint-plan.json`` in the deployment directory.
    """
    nodes = []
    for node in plan["nodes"]:
        sources = [{key: source[key] for key in ("path", "layout", "commit", "branches", "home", "files")}
                   for source in node["sources"]]
        if node["mode"] == "in-place":
            info = node["in_place"]
            sources = [{"path": node["path"], "layout": info["layout"], "commit": info["commit"],
                        "branches": info["branches"], "home": info["home"], "files": len(node["files"])}]
        search = {"complete": node["search"]["complete"], "seconds": node["search"]["seconds"],
                  "not_searched": list(node["search"].get("network") or []),
                  "unvisited": dict(node["search"].get("unvisited") or {})}
        if node["search"].get("failed"):
            search["error"] = node["search"]["failed"]
        item = {"rank": node["rank"], "host": node["host"], "hostname": node["hostname"], "mode": node["mode"],
                "path": node["path"], "sources": sources,
                "bytes": {key: node["bytes"][key] for key in BYTE_KEYS},
                "free_bytes": node["free_bytes"], "required_bytes": node["required_bytes"], "search": search,
                "not_used": [{key: entry.get(key) for key in ("path", "reason", "option")}
                             for entry in node.get("not_used") or []]}
        if node["mode"] == "in-place":
            item["bytes"]["in_place"] = node["bytes"]["in_place"]
            item["in_place"] = (node.get("in_place") or {}).get("state")
        nodes.append(item)
    return {"repository": plan["repository"], "revision": plan["revision"], "hub_files": plan["hub_files"],
            "hub_bytes": plan["hub_bytes"], "approval": plan.get("approval"), "reviewed": bool(plan.get("reviewed")),
            "command": plan.get("command") or COMMAND, "nodes": nodes,
            "refreshed_receipts": plan["refreshed_receipts"]}


def describe(plan):
    """The printed plan, one string per line.

    The last line says what is downloaded, so it sits directly above any prompt.
    A Spark whose block equals an earlier Spark's apart from free space is shown
    as ``as Node N``.
    """
    sizes = plan["required"]["sizes"]
    lines = [f"Checkpoint {plan['repository']} at {plan['revision'][:12]}: {len(sizes)} files, "
             f"{_human(sum(sizes.values()))}"]
    searched = _search_sentence(plan)
    if searched:
        lines.append(searched)
    lines.append("")
    shown = {}
    for node in plan["nodes"]:
        signature = tuple(_node_lines(plan, node, generic=True))
        if signature in shown:
            free = f" ({_free(node['free_bytes'])} free)" if node["free_bytes"] is not None else ""
            lines.append(f"Node {node['rank']} {node['hostname']}: as Node {shown[signature]}{free}")
            continue
        shown[signature] = node["rank"]
        lines.extend(_node_lines(plan, node))
    lines.append("")
    lines.extend(_closing(plan))
    return lines


def _search_sentence(plan):
    done, sentences, network = [], [], []
    for node in plan["nodes"]:
        search = node["search"]
        if search.get("failed"):
            sentences.append(f"The search on {node['hostname']} failed: {search['failed']}.")
            continue
        seconds = search.get("seconds")
        done.append(f"{node['hostname']} in {seconds:.1f} s" if isinstance(seconds, (int, float)) else node["hostname"])
        if not search.get("complete"):
            limit = "entry" if search.get("stopped") == "entries" else "time"
            unvisited = _unvisited(search.get("unvisited"))
            sentences.append(f"The search on {node['hostname']} stopped at its {limit} limit"
                             + (f" (not searched: {unvisited})." if unvisited else "."))
        network += [path for path in search.get("network") or [] if path not in network]
    if done:
        sentences.insert(0, "Searched " + _and(done) + ".")
    if network:
        sentences.append(f"Not searched: {', '.join(sorted(network))} (network storage; name a copy there with "
                         f"--model-path N=PATH).")
    return " ".join(sentences)


def _unvisited(value):
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    parts = []
    for root, item in sorted(value.items()):
        number = item.get("count", len(item.get("paths") or ())) if isinstance(item, dict) else \
            len(item) if isinstance(item, (list, tuple)) else int(item)
        parts.append(f"{_plural(number, 'folder')} under {root}")
    return ", ".join(parts)


def _node_lines(plan, node, generic=False):
    name = "Node" if generic else f"Node {node['rank']} {node['hostname']}"
    if node["mode"] == "in-place":
        return _in_place_lines(node, name)
    files, total = node["files"], len(plan["required"]["sizes"])
    local = [entry for entry in files.values() if entry["action"] in ("link", "copy")]
    present = [entry for entry in files.values() if entry["action"] == "present"]
    transfers = _transfer_texts(plan, node)
    needs = f"needs {_gib1(node['required_bytes'])} free" + (f" on {node['mount_point']}" if node["mount_point"] else "")
    parts = _allowances(node)
    free = f"{_free(node['free_bytes'])} free" if not generic and node["free_bytes"] is not None else None
    if parts or free:
        needs += " (" + ", ".join(parts) + ("; " if parts and free else "") + (free or "") + ")"
    lines = []
    if local or present:
        lines.append(f"{name} -> {node['path']}")
        for source in node["sources"]:
            lines.extend(_source_lines(plan, node, source))
        if present:
            lines.append(f"    {len(present)} of {total} files already in place")
        actions = _action_text(node)
        if actions:
            lines.append("    " + actions)
        lines.extend("    " + text for text in transfers)
        lines.append("    " + needs)
    else:
        search = node["search"]
        if search.get("failed"):
            lines.append(f"{name}: search failed: {search['failed']}")
        else:
            entries = search.get("entries")
            seconds = search.get("seconds")
            detail = " ".join(part for part in (
                f"{entries:,} folder entries" if isinstance(entries, int) else "",
                f"in {seconds:.1f} s" if isinstance(seconds, (int, float)) else "") if part)
            lines.append(f"{name}: no copy found" + (f" (searched {detail})" if detail else ""))
        if len(transfers) == 1:
            lines.append(f"    {transfers[0]}; {needs}")
        else:
            lines.extend("    " + text for text in transfers)
            lines.append("    " + needs)
    for item in node["not_used"]:
        lines.append(f"    not used: {item['path']} ({item['reason']})"
                     + (f"; use it with {item['option']}" if item.get("option") else ""))
    return lines


def _allowances(node):
    """The parts of a Spark's free-space figure when it includes an allowance, else an empty list."""
    storage = node.get("storage") or {}
    if not storage.get("cache_bytes") and not storage.get("image_bytes"):
        return []
    parts = []
    files = storage.get("written_bytes", 0) + storage.get("largest_bytes", 0)
    if files:
        parts.append(f"{_human(files)} for checkpoint files")
    if storage.get("cache_bytes"):
        parts.append(f"{storage['cache_bytes'] // GIB} GiB for the compile cache")
    if storage.get("image_bytes"):
        parts.append(f"{storage['image_bytes'] // GIB} GiB for the image")
    return parts


def _in_place_lines(node, name):
    info = node["in_place"] or {}
    entry = info.get("named")
    named = ("named with " + option(entry, abbreviate=True)) if entry else "named for this deployment"
    directory = node.get("directory") or ""
    base = "/srv/sparkring" if directory.startswith("/srv/sparkring/") else (node.get("probe_path") or directory)
    where = f"; on another filesystem than {base}" if info.get("other_filesystem") else ""
    state = info.get("state") or "exact"
    if state == "unchecked":
        detail = ("; not checked, because the search on this Spark failed; SparkRing verifies every file before the "
                  "model starts")
    elif state == "changed":
        detail = where + "; it no longer holds exactly the pinned files (see below)"
    else:
        detail = (where + ", and it" if where else "; it") + " holds exactly the pinned files"
    return [f"{name}: serve {node['path']} in place, read-only",
            f"    {named}{detail}",
            "    The model will not start while that folder is changed or missing."]


def _source_lines(plan, node, source):
    lines = [f"    from {source['path']}", "        " + _source_description(plan, source)]
    used = [(name, entry) for name, entry in node["files"].items()
            if entry.get("candidate") == source["path"] and entry["action"] in ("link", "copy")]
    groups = {}
    for name, entry in used:
        groups.setdefault(entry["evidence"] if entry["evidence"] in EVIDENCE_TEXT else "size", []).append(name)
    ordered = [(cls, groups[cls]) for cls in EVIDENCE_CLASSES if cls in groups]
    if len(ordered) == 1:
        cls, names = ordered[0]
        text = f"{len(names)} of {len(plan['required']['sizes'])} files {EVIDENCE_TEXT[cls][0]}"
    else:
        parts = []
        for cls, names in ordered:
            noun = ("weight file" if all(weight(n) for n in names) else
                    "small file" if not any(weight(n) for n in names) else "file")
            parts.append(f"{_plural(len(names), noun)} {EVIDENCE_TEXT[cls][0 if len(names) != 1 else 1]}")
        text = "; ".join(parts)
    if source["differs"]:
        differs = source["differs"]
        verb = "differs" if len(differs) == 1 else "differ"
        text += f"; {_names(differs)} {verb} from the pinned revision"
    lines.append("        " + text)
    unverified = [(name, entry) for name, entry in used if entry["evidence"] in ("size", "network")]
    if unverified:
        actions = {entry["action"] for _, entry in unverified}
        before = "linking" if actions == {"link"} else "copying" if actions == {"copy"} else "linking or copying"
        lines.append(f"        SparkRing hashes them before {before}. If any differs, SparkRing stops before "
                     f"downloading a replacement.")
    return lines


def _source_description(plan, source):
    layout = source.get("layout")
    parts = [LAYOUT_NAMES.get(layout, layout or "folder")]
    commit = source.get("commit")
    if commit and not source.get("sparkring"):
        noun = "snapshot" if str(layout).startswith("hf-cache") else "commit"
        branches = source.get("branches") or []
        qualifier = ("the pinned revision" if commit == plan["revision"] else
                     ("branch " if len(branches) == 1 else "branches ") + _and(branches) if branches else None)
        parts.append(f"{noun} {commit[:12]}" + (f" ({qualifier})" if qualifier else ""))
    home = source.get("home")
    if home:
        whose = {"operator": " (the operator's)", "service": " (a service account)",
                 "private": " (another account)"}.get(home.get("kind"), "")
        parts.append(f"in {home.get('account')}'s home{whose}")
    owners = source.get("owners") or []
    if owners and not source.get("sparkring") and not (home and set(owners) == {home.get("account")}):
        parts.append("files owned by " + _and(owners))
    return ", ".join(parts)


def _action_text(node):
    linked = [name for name, entry in node["files"].items() if entry["action"] == "link"]
    copied = [name for name, entry in node["files"].items() if entry["action"] == "copy"]
    parts = []
    if linked:
        noun = "weight file" if all(weight(n) for n in linked) else "file"
        parts.append(f"hard-link {_plural(len(linked), noun)} (no copy, no extra space)")
    if copied:
        noun = ("other file" if linked and not any(weight(n) for n in copied) else
                "weight file" if all(weight(n) for n in copied) else "file")
        size = sum(node["files"][n]["size"] for n in copied)
        parts.append(f"copy {_plural(len(copied), noun)} ({_human(size)})")
    return "; ".join(parts)


def _transfer_texts(plan, node):
    files, hub = node["files"], set(plan["hub_files"])
    texts = []
    groups = {}
    for name, entry in files.items():
        if entry["action"] == "pool":
            groups.setdefault((entry["from"], entry["transport"]), []).append(name)
    for (source, transport), names in sorted(groups.items()):
        texts.append(f"receives {_files(names)} ({_human(sum(files[n]['size'] for n in names))}) from Node {source} "
                     f"over {'the fabric' if transport == 'fabric' else 'SSH (rsync)'}")
    received = {}
    for name, entry in files.items():
        if entry["action"] == "receive" and name not in hub:
            received.setdefault(entry["from"], []).append(name)
    for source, names in sorted(received.items()):
        texts.append(f"receives {_files(names)} ({_human(sum(files[n]['size'] for n in names))}) from Node {source} "
                     f"over the fabric")
    fetched = sorted(name for name, entry in files.items()
                     if entry["action"] == "hub" or (entry["action"] == "receive" and name in hub))
    if fetched:
        size = _human(sum(files[n]["size"] for n in fetched))
        if any(files[n]["action"] == "hub" for n in fetched):
            texts.append(f"downloads {_files(fetched)} ({size}) from huggingface.co")
        else:
            verb = "is" if len(fetched) == 1 else "are"
            texts.append(f"{_files(fetched)} ({size}) {verb} downloaded once on Node 0 from huggingface.co and "
                         f"copied over the fabric")
    return texts


def _closing(plan):
    lines = []
    nodes, sizes = plan["nodes"], plan["required"]["sizes"]
    if len(nodes) > 1 and plan["hub_files"] and set(plan["hub_files"]) == set(sizes):
        lines.append(f"No Spark holds the checkpoint: Node 0 downloads it once from huggingface.co "
                     f"({_human(plan['hub_bytes'])}) and copies it to the other Sparks over the fabric.")
    entries = [entry for node in nodes for entry in node["files"].values()]
    user = [entry for entry in entries if entry["action"] in ("link", "copy") and not entry.get("sparkring")]
    user_links = [entry for entry in user if entry["action"] == "link"]
    if plan["retained"]:
        folders = {entry["candidate"] for entry in user_links}
        lines.append(f"Also updates the recorded file identities of the retained deployments that serve the linked "
                     f"{'folder' if len(folders) == 1 else 'folders'}: {', '.join(plan['retained'])}")
    if any(entry["action"] in ("link", "copy") for entry in entries):
        lines.append("SparkRing checks the SHA-256 of every file before it links or copies it (about 20 s to 5 "
                     "minutes per Spark); later starts compare recorded file identities and re-hash any file that "
                     "changed.")
    if user_links:
        lines.append("SparkRing only reads your copies and adds hard links to their weight files; it never writes, "
                     "moves or deletes them.")
        lines.append("While this checkpoint directory exists, deleting a linked copy does not free its disk space "
                     "(see sudo sparkring checkpoints).")
    elif user or any(node["mode"] == "in-place" for node in nodes):
        lines.append("SparkRing only reads your copies; it never writes, moves or deletes them.")
    lines.append(f"Downloads {_human(plan['hub_bytes'])} from huggingface.co on Node 0." if plan["hub_files"]
                 else "Nothing is downloaded.")
    return lines


# Text helpers

def _gib1(value):
    return f"{value / GIB:.1f} GiB"


def _free(value):
    return f"{round(value / GIB)} GiB"


def _amount(value):
    return f"{value / GIB:.2f} GiB" if value >= GIB else _human(value)


def _human(value):
    """Sizes as printed: GiB with one decimal from 1 GiB, else decimal MB or KB to three digits."""
    if value >= GIB:
        return _gib1(value)
    for unit, scale in (("MB", 10 ** 6), ("KB", 10 ** 3)):
        if value >= scale:
            number = value / scale
            text = f"{number:.0f}" if number >= 99.95 else f"{number:.1f}" if number >= 9.995 else f"{number:.2f}"
            return f"{text} {unit}"
    return f"{value} bytes"


def _plural(number, noun):
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _files(names):
    return names[0] if len(names) == 1 else f"{len(names)} files"


def _names(names, noun="files"):
    names = list(names)
    return _and(names) if len(names) <= 3 else f"{len(names)} {noun}"


def _and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _series(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]
