"""A derived checkpoint's part of ``sudo sparkring install``: survey, plan section and plan text.

A derived checkpoint (``runtime/common/derived_checkpoint.py``) is served from
its own SparkRing checkpoint directory on every Spark, beside the base's. The
installation first acquires the base with the ordinary checkpoint plan
(``runtime.host.checkpoint_plan``); this module adds what follows:

- every Spark hard-links the files the derived checkpoint keeps from its base
  (``derive-link``), so they take no space;
- the files the recipe writes come from a Spark that holds them, or, when no
  Spark does, Node A writes them: it downloads the donor files it lacks from
  huggingface.co into the donor's own checkpoint directory (``derive-donor``)
  and runs the recipe in the installer image, CPU only and without network
  (``derive-run``);
- the other Sparks receive the recipe's files along the fabric ring, as they
  receive checkpoint files from a donor Spark, with rsync through Node A as the
  fallback (``install_assets.Assets.derive``).

Node A derives once and the other Sparks receive the result, rather than each
Spark deriving its own: only Node A then needs the donor files on disk and a
recipe run, and the other Sparks use the receive path that every checkpoint
transfer uses, which places a file only after its SHA-256 equals the pin. That
is as safe as deriving on each Spark, because the manifest pins the bytes that
Node A's byte-exact check produced. The other Sparks receive the recipe's
5.6 GiB over the fabric instead of the donor's 2.6 GiB.

``probe_source`` reads, on one Spark, the placed files of named SparkRing
checkpoint directories from their journals; ``section`` turns those readings
into the plan's ``derivation`` section, which ``checkpoint_plan.plan`` counts
in each Spark's free-space need and which bounds the derivation's downloads
and writes at run time (``bound``).
"""
from __future__ import annotations

import inspect
import json
import subprocess

from runtime.common import derived_checkpoint
from runtime.host import checkpoint_plan

# The probe runs as root and reads its source on stdin.
PROBE_COMMAND = ["sudo", "-n", "python3", "-I", "-B", "-"]
PROBE_TIMEOUT = 60


def probe(paths):
    """Each named SparkRing checkpoint directory's placed files, as its journal records them.

    Shipped as source and run as root; it imports only the standard library,
    reads ``owner.json`` and ``journal.json`` in each directory's state
    directory without following symlinks, and calls ``lstat``. For each path:
    ``state`` (``absent``, ``empty``, ``owned`` or ``foreign``), the owner's
    ``repository`` and ``revision``, ``files`` (each journaled name whose
    current five stats equal the journal's, with ``[size, sha256]``), and the
    ``device`` and ``free_bytes`` of the nearest existing ancestor.
    """
    import json
    import os
    import stat

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

    def lstat(path):
        try:
            return os.lstat(path)
        except OSError:
            return None

    def document(path):
        try:
            fd = os.open(path, flags)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            data = b""
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                data += block
                if len(data) > 16 << 20:
                    return None
        finally:
            os.close(fd)
        try:
            return json.loads(data)
        except ValueError:
            return None

    result = {}
    for path in paths:
        entry = {"state": "absent", "repository": None, "revision": None, "files": {}, "device": None,
                 "free_bytes": None}
        nearest = path
        while lstat(nearest) is None and nearest != "/":
            nearest = os.path.dirname(nearest)
        info = lstat(nearest)
        if info is not None:
            entry["device"] = info.st_dev
            try:
                space = os.statvfs(nearest)
                entry["free_bytes"] = space.f_bavail * space.f_frsize
            except (OSError, AttributeError):
                pass
        here = lstat(path)
        if here is not None:
            parent, name = os.path.split(path)
            state = os.path.join(parent, "." + name + ".sparkring")
            owner = document(os.path.join(state, "owner.json"))
            if (stat.S_ISDIR(here.st_mode) and isinstance(owner, dict) and owner.get("path") == path
                    and owner.get("directory") == [here.st_dev, here.st_ino]):
                journal = document(os.path.join(state, "journal.json"))
                files = journal.get("files") if isinstance(journal, dict) else None
                entry.update(state="owned", repository=owner.get("repository"), revision=owner.get("revision"))
                for file, record in (files if isinstance(files, dict) else {}).items():
                    current = lstat(os.path.join(path, file))
                    if (isinstance(record, dict) and record.get("state") == "placed" and current is not None
                            and stat.S_ISREG(current.st_mode) and record.get("stats") == [
                                current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
                                current.st_ctime_ns]):
                        entry["files"][file] = [current.st_size, record.get("sha256")]
            elif stat.S_ISDIR(here.st_mode) and not os.listdir(path):
                entry["state"] = "empty"
            else:
                entry["state"] = "foreign"
        result[path] = entry
    return result


def probe_source(paths):
    """Self-contained Python source that prints ``probe(paths)`` as JSON."""
    return inspect.getsource(probe) + "\n\nimport json\nprint(json.dumps(probe(" + repr(list(paths)) + ")))\n"


def paths(manifest, rows):
    """``(derived directories per rank, donor directory)`` beside each rank's base directory; Node A holds the donor."""
    return ([derived_checkpoint.directory(row["model"], manifest) for row in rows],
            derived_checkpoint.directory(rows[0]["model"], manifest["donor"]))


def survey(manifest, rows, *, invoke):
    """Every Spark's probe of its derived directory, and Node A's of the donor directory too, all ranks at once.

    Returns one ``{path: entry}`` mapping per rank, or ``{"error": text}`` for
    a Spark whose probe failed; the plan then takes that Spark to hold nothing.
    """
    import concurrent.futures
    derived, donor = paths(manifest, rows)

    def one(rank):
        wanted = [derived[rank]] + ([donor] if rank == 0 else [])
        try:
            found = json.loads(invoke(rows[rank]["host"], list(PROBE_COMMAND), data=probe_source(wanted),
                                      timeout=PROBE_TIMEOUT))
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            lines = [line.strip() for line in str(error).splitlines() if line.strip()]
            return {"error": (lines[-1] if lines else type(error).__name__)[:300]}
        if not isinstance(found, dict) or set(found) != set(wanted):
            return {"error": "invalid probe output"}
        return found

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return list(pool.map(one, range(len(rows))))


def _placed(entry, required, repository, revision):
    """Names of ``required`` that a probed directory holds with their pinned size and SHA-256."""
    if not isinstance(entry, dict) or entry.get("state") != "owned" or (
            entry.get("repository"), entry.get("revision")) != (repository, revision):
        return set()
    return {name for name, value in (entry.get("files") or {}).items()
            if name in required and isinstance(value, list) and len(value) == 2
            and value == [required[name]["size"], required[name]["sha256"]]}


def section(name, manifest, donor_pins, rows, probes, hostnames=None):
    """The plan's ``derivation`` section for derived checkpoint ``name``.

    ``rows`` are the deployment rows, whose ``model`` is each Spark's base
    directory; ``probes`` the ``survey`` result. Every Spark links the base
    files its derived directory lacks. The recipe's files follow
    ``checkpoint_plan``'s distribution: the lowest-numbered Spark holding all
    of them is the source. Without one, Node A pools them from the other Sparks
    when together they hold every file, and otherwise derives them, downloading
    the donor files its donor directory lacks; the recipe writes every file
    into staging, so all count. Each node's ``written`` lists the sizes of the
    files it writes.
    """
    required = derived_checkpoint.required(manifest)
    recipe = derived_checkpoint.files_of(manifest, "recipe")
    kept = derived_checkpoint.files_of(manifest, "base")
    derived, donor_path = paths(manifest, rows)
    count = len(rows)
    readings = [probe if isinstance(probe, dict) and "error" not in probe else {} for probe in probes]
    held = [_placed(readings[rank].get(derived[rank]), required, manifest["repository"], manifest["revision"])
            for rank in range(count)]
    distribution = checkpoint_plan._distribute(count, [names & set(recipe) for names in held], recipe)
    derive = bool(distribution["hub"])
    donor_files = {key: donor_pins["files"][key]["size"] for key in manifest["donor"]["files"]}
    donor_required = {key: {"size": donor_pins["files"][key]["size"], "sha256": donor_pins["files"][key]["sha256"]}
                      for key in donor_files}
    donor_reading = readings[0].get(donor_path) or {}
    present = _placed(donor_reading, donor_required, manifest["donor"]["repository"], manifest["donor"]["revision"])
    download = sorted(set(donor_files) - present) if derive else []
    nodes = []
    for rank, row in enumerate(rows):
        reading = readings[rank].get(derived[rank]) or {}
        written, receive = [], []
        if rank == 0 and derive:
            # The recipe writes every file into staging at once, so Node A derives instead of pooling some.
            written += list(recipe.values())
            written += [donor_files[key] for key in download]
        elif rank == 0:
            written += [recipe[key] for item in distribution["pool"] for key in item["names"]]
        for item in distribution["receive"]:
            if item["target"] == rank:
                written += [recipe[key] for key in item["names"]]
                receive.append({"from": item["source"], "files": len(item["names"]),
                                "bytes": sum(recipe[key] for key in item["names"]), "transport": item["transport"]})
        link = sorted(set(kept) - held[rank])
        nodes.append({"rank": rank, "host": row["host"], "hostname": (hostnames or {}).get(rank) or row["host"],
                      "path": derived[rank], "state": reading.get("state", "unknown"),
                      "probe_error": (probes[rank] or {}).get("error") if isinstance(probes[rank], dict) else None,
                      "present": len(held[rank]), "link": len(link), "link_bytes": sum(kept[key] for key in link),
                      "derive": sorted(set(recipe) - held[0]) if rank == 0 and derive else [],
                      "pool": [{"from": item["source"], "files": len(item["names"]), "transport": item["transport"]}
                               for item in distribution["pool"]] if rank == 0 and not derive else [],
                      "receive": receive, "written": written, "write_bytes": sum(written),
                      "device": reading.get("device"), "free_bytes": reading.get("free_bytes")})
    return {"checkpoint": name, "repository": manifest["repository"], "revision": manifest["revision"],
            "base": dict(manifest["base"]), "recipe": dict(manifest["recipe"]),
            "files": {"total": len(required), "bytes": sum(item["size"] for item in required.values()),
                      "kept": len(kept), "recipe": len(recipe), "recipe_bytes": sum(recipe.values())},
            "donor": {"repository": manifest["donor"]["repository"], "revision": manifest["donor"]["revision"],
                      "path": donor_path, "state": donor_reading.get("state", "unknown"), "files": donor_files,
                      "present": sorted(present), "download": download,
                      "download_bytes": sum(donor_files[key] for key in download)},
            "derive": derive, "source": distribution["donor"], "nodes": nodes}


def problems(section_value, plan_nodes):
    """Requests for input before approval.

    A derived or donor directory that SparkRing did not create, a base served
    in place, and a derived directory on another filesystem than the base's
    each stop the plan.
    """
    found = []
    donor = section_value["donor"]
    if section_value["derive"] and donor.get("state") == "foreign":
        node = section_value["nodes"][0]
        found.append({"field": "storage", "rank": 0, "message": (
            f"Node 0 {node['hostname']}: {donor['path']} holds files that SparkRing did not place, so the donor files "
            "cannot be downloaded there. Move them away or remove the directory. Nothing has been changed.")})
    for node, base in zip(section_value["nodes"], plan_nodes, strict=True):
        name = f"Node {node['rank']} {node['hostname']}"
        if node["state"] == "foreign":
            found.append({"field": "storage", "rank": node["rank"], "message": (
                f"{name}: {node['path']} is not empty and was not created by SparkRing; SparkRing does not adopt a "
                "directory it did not create. Move it away. Nothing has been changed.")})
        elif base["mode"] == "in-place":
            found.append({"field": "model_path", "rank": node["rank"], "message": (
                f"{name}: {base['path']} would be served in place, but checkpoint {section_value['checkpoint']} "
                "hard-links its base's files into a directory beside SparkRing's checkpoint directory of the base. "
                "Install it without naming that copy, so that SparkRing links or copies its files into its own "
                "directory. Nothing has been changed.")})
        elif (node["device"] is not None and base.get("device") is not None and node["device"] != base["device"]):
            found.append({"field": "storage", "rank": node["rank"], "message": (
                f"{name}: {node['path']} lies on another filesystem than the base's checkpoint directory "
                f"{base['path']}, so the derived checkpoint cannot hard-link the base's files. Keep "
                "/srv/sparkring/<cluster>/checkpoints on one filesystem. Nothing has been changed.")})
    return found


def bound(approved, fresh_writes, tolerance=checkpoint_plan.TOLERANCE_BYTES):
    """Items by which the derivation's writes exceed the approved section; empty when within it."""
    items = []
    for node, writes in zip(approved["nodes"], fresh_writes, strict=True):
        if writes > node["write_bytes"] + tolerance:
            items.append({"kind": "writes", "rank": node["rank"], "bytes": writes, "text": (
                f"Node {node['rank']} {node['hostname']}: the derived checkpoint needs "
                f"{checkpoint_plan._human(writes)} of writes instead of the planned "
                f"{checkpoint_plan._human(node['write_bytes'])}")})
    return items


def envelope(reviewed, fresh):
    """Differences by which a fresh derivation section leaves the reviewed one, as ``checkpoint_plan.envelope`` lists them."""
    if reviewed is None and fresh is None:
        return []
    if reviewed is None or fresh is None or (reviewed["repository"], reviewed["revision"]) != (
            fresh["repository"], fresh["revision"]):
        return [{"kind": "checkpoint", "rank": None, "text": "the reviewed plan derives another checkpoint"}]
    items = []
    extra = sorted(set(fresh["donor"]["download"]) - set(reviewed["donor"]["download"]))
    size = sum(fresh["donor"]["files"][key] for key in extra)
    if extra:
        items.append({"kind": "download", "rank": 0, "names": extra, "bytes": size, "text": (
            f"Node 0 would download {checkpoint_plan._amount(size)} of donor files from huggingface.co "
            f"({', '.join(extra)})")})
    # Node A's growth that its extra downloads explain is reported once, as the download.
    items += bound(reviewed, [node["write_bytes"] - (size if node["rank"] == 0 else 0) for node in fresh["nodes"]])
    return items


def bounded(fresh, reviewed):
    """The fresh section with the reviewed section's downloads and per-Spark writes as its bound."""
    result = json.loads(json.dumps(fresh))
    result["donor"]["download"] = list(reviewed["donor"]["download"])
    result["donor"]["download_bytes"] = reviewed["donor"]["download_bytes"]
    for mine, theirs in zip(result["nodes"], reviewed["nodes"], strict=True):
        mine["write_bytes"] = theirs["write_bytes"]
    return result


def summary(section_value):
    """The derivation part of ``sparkring install --json``'s ``checkpoint`` value."""
    return {"checkpoint": section_value["checkpoint"], "repository": section_value["repository"],
            "revision": section_value["revision"], "derive": section_value["derive"],
            "download": list(section_value["donor"]["download"]),
            "download_bytes": section_value["donor"]["download_bytes"],
            "nodes": [{key: node[key] for key in ("rank", "host", "hostname", "path", "present", "link", "write_bytes")}
                      for node in section_value["nodes"]]}


def describe(section_value):
    """The derivation's lines of the printed plan."""
    files = section_value["files"]
    human = checkpoint_plan._human
    donor = section_value["donor"]
    lines = [f"Derived checkpoint {section_value['checkpoint']} ({section_value['repository']} at "
             f"{section_value['revision'][:12]}): {files['total']} files, {human(files['bytes'])}; "
             f"{files['kept']} files are hard-linked from the base, {files['recipe']} "
             f"({human(files['recipe_bytes'])}) are written by {section_value['recipe']['path']}"]
    for node in section_value["nodes"]:
        name = f"Node {node['rank']} {node['hostname']}"
        if node["present"] == files["total"]:
            lines.append(f"{name}: {node['path']} holds every file; verified before the model starts")
            continue
        parts = []
        if node["link"]:
            parts.append(f"hard-links {checkpoint_plan._plural(node['link'], 'file')} from the base (no extra space)")
        if node["derive"]:
            if donor["download"]:
                parts.append(f"downloads {checkpoint_plan._files(donor['download'])} of {donor['repository']} at "
                             f"{donor['revision'][:12]} ({human(donor['download_bytes'])}) from huggingface.co into "
                             f"{donor['path']}")
            parts.append("derives " + checkpoint_plan._plural(len(node["derive"]), "file") + " ("
                         + human(sum(node["written"]) - donor["download_bytes"]) + ") in the installer image, CPU only "
                         "and without network, after checking that the donor's MXFP8 tensors are the base's weights")
        for item in node["pool"]:
            parts.append(f"receives {checkpoint_plan._plural(item['files'], 'file')} from Node {item['from']}")
        for item in node["receive"]:
            parts.append(f"receives {checkpoint_plan._plural(item['files'], 'file')} ({human(item['bytes'])}) from "
                         f"Node {item['from']} over the fabric")
        lines.append(f"{name} -> {node['path']}")
        lines.extend("    " + part for part in parts)
        if node.get("probe_error"):
            lines.append(f"    its directory could not be read ({node['probe_error']}); planned as empty")
    lines.append("Every file is placed only after its SHA-256 equals the derived checkpoint's manifest.")
    return lines
