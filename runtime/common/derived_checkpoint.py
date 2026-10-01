"""Derived checkpoints: checkpoints that the installer builds on the Sparks from published ones.

A profile's checkpoints table (``runtime.common.qwen_flash_next.checkpoint_names``)
may list an entry with a ``derived`` object, ``{"base": NAME, "donor": NAME}``,
naming two other entries of the same table. Such an entry describes a checkpoint
that no repository publishes: the installer writes it on every Spark from the
pinned files of its *base* and some files of its *donor*, with a recipe module of
this repository, and serves it. Its ``model`` names the derived checkpoint, not a
Hugging Face repository:

- ``repository`` is ``sparkring-derived/<name>``, a name SparkRing gives it;
- ``revision`` is its identity (``identity``): the first 40 hexadecimal digits of
  the SHA-256 of the base's repository and revision, the donor's repository,
  revision and files, and the recipe's path and SHA-256;
- ``config_sha256`` and ``index_sha256`` are the derived ``config.json`` and weight
  index, which serving checks before the model starts.

Manifest
--------
``profiles/checkpoints/<owner>--<name>/<revision>.json`` of the derived
``model`` (schema ``sparkring-derived-checkpoint/v1``) pins the derived checkpoint:

- ``repository``, ``revision``: as in the entry's ``model``;
- ``base``, ``donor``: ``{"repository", "revision"}`` of the two entries' models;
  the donor's ``files`` are the donor files the recipe reads, each a required file
  of the donor's pin manifest;
- ``recipe``: ``{"path", "sha256"}`` of the recipe module, a file below
  ``runtime/`` (checked out with LF line endings) whose bytes must hash to
  ``sha256``;
- ``transform``: what the recipe must find and do, passed to it in the record;
- ``index``: the weight index's name;
- ``files``: every file of the derived checkpoint with its ``size``, ``sha256``
  and ``origin``: ``base`` for a file of the base that the derived checkpoint
  keeps unchanged (hard-linked from the base's checkpoint directory; its pins
  equal the base's), ``recipe`` for a file the recipe writes.

The recipe writes ``derivation.json``, the *record*: the derived identity, base,
donor, recipe and transform (``record``), as canonical JSON (``record_bytes``).
The manifest pins its bytes like those of every other file.

Directories
-----------
A derived checkpoint's directory, and the directory holding its donor files, lie
beside the base's SparkRing checkpoint directory in the cluster's checkpoints
root (``directory``), so the unchanged files can be hard-linked from the base on
the same filesystem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re

from runtime.common import profiles

SCHEMA = "sparkring-derived-checkpoint/v1"
RECORD_SCHEMA = "sparkring-derivation/v1"
OWNER = "sparkring-derived"
RECORD = "derivation.json"
ORIGINS = ("base", "recipe")
KEYS = {"schema", "repository", "revision", "base", "donor", "recipe", "transform", "index", "files"}
_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"


def entry(configuration, name):
    """The ``derived`` object of checkpoint ``name`` in a serving configuration, or None."""
    value = ((configuration.get("checkpoints") or {}).get(name) or {}).get("derived")
    return value if isinstance(value, dict) else None


def identity(base, donor, recipe):
    """The derived revision: 40 hexadecimal digits of the SHA-256 of the derivation's inputs."""
    document = {"base": {"repository": base["repository"], "revision": base["revision"]},
                "donor": {"repository": donor["repository"], "revision": donor["revision"],
                          "files": list(donor["files"])},
                "recipe": {"path": recipe["path"], "sha256": recipe["sha256"]}}
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:40]


def slug(repository):
    """``<owner>--<name>`` of a repository name, as checkpoint directories and pin manifests use it."""
    if (not isinstance(repository, str) or "--" in repository or ".." in repository
            or not re.fullmatch(_PART + "/" + _PART, repository)):
        raise ValueError(f"{repository!r} is not an owner/name checkpoint repository")
    return repository.replace("/", "--")


def manifest_path(model, root=None):
    """The manifest of derived ``model`` (``repository``, ``revision``)."""
    return Path(root if root is not None else profiles.ROOT) / "profiles/checkpoints" / slug(
        model["repository"]) / (model["revision"] + ".json")


def record(manifest):
    """The ``derivation.json`` document of a manifest."""
    return {"schema": RECORD_SCHEMA,
            "checkpoint": {"repository": manifest["repository"], "revision": manifest["revision"]},
            "base": dict(manifest["base"]),
            "donor": {"repository": manifest["donor"]["repository"], "revision": manifest["donor"]["revision"]},
            "recipe": dict(manifest["recipe"]), **manifest["transform"]}


def record_bytes(value):
    """Canonical JSON of a record, as the recipe writes ``derivation.json``."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def required(manifest):
    """``{name: {"size", "sha256"}}`` of every file of the derived checkpoint."""
    return {name: {"size": item["size"], "sha256": item["sha256"]} for name, item in sorted(manifest["files"].items())}


def files_of(manifest, origin):
    """``{name: size}`` of the files with ``origin`` (``base`` or ``recipe``)."""
    return {name: item["size"] for name, item in sorted(manifest["files"].items()) if item["origin"] == origin}


def directory(base, model):
    """The directory beside base checkpoint directory ``base`` for checkpoint ``model``.

    ``base`` must be a SparkRing checkpoint directory,
    ``<root>/<owner>--<name>/<revision>``; the result is
    ``<root>/<slug of model>/<revision of model>``. A base served in place from a
    named copy has no such root, so a derived checkpoint refuses it.
    """
    base = posixpath.normpath(base)
    parent, revision = posixpath.split(base)
    root, name = posixpath.split(parent)
    if not base.startswith("/") or not re.fullmatch(r"[0-9a-f]{40}", revision) or "--" not in name or root == "/":
        raise ValueError(f"{base} is not a SparkRing checkpoint directory; a derived checkpoint is written beside "
                         "its base's SparkRing checkpoint directory, never beside a named copy")
    return posixpath.join(root, slug(model["repository"]), model["revision"])


def listed(profiles_ids=None, *, root=None):
    """The derived checkpoints that installer profiles list, by ``(repository, revision)``.

    Each value holds the checkpoint ``name``, the ``profiles`` that list it and
    the ``base`` and ``donor`` models' ``repository`` and ``revision``, read
    from the profiles' checkpoint tables. ``profiles_ids`` defaults to the
    profiles ``sparkring install`` offers.
    """
    if profiles_ids is None:
        from runtime.common import installer_image
        profiles_ids = installer_image.SUPPORTED
    found = {}
    for profile_id in sorted(profiles_ids):
        try:
            metadata, _ = profiles.load(profile_id, Path(root if root is not None else profiles.ROOT))
            configuration = profiles.read_json(profiles.local_path(metadata["configuration"]["path"],
                                                                   Path(root if root is not None else profiles.ROOT)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        table = configuration.get("checkpoints") or {}
        for name, value in table.items():
            derived = value.get("derived") if isinstance(value, dict) else None
            if not isinstance(derived, dict) or derived.get("base") not in table or derived.get("donor") not in table:
                continue
            model = value["model"]
            item = found.setdefault((model["repository"], model["revision"]), {
                "name": name, "profiles": [],
                "base": {key: table[derived["base"]]["model"][key] for key in ("repository", "revision")},
                "donor": {key: table[derived["donor"]]["model"][key] for key in ("repository", "revision")}})
            item["profiles"].append(profile_id)
    return found


def view(card, manifest):
    """The selection card of the derived checkpoint itself: its repository and revision instead of the base's."""
    return {**card, "model_repository": manifest["repository"], "model_revision": manifest["revision"]}


def donor_card(card, manifest, *, root=None):
    """The selection card of the donor checkpoint, whose pin manifest and settings its entry names."""
    name = entry(_configuration(card, root), card["target_variant"])["donor"]
    return {**card, "model_repository": manifest["donor"]["repository"],
            "model_revision": manifest["donor"]["revision"], "target_variant": name}


def _configuration(card, root):
    return profiles.read_json(profiles.local_path(card["configuration"], Path(root if root is not None
                                                                               else profiles.ROOT)))


def model_of(card, *, root=None):
    """The derived checkpoint's ``model`` when the card's checkpoint is derived, else None.

    Reads only the profile configuration; ``load`` validates the manifest.
    """
    if not card.get("target_variant") or not card.get("configuration"):
        return None
    configuration = _configuration(card, root)
    value = (configuration.get("checkpoints") or {}).get(card["target_variant"]) if isinstance(configuration,
                                                                                               dict) else None
    return dict(value["model"]) if isinstance(value, dict) and "derived" in value else None


def load(card, *, root=None):
    """The validated manifest of the card's checkpoint when the profile derives it, else None.

    ``card`` is a selection whose ``target_variant`` names the checkpoint; its
    repository and revision may be the base's (``runtime.common.setup.selection``)
    or the derived checkpoint's (``view``).
    """
    root = Path(root if root is not None else profiles.ROOT)
    configuration = _configuration(card, root)
    derived = entry(configuration, card.get("target_variant"))
    if derived is None:
        return None
    table = configuration["checkpoints"]
    model = table[card["target_variant"]]["model"]
    path = manifest_path(model, root)
    if not path.is_file():
        raise ValueError(f"No manifest {path.relative_to(root).as_posix()} for derived checkpoint "
                         f"{card['target_variant']}")
    manifest = profiles.read_json(path)
    from runtime.common import installer
    base_card = {**card, "model_repository": table[derived["base"]]["model"]["repository"],
                 "model_revision": table[derived["base"]]["model"]["revision"], "target_variant": derived["base"]}
    donor = {**card, "model_repository": table[derived["donor"]]["model"]["repository"],
             "model_revision": table[derived["donor"]]["model"]["revision"], "target_variant": derived["donor"]}
    check(manifest, model, installer.checkpoint_pins(base_card, root=root),
          installer.checkpoint_pins(donor, root=root), root=root,
          name=path.relative_to(root).as_posix())
    return manifest


def check(manifest, model, base_pins, donor_pins, *, root=None, name="the derived checkpoint manifest"):
    """Refuse a manifest that does not pin exactly what ``model`` and the two pin manifests give.

    ``model`` is the profile entry's derived ``model``; ``base_pins`` and
    ``donor_pins`` are the validated pin manifests of the base and donor
    entries. Each refusal names the manifest and the field or file.
    """
    root = Path(root if root is not None else profiles.ROOT)

    def require(condition, reason):
        if not condition:
            raise ValueError(f"{name}: {reason}")

    require(isinstance(manifest, dict) and set(manifest) == KEYS, "expected exactly " + ", ".join(sorted(KEYS)))
    require(manifest["schema"] == SCHEMA, "expected " + SCHEMA)
    require((manifest["repository"], manifest["revision"]) == (model["repository"], model["revision"]),
            "names another checkpoint than the profile's entry")
    require(manifest["repository"].split("/")[0] == OWNER, f"a derived checkpoint's repository is {OWNER}/<name>")
    for key, pins in (("base", base_pins), ("donor", donor_pins)):
        value = manifest[key]
        require(isinstance(value, dict) and value.get("repository") == pins["repository"]
                and value.get("revision") == pins["revision"], f"{key} differs from the profile's {key} entry")
    require(set(manifest["base"]) == {"repository", "revision"}, "base holds repository and revision only")
    donor = manifest["donor"]
    require(set(donor) == {"repository", "revision", "files"}, "donor holds repository, revision and files")
    donor_required = set(donor_pins["files"]) - set(donor_pins["optional"])
    require(isinstance(donor["files"], list) and donor["files"] and donor["files"] == sorted(set(donor["files"]))
            and set(donor["files"]) <= donor_required, "donor files must be sorted, distinct required donor files")
    recipe = manifest["recipe"]
    require(isinstance(recipe, dict) and set(recipe) == {"path", "sha256"} and isinstance(recipe["path"], str)
            and recipe["path"].startswith("runtime/"), "recipe holds the path of a module below runtime/ and its sha256")
    recipe_file = profiles.local_path(recipe["path"], root)
    require(hashlib.sha256(recipe_file.read_bytes()).hexdigest() == recipe["sha256"],
            f"{recipe['path']} differs from the pinned recipe")
    expected = identity(manifest["base"], donor, recipe)
    require(manifest["revision"] == expected, f"revision is not the identity of its base, donor and recipe, {expected}")
    require(isinstance(manifest["transform"], dict) and not {"schema", "checkpoint", "base", "donor", "recipe"}
            & set(manifest["transform"]), "transform must not repeat the record's identity fields")
    files = manifest["files"]
    require(isinstance(files, dict) and files, "files must map each file name to its pins")
    base_required = {key: entry for key, entry in base_pins["files"].items() if key not in base_pins["optional"]}
    for key, item in files.items():
        require(isinstance(item, dict) and set(item) == {"origin", "size", "sha256"} and item["origin"] in ORIGINS
                and type(item["size"]) is int and item["size"] > 0 and isinstance(item["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None
                and PurePosixPath(key).as_posix() == key and ".." not in key.split("/"),
                f"{key}: expected origin, a positive size and a sha256")
        if item["origin"] == "base":
            require(key in base_required and (item["size"], item["sha256"]) == (
                base_required[key]["size"], base_required[key]["sha256"]), f"{key} differs from the base's pins")
        else:
            require(key in base_required or key == RECORD, f"{key} is neither a base file nor {RECORD}")
    require(set(base_required) <= set(files), "the derived checkpoint lacks base files: "
            + ", ".join(sorted(set(base_required) - set(files))[:5]))
    require(files.get(RECORD, {}).get("origin") == "recipe", f"{RECORD} must be a recipe file")
    written = record_bytes(record(manifest))
    require((files[RECORD]["size"], files[RECORD]["sha256"]) == (len(written), hashlib.sha256(written).hexdigest()),
            f"{RECORD} differs from the record of this manifest, {len(written)} bytes with SHA-256 "
            f"{hashlib.sha256(written).hexdigest()}")
    require(manifest["index"] in files and files[manifest["index"]]["origin"] == "recipe",
            "index must name a recipe file")
    require(files.get("config.json", {}).get("sha256") == model["config_sha256"]
            and files[manifest["index"]]["sha256"] == model["index_sha256"],
            "config.json or the index differs from the profile entry's model")
    return manifest
