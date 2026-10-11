#!/usr/bin/env python3
"""Write the manifest of a profile's NVFP4-CSF serving directory.

A profile that serves a checkpoint in Local Inference Lab's NVFP4-CSF
container layout lists, as its default checkpoint, a derived checkpoint
without a base entry or donor (``"derived": {}``): the serving directory that
``runtime/common/csf_serving.py`` describes. Its manifest
(``sparkring-derived-checkpoint/v1``,
``profiles/checkpoints/sparkring-derived--<name>/<revision>.json``) pins

- every required file of the container, hard-linked from the container's
  SparkRing checkpoint directory: under its own name, or for the files of
  ``csf_serving.layout`` at the root with the container file it links as
  ``source``;
- ``config.json`` and ``derivation.json``, which the recipe writes.

The revision is ``derived_checkpoint.identity`` of the container, no donor and
the recipe. The serving ``config.json`` depends on the container's
``metadata/config.json``, which this script downloads from the Hub together
with ``manifest.json`` (for the checkpoint family), each after its size and
SHA-256 are checked against the container's pin manifest; ``--source-config``
and ``--container-manifest`` read local copies instead. No weight file is
read. The script prints the ``model`` object of the profile's default entry.

Usage::

    python scripts/pin_csf_serving.py --profile glm53-nvfp4-csf-tp8 \\
        --repository sparkring-derived/GLM-5.3-NVFP4-CSF-Serving
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import csf_serving, derived_checkpoint, installer, profiles  # noqa: E402
from scripts import pin_checkpoint  # noqa: E402

RECIPE = "runtime/common/csf_serving.py"
INDEX = "model.safetensors.index.json"


def transform(pins, container_manifest):
    """The record fields the recipe checks and applies, from the pin manifest and the container's manifest.json."""
    if not isinstance(container_manifest, dict) or container_manifest.get("schema") != csf_serving.LAYOUT:
        raise ValueError(f"manifest.json does not name the layout {csf_serving.LAYOUT}")
    return {"checkpoint_root": csf_serving.CHECKPOINT_ROOT, "family": container_manifest["family"],
            "layout": csf_serving.LAYOUT,
            "source_config": {"name": csf_serving.METADATA + csf_serving.CONFIG,
                              "sha256": pins["files"][csf_serving.METADATA + csf_serving.CONFIG]["sha256"]}}


def manifest(pins, repository, transform_value, config, *, root=ROOT):
    """The serving directory's manifest; ``config`` is the ``{"size", "sha256"}`` of the written config.json."""
    recipe_bytes = (Path(root) / RECIPE).read_bytes()
    recipe = {"path": RECIPE, "sha256": hashlib.sha256(recipe_bytes).hexdigest()}
    base = {"repository": pins["repository"], "revision": pins["revision"]}
    required = sorted(set(pins["files"]) - set(pins["optional"]))
    links = csf_serving.layout(required)
    files = {}
    for name in required:
        if name != csf_serving.CONFIG and name not in links.values():
            files[name] = {"origin": "base", "size": pins["files"][name]["size"], "sha256": pins["files"][name]["sha256"]}
    for name, source in links.items():
        files[name] = {"origin": "base", "size": pins["files"][source]["size"],
                       "sha256": pins["files"][source]["sha256"], "source": source}
    files[csf_serving.CONFIG] = {"origin": "recipe", "size": config["size"], "sha256": config["sha256"]}
    value = {"schema": derived_checkpoint.SCHEMA, "repository": repository,
             "revision": derived_checkpoint.identity(base, None, recipe), "base": base, "donor": None,
             "recipe": recipe, "transform": transform_value, "index": INDEX, "files": {}}
    record = derived_checkpoint.record_bytes(derived_checkpoint.record(value))
    files[derived_checkpoint.RECORD] = {"origin": "recipe", "size": len(record),
                                        "sha256": hashlib.sha256(record).hexdigest()}
    value["files"] = dict(sorted(files.items()))
    return value


def model(value, pins):
    """The ``model`` of the profile's default entry for manifest ``value``."""
    return {"repository": value["repository"], "revision": value["revision"],
            "config_sha256": value["files"][csf_serving.CONFIG]["sha256"],
            "index_sha256": pins["files"][csf_serving.METADATA + INDEX]["sha256"]}


def encode(value):
    """Manifest text: one line per top-level field and per file, for readable diffs."""
    lines = ["{"]
    for key in ("schema", "repository", "revision", "base", "donor", "recipe", "transform", "index"):
        lines.append(f"  {json.dumps(key)}: {json.dumps(value[key])},")
    lines.append('  "files": {')
    lines.append(",\n".join(f"    {json.dumps(name)}: {json.dumps(entry)}" for name, entry in value["files"].items()))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _checked(data, pins, name):
    entry = pins["files"][name]
    if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{name} differs from the container's pin manifest")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True, help="the profile whose default entry is the serving directory")
    parser.add_argument("--repository", help="the serving directory's sparkring-derived/<name> repository "
                                             "(default: the one the profile's default entry names)")
    parser.add_argument("--source-config", type=Path, help="a local copy of the container's metadata/config.json")
    parser.add_argument("--container-manifest", type=Path, help="a local copy of the container's manifest.json")
    parser.add_argument("--endpoint", default=pin_checkpoint.ENDPOINT)
    args = parser.parse_args(argv)
    try:
        card = installer.setup.selection(args.profile)
        pins = installer.checkpoint_pins(card)
        configuration = profiles.read_json(profiles.local_path(card["configuration"]))
        entry = configuration["checkpoints"][configuration["checkpoint"]]
        if entry.get("derived") != {}:
            raise ValueError(f"{args.profile}'s default checkpoint is not derived from the profile's model alone")
        repository = args.repository or entry["model"]["repository"]
        hub = pin_checkpoint.Hub(args.endpoint)

        def read(path, name):
            if path is not None:
                return _checked(path.read_bytes(), pins, name)
            return _checked(pin_checkpoint.fetch(hub, pins["repository"], pins["revision"], name,
                                                 pins["files"][name]["size"]), pins, name)

        source = read(args.source_config, csf_serving.METADATA + csf_serving.CONFIG)
        layout = json.loads(read(args.container_manifest, "manifest.json"))
        written = csf_serving.serving_config(source)
        value = manifest(pins, repository, transform(pins, layout),
                         {"size": len(written), "sha256": hashlib.sha256(written).hexdigest()})
        path = derived_checkpoint.manifest_path(value)
        pin_checkpoint.replace(path, encode(value))
    except (ValueError, OSError, KeyError) as error:
        print(f"pin_csf_serving: {error}", file=sys.stderr)
        return 1
    print(path.relative_to(ROOT).as_posix())
    print(json.dumps(model(value, pins), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
