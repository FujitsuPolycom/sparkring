"""Write the serving directory of a checkpoint in Local Inference Lab's NVFP4-CSF container layout.

A checkpoint in the container layout (``lil-nvfp4-csf-checkpoint/1``, such as
``local-inference-lab/GLM-5.3-NVFP4-CSF``) stores its routed experts' NVFP4
block scales losslessly compressed. Its root holds ``manifest.json`` and
``build-contract.json``, which describe the container, and ``tensors/``, the
source checkpoint's safetensors shards under their source names; the source's
configuration, tokenizer and weight index are byte copies in ``metadata/``.
The root holds no weight index, so no loader reads it as an ordinary
checkpoint.

vLLM reads the container with ``--quantization nvfp4_csf --load-format
nvfp4_csf`` from a *serving directory*: the files of ``metadata/`` with
``config.json``'s ``quantization_config`` replaced by::

    {"quant_method": "nvfp4_csf", "format_version": 1,
     "checkpoint_root": ROOT, "source_quantization_config": SOURCE}

where ``SOURCE`` is the source's ``quantization_config`` and ``ROOT`` the
absolute path of the container as the server process sees it. The reader
(vLLM ``nvfp4_csf_loader``) resolves ``ROOT``, reads ``manifest.json`` and
``build-contract.json`` there and opens the shards in ``ROOT/tensors``.

SparkRing's serving directory
-----------------------------
The installer builds the serving directory as a derived checkpoint without a
donor (``runtime/common/derived_checkpoint.py``) beside the container's
SparkRing checkpoint directory, and the rank container mounts it read-only at
``/models/target`` (``runtime.common.toolchain_profiles.container_spec``). A
path on the Spark is not visible inside that container, so the serving
directory is also the container root that ``ROOT`` names: it holds

- every required file of the container as a hard link to the container's
  file, which takes no space: under its own name (``manifest.json``,
  ``build-contract.json`` and ``tensors/`` among them), except the files of
  ``metadata/`` other than ``config.json``, which are linked at the root
  under their own base names instead (``layout``), where vLLM and the
  tokenizer read them;
- ``config.json``, which this recipe writes with ``ROOT`` set to
  ``/models/target`` (``CHECKPOINT_ROOT``), and ``derivation.json``, the
  record.

Each file of the container is linked once: a second link of one inode in the
directory would change the change time that the directory's journal recorded
for the first (``runtime.host.checkpoint_place``).

``ROOT`` is the same on every Spark, so the written ``config.json`` has one
SHA-256 that the profile pins (``config_sha256``), and the installer checks it
before the model starts as for any checkpoint.

Running
-------
The installer runs this file's text with ``python3 -c`` inside the
deployment's installer image, CPU-only, without network and with the
container's SparkRing checkpoint directory mounted read-only::

    python3 -c "$(cat csf_serving.py)" --base /base --out /out --record '{...}'

The record's ``layout``, ``family``, ``checkpoint_root`` and
``source_config`` (``{"name", "sha256"}``) are what the recipe must find and
write. It refuses a base whose ``manifest.json`` names another layout or
family, or whose source configuration differs from the record or from the
digest ``manifest.json`` records for it, and an output directory that is not
empty. The output bytes depend only on the source configuration, the record
and this file; JSON is written with Python's ``json`` module in a fixed form.
The derived checkpoint's manifest pins this file's SHA-256, so any change to
it, ``layout`` included, gives the serving directory another identity. Only
the standard library is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

LAYOUT = "lil-nvfp4-csf-checkpoint/1"
# Where the rank container mounts the serving directory, which is also the container root.
CHECKPOINT_ROOT = "/models/target"
METADATA = "metadata/"
CONFIG = "config.json"
RECORD = "derivation.json"
QUANT_METHOD = "nvfp4_csf"
FORMAT_VERSION = 1


class RecipeError(ValueError):
    """An input that the recipe refuses; the message names the file or field."""


def layout(required):
    """``{name: base file}`` of the files that a serving directory links at its root instead of in ``metadata/``.

    ``required`` is the container's required file names (its pin manifest's
    ``files`` minus ``optional``). Each required ``metadata/<name>`` other
    than ``metadata/config.json`` is linked as ``<name>`` and not under its
    own name; a name the container already holds at its root is refused
    rather than shadowed. Every other required file keeps its own name.
    """
    names = set(required)
    links = {}
    for name in sorted(names):
        if not name.startswith(METADATA) or "/" in name[len(METADATA):] or name == METADATA + CONFIG:
            continue
        target = name[len(METADATA):]
        if target in names:
            raise RecipeError(f"The container holds both {target} and {name}; the serving directory cannot link "
                              f"{name} at its root")
        links[target] = name
    if METADATA + CONFIG not in names or METADATA + "model.safetensors.index.json" not in names:
        raise RecipeError("The container's metadata lacks config.json or model.safetensors.index.json")
    return links


def serving_config(source, root=CHECKPOINT_ROOT):
    """The serving directory's ``config.json`` from the source's ``config.json`` bytes.

    Every key keeps its position and value except ``quantization_config``,
    which becomes the ``nvfp4_csf`` wrapper around the source's.
    """
    try:
        document = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RecipeError(f"The source {CONFIG} is not JSON: {error}") from None
    if not isinstance(document, dict) or not isinstance(document.get("quantization_config"), dict):
        raise RecipeError(f"The source {CONFIG} has no quantization_config")
    original = document["quantization_config"]
    if original.get("quant_method") == QUANT_METHOD:
        raise RecipeError(f"The source {CONFIG} is already a serving configuration")
    if not isinstance(root, str) or not root.startswith("/"):
        raise RecipeError("checkpoint_root must be an absolute path")
    document["quantization_config"] = {"quant_method": QUANT_METHOD, "format_version": FORMAT_VERSION,
                                       "checkpoint_root": root, "source_quantization_config": original}
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def record_bytes(record):
    """``derivation.json``: the record as canonical JSON (``derived_checkpoint.record_bytes``)."""
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read(path, limit=64 << 20):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    try:
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > limit:
                raise RecipeError(f"{path} exceeds {limit} bytes")
            chunks.append(chunk)
    finally:
        os.close(fd)


def _create(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(data)


def check_base(base, record):
    """The source ``config.json`` bytes after checking the container against the record."""
    expected = record.get("source_config") if isinstance(record, dict) else None
    if (not isinstance(expected, dict) or set(expected) != {"name", "sha256"}
            or expected["name"] != METADATA + CONFIG or record.get("layout") != LAYOUT
            or not isinstance(record.get("family"), str) or not isinstance(record.get("checkpoint_root"), str)):
        raise RecipeError(f"The record must name the layout {LAYOUT}, a family, the checkpoint root and the source "
                          f"configuration {METADATA}{CONFIG} with its SHA-256")
    try:
        manifest = json.loads(_read(os.path.join(base, "manifest.json")).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RecipeError(f"The base's manifest.json cannot be read: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != LAYOUT or manifest.get("family") != record["family"]:
        raise RecipeError(f"The base's manifest.json names layout {manifest.get('schema')!r} and family "
                          f"{manifest.get('family')!r}, not {LAYOUT} and {record['family']}")
    source = _read(os.path.join(base, expected["name"]))
    digest = hashlib.sha256(source).hexdigest()
    recorded = (manifest.get("metadata_sha256") or {}).get(CONFIG)
    if digest != expected["sha256"] or digest != recorded:
        raise RecipeError(f"The base's {expected['name']} has SHA-256 {digest}; the record names {expected['sha256']} "
                          f"and the container's manifest {recorded}. Nothing was written.")
    return source


def derive(base, out, record, log=print):
    """Write ``config.json`` and ``derivation.json`` into the empty directory ``out``; returns their sizes."""
    if os.listdir(out):
        raise RecipeError(f"{out} is not empty")
    source = check_base(base, record)
    sizes = {CONFIG: _create(os.path.join(out, CONFIG), serving_config(source, record["checkpoint_root"])),
             RECORD: _create(os.path.join(out, RECORD), record_bytes(record))}
    log(f"wrote {CONFIG} with checkpoint_root {record['checkpoint_root']} and {RECORD}")
    return sizes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--record", required=True, help="the serving directory's record, as JSON")
    args = parser.parse_args(argv)
    try:
        derive(args.base, args.out, json.loads(args.record), log=lambda text: print(text, flush=True))
    except RecipeError as error:
        print(f"csf_serving: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
