"""Derive a Qwen3.8-Flash-Next checkpoint whose text attention projections are MXFP8.

Inputs
------
- The *base*: a checkpoint directory whose text attention projections are BF16
  ``weight`` tensors. For the profiles' derived checkpoint it is QAD step 5500 of
  ``local-inference-lab/Qwen3.8-Flash-Next-NVFP4``.
- The *donor*: a directory holding the weight index and the shards of a checkpoint
  that stores the same projections as MXFP8 block32, an E4M3 ``weight`` with one
  UE8M0 ``weight_scale`` byte per 32 values (QAD step 4000 of the same repository).
- The *record*: the derived checkpoint's provenance document, which also states the
  projections and shards the recipe must find (``modules`` and
  ``rewritten_shards``).

The projections are the ``weight`` tensors whose names ``TARGET`` matches: in each
Gated DeltaNet layer ``linear_attn.in_proj_qkv``, ``in_proj_z``, ``in_proj_a``,
``in_proj_b`` and ``out_proj``, and in each full-attention layer
``self_attn.q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` and
``indexer.index_qk_proj``.

Transform
---------
For every projection the recipe quantizes the base's BF16 weight with vLLM's
``_mxfp8_e4m3_quantize_torch`` and requires the result to equal the donor's
``weight`` and ``weight_scale`` bytes and shapes exactly. Only then does the
derived checkpoint store the donor's two tensors in place of the BF16 weight; a
projection that fails the check stops the recipe with its module name before its
shard is written. The donor's tensors are therefore the base's own frozen weights
in MXFP8, not other weights.

The recipe writes into the output directory, which must be empty:

- each base shard that holds a projection, with the donor's ``weight`` and
  ``weight_scale`` in place of the BF16 ``weight`` and every other tensor and the
  header metadata unchanged;
- ``config.json`` and ``hf_quant_config.json``: the base's documents with one
  ``{"quant_algo": "MXFP8", "group_size": 32}`` entry per projection added under
  ``quantized_layers``;
- ``model.safetensors.index.json``: the base's weight map plus each
  ``weight_scale``, and the total size of all shards;
- ``derivation.json``: the record, as canonical JSON.

Every other file of the base is unchanged; the installer hard-links those files
into the derived checkpoint directory.

Determinism
-----------
The output bytes depend only on the input bytes, the record and this file. JSON
documents are written with Python's ``json`` module in a fixed form. Shards are
written by this module's safetensors writer (``write_shard``), which orders
tensors as the safetensors library does (descending dtype, then name), writes the
header as compact JSON with ``__metadata__`` first and its keys sorted, and pads
the header with spaces to a multiple of 8 bytes. For a header with at most one
metadata entry this equals ``safetensors.torch.save_file`` byte for byte, which
``runtime/common/test_mxfp8_attention.py`` checks; with more entries the library's
own order is not fixed, while this writer's is. The derived checkpoint's manifest
pins the SHA-256 of every output file.

Running
-------
The installer runs this file's text with ``python3 -c`` inside the deployment's
installer image, CPU-only, without network and with the base and donor mounted
read-only::

    python3 -c "$(cat mxfp8_attention.py)" --base /base --donor /donor --out /out --record '{...}'

Only the requantization check imports torch and vLLM (``vllm_quantizer``); the
rest uses the standard library. The checkpoint manifest pins this file's
SHA-256, so any change to it gives the derived checkpoint a new identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys

TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_(qkv|z|a|b)|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj))\.weight$"
)
ENTRY = {"quant_algo": "MXFP8", "group_size": 32}
INDEX = "model.safetensors.index.json"
CONFIGS = ("config.json", "hf_quant_config.json")
RECORD = "derivation.json"
# Dtypes of the safetensors format in the library's order; tensors are stored by
# descending position, then by name. Other dtypes are refused rather than guessed.
DTYPES = ("BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64",
          "I64", "U64")
ITEM_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
              "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8}
HEADER_LIMIT = 100 << 20
BLOCK = 16 << 20


class RecipeError(ValueError):
    """An input that the recipe refuses; the message names the file or module."""


# safetensors reading and writing -------------------------------------------------

def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_header(path):
    """``(metadata, tensors, data_start)`` of safetensors file ``path``.

    ``tensors`` maps each name to ``(dtype, shape, start, end)``, offsets relative
    to ``data_start``; ``metadata`` is the header's ``__metadata__`` or None.
    """
    with open(path, "rb") as stream:
        head = stream.read(8)
        if len(head) != 8:
            raise RecipeError(f"{path} is not a safetensors file")
        (length,) = struct.unpack("<Q", head)
        if length > HEADER_LIMIT:
            raise RecipeError(f"{path} declares a {length}-byte header")
        text = stream.read(length)
    if len(text) != length:
        raise RecipeError(f"{path} ends inside its header")
    document = json.loads(text, object_pairs_hook=_unique)
    metadata = document.pop("__metadata__", None)
    tensors = {}
    for name, info in document.items():
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in ITEM_BYTES:
            raise RecipeError(f"{path}: {name} has dtype {dtype!r}, which this recipe does not write")
        if (not isinstance(shape, list) or not all(type(size) is int and size >= 0 for size in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(value) is int for value in offsets) or not 0 <= offsets[0] <= offsets[1]):
            raise RecipeError(f"{path}: {name} has a malformed header entry")
        count = 1
        for size in shape:
            count *= size
        if offsets[1] - offsets[0] != count * ITEM_BYTES[dtype]:
            raise RecipeError(f"{path}: {name} has {offsets[1] - offsets[0]} bytes, not {count} {dtype} values")
        tensors[name] = (dtype, list(shape), offsets[0], offsets[1])
    if metadata is not None and (not isinstance(metadata, dict)
                                 or not all(isinstance(v, str) for v in metadata.values())):
        raise RecipeError(f"{path}: __metadata__ must map names to strings")
    return metadata, tensors, 8 + length


def read_tensor(path, data_start, start, end):
    """The bytes of one tensor, read at its offsets."""
    with open(path, "rb") as stream:
        stream.seek(data_start + start)
        data = stream.read(end - start)
    if len(data) != end - start:
        raise RecipeError(f"{path} ends inside a tensor")
    return data


def layout(tensors, metadata):
    """``(header, order)`` of a shard holding ``tensors``: ``{name: (dtype, shape, size)}``.

    ``header`` is the 8-byte length and the padded JSON header; ``order`` lists
    the names in storage order. The order and header form are the safetensors
    library's (see the module description).
    """
    for name, (dtype, _, _) in tensors.items():
        if dtype not in DTYPES:
            raise RecipeError(f"{name} has dtype {dtype!r}, which this recipe does not write")
    order = sorted(tensors, key=lambda name: (-DTYPES.index(tensors[name][0]), name))
    parts, offset = [], 0
    if metadata is not None:
        parts.append(json.dumps("__metadata__") + ":"
                     + json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    for name in order:
        dtype, shape, size = tensors[name]
        info = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        parts.append(json.dumps(name, ensure_ascii=False) + ":" + json.dumps(info, separators=(",", ":")))
        offset += size
    text = ("{" + ",".join(parts) + "}").encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text, order


def write_shard(path, tensors, metadata):
    """Write a safetensors shard; returns its size in bytes.

    ``tensors`` maps each name to ``(dtype, shape, source)``, where ``source`` is
    ``bytes`` or ``(file, absolute start, size)`` copied from an existing file. The
    file is created exclusively and synced before it is closed.
    """
    sized = {}
    for name, (dtype, shape, source) in tensors.items():
        sized[name] = (dtype, shape, len(source) if isinstance(source, bytes) else source[2])
    header, order = layout(sized, metadata)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        written = _write(fd, header)
        for name in order:
            source = tensors[name][2]
            if isinstance(source, bytes):
                written += _write(fd, source)
                continue
            file, start, size = source
            with open(file, "rb") as stream:
                stream.seek(start)
                left = size
                while left:
                    block = stream.read(min(BLOCK, left))
                    if not block:
                        raise RecipeError(f"{file} ends inside {name}")
                    written += _write(fd, block)
                    left -= len(block)
        os.fsync(fd)
    finally:
        os.close(fd)
    return written


def _write(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]
    return len(data)


# JSON documents --------------------------------------------------------------------

def read_json(path):
    with open(path, "rb") as stream:
        return json.loads(stream.read())


def json_bytes(value):
    """The form ``config.json``, ``hf_quant_config.json`` and the index are written in."""
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def quantization(document):
    """The object of a configuration document that holds ``quantized_layers``.

    ``config.json`` keeps it in ``quantization_config``; ModelOpt's
    ``hf_quant_config.json`` in ``quantization`` or, in flat form, at the top level.
    """
    if "quantization_config" in document:
        return document["quantization_config"]
    if "quantization" in document:
        return document["quantization"]
    return document


def quantized_config(text, modules, name):
    """``name``'s bytes with an ``ENTRY`` for every module of ``modules`` under ``quantized_layers``."""
    document = json.loads(text)
    layers = quantization(document).setdefault("quantized_layers", {})
    clash = [module for module in modules if module in layers]
    if clash:
        raise RecipeError(f"{name} already lists {', '.join(clash[:3])}")
    layers.update({module: dict(ENTRY) for module in modules})
    return json_bytes(document)


def derived_index(index, scales, total_size):
    """The weight index with each ``weight_scale`` of ``scales`` (name to shard) and the new total size."""
    weight_map = dict(index["weight_map"])
    weight_map.update(scales)
    return json_bytes({"metadata": {**index.get("metadata", {}), "total_size": total_size},
                       "weight_map": dict(sorted(weight_map.items()))})


def record_bytes(record):
    """``derivation.json``: the record as canonical JSON."""
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def modules_digest(modules):
    """SHA-256 of the module names joined by newlines, in the order the recipe treats them."""
    return hashlib.sha256("\n".join(modules).encode()).hexdigest()


# The derivation ----------------------------------------------------------------------

def plan(base_index, donor_index):
    """``(modules, shards)``: the projections in recipe order and, per base shard, its projection weights.

    Refuses a donor without a projection's ``weight`` or ``weight_scale`` and a
    base that already holds a ``weight_scale`` for one.
    """
    source, donor = base_index["weight_map"], donor_index["weight_map"]
    targets = sorted(name for name in source if TARGET.match(name))
    modules = [name[:-len(".weight")] for name in targets]
    for module in modules:
        for suffix in (".weight", ".weight_scale"):
            if module + suffix not in donor:
                raise RecipeError(f"The donor lacks {module}{suffix}")
        if module + ".weight_scale" in source:
            raise RecipeError(f"The base already holds {module}.weight_scale")
    shards = {}
    for name in targets:
        shards.setdefault(source[name], []).append(name)
    return modules, shards


def check_record(record, modules, shards):
    """Refuse a record whose modules, format or rewritten shards differ from what the inputs give."""
    expected = {"count": len(modules), "sha256": modules_digest(modules)}
    if not isinstance(record, dict) or record.get("modules") != expected:
        raise RecipeError(f"The inputs hold {len(modules)} projections with digest {expected['sha256']}, which the "
                          "derived checkpoint's record does not name")
    if record.get("format") != ENTRY:
        raise RecipeError("The record names another format than MXFP8 block 32")
    if record.get("rewritten_shards") != sorted(shards):
        raise RecipeError("The projections lie in shards " + ", ".join(sorted(shards))
                          + ", not in the record's rewritten shards")


def derive(base, donor, out, record, quantize, log=print):
    """Write the derived files into the empty directory ``out``; returns their names and sizes.

    ``quantize(data, shape)`` returns ``(shape, weight bytes, scale bytes)`` of the
    MXFP8 quantization of BF16 tensor bytes ``data``; ``vllm_quantizer`` supplies
    vLLM's. ``record`` is written as ``derivation.json`` after the checks.
    """
    if os.listdir(out):
        raise RecipeError(f"{out} is not empty")
    base_index = read_json(os.path.join(base, INDEX))
    donor_index = read_json(os.path.join(donor, INDEX))
    modules, shards = plan(base_index, donor_index)
    check_record(record, modules, shards)
    donor_headers = {}

    def donor_tensor(name):
        shard = donor_index["weight_map"][name]
        if shard not in donor_headers:
            donor_headers[shard] = read_header(os.path.join(donor, shard))
        _, entries, start = donor_headers[shard]
        dtype, shape, begin, end = entries[name]
        return dtype, shape, read_tensor(os.path.join(donor, shard), start, begin, end)

    sizes, scales = {}, {}
    for shard in sorted(shards):
        path = os.path.join(base, shard)
        metadata, entries, start = read_header(path)
        tensors = {name: (dtype, shape, (path, start + begin, end - begin))
                   for name, (dtype, shape, begin, end) in entries.items()}
        for name in shards[shard]:
            module = name[:-len(".weight")]
            dtype, shape, begin, end = entries[name]
            if dtype != "BF16":
                raise RecipeError(f"{module}: the base stores its weight as {dtype}, not BF16")
            weight_dtype, weight_shape, weight = donor_tensor(name)
            scale_dtype, scale_shape, scale = donor_tensor(module + ".weight_scale")
            if (weight_dtype, scale_dtype) != ("F8_E4M3", "U8"):
                raise RecipeError(f"{module}: the donor stores {weight_dtype} and {scale_dtype}, not F8_E4M3 and U8")
            shape_q, quantized, scale_q = quantize(read_tensor(path, start, begin, end), shape)
            if list(shape_q) != weight_shape or quantized != weight or scale_q != scale:
                raise RecipeError(f"{module}: quantizing the base's BF16 weight does not reproduce the donor's MXFP8 "
                                  "weight and scale, so the donor is not this base's weight in MXFP8; nothing was "
                                  f"written for {shard}")
            tensors[name] = ("F8_E4M3", weight_shape, weight)
            tensors[module + ".weight_scale"] = ("U8", scale_shape, scale)
            scales[module + ".weight_scale"] = shard
        sizes[shard] = write_shard(os.path.join(out, shard), tensors, metadata)
        log(f"rewrote {shard}: {len(shards[shard])} projections")
    for name in CONFIGS:
        with open(os.path.join(base, name), "rb") as stream:
            data = quantized_config(stream.read(), modules, name)
        sizes[name] = _create(os.path.join(out, name), data)
    total = 0
    for shard in sorted(set(base_index["weight_map"].values())):
        total += sizes[shard] if shard in sizes else os.stat(os.path.join(base, shard)).st_size
    sizes[INDEX] = _create(os.path.join(out, INDEX), derived_index(base_index, scales, total))
    sizes[RECORD] = _create(os.path.join(out, RECORD), record_bytes(record))
    log(f"derived {len(modules)} projections in {len(shards)} shards")
    return sizes


def _create(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        _write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(data)


def vllm_quantizer():
    """``quantize`` for ``derive`` from vLLM's ``_mxfp8_e4m3_quantize_torch``, run on the CPU."""
    import torch
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_torch

    def quantize(data, shape):
        source = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)
        weight, scale = _mxfp8_e4m3_quantize_torch(source, False)
        return (list(weight.shape), weight.contiguous().view(torch.uint8).numpy().tobytes(),
                scale.contiguous().view(torch.uint8).numpy().tobytes())
    return quantize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--record", required=True, help="the derived checkpoint's record, as JSON")
    args = parser.parse_args(argv)
    try:
        derive(args.base, args.donor, args.out, json.loads(args.record), vllm_quantizer(),
               log=lambda text: print(text, flush=True))
    except RecipeError as error:
        print(f"mxfp8_attention: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
