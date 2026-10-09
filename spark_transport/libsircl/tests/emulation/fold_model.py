"""Host model of the fold pack (kernels/sircl_fold.cu): inputs, element encodings and the rank-ordered fold.

Every NCCL datatype is modelled with NumPy. Integers fold in their own type, sums and products wrapping
through the unsigned twin; avg is the wrapped sum divided by the group size, truncated toward zero.
float16, bfloat16 and the two float8 formats are exact float32 values folded in float32 and rounded once
to the element type (round to nearest even, torch's CPU conversions for the formats NumPy lacks); float32
and float64 fold in their own type. Max and min keep the earlier rank's value on ties.
"""
from __future__ import annotations

import numpy as np

# name: (NCCL datatype enum, element bytes, NumPy type or None for the formats only torch converts)
TYPES = {
    "int8": (0, 1, np.int8), "uint8": (1, 1, np.uint8), "int32": (2, 4, np.int32), "uint32": (3, 4, np.uint32),
    "int64": (4, 8, np.int64), "uint64": (5, 8, np.uint64), "float16": (6, 2, np.float16),
    "float32": (7, 4, np.float32), "float64": (8, 8, np.float64), "bfloat16": (9, 2, None),
    "float8e4m3": (10, 1, None), "float8e5m2": (11, 1, None),
}
OPS = {"sum": 0, "prod": 1, "max": 2, "min": 3, "avg": 4}
UNSIGNED = {np.int8: np.uint8, np.uint8: np.uint8, np.int32: np.uint32, np.uint32: np.uint32, np.int64: np.uint64,
            np.uint64: np.uint64}


def _torch_format(torch, name):
    return {"bfloat16": torch.bfloat16, "float8e4m3": torch.float8_e4m3fn, "float8e5m2": torch.float8_e5m2}[name]


def is_integer(name: str) -> bool:
    kind = TYPES[name][2]
    return kind is not None and np.issubdtype(kind, np.integer)


def encode(torch, name: str, values) -> bytes:
    """The element bytes of `values`, rounded to nearest even where the type is narrower."""
    kind = TYPES[name][2]
    if kind is not None:
        return np.ascontiguousarray(values.astype(kind)).tobytes()
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(_torch_format(torch, name))
    return tensor.view(torch.uint8).numpy().tobytes()


def decode(torch, name: str, data: bytes):
    """Model values of element bytes: the integer type itself, or exact floating values."""
    kind = TYPES[name][2]
    if kind is not None:
        return np.frombuffer(data, dtype=kind).copy()
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(_torch_format(torch, name))
    return tensor.float().numpy()


def inputs(torch, name: str, op: str, world: int, count: int, seed: int):
    """Every rank's (bytes, model values): integers over the whole range; floating values from a normal
    distribution, or uniform in [0.5, 1.5] for products."""
    kind = TYPES[name][2]
    result = []
    for rank in range(world):
        rng = np.random.default_rng(seed * 1009 + rank)
        if is_integer(name):
            info = np.iinfo(kind)
            values = rng.integers(info.min, info.max, size=count, dtype=kind, endpoint=True)
            result.append((values.tobytes(), values))
            continue
        raw = rng.uniform(0.5, 1.5, count) if op == "prod" else rng.standard_normal(count)
        data = encode(torch, name, raw if name == "float64" else raw.astype(np.float32))
        result.append((data, decode(torch, name, data)))
    return result


def fold(torch, name: str, op: str, values, world: int) -> bytes:
    """The bytes every rank stores: the rank-ordered fold of every rank's model values."""
    kind = TYPES[name][2]
    if is_integer(name):
        twin = UNSIGNED[kind]
        if op in ("sum", "avg", "prod"):
            acc = values[0].view(twin).copy()
            for value in values[1:]:
                acc = acc * value.view(twin) if op == "prod" else acc + value.view(twin)
            acc = acc.view(kind)
            if op == "avg":
                if np.issubdtype(kind, np.signedinteger):
                    wide = acc.astype(np.int64)
                    quotient = wide // world
                    quotient = quotient + (((wide % world) != 0) & (wide < 0))
                    acc = quotient.astype(kind)
                else:
                    acc = acc // kind(world)
        else:
            acc = values[0].copy()
            for value in values[1:]:
                acc = np.where(value > acc, value, acc) if op == "max" else np.where(value < acc, value, acc)
        return np.ascontiguousarray(acc.astype(kind)).tobytes()
    wide = np.float64 if name == "float64" else np.float32
    acc = values[0].astype(wide)
    for value in values[1:]:
        value = value.astype(wide)
        if op in ("sum", "avg"):
            acc = acc + value
        elif op == "prod":
            acc = acc * value
        elif op == "max":
            acc = np.where(value > acc, value, acc)
        else:
            acc = np.where(value < acc, value, acc)
    if op == "avg":
        acc = acc / wide(world)
    return encode(torch, name, acc)
