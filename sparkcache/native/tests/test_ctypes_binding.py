from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "python" / "spark_cache_native.py"
STAGED_BINDING = ROOT.parent / "spark_cache_native.py"
SPEC = importlib.util.spec_from_file_location("spark_cache_native", BINDING)
assert SPEC is not None and SPEC.loader is not None
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def test_staged_and_canonical_bindings_are_identical() -> None:
    assert STAGED_BINDING.read_bytes() == BINDING.read_bytes()


def test_ctypes_sizes_match_fixed_c_abi() -> None:
    assert ctypes.sizeof(native.PlacementConfig) == 48
    assert ctypes.sizeof(native.DestinationDescriptor) == 32
    assert ctypes.sizeof(native.ChunkDescriptor) == 64
    assert ctypes.sizeof(native.TransposedSource) == 16
    assert ctypes.sizeof(native.PlacementStats) == 56
    assert ctypes.sizeof(native.AbiInfo) == 64
    assert ctypes.sizeof(native.ArenaView) == 40


def test_integer_arena_address_becomes_writable_byte_view() -> None:
    owner = (ctypes.c_ubyte * 32)()
    arena = native.ArenaView(
        host_address=ctypes.addressof(owner),
        capacity_bytes=ctypes.sizeof(owner),
    )
    target = native.arena_memoryview(arena, length=16)
    target[:] = bytes(range(16))
    assert bytes(owner[:16]) == bytes(range(16))
    target.release()


def test_arena_view_rejects_invalid_address_or_length() -> None:
    with pytest.raises(ValueError):
        native.arena_memoryview(native.ArenaView(capacity_bytes=16))
    owner = (ctypes.c_ubyte * 8)()
    arena = native.ArenaView(
        host_address=ctypes.addressof(owner),
        capacity_bytes=ctypes.sizeof(owner),
    )
    with pytest.raises(ValueError):
        native.arena_memoryview(arena, length=9)


def test_pread_exact_fills_external_buffer_without_bytes_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = bytes(range(64))
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)
    owner = (ctypes.c_ubyte * len(payload))()
    arena = native.ArenaView(
        host_address=ctypes.addressof(owner),
        capacity_bytes=ctypes.sizeof(owner),
    )
    target = native.arena_memoryview(arena)

    if not hasattr(os, "preadv"):

        def fake_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
            os.lseek(fd, offset, os.SEEK_SET)
            data = os.read(fd, len(buffers[0]))
            buffers[0][: len(data)] = data
            return len(data)

        monkeypatch.setattr(os, "preadv", fake_preadv, raising=False)

    fd = os.open(source, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        native.pread_exact(fd, target)
    finally:
        os.close(fd)
        target.release()
    assert bytes(owner) == payload
