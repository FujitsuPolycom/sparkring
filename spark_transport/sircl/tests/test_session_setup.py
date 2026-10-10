"""Session setup helpers offline: which native library is the verbs stand-in."""

from __future__ import annotations

import ctypes

from sparkring_sircl.oneshot import _proxy


class _Library:
    def __init__(self, names):
        for name in names:
            setattr(self, name, object())


def test_stand_in_library_is_the_one_that_exports_the_stand_in(monkeypatch, tmp_path):
    monkeypatch.delenv("SIRCL_NATIVE_LIBRARY", raising=False)
    assert not _proxy.stand_in_library()                       # the cached build: the real library
    assert not _proxy.stand_in_library(tmp_path / "missing.so")
    loaded = {}
    monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.setdefault(path, _Library(
        ("roce_abi_version", "fv_add_device") if "fake" in path else ("roce_abi_version",))))
    assert _proxy.stand_in_library(tmp_path / "fake.so")
    assert not _proxy.stand_in_library(tmp_path / "roce.so")
    monkeypatch.setenv("SIRCL_NATIVE_LIBRARY", str(tmp_path / "fake.so"))
    assert _proxy.stand_in_library()
