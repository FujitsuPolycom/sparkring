"""A stand-in for a loaded native library (``ctypes.CDLL``) of a given ABI and local features, for the bindings'
load checks: every function returns 0 except the ABI version and the feature identity, which a library from an
earlier source lacks altogether (``features=None``)."""

from __future__ import annotations


class _Function:
    def __init__(self, value: int) -> None:
        self.value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args) -> int:
        return self.value


class NativeStub:
    def __init__(self, prefix: str, abi: int, features: int | None) -> None:
        self._prefix, self._abi, self._features = prefix, abi, features
        self._functions: dict[str, _Function] = {}

    def __getattr__(self, name: str) -> _Function:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == f"{self._prefix}_local_features" and self._features is None:
            raise AttributeError(name)
        value = {f"{self._prefix}_abi_version": self._abi, f"{self._prefix}_local_features": self._features}.get(name, 0)
        return self._functions.setdefault(name, _Function(value))
