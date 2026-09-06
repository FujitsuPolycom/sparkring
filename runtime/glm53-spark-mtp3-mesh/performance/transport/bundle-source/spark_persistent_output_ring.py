from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputSignature:
    shape: tuple[int, ...]
    dtype: object
    device: object


class PersistentOutputRing:
    """Lazily allocate and cycle stable CUDA output buffers.

    One ring is owned by one fixed-payload native transport session.  The
    caller remains responsible for submitting producers and consumers in
    CUDA stream order before a slot is reused.
    """

    def __init__(self, slots: int) -> None:
        if slots <= 0:
            raise ValueError("persistent output ring slots must be positive")
        self._slots = slots
        self._signature: OutputSignature | None = None
        self._storage: Any | None = None
        self._outputs: tuple[Any, ...] = ()
        self._next = 0
        self._acquires = 0

    @property
    def slots(self) -> int:
        return self._slots

    @property
    def acquires(self) -> int:
        return self._acquires

    @property
    def wraps(self) -> int:
        return self._acquires // self._slots

    @staticmethod
    def _signature_for(tensor: Any) -> OutputSignature:
        return OutputSignature(
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def acquire(self, tensor: Any, torch_module: Any) -> Any:
        signature = self._signature_for(tensor)
        if self._signature is None:
            storage = torch_module.empty(
                (self._slots, *signature.shape),
                dtype=signature.dtype,
                device=signature.device,
            )
            outputs = tuple(storage.unbind(0))
            if len(outputs) != self._slots:
                raise RuntimeError(
                    "persistent output storage returned the wrong slot count"
                )
            self._storage = storage
            self._outputs = outputs
            self._signature = signature
        elif signature != self._signature:
            raise RuntimeError(
                "persistent output session signature changed: "
                f"expected={self._signature} observed={signature}"
            )

        output = self._outputs[self._next]
        self._next = (self._next + 1) % self._slots
        self._acquires += 1
        return output
