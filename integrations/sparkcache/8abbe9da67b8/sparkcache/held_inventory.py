"""Mutation-tracked digest inventory for bounded worker status reporting."""

from collections.abc import Iterable, Iterator, MutableSet


class HeldInventory(MutableSet[str]):
    """Set of offered digests with a revision for content changes.

    Callers serialize reads and mutations with the connector's worker-state
    lock. The revision belongs to this object: replacing an inventory requires
    comparing its identity as well as its revision. Input collections are
    copied so mutations cannot bypass revision tracking through another owner.
    """

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._values = set(values)
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def __contains__(self, value: object) -> bool:
        return value in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: str) -> None:
        before = len(self._values)
        self._values.add(value)
        self._revision += len(self._values) != before

    def discard(self, value: str) -> None:
        before = len(self._values)
        self._values.discard(value)
        self._revision += len(self._values) != before

    def remove(self, value: str) -> None:
        self._values.remove(value)
        self._revision += 1

    def pop(self) -> str:
        value = self._values.pop()
        self._revision += 1
        return value

    def clear(self) -> None:
        if self._values:
            self._values.clear()
            self._revision += 1

    def update(self, *values: Iterable[str]) -> None:
        before = len(self._values)
        try:
            self._values.update(*values)
        finally:
            # Iterators can fail after yielding elements. A partial mutation
            # must still invalidate the report's inventory snapshot.
            self._revision += len(self._values) != before

    def difference_update(self, *values: Iterable[str]) -> None:
        before = len(self._values)
        try:
            self._values.difference_update(
                *(self._values if other is self else other for other in values)
            )
        finally:
            self._revision += len(self._values) != before

    def intersection_update(self, *values: Iterable[str]) -> None:
        before = len(self._values)
        try:
            self._values.intersection_update(*values)
        finally:
            self._revision += len(self._values) != before

    def symmetric_difference_update(self, values: Iterable[str]) -> None:
        other = set(values)
        if other:
            self._values.symmetric_difference_update(other)
            self._revision += 1

    def __ior__(self, values: Iterable[str]) -> "HeldInventory":
        self.update(values)
        return self

    def __isub__(self, values: Iterable[str]) -> "HeldInventory":
        self.difference_update(values)
        return self

    def __iand__(self, values: Iterable[str]) -> "HeldInventory":
        self.intersection_update(values)
        return self

    def __ixor__(self, values: Iterable[str]) -> "HeldInventory":
        self.symmetric_difference_update(values)
        return self

    def copy(self) -> set[str]:
        return self._values.copy()
