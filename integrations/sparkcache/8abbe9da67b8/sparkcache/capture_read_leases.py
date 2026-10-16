"""GPU-free ownership accounting for scheduler-issued capture reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import uuid


@dataclass
class _ReadLease:
    block_ids: tuple[int, ...]
    completed_ranks: set[int] = field(default_factory=set)
    quarantined: bool = False


class CaptureReadLeases:
    """Pin exact source blocks until every physical worker completes its read."""

    def __init__(self, pool: Any, *, ranks: int, max_jobs: int):
        if ranks <= 0 or max_jobs <= 0:
            raise ValueError("capture leases require positive rank and job bounds")
        self.pool = pool
        self.ranks = ranks
        self.max_jobs = max_jobs
        self._epoch = uuid.uuid4().hex
        self._sequence = 0
        self._leases: dict[str, _ReadLease] = {}
        self.disabled = False

    def __bool__(self) -> bool:
        return any(not lease.quarantined for lease in self._leases.values())

    def reserve(self, block_ids: Iterable[int]) -> str | None:
        """Take one reference per physical block, or decline optional work."""
        if self.disabled or len(self._leases) >= self.max_jobs:
            return None
        supplied_ids = tuple(block_ids)
        if not supplied_ids or any(
            type(bid) is not int or not 0 < bid < len(self.pool.blocks)
            for bid in supplied_ids
        ):
            raise ValueError("capture source contains an invalid or null block")
        ids = tuple(dict.fromkeys(supplied_ids))
        blocks = [self.pool.blocks[bid] for bid in ids]
        if any(block.is_null or block.block_hash is None for block in blocks):
            raise ValueError("capture sources must be hash-proven immutable blocks")
        self.pool.touch(blocks)
        self._sequence += 1
        job = f"{self._epoch}:{self._sequence}"
        self._leases[job] = _ReadLease(ids)
        return job

    def complete(self, job: str, ranks: Iterable[int]) -> bool:
        """Release after distinct rank acknowledgements; repeated ACKs are inert."""
        lease = self._leases.get(job)
        if lease is None:
            return False
        supplied_ranks = tuple(ranks)
        if any(
            type(rank) is not int or not 0 <= rank < self.ranks
            for rank in supplied_ranks
        ):
            raise ValueError("capture completion names an invalid physical rank")
        received = set(supplied_ranks)
        lease.completed_ranks.update(received)
        if len(lease.completed_ranks) != self.ranks:
            return False
        self.pool.free_blocks(
            self.pool.blocks[bid] for bid in reversed(lease.block_ids)
        )
        del self._leases[job]
        return True

    def quarantine(self, job: str, ranks: Iterable[int]) -> None:
        """Keep uncertain sources pinned without spinning the idle engine."""
        lease = self._leases.get(job)
        if lease is None:
            return
        if any(type(rank) is not int or not 0 <= rank < self.ranks for rank in ranks):
            raise ValueError("capture failure names an invalid physical rank")
        lease.quarantined = True
        self.disabled = True
