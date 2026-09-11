"""Check contributor coverage while interleaving complete ring exchange stages.

Each stage delivers all four ranks' shards atomically. This model checks routing,
not asynchronous rail completion, credit publication, or memory visibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Flow:
    direction: int
    tile: int
    stage: int = 0
    contributors: list[list[int]] = field(
        default_factory=lambda: [[1 << rank for _ in range(4)] for rank in range(4)]
    )


def send_shard(rank: int, direction: int, stage: int) -> int:
    # Reduce-scatter and all-gather traverse the same shard order modulo four.
    return (rank - direction * stage) % 4


def simulate(seed: int) -> tuple[bool, int]:
    rng = random.Random(seed)
    flows = [Flow(direction, tile) for direction in (1, -1) for tile in range(4)]
    completed = 0
    while completed < len(flows):
        runnable = [flow for flow in flows if flow.stage < 6]
        flow = rng.choice(runnable)
        stage = flow.stage
        before = [row[:] for row in flow.contributors]
        for rank in range(4):
            receiver = (rank + flow.direction) % 4
            shard = send_shard(rank, flow.direction, stage)
            source = before[rank][shard]
            if stage < 3:
                flow.contributors[receiver][shard] |= source
            else:
                assert source == 15
                flow.contributors[receiver][shard] = source
        flow.stage += 1
        if flow.stage == 6:
            completed += 1
    return all(mask == 15 for flow in flows for row in flow.contributors for mask in row), completed


def classify_token(observed: int, expected: int, previous: int) -> str:
    if observed > expected:
        return "fatal_future"
    if observed < previous:
        return "fatal_regression"
    if observed < expected:
        return "pending"
    return "ready"
