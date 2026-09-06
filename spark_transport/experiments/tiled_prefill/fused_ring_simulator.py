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
    acknowledgement: list[int] = field(default_factory=lambda: [0] * 8)
    observed: list[int] = field(default_factory=lambda: [0] * 8)


def send_shard(rank: int, direction: int, stage: int) -> int:
    if stage < 3:
        return (rank - direction * stage) % 4
    hop = stage - 3
    return (rank + direction - direction * hop) % 4


def simulate(seed: int) -> tuple[bool, int]:
    rng = random.Random(seed)
    flows = [Flow(direction, tile) for direction in (1, -1) for tile in range(4)]
    completed = 0
    while completed < len(flows):
        runnable = [flow for flow in flows if flow.stage < 6]
        flow = rng.choice(runnable)
        stage = flow.stage
        # Stages two onward reuse the parity bank from stage-2. Both remote
        # consumption credit and both-local-CQ observation are mandatory.
        if stage >= 2:
            for rank in range(4):
                prior = (stage - 2) * 4 + flow.tile
                slot = prior % 8
                assert flow.acknowledgement[slot] >= prior + 1
                assert flow.observed[slot] == prior + 1

        events = [(rank, kind) for rank in range(4) for kind in ("r0", "r1", "cq")]
        rng.shuffle(events)
        seen = {rank: set() for rank in range(4)}
        before = [row[:] for row in flow.contributors]
        for rank, kind in events:
            seen[rank].add(kind)
        assert all(value == {"r0", "r1", "cq"} for value in seen.values())
        for rank in range(4):
            receiver = (rank + flow.direction) % 4
            shard = send_shard(rank, flow.direction, stage)
            source = before[rank][shard]
            if stage < 3:
                flow.contributors[receiver][shard] |= source
            else:
                assert source == 15
                flow.contributors[receiver][shard] = source
            ordinal = stage * 4 + flow.tile
            slot = ordinal % 8
            flow.observed[slot] = ordinal + 1
            flow.acknowledgement[slot] = ordinal + 1
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
