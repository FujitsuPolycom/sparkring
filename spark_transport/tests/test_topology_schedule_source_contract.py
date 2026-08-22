"""Keep the published four-rank topology aligned with the TP4 round schedule."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_direct_cycle_edges_match_tp4_round_masks() -> None:
    topology = (ROOT / "src" / "topology.cpp").read_text(encoding="utf-8")
    schedule = (ROOT / "src" / "tp4_schedule.cpp").read_text(encoding="utf-8")

    configured = {
        tuple(sorted((int(left), int(right))))
        for left, right in re.findall(
            r"add_pair\((\d+),\s*(\d+),\s*\d+\);", topology
        )
    }
    masks = re.search(
        r"round\s*==\s*0\s*\?\s*(\d+)U\s*:\s*(\d+)U", schedule
    )
    assert masks is not None
    expected = {
        tuple(sorted((rank, rank ^ int(mask))))
        for rank in range(4)
        for mask in masks.groups()
    }

    assert configured == expected == {(0, 1), (1, 2), (2, 3), (0, 3)}
