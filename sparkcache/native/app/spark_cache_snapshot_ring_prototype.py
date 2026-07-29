#!/usr/bin/env python3
"""PROTOTYPE TUI for the fail-open snapshot-ring state model.

Question: can two or three staging slots remain safe when GPU completion,
writer ownership, ring saturation, and context cancellation race?
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from snapshot_ring_state_prototype import SnapshotRing  # noqa: E402


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def render(ring: SnapshotRing, message: str) -> None:
    print("\x1b[2J\x1b[H", end="")
    print(f"{BOLD}SparkCache snapshot-ring prototype{RESET}")
    print(
        f"{DIM}Question: can saturation/cancellation remain fail-open "
        f"without reusing owned bytes?{RESET}\n"
    )
    for index, slot in enumerate(ring.slots):
        print(
            f"{BOLD}slot {index}{RESET}: state={slot.state.value:<11} "
            f"generation={slot.generation:<3} context={slot.context:<3} "
            f"discarded={slot.discarded}"
        )
    print(f"\n{BOLD}would_block{RESET}: {ring.would_block}")
    print(f"{BOLD}last action{RESET}: {message}")
    print(
        "\n"
        f"{BOLD}s CONTEXT{RESET} submit  "
        f"{BOLD}c SLOT{RESET} complete  "
        f"{BOLD}w SLOT{RESET} claim writer\n"
        f"{BOLD}r SLOT{RESET} release  "
        f"{BOLD}a CONTEXT{RESET} abandon  "
        f"{BOLD}q{RESET} quit"
    )


def main() -> int:
    ring = SnapshotRing(3)
    message = "ready"
    while True:
        render(ring, message)
        try:
            parts = input("> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not parts:
            continue
        try:
            if parts[0] == "q":
                return 0
            if len(parts) != 2:
                message = "expected an action and integer"
            elif parts[0] == "s":
                message = ring.submit(int(parts[1]))
            elif parts[0] == "c":
                message = ring.complete(int(parts[1]))
            elif parts[0] == "w":
                message = ring.claim(int(parts[1]))
            elif parts[0] == "r":
                message = ring.release(int(parts[1]))
            elif parts[0] == "a":
                message = ring.abandon(int(parts[1]))
            else:
                message = "unknown action"
        except (IndexError, ValueError) as error:
            message = str(error)


if __name__ == "__main__":
    raise SystemExit(main())
