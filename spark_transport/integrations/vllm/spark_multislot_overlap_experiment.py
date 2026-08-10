"""Feasibility contract for multi-slot ingress overlap.

**This module is a feasibility/design contract, not a performance
simulator.** It does not model, estimate, or project speedup. The
previous timing model was removed because it modeled an overlap
pattern that the actual native execution mechanics cannot produce
without unimplemented changes.

## What the native code actually does

``tp4_session.cpp`` (lines 640-673, ``progress()``): one progress
thread synchronously executes round 0 then round 1 for each
collective. ``gpu_tp4_tensor.cu`` (lines 155-243,
``tp4_tensor_all_reduce``): one GPU kernel on the caller stream
performs both rounds — staging, RDMA doorbell, wait, reduce, RDMA
doorbell, wait — in a single kernel launch. There is no overlap
between consecutive collectives.

## What multi-slot would require (not implemented)

To overlap round 0 of collective N+1 with round 1 of collective N,
the following capabilities would be needed:

1. **Interleavable round progress**: round 0 of collective N+1 must
   be able to progress while round 1 of collective N is still in
   flight. The current single progress thread
   (``tp4_session.cpp`` lines 640-673) serializes both rounds per
   collective.
2. **Safe per-slot ownership**: buffer ownership must prevent the
   GPU kernel of collective N+1 from overwriting ``send0`` while
   the verbs thread is still reading it for collective N
   (``gpu_tp4_tensor.cu`` lines 195-198).
3. **GPU execution concurrency**: the GPU must be able to execute
   round 0 of collective N+1 concurrently with or ordered relative
   to round 1 of collective N. The current single kernel on the
   caller stream (``gpu_tp4_tensor.cu`` lines 155-243) prevents
   this.

Separate progress threads/streams are possible mechanisms, not
mandatory architecture. Event-driven single-thread and persistent
or multi-collective-kernel alternatives are also valid.

None of these changes exist in the current source. This module
defines the contract that any future implementation must satisfy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence


class ExperimentInputError(ValueError):
    """Stable error for invalid experiment inputs (no traceback from CLI)."""


def _validate_positive_int(value: int, name: str, *, maximum: int | None = None) -> int:
    """Reject bool, negative, zero, or out-of-range integers."""
    if isinstance(value, bool):
        raise ExperimentInputError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise ExperimentInputError(f"{name} must be an int")
    if value <= 0:
        raise ExperimentInputError(f"{name} must be positive, got {value}")
    if maximum is not None and value > maximum:
        raise ExperimentInputError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class FeasibilityRequirement:
    """One requirement that must be satisfied before multi-slot overlap
    can be implemented."""

    identifier: str
    description: str
    source_citation: str
    status: str  # "unimplemented", "proposed", "satisfied"


@dataclass
class FeasibilityContract:
    """Contract for multi-slot ingress overlap feasibility.

    This is **not** a performance model. It lists the unimplemented
    changes required, cites the conflicting source, and provides a
    checklist for any future implementation.
    """

    requirements: list[FeasibilityRequirement] = field(default_factory=list)

    @property
    def all_satisfied(self) -> bool:
        return all(r.status == "satisfied" for r in self.requirements)

    @property
    def any_unimplemented(self) -> bool:
        return any(r.status == "unimplemented" for r in self.requirements)


def default_requirements() -> list[FeasibilityRequirement]:
    """Return the default feasibility requirements.

    Each requirement cites the exact source location that would need
    to change. All are currently **unimplemented**.
    """
    return [
        FeasibilityRequirement(
            identifier="interleavable_round_progress",
            description=(
                "Round 0 of collective N+1 must be able to progress "
                "while round 1 of collective N is still in flight. "
                "The current single progress thread serializes both "
                "rounds per collective (tp4_session.cpp:640-673). "
                "Possible mechanisms: split progress into per-round "
                "functions with separate workers; event-driven "
                "single-thread with non-blocking round dispatch; "
                "persistent multi-collective kernel that processes "
                "multiple collectives' rounds in one launch."
            ),
            source_citation="tp4_session.cpp:640-673",
            status="unimplemented",
        ),
        FeasibilityRequirement(
            identifier="safe_per_slot_ownership",
            description=(
                "Buffer ownership must prevent the GPU kernel of "
                "collective N+1 from overwriting send0 while the "
                "verbs thread is still reading it for collective N "
                "(gpu_tp4_tensor.cu:195-198). The current per-buffer "
                "DoorbellControl (gpu_doorbell.hpp:8-17) must be "
                "extended to per-slot or a new ownership protocol "
                "must be designed."
            ),
            source_citation="gpu_tp4_tensor.cu:195-198",
            status="unimplemented",
        ),
        FeasibilityRequirement(
            identifier="gpu_execution_concurrency",
            description=(
                "The GPU must be able to execute round 0 of "
                "collective N+1 concurrently with or ordered relative "
                "to round 1 of collective N. The current single "
                "kernel on the caller stream (gpu_tp4_tensor.cu:155-243) "
                "prevents this. Possible mechanisms: split into separate "
                "kernels on separate streams; persistent kernel that "
                "processes an internal queue of round operations; "
                "multi-collective kernel that batches rounds from "
                "different collectives."
            ),
            source_citation="gpu_tp4_tensor.cu:155-243",
            status="unimplemented",
        ),
    ]


def build_contract() -> FeasibilityContract:
    """Build the feasibility contract with all default requirements."""
    return FeasibilityContract(requirements=default_requirements())


def render_report(contract: FeasibilityContract) -> str:
    """Render a human-readable feasibility report."""
    lines = [
        "MULTISLOT_INGRESS_FEASIBILITY_CONTRACT",
        "This is a feasibility contract, not a performance model.",
        "No performance estimate is provided.",
        "",
        f"requirements={len(contract.requirements)} "
        f"satisfied={sum(1 for r in contract.requirements if r.status == 'satisfied')} "
        f"unimplemented={sum(1 for r in contract.requirements if r.status == 'unimplemented')}",
        "",
        "| id | description | source | status |",
        "|---|---|---|---|",
    ]
    for req in contract.requirements:
        desc = req.description[:80]
        lines.append(
            f"| {req.identifier} | {desc} | "
            f"{req.source_citation} | {req.status} |"
        )
    if contract.any_unimplemented:
        lines.append("")
        lines.append(
            "CONCLUSION: Multi-slot overlap is NOT feasible with the "
            "current native code. All requirements above must be "
            "implemented before any overlap can occur."
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feasibility contract for multi-slot ingress overlap",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _parse_args(argv)
        contract = build_contract()
        print(render_report(contract))
        return 0
    except ExperimentInputError as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
