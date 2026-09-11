"""Bounded timing of target verification and speculative draft-generation phases.

The q2r_phase_timing package name and SPARK_Q2R_* options identify this probe;
manager ownership and measured call boundaries define its measurement roles.
"""

from .phase_timing import (
    DrainResult,
    PhaseDescriptor,
    PhaseKind,
    PhaseTimingCollector,
    SnapshotMismatch,
    snapshot_delta,
)

__all__ = [
    "DrainResult",
    "PhaseDescriptor",
    "PhaseKind",
    "PhaseTimingCollector",
    "SnapshotMismatch",
    "snapshot_delta",
]
