"""Bounded, offline-tested phase timing primitives for Q-2R."""

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
