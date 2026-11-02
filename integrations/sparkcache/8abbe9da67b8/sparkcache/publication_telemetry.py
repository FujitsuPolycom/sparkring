"""Observational byte accounting for persistent-cache publication.

The counters in this module describe SparkCache's host-side publication path.
They never participate in cache admission, identity, restore verification, or
serving decisions. In particular, ``staged_write_bytes`` is not a device-write
or NAND-write counter; compare it with an NVMe Data Units Written delta when a
physical write-amplification estimate is required.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Literal

PUBLICATION_TELEMETRY_SCHEMA = "sparkcache-publication-telemetry/v1"
PUBLICATION_RECEIPT_SCHEMA = "sparkcache-publication-receipt/v1"

PublicationKind = Literal[
    "complete_snapshot",
    "prefix_alias",
    "row_tail",
    "page_snapshot",
    "page_delta",
]
PublicationOutcome = Literal["committed", "aborted", "failed"]


@dataclass(frozen=True)
class PublicationByteReceipt:
    """Byte accounting for one terminal publication attempt.

    ``logical_payload_bytes`` is the encoded state payload represented by the
    result, excluding manifests and descriptor-chain metadata. A row tail or
    page delta counts only its extension payload. ``reused_base_bytes`` reports
    base payload referenced without staging it again.

    ``unique_object_bytes`` counts complete immutable files newly linked or
    repaired by the attempt, including metadata roots. ``deduplicated_bytes``
    counts identical immutable files that already existed. An aborted or
    failed attempt may have unique objects that are not reachable from a
    manifest; ``uncommitted_unique_object_bytes`` makes that distinction.
    ``staged_write_bytes`` counts payload bytes submitted to temporary-file
    writes, including files later deduplicated or left unreferenced.
    """

    schema: str
    kind: PublicationKind
    outcome: PublicationOutcome
    logical_payload_bytes: int
    reused_base_bytes: int
    unique_object_bytes: int
    committed_unique_object_bytes: int
    uncommitted_unique_object_bytes: int
    staged_write_bytes: int
    deduplicated_bytes: int
    staged_objects: int
    unique_objects: int
    deduplicated_objects: int

    def as_dict(self) -> dict[str, str | int]:
        """Return the stable JSON-compatible receipt representation."""

        return asdict(self)

    def format_compact(self) -> str:
        """Format one attempt as a compact operator log line."""

        return (
            "sparkcache: publish "
            f"kind={self.kind} outcome={self.outcome} "
            f"payload={self.logical_payload_bytes}B "
            f"unique={self.committed_unique_object_bytes}B "
            f"staged={self.staged_write_bytes}B "
            f"dedup={self.deduplicated_bytes}B "
            f"reused_base={self.reused_base_bytes}B"
        )


@dataclass(frozen=True)
class PublicationTelemetrySnapshot:
    """Monotonic process-local totals for one ``ManifestStore`` instance."""

    schema: str
    publication_attempts: int
    committed_publications: int
    aborted_publications: int
    failed_publications: int
    aborted_unique_object_bytes: int
    aborted_staged_write_bytes: int
    failed_unique_object_bytes: int
    failed_staged_write_bytes: int
    logical_payload_bytes: int
    reused_base_bytes: int
    unique_object_bytes: int
    committed_unique_object_bytes: int
    uncommitted_unique_object_bytes: int
    staged_write_bytes: int
    deduplicated_bytes: int
    staged_objects: int
    unique_objects: int
    deduplicated_objects: int

    def as_dict(self) -> dict[str, str | int]:
        """Return the stable JSON-compatible cumulative representation."""

        return asdict(self)

    def format_compact(self) -> str:
        """Format exact counters as one compact operator log line."""

        return (
            "sparkcache: publication "
            f"commits={self.committed_publications} "
            f"payload={self.logical_payload_bytes}B "
            f"unique={self.committed_unique_object_bytes}B "
            f"staged={self.staged_write_bytes}B "
            f"dedup={self.deduplicated_bytes}B "
            f"reused_base={self.reused_base_bytes}B "
            f"aborted={self.aborted_publications} "
            f"aborted_staged={self.aborted_staged_write_bytes}B "
            f"failed={self.failed_publications} "
            f"failed_staged={self.failed_staged_write_bytes}B"
        )


class PublicationAttempt:
    """Mutable accounting owned by one publication operation."""

    def __init__(self, telemetry: "PublicationTelemetry", kind: PublicationKind) -> None:
        self._telemetry = telemetry
        self.kind = kind
        self.logical_payload_bytes = 0
        self.reused_base_bytes = 0
        self.unique_object_bytes = 0
        self.staged_write_bytes = 0
        self.deduplicated_bytes = 0
        self.staged_objects = 0
        self.unique_objects = 0
        self.deduplicated_objects = 0
        self._terminal: PublicationByteReceipt | None = None
        self._lock = threading.Lock()

    def describe_payload(self, logical_bytes: int, reused_base_bytes: int = 0) -> None:
        """Set semantic payload sizes before publication becomes visible."""

        if logical_bytes < 0 or reused_base_bytes < 0:
            raise ValueError("publication byte counts must be non-negative")
        with self._lock:
            self.logical_payload_bytes = logical_bytes
            self.reused_base_bytes = reused_base_bytes

    def record_staged(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("staged byte count must be non-negative")
        with self._lock:
            self.staged_write_bytes += byte_count
            self.staged_objects += 1

    def record_unique(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("unique byte count must be non-negative")
        with self._lock:
            self.unique_object_bytes += byte_count
            self.unique_objects += 1

    def record_deduplicated(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("deduplicated byte count must be non-negative")
        with self._lock:
            self.deduplicated_bytes += byte_count
            self.deduplicated_objects += 1

    @property
    def has_activity(self) -> bool:
        """Return whether the operation reached payload planning or writing."""

        with self._lock:
            return bool(
                self.logical_payload_bytes
                or self.reused_base_bytes
                or self.staged_objects
                or self.unique_objects
                or self.deduplicated_objects
            )

    def finish(self, outcome: PublicationOutcome) -> PublicationByteReceipt:
        """Record one terminal outcome exactly once and return its receipt."""

        with self._lock:
            if self._terminal is not None:
                return self._terminal
            committed_unique = self.unique_object_bytes if outcome == "committed" else 0
            uncommitted_unique = self.unique_object_bytes - committed_unique
            receipt = PublicationByteReceipt(
                schema=PUBLICATION_RECEIPT_SCHEMA,
                kind=self.kind,
                outcome=outcome,
                logical_payload_bytes=(
                    self.logical_payload_bytes if outcome == "committed" else 0
                ),
                reused_base_bytes=(self.reused_base_bytes if outcome == "committed" else 0),
                unique_object_bytes=self.unique_object_bytes,
                committed_unique_object_bytes=committed_unique,
                uncommitted_unique_object_bytes=uncommitted_unique,
                staged_write_bytes=self.staged_write_bytes,
                deduplicated_bytes=self.deduplicated_bytes,
                staged_objects=self.staged_objects,
                unique_objects=self.unique_objects,
                deduplicated_objects=self.deduplicated_objects,
            )
            self._terminal = receipt
        self._telemetry._record(receipt)
        return receipt


class PublicationTelemetry:
    """Thread-safe monotonic publication accounting for one cache store."""

    _FIELDS = (
        "logical_payload_bytes",
        "reused_base_bytes",
        "unique_object_bytes",
        "committed_unique_object_bytes",
        "uncommitted_unique_object_bytes",
        "staged_write_bytes",
        "deduplicated_bytes",
        "staged_objects",
        "unique_objects",
        "deduplicated_objects",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = {field: 0 for field in self._FIELDS}
        self._attempts = 0
        self._committed = 0
        self._aborted = 0
        self._failed = 0
        self._aborted_unique_object_bytes = 0
        self._aborted_staged_write_bytes = 0
        self._failed_unique_object_bytes = 0
        self._failed_staged_write_bytes = 0

    def begin(self, kind: PublicationKind) -> PublicationAttempt:
        """Create an uncounted attempt; terminal accounting occurs at finish."""

        return PublicationAttempt(self, kind)

    def _record(self, receipt: PublicationByteReceipt) -> None:
        with self._lock:
            self._attempts += 1
            if receipt.outcome == "committed":
                self._committed += 1
            elif receipt.outcome == "aborted":
                self._aborted += 1
                self._aborted_unique_object_bytes += receipt.unique_object_bytes
                self._aborted_staged_write_bytes += receipt.staged_write_bytes
            else:
                self._failed += 1
                self._failed_unique_object_bytes += receipt.unique_object_bytes
                self._failed_staged_write_bytes += receipt.staged_write_bytes
            for field in self._FIELDS:
                self._totals[field] += int(getattr(receipt, field))

    def snapshot(self) -> PublicationTelemetrySnapshot:
        """Read an atomic cumulative snapshot without resetting counters."""

        with self._lock:
            return PublicationTelemetrySnapshot(
                schema=PUBLICATION_TELEMETRY_SCHEMA,
                publication_attempts=self._attempts,
                committed_publications=self._committed,
                aborted_publications=self._aborted,
                failed_publications=self._failed,
                aborted_unique_object_bytes=self._aborted_unique_object_bytes,
                aborted_staged_write_bytes=self._aborted_staged_write_bytes,
                failed_unique_object_bytes=self._failed_unique_object_bytes,
                failed_staged_write_bytes=self._failed_staged_write_bytes,
                **self._totals,
            )
