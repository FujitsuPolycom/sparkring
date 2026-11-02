"""Stable public import for the manifest-store interface."""

from __future__ import annotations

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    CapacityPolicy,
    CommitConflict,
    ContextChunk,
    EntryKey,
    IncompleteEntry,
    LookupResult,
    MaintenanceReport,
    ManifestStore,
    PageDeltaDepthExceeded,
    StateRecord,
    validate_clear_once_request,
)
from sparkcache.publication_telemetry import (
    PUBLICATION_RECEIPT_SCHEMA,
    PUBLICATION_TELEMETRY_SCHEMA,
    PublicationByteReceipt,
    PublicationTelemetry,
    PublicationTelemetrySnapshot,
)

__all__ = [
    "CacheIdentity",
    "CapacityPolicy",
    "CommitConflict",
    "ContextChunk",
    "EntryKey",
    "IncompleteEntry",
    "LookupResult",
    "MaintenanceReport",
    "ManifestStore",
    "PageDeltaDepthExceeded",
    "PublicationByteReceipt",
    "PublicationTelemetry",
    "PublicationTelemetrySnapshot",
    "PUBLICATION_RECEIPT_SCHEMA",
    "PUBLICATION_TELEMETRY_SCHEMA",
    "StateRecord",
    "validate_clear_once_request",
]
