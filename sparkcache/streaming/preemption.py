"""Fail-closed preemption bridge for streaming snapshot block leases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .block_lease import BlockLeaseRegistry


class PreemptionMetadataError(RuntimeError):
    """The connector metadata cannot prove which requests may be overwritten."""


@dataclass(frozen=True, slots=True)
class StreamingPreemptionMetadata:
    """Deterministic field to add to ``SparkCacheConnectorMetadata``.

    Scheduler-side construction must use:

    ``tuple(sorted(scheduler_output.preempted_req_ids or ()))``
    """

    preempted_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_request_ids(self.preempted_request_ids)


def _validate_request_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise PreemptionMetadataError(
            "preempted_request_ids must be a deterministic tuple"
        )
    if any(not isinstance(request_id, str) or not request_id for request_id in value):
        raise PreemptionMetadataError(
            "preempted_request_ids must contain non-empty strings"
        )
    if len(set(value)) != len(value):
        raise PreemptionMetadataError("preempted_request_ids must be unique")
    if value != tuple(sorted(value)):
        raise PreemptionMetadataError("preempted_request_ids must be sorted")
    return value


class PreemptionDrainAdapter:
    """Drain affected gathers before vLLM may overwrite their blocks."""

    def __init__(
        self,
        leases: BlockLeaseRegistry,
        abandon_publication: Callable[[str], None],
    ) -> None:
        self._leases = leases
        self._abandon_publication = abandon_publication

    def handle_preemptions(self, kv_connector_metadata: Any) -> int:
        """Synchronize armed work, cancel reservations, then abandon journals.

        Missing or malformed metadata fails closed. A fence synchronization
        failure propagates ``LeaseFenceError`` from the registry, retains the
        affected lease, and prevents this method from returning to vLLM.
        """

        if not hasattr(kv_connector_metadata, "preempted_request_ids"):
            raise PreemptionMetadataError(
                "SparkCache metadata lacks preempted_request_ids"
            )
        request_ids = _validate_request_ids(
            kv_connector_metadata.preempted_request_ids
        )
        released = 0
        for request_id in request_ids:
            # abort_request synchronizes all ARMED events before it releases
            # any associated block IDs. RESERVED work is cancelled under the
            # same lock that guards submit(), closing submit-vs-preempt races.
            released += self._leases.abort_request(request_id)
            self._abandon_publication(request_id)
        return released
