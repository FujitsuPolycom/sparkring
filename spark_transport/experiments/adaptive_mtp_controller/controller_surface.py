"""Fail-closed scheduler-local control for vLLM adaptive speculation.

This module deliberately has no vLLM imports.  The runtime installer owns the
version-specific monkeypatch; this module owns the small, testable state and
safety contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControllerTelemetry:
    """Lifetime reset telemetry attached to one scheduler controller.

    Exact per-depth proposal and acceptance accounting belongs to the
    true-draft/Observer interval snapshots.  Keeping it out of this reset
    patch avoids assigning the scheduler's *next* proposal K to the current
    verification result.
    """

    depth_ladder: tuple[int, ...]
    reset_count: int = 0
    resets_by_reason: dict[str, int] = field(default_factory=dict)
    last_reset_reason: str | None = None

    def record_reset(self, reason: str) -> None:
        if not reason or not reason.replace("_", "").isalnum():
            raise ValueError(f"invalid adaptive-MTP reset reason: {reason!r}")
        self.reset_count += 1
        self.resets_by_reason[reason] = self.resets_by_reason.get(reason, 0) + 1
        self.last_reset_reason = reason

class AdaptiveMtpControlSurface:
    """Status/reset facade executed in the EngineCore utility-RPC thread."""

    def __init__(
        self,
        engine_core: Any,
        *,
        telemetry: ControllerTelemetry,
    ) -> None:
        self._engine_core = engine_core
        self._telemetry = telemetry

    def control(self, action: str) -> dict[str, Any]:
        if action == "status":
            return self._status()
        if action == "reset":
            return self._reset(reason="manual")
        raise ValueError(f"unsupported adaptive-MTP control action: {action!r}")

    def reset_idle_epoch_if_safe(self) -> dict[str, Any]:
        """Reset before the first request of a genuinely idle engine epoch.

        Resumable/streaming sessions can leave a known request behind while the
        engine is otherwise waiting for work.  That is not a new workload
        epoch, so automatic reset must skip it rather than mutating shared
        speculative state or failing the request.
        """

        status = self._status()
        if not status["enabled"] or not status["idle"]["safe_to_reset"]:
            return {
                "reset": False,
                "reset_reason": "idle_epoch",
                "status": status,
            }
        return self._reset(reason="idle_epoch")

    def _status(self) -> dict[str, Any]:
        scheduler = self._engine_core.scheduler
        controller = scheduler.acceptance_length_controller
        if controller is None:
            return {
                "enabled": False,
                "idle": self._idle_status(scheduler),
            }

        raw_k = int(controller.num_spec_tokens)
        maximum = int(controller.max_num_spec_tokens)
        result = {
            "enabled": True,
            "raw_k": raw_k,
            "floor_snapped_k": self._floor_snap(raw_k, maximum),
            "configured_max_k": maximum,
            "observation_window": int(controller.observation_window),
            "window": {
                "observation_steps": int(controller._num_observation_steps),
                "drafted_rounds": int(controller._num_drafts),
                "attempted_tokens": int(controller._num_draft_tokens),
                "accepted_tokens": int(controller._num_accepted_tokens),
            },
            "idle": self._idle_status(scheduler),
            "reset_count": self._telemetry.reset_count,
            "resets_by_reason": dict(
                sorted(self._telemetry.resets_by_reason.items())
            ),
        }
        if self._telemetry.last_reset_reason is not None:
            result["last_reset_reason"] = self._telemetry.last_reset_reason
        return result

    def _reset(self, *, reason: str) -> dict[str, Any]:
        scheduler = self._engine_core.scheduler
        controller = scheduler.acceptance_length_controller
        if controller is None:
            raise RuntimeError("adaptive speculative decoding is not enabled")

        idle = self._idle_status(scheduler)
        if not idle["safe_to_reset"]:
            raise RuntimeError(
                "adaptive-MTP reset refused while scheduler state is in flight: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in idle.items()
                    if key != "safe_to_reset" and value
                )
            )

        # Validate the complete mutation ABI before changing any state.
        for attr in (
            "max_num_spec_tokens",
            "num_spec_tokens",
            "_num_observation_steps",
            "_num_drafts",
            "_num_draft_tokens",
            "_num_accepted_tokens",
        ):
            if not hasattr(controller, attr):
                raise RuntimeError(
                    f"adaptive-MTP controller ABI mismatch: missing {attr!r}"
                )
        reset_window = getattr(controller, "_reset_window", None)
        if not callable(reset_window):
            raise RuntimeError(
                "adaptive-MTP controller ABI mismatch: _reset_window is not callable"
            )

        mutable_fields = (
            "num_spec_tokens",
            "_num_observation_steps",
            "_num_drafts",
            "_num_draft_tokens",
            "_num_accepted_tokens",
        )
        previous = {
            name: getattr(controller, name)
            for name in mutable_fields
        }
        try:
            controller.num_spec_tokens = int(controller.max_num_spec_tokens)
            reset_window()
        except Exception:
            for name, value in previous.items():
                setattr(controller, name, value)
            raise
        self._telemetry.record_reset(reason)
        result = self._status()
        result["reset"] = True
        result["reset_reason"] = reason
        return result

    def _floor_snap(self, raw_k: int, maximum: int) -> int:
        candidates = tuple(
            depth
            for depth in self._telemetry.depth_ladder
            if 0 < depth <= maximum and depth <= raw_k
        )
        return max(candidates) if candidates else raw_k

    def _idle_status(self, scheduler: Any) -> dict[str, Any]:
        running = len(scheduler.running)
        waiting = len(scheduler.waiting)
        skipped_waiting = len(getattr(scheduler, "skipped_waiting", ()))
        requests = len(scheduler.requests)
        deferred_requests = sum(
            len(getattr(scheduler, name, ()))
            for name in (
                "deferred_requests",
                "deferred_request_queue",
                "pending_structured_output_requests",
            )
        )
        deferred_frees = len(getattr(scheduler, "deferred_frees", ()))
        streaming_waiters = int(
            getattr(scheduler, "num_waiting_for_streaming_input", 0)
        )
        batch_queue = getattr(self._engine_core, "batch_queue", None)
        queued_batches = len(batch_queue) if batch_queue is not None else 0

        request_values = tuple(scheduler.requests.values())
        speculative_placeholders = sum(
            max(int(getattr(request, "num_output_placeholders", 0)), 0)
            for request in request_values
        )
        speculative_token_rows = sum(
            bool(getattr(request, "spec_token_ids", ()))
            for request in request_values
        )
        async_discard_frames = sum(
            max(int(getattr(request, "async_tokens_to_discard", 0)), 0)
            for request in request_values
        )
        in_flight_tokens = sum(
            max(int(getattr(request, "num_in_flight_tokens", 0)), 0)
            for request in request_values
        )
        safe = not any(
            (
                running,
                waiting,
                skipped_waiting,
                requests,
                deferred_requests,
                deferred_frees,
                streaming_waiters,
                queued_batches,
                speculative_placeholders,
                speculative_token_rows,
                async_discard_frames,
                in_flight_tokens,
            )
        )
        return {
            "safe_to_reset": safe,
            "running": running,
            "waiting": waiting,
            "deferred_waiting": skipped_waiting,
            "known_requests": requests,
            "deferred_requests": deferred_requests,
            "deferred_frees": deferred_frees,
            "streaming_waiters": streaming_waiters,
            "queued_batches": queued_batches,
            "speculative_placeholders": speculative_placeholders,
            "speculative_token_rows": speculative_token_rows,
            "async_discard_frames": async_discard_frames,
            "in_flight_tokens": in_flight_tokens,
        }
