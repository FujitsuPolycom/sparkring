from __future__ import annotations

from spark_transport.experiments.adaptive_mtp_controller.controller_surface import (
    AdaptiveMtpControlSurface,
    ControllerTelemetry,
)


class FakeController:
    def __init__(self, maximum: int = 4, window: int = 32) -> None:
        self.max_num_spec_tokens = maximum
        self.observation_window = window
        self.num_spec_tokens = maximum
        self._num_observation_steps = 3
        self._num_drafts = 3
        self._num_draft_tokens = 12
        self._num_accepted_tokens = 8

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0


class FakeScheduler:
    def __init__(self) -> None:
        self.acceptance_length_controller = FakeController()
        self.running: list[object] = []
        self.waiting: list[object] = []
        self.skipped_waiting: list[object] = []
        self.requests: dict[str, object] = {}
        self.deferred_requests: list[object] = []
        self.deferred_frees: list[object] = []
        self.num_waiting_for_streaming_input = 0


class FakeEngineCore:
    def __init__(self) -> None:
        self.scheduler = FakeScheduler()
        self.batch_queue: list[object] = []


def test_status_reports_controller_and_window_state() -> None:
    core = FakeEngineCore()
    telemetry = ControllerTelemetry(depth_ladder=(2, 4))
    surface = AdaptiveMtpControlSurface(core, telemetry=telemetry)

    status = surface.control("status")

    assert status["enabled"] is True
    assert status["raw_k"] == 4
    assert status["floor_snapped_k"] == 4
    assert status["configured_max_k"] == 4
    assert status["observation_window"] == 32
    assert status["window"] == {
        "observation_steps": 3,
        "drafted_rounds": 3,
        "attempted_tokens": 12,
        "accepted_tokens": 8,
    }
    assert status["idle"]["safe_to_reset"] is True


def test_reset_restores_maximum_and_clears_only_window_accumulators() -> None:
    core = FakeEngineCore()
    controller = core.scheduler.acceptance_length_controller
    controller.num_spec_tokens = 2
    telemetry = ControllerTelemetry(depth_ladder=(2, 4))
    surface = AdaptiveMtpControlSurface(core, telemetry=telemetry)

    result = surface.control("reset")

    assert result["reset"] is True
    assert result["raw_k"] == 4
    assert result["window"] == {
        "observation_steps": 0,
        "drafted_rounds": 0,
        "attempted_tokens": 0,
        "accepted_tokens": 0,
    }
    assert result["reset_count"] == 1
    assert result["resets_by_reason"] == {"manual": 1}
    assert result["last_reset_reason"] == "manual"
    assert result["reset_reason"] == "manual"


def test_reset_refuses_every_in_flight_scheduler_surface_without_mutation() -> None:
    class PlaceholderRequest:
        num_output_placeholders = 5
        spec_token_ids = [-1, -1, -1, -1]
        async_tokens_to_discard = 1
        num_in_flight_tokens = 5

    blockers = {
        "running": lambda core: core.scheduler.running.append(object()),
        "waiting": lambda core: core.scheduler.waiting.append(object()),
        "deferred_waiting": lambda core: core.scheduler.skipped_waiting.append(object()),
        "deferred_requests": lambda core: core.scheduler.deferred_requests.append(
            object()
        ),
        "queued_batches": lambda core: core.batch_queue.append(object()),
        "speculative_placeholders": lambda core: core.scheduler.requests.update(
            {"request-1": PlaceholderRequest()}
        ),
    }

    for expected_blocker, arm in blockers.items():
        core = FakeEngineCore()
        controller = core.scheduler.acceptance_length_controller
        controller.num_spec_tokens = 2
        telemetry = ControllerTelemetry(depth_ladder=(2, 4))
        surface = AdaptiveMtpControlSurface(core, telemetry=telemetry)
        arm(core)

        try:
            surface.control("reset")
        except RuntimeError as exc:
            assert expected_blocker in str(exc)
        else:
            raise AssertionError(f"reset did not refuse blocker {expected_blocker}")

        assert controller.num_spec_tokens == 2
        assert controller._num_observation_steps == 3
        assert telemetry.reset_count == 0


def test_reset_rolls_back_if_exact_controller_reset_raises() -> None:
    class ExplodingController(FakeController):
        def _reset_window(self) -> None:
            self._num_observation_steps = 0
            raise RuntimeError("synthetic reset failure")

    core = FakeEngineCore()
    core.scheduler.acceptance_length_controller = ExplodingController()
    controller = core.scheduler.acceptance_length_controller
    controller.num_spec_tokens = 2
    telemetry = ControllerTelemetry(depth_ladder=(2, 4))
    surface = AdaptiveMtpControlSurface(core, telemetry=telemetry)

    try:
        surface.control("reset")
    except RuntimeError as exc:
        assert "synthetic reset failure" in str(exc)
    else:
        raise AssertionError("controller reset failure was swallowed")

    assert controller.num_spec_tokens == 2
    assert controller._num_observation_steps == 3
    assert controller._num_drafts == 3
    assert controller._num_draft_tokens == 12
    assert controller._num_accepted_tokens == 8
    assert telemetry.reset_count == 0


def test_idle_epoch_reset_is_distinct_and_skips_retained_sessions() -> None:
    core = FakeEngineCore()
    controller = core.scheduler.acceptance_length_controller
    controller.num_spec_tokens = 2
    telemetry = ControllerTelemetry(depth_ladder=(2, 4))
    surface = AdaptiveMtpControlSurface(core, telemetry=telemetry)

    reset = surface.reset_idle_epoch_if_safe()
    assert reset["reset"] is True
    assert reset["reset_reason"] == "idle_epoch"
    assert telemetry.resets_by_reason == {"idle_epoch": 1}

    controller.num_spec_tokens = 2
    core.scheduler.requests["stream-session"] = object()
    skipped = surface.reset_idle_epoch_if_safe()
    assert skipped["reset"] is False
    assert skipped["reset_reason"] == "idle_epoch"
    assert controller.num_spec_tokens == 2
    assert telemetry.reset_count == 1
