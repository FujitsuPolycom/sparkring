from __future__ import annotations

from dataclasses import dataclass

from spark_transport.experiments.adaptive_mtp_controller.runtime_installer import (
    RuntimeInstaller,
    RuntimeTypes,
)


@dataclass(frozen=True)
class FakeUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int


class FakeController:
    def __init__(self, max_num_spec_tokens: int, observation_window: int) -> None:
        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        self.num_spec_tokens = max_num_spec_tokens
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> FakeUpdate | None:
        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if self._num_observation_steps < self.observation_window:
            return None
        previous = self.num_spec_tokens
        self.num_spec_tokens = 3
        self._reset_window()
        return FakeUpdate(previous, self.num_spec_tokens)

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0


class FakeSchedulerOutput:
    num_spec_tokens_to_schedule = 2


class FakeScheduler:
    def __init__(self) -> None:
        self.acceptance_length_controller = FakeController(4, 2)
        self.num_spec_tokens = 4
        self.running: list[object] = []
        self.waiting: list[object] = []
        self.skipped_waiting: list[object] = []
        self.requests: dict[str, object] = {}
        self.deferred_frees: list[object] = []
        self.num_waiting_for_streaming_input = 0

    def update_from_output(
        self,
        scheduler_output: FakeSchedulerOutput,
        model_runner_output: object,
    ) -> str:
        del scheduler_output, model_runner_output
        self.acceptance_length_controller.observe_batch(
            num_drafts=2,
            num_draft_tokens=4,
            num_accepted_tokens=3,
        )
        return "updated"


class FakeEngineCore:
    def __init__(self) -> None:
        self.scheduler = FakeScheduler()
        self.batch_queue: list[object] = []

    def has_work(self) -> bool:
        return bool(self.scheduler.requests or self.batch_queue)

    def add_request(self, request: object, request_wave: int = 0) -> int:
        request_id = getattr(request, "request_id")
        self.scheduler.requests[request_id] = request
        return request_wave


def test_installer_only_wraps_engine_idle_epoch_and_exposes_utility() -> None:
    types = RuntimeTypes(
        controller=FakeController,
        scheduler=FakeScheduler,
        engine_core=FakeEngineCore,
    )
    installer = RuntimeInstaller(types, depth_ladder=(2, 4))
    original_observe = FakeController.observe_batch
    original_update = FakeScheduler.update_from_output
    original_add_request = FakeEngineCore.add_request

    installer.install()
    try:
        core = FakeEngineCore()
        status = core.spark_adaptive_mtp_control("status")

        assert status["raw_k"] == 4
        assert status["window"]["observation_steps"] == 0
        assert FakeController.observe_batch is original_observe
        assert FakeScheduler.update_from_output is original_update
    finally:
        installer.uninstall()

    assert FakeController.observe_batch is original_observe
    assert FakeScheduler.update_from_output is original_update
    assert FakeEngineCore.add_request is original_add_request
    assert not hasattr(FakeEngineCore, "spark_adaptive_mtp_control")


def test_first_request_of_each_idle_epoch_resets_controller_automatically() -> None:
    class Request:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

    installer = RuntimeInstaller(
        RuntimeTypes(FakeController, FakeScheduler, FakeEngineCore),
        depth_ladder=(2, 4),
    )
    installer.install()
    try:
        core = FakeEngineCore()
        controller = core.scheduler.acceptance_length_controller
        controller.num_spec_tokens = 2
        controller._num_observation_steps = 7

        assert core.add_request(Request("first"), request_wave=11) == 11
        status = core.spark_adaptive_mtp_control("status")
        assert controller.num_spec_tokens == 4
        assert controller._num_observation_steps == 0
        assert status["resets_by_reason"] == {"idle_epoch": 1}

        controller.num_spec_tokens = 2
        core.add_request(Request("concurrent"))
        assert controller.num_spec_tokens == 2
        status = core.spark_adaptive_mtp_control("status")
        assert status["resets_by_reason"] == {"idle_epoch": 1}

        core.scheduler.requests.clear()
        core.add_request(Request("next-epoch"))
        assert controller.num_spec_tokens == 4
        status = core.spark_adaptive_mtp_control("status")
        assert status["resets_by_reason"] == {"idle_epoch": 2}
    finally:
        installer.uninstall()


def test_idle_waiting_stream_session_is_not_reset_or_rejected() -> None:
    class Request:
        request_id = "new-input"

    installer = RuntimeInstaller(
        RuntimeTypes(FakeController, FakeScheduler, FakeEngineCore),
        depth_ladder=(2, 4),
    )
    installer.install()
    try:
        core = FakeEngineCore()
        controller = core.scheduler.acceptance_length_controller
        controller.num_spec_tokens = 2
        core.scheduler.requests["stream-session"] = object()

        # Simulate an EngineCore that is waiting for the next streaming chunk
        # even though the scheduler retains the logical session.
        core.has_work = lambda: False
        core.add_request(Request())

        assert controller.num_spec_tokens == 2
        status = core.spark_adaptive_mtp_control("status")
        assert status["reset_count"] == 0
    finally:
        installer.uninstall()

