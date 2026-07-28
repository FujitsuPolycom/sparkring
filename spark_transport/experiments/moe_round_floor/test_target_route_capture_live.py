from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from spark_transport.experiments.moe_round_floor.target_route_capture_live import (
    LiveDependencies,
    LiveInstallConfig,
    LiveInstallError,
    SourcePinnedLiveInstaller,
    install_opt_in,
    source_sha256,
    uninstall_opt_in,
)


class FakeCuda:
    @staticmethod
    def is_current_stream_capturing() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()


class FakeBaseRouter:
    def __init__(self) -> None:
        self.capture_fn = None

    def set_capture_fn(self, callback: Any) -> None:
        self.capture_fn = callback


class WrongRouter:
    pass


class FakeMoERunner:
    def __init__(self, layer_id: int, router: Any | None = None) -> None:
        self.layer_id = layer_id
        self.router = router if router is not None else FakeBaseRouter()


class FakeGPUModelRunner:
    def __init__(self, modules: list[Any]) -> None:
        self.device = "cuda:0"
        self.model_config = SimpleNamespace(enable_return_routed_experts=False)
        self.compilation_config = SimpleNamespace(
            static_forward_context={
                (
                    "model.layers."
                    f"{getattr(module, 'layer_id', 'bad')}"
                    f".mlp.experts#{index}"
                ): module
                for index, module in enumerate(modules)
            }
        )
        self.sampled = object()
        self.rejected = object()
        self.sample_calls = 0

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        self.kv_cache_config = kv_cache_config

    def sample(self, *args: Any, **kwargs: Any) -> tuple[object, object, object]:
        del args, kwargs
        self.sample_calls += 1
        return object(), self.sampled, self.rejected


class FakeWorker:
    def __init__(self, runner: FakeGPUModelRunner) -> None:
        self.model_runner = runner
        self.initialized = 0

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        self.kv_cache_config = kv_cache_config
        self.initialized += 1


class FakeCapture:
    def __init__(
        self,
        *,
        torch_module: Any,
        device: Any,
        config: Any,
        events: list[str],
    ) -> None:
        del torch_module, device, config
        events.append("allocate")
        self.events = events
        self.bound: list[int] = []
        self.arm_calls: list[dict[str, Any]] = []
        self.rejection_calls: list[dict[str, Any]] = []

    def make_base_router_callback(
        self, *, routed_layer_index: int, model_role: str, stream_slot: int
    ):
        assert model_role == "target"
        assert stream_slot == 0
        self.bound.append(routed_layer_index)

        def callback(topk_ids: Any) -> None:
            del topk_ids

        return callback

    def begin_request(self, **kwargs: Any) -> None:
        self.arm_calls.append(kwargs)

    def record_rejection(
        self,
        num_sampled: Any,
        num_rejected: Any,
        *,
        stream_slot: int,
    ) -> None:
        self.rejection_calls.append(
            {
                "num_sampled": num_sampled,
                "num_rejected": num_rejected,
                "stream_slot": stream_slot,
            }
        )

    def disarm(self, *, stream_slot: int) -> None:
        self.events.append(f"disarm:{stream_slot}")

    def read_counters(self, *, timed_execution_complete: bool) -> dict[str, int]:
        assert timed_execution_complete
        return {"rounds_claimed": 123}

    def drain_jsonl(
        self, path: Any, provenance: Any, *, timed_execution_complete: bool
    ) -> dict[str, int]:
        del path, provenance
        assert timed_execution_complete
        return {"rounds_completed": 123}


@dataclass
class Harness:
    installer: SourcePinnedLiveInstaller
    events: list[str]
    captures: list[FakeCapture]


def make_modules(
    layer_ids: list[int] | None = None,
    *,
    include_draft: bool = True,
) -> list[FakeMoERunner]:
    modules = [
        FakeMoERunner(layer_id)
        for layer_id in (layer_ids if layer_ids is not None else list(range(3, 78)))
    ]
    if include_draft:
        modules.append(FakeMoERunner(78))
    return modules


def make_harness(modules: list[Any]) -> Harness:
    events: list[str] = []
    captures: list[FakeCapture] = []

    def extension_loader(torch_module: Any, *, name: str) -> object:
        assert torch_module is FakeTorch
        assert name == "test-target-route-extension"
        events.append("load")
        return object()

    def capture_factory(**kwargs: Any) -> FakeCapture:
        capture = FakeCapture(**kwargs, events=events)
        captures.append(capture)
        return capture

    config = LiveInstallConfig(
        extension_name="test-target-route-extension",
        worker_initialize_sha256=source_sha256(
            FakeWorker.initialize_from_config
        ),
        runner_initialize_kv_sha256=source_sha256(
            FakeGPUModelRunner.initialize_kv_cache
        ),
        runner_sample_sha256=source_sha256(FakeGPUModelRunner.sample),
    )
    dependencies = LiveDependencies(
        torch_module=FakeTorch,
        moe_runner_type=FakeMoERunner,
        base_router_type=FakeBaseRouter,
        extension_loader=extension_loader,
        capture_factory=capture_factory,
    )
    installer = SourcePinnedLiveInstaller(
        worker_type=FakeWorker,
        runner_type=FakeGPUModelRunner,
        dependencies=dependencies,
        config=config,
    )
    return Harness(installer=installer, events=events, captures=captures)


def test_binds_exact_target_layers_once_after_original_initialization() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    original = FakeWorker.initialize_from_config
    original_sample = FakeGPUModelRunner.sample
    harness.installer.install()
    try:
        runner = FakeGPUModelRunner(modules)
        worker = FakeWorker(runner)
        worker.initialize_from_config("kv")
        assert worker.initialized == 1
        assert harness.events == ["load", "allocate"]
        assert harness.captures[0].bound == list(range(75))
        assert harness.installer.controller.bound_layer_ids == tuple(range(3, 78))
        assert all(
            module.router.capture_fn is not None
            for module in modules
            if module.layer_id in range(3, 78)
        )
        assert modules[-1].layer_id == 78
        assert modules[-1].router.capture_fn is None
        harness.installer.controller.arm_salted(
            request_slot=0,
            request_key="salted",
        )
        input_batch = SimpleNamespace(num_reqs=1, num_draft_tokens=5)
        result = runner.sample("hidden", input_batch)
        assert result[1] is runner.sampled
        assert result[2] is runner.rejected
        assert runner.sample_calls == 1
        assert harness.captures[0].rejection_calls == [
            {
                "num_sampled": runner.sampled,
                "num_rejected": runner.rejected,
                "stream_slot": 0,
            }
        ]

        worker.initialize_from_config("kv-again")
        assert worker.initialized == 2
        assert harness.events == ["load", "allocate"]
        assert harness.captures[0].bound == list(range(75))
    finally:
        harness.installer.uninstall()
    assert FakeWorker.initialize_from_config is original
    assert FakeGPUModelRunner.sample is original_sample
    assert all(module.router.capture_fn is None for module in modules)


def test_target_selector_excludes_one_explicit_mtp_draft_runner() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    harness.installer.install()
    try:
        FakeWorker(FakeGPUModelRunner(modules)).initialize_from_config("kv")
        assert harness.installer.controller.bound_layer_ids == tuple(range(3, 78))
        assert len(modules) == 76
        draft = modules[-1]
        assert draft.layer_id == 78
        assert draft.router.capture_fn is None
    finally:
        harness.installer.uninstall()


@pytest.mark.parametrize(
    ("modules", "message"),
    (
        (make_modules(list(range(3, 77))), "exactly 75"),
        (
            make_modules(list(range(3, 77)) + [76]),
            "duplicate target layer IDs",
        ),
        (
            make_modules(list(range(3, 77)) + [79]),
            "unexpected MoERunner ownership",
        ),
    ),
)
def test_missing_duplicate_or_wrong_layer_fails_before_load_or_binding(
    modules: list[FakeMoERunner], message: str
) -> None:
    harness = make_harness(modules)
    harness.installer.install()
    try:
        worker = FakeWorker(FakeGPUModelRunner(modules))
        with pytest.raises(LiveInstallError, match=message):
            worker.initialize_from_config("kv")
        assert harness.events == []
        assert harness.captures == []
        assert all(module.router.capture_fn is None for module in modules)
    finally:
        harness.installer.uninstall()


def test_wrong_router_and_existing_callback_fail_before_load() -> None:
    for mutate, message in (
        (
            lambda modules: setattr(modules[10], "router", WrongRouter()),
            "lacks BaseRouter",
        ),
        (
            lambda modules: setattr(
                modules[10].router, "capture_fn", lambda value: value
            ),
            "already has a callback",
        ),
    ):
        modules = make_modules()
        mutate(modules)
        harness = make_harness(modules)
        harness.installer.install()
        try:
            with pytest.raises(LiveInstallError, match=message):
                FakeWorker(FakeGPUModelRunner(modules)).initialize_from_config("kv")
            assert harness.events == []
        finally:
            harness.installer.uninstall()


def test_source_mismatch_mutates_no_worker_method() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    original = FakeWorker.initialize_from_config
    original_sample = FakeGPUModelRunner.sample
    bad = LiveInstallConfig(
        extension_name="test-target-route-extension",
        worker_initialize_sha256="0" * 64,
        runner_initialize_kv_sha256=source_sha256(
            FakeGPUModelRunner.initialize_kv_cache
        ),
        runner_sample_sha256=source_sha256(FakeGPUModelRunner.sample),
    )
    harness.installer.config = bad
    with pytest.raises(LiveInstallError, match="source mismatch"):
        harness.installer.install()
    assert FakeWorker.initialize_from_config is original
    assert FakeGPUModelRunner.sample is original_sample


def test_sample_source_mismatch_mutates_no_live_method() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    original_initialize = FakeWorker.initialize_from_config
    original_sample = FakeGPUModelRunner.sample
    harness.installer.config = LiveInstallConfig(
        extension_name="test-target-route-extension",
        worker_initialize_sha256=source_sha256(
            FakeWorker.initialize_from_config
        ),
        runner_initialize_kv_sha256=source_sha256(
            FakeGPUModelRunner.initialize_kv_cache
        ),
        runner_sample_sha256="0" * 64,
    )
    with pytest.raises(
        LiveInstallError,
        match="GPUModelRunner.sample source mismatch",
    ):
        harness.installer.install()
    assert FakeWorker.initialize_from_config is original_initialize
    assert FakeGPUModelRunner.sample is original_sample


def test_sample_before_runner_attachment_is_inert() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    harness.installer.install()
    try:
        runner = FakeGPUModelRunner(modules)
        result = runner.sample("profile")
        assert result[1] is runner.sampled
        assert result[2] is runner.rejected
        assert harness.captures == []
    finally:
        harness.installer.uninstall()


def test_attached_sample_is_inert_until_armed_and_skips_q1() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    harness.installer.install()
    try:
        runner = FakeGPUModelRunner(modules)
        FakeWorker(runner).initialize_from_config("kv")
        verify = SimpleNamespace(num_reqs=1, num_draft_tokens=5)
        runner.sample("hidden", verify)
        assert harness.captures[0].rejection_calls == []

        harness.installer.controller.arm_salted(
            request_slot=0,
            request_key="salted",
        )
        q1 = SimpleNamespace(num_reqs=1, num_draft_tokens=0)
        runner.sample("hidden", q1)
        assert harness.captures[0].rejection_calls == []
    finally:
        harness.installer.uninstall()


def test_armed_multi_request_sample_fails_closed() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    harness.installer.install()
    try:
        runner = FakeGPUModelRunner(modules)
        FakeWorker(runner).initialize_from_config("kv")
        harness.installer.controller.arm_salted(
            request_slot=0,
            request_key="salted",
        )
        batch = SimpleNamespace(num_reqs=2, num_draft_tokens=10)
        with pytest.raises(LiveInstallError, match="exactly one request"):
            runner.sample("hidden", batch)
        assert harness.captures[0].rejection_calls == []
    finally:
        harness.installer.uninstall()


def test_deployed_source_pins_are_immutable_exact_values() -> None:
    from spark_transport.experiments.moe_round_floor import (
        target_route_capture_live as live,
    )

    assert live.DEPLOYED_WORKER_INITIALIZE_SHA256 == (
        "196bbe8208eb5ba56f0e2eb97c0d8922351f1963ac6dbd3466eae94378864ad9"
    )
    assert live.DEPLOYED_RUNNER_INITIALIZE_KV_SHA256 == (
        "c606851a60fef594fb231c7c68e695d3a1d52396d2e12a0304819bef8c21e808"
    )
    assert live.DEPLOYED_RUNNER_SAMPLE_SHA256 == (
        "4d5ce613197dfa32ab5cce9472ef966ce4bca45f8a41edc87b79527908e9b07d"
    )
    LiveInstallConfig().validate()


def test_sample_wrapper_has_no_device_readback_or_sync() -> None:
    source = inspect.getsource(SourcePinnedLiveInstaller.install)
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        "synchronize(",
    ):
        assert forbidden not in source
    assert "self._controller.record_rejection(" in source


def test_builtin_capturer_and_second_target_runner_are_rejected() -> None:
    first_modules = make_modules()
    harness = make_harness(first_modules)
    harness.installer.install()
    try:
        first_worker = FakeWorker(FakeGPUModelRunner(first_modules))
        first_worker.initialize_from_config("kv")
        second_runner = FakeGPUModelRunner(make_modules())
        with pytest.raises(LiveInstallError, match="two target"):
            FakeWorker(second_runner).initialize_from_config("kv")
    finally:
        harness.installer.uninstall()

    modules = make_modules()
    runner = FakeGPUModelRunner(modules)
    runner.model_config.enable_return_routed_experts = True
    harness = make_harness(modules)
    harness.installer.install()
    try:
        with pytest.raises(LiveInstallError, match="must remain disabled"):
            FakeWorker(runner).initialize_from_config("kv")
        assert harness.events == []
    finally:
        harness.installer.uninstall()


def test_process_global_controller_entrypoint_is_explicit_and_reversible() -> None:
    modules = make_modules()
    harness = make_harness(modules)
    installer = install_opt_in(
        worker_type=FakeWorker,
        runner_type=FakeGPUModelRunner,
        dependencies=harness.installer.dependencies,
        config=harness.installer.config,
    )
    try:
        FakeWorker(FakeGPUModelRunner(modules)).initialize_from_config("kv")
        assert installer.controller.bound_layer_ids == tuple(range(3, 78))
    finally:
        uninstall_opt_in()
    assert all(module.router.capture_fn is None for module in modules)
