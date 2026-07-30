from __future__ import annotations

from types import SimpleNamespace

import pytest

import spark_nf3_startup_profile_cap as profile_cap


class FakeRange:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class FakeCompilationConfig:
    def __init__(
        self,
        *,
        endpoints: list[int] | None = None,
        capture_sizes: list[int] | None = None,
        compile_sizes: list[int] | None = None,
        max_capture_size: int = 32,
    ) -> None:
        self.compile_ranges_endpoints = (
            [4096] if endpoints is None else endpoints
        )
        self.cudagraph_capture_sizes = (
            [1, 2, 4, 8, 16, 24, 32]
            if capture_sizes is None
            else capture_sizes
        )
        self.compile_sizes = [] if compile_sizes is None else compile_sizes
        self.max_cudagraph_capture_size = max_capture_size

    def get_compile_ranges(self) -> list[FakeRange]:
        endpoints = sorted(set(self.compile_ranges_endpoints))
        starts = [0] + endpoints[:-1]
        return [
            FakeRange(start + 1, end)
            for start, end in zip(starts, endpoints)
        ]


class FakeRunner:
    observations: list[tuple[int, int]]
    should_fail: bool

    def __init__(
        self,
        *,
        runtime_max: int = 4096,
        scheduler_max: int = 4096,
        cache_bytes: int | None = 12_000_000_000,
        should_fail: bool = False,
        compilation_config: FakeCompilationConfig | None = None,
    ) -> None:
        self.max_num_tokens = runtime_max
        self.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=scheduler_max
        )
        self.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=cache_bytes
        )
        self.compilation_config = (
            compilation_config or FakeCompilationConfig()
        )
        self.should_fail = should_fail
        self.observations = []

    def profile_run(self) -> str:
        self.observations.append(
            (
                self.max_num_tokens,
                self.scheduler_config.max_num_batched_tokens,
            )
        )
        if self.should_fail:
            raise RuntimeError("synthetic profile failure")
        return "profiled"


class OtherFakeRunner(FakeRunner):
    def profile_run(self) -> str:
        return super().profile_run()


class FakeWorker:
    def __init__(
        self,
        *,
        runtime_batched: int = 4096,
        runtime_scheduled: int | None = 4096,
        should_fail: bool = False,
        model_runner: FakeRunner | None = None,
        use_v2_model_runner: bool = True,
    ) -> None:
        self.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=runtime_batched,
            max_num_scheduled_tokens=runtime_scheduled,
        )
        self.vllm_config = SimpleNamespace(
            scheduler_config=self.scheduler_config
        )
        self.should_fail = should_fail
        self.model_runner = model_runner
        self.use_v2_model_runner = use_v2_model_runner
        self.memory_profile_calls = 0
        self.observations: list[tuple[int, int | None]] = []
        self.dummy_run_sizes: list[int] = []

    def determine_available_memory(self) -> str:
        self.memory_profile_calls += 1
        if self.model_runner is None:
            return "memory-profiled"
        return self.model_runner.profile_run()

    def _compile_or_warm_up_model_impl(self) -> str:
        batched = int(self.scheduler_config.max_num_batched_tokens)
        scheduled_raw = self.scheduler_config.max_num_scheduled_tokens
        scheduled = (
            None if scheduled_raw is None else int(scheduled_raw)
        )
        self.observations.append((batched, scheduled))
        self.dummy_run_sizes.append(
            max(batched, scheduled if scheduled is not None else batched)
        )
        if self.should_fail:
            raise RuntimeError("synthetic compile warmup failure")
        return "warmed"


@pytest.fixture(autouse=True)
def reset_patch() -> None:
    profile_cap._reset_for_tests(FakeRunner, FakeWorker)
    yield
    profile_cap._reset_for_tests(FakeRunner, FakeWorker)


def test_cap_changes_only_profile_width_and_restores_runtime() -> None:
    assert profile_cap._install_on_class(FakeRunner, 256)
    runner = FakeRunner()

    assert runner.profile_run() == "profiled"
    assert runner.observations == [(256, 4096)]
    assert runner.max_num_tokens == 4096
    assert runner.scheduler_config.max_num_batched_tokens == 4096
    snapshot = profile_cap.startup_profile_cap_snapshot()
    snapshot["last_elapsed_seconds"] = None
    assert {
        key: snapshot[key]
        for key in (
            "installed",
            "profile_calls",
            "capped_calls",
            "last_runtime_max_tokens",
            "last_profile_max_tokens",
            "last_elapsed_seconds",
            "compile_warmup_installed",
            "compile_warmup_calls",
            "last_worker_runtime_batched_tokens",
            "last_worker_runtime_scheduled_tokens",
            "source_sha256",
            "compile_warmup_source_sha256",
            "version",
        )
    } == {
        "installed": True,
        "profile_calls": 1,
        "capped_calls": 1,
        "last_runtime_max_tokens": 4096,
        "last_profile_max_tokens": 256,
        "last_elapsed_seconds": None,
        "compile_warmup_installed": False,
        "compile_warmup_calls": 0,
        "last_worker_runtime_batched_tokens": None,
        "last_worker_runtime_scheduled_tokens": None,
        "source_sha256": profile_cap._EXPECTED_PROFILE_RUN_SHA256,
        "compile_warmup_source_sha256": (
            profile_cap._EXPECTED_COMPILE_WARMUP_SHA256
        ),
        "version": profile_cap._EXPECTED_VERSION,
    }


def test_profile_cap_log_identifies_selected_runner_kind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    assert profile_cap._install_on_class(
        FakeRunner,
        256,
        runner_kind="v2",
    )

    assert FakeRunner().profile_run() == "profiled"
    assert "runner_kind=v2" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("0", "v1"), ("1", "v2")),
)
def test_runner_kind_requires_explicit_environment_selection(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: str,
) -> None:
    monkeypatch.setenv(profile_cap._V2_RUNNER_ENV, raw)
    assert profile_cap._runner_kind_from_environment() == expected


@pytest.mark.parametrize("raw", (None, "", "true", "2"))
def test_runner_kind_rejects_implicit_or_invalid_selection(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
) -> None:
    if raw is None:
        monkeypatch.delenv(profile_cap._V2_RUNNER_ENV, raising=False)
    else:
        monkeypatch.setenv(profile_cap._V2_RUNNER_ENV, raw)
    with pytest.raises(RuntimeError, match="explicitly 0 or 1"):
        profile_cap._runner_kind_from_environment()


@pytest.mark.parametrize(
    ("runner_kind", "expected_sha256"),
    (
        ("v1", profile_cap._EXPECTED_PROFILE_RUN_SHA256),
        ("v2", profile_cap._EXPECTED_V2_PROFILE_RUN_SHA256),
    ),
)
def test_install_owns_only_selected_runner_and_reports_selected_hash(
    monkeypatch: pytest.MonkeyPatch,
    runner_kind: str,
    expected_sha256: str,
) -> None:
    monkeypatch.setenv(profile_cap._CAP_ENV, "256")
    monkeypatch.setenv(profile_cap._SINGLE_RANGE_ENV, "0")
    bindings = profile_cap._RuntimeBindings(
        version=profile_cap._EXPECTED_VERSION,
        runner_cls=FakeRunner,
        runner_kind=runner_kind,
        v1_runner_cls=(
            FakeRunner if runner_kind == "v1" else OtherFakeRunner
        ),
        v2_runner_cls=(
            FakeRunner if runner_kind == "v2" else OtherFakeRunner
        ),
        worker_cls=FakeWorker,
    )
    monkeypatch.setattr(profile_cap, "_load_bindings", lambda: bindings)
    monkeypatch.setattr(profile_cap, "_attest", lambda _bindings: None)

    assert profile_cap.install() is True
    snapshot = profile_cap.startup_profile_cap_snapshot()
    assert snapshot["runner_kind"] == runner_kind
    assert snapshot["v1_runner_owned"] is (runner_kind == "v1")
    assert snapshot["v2_runner_owned"] is (runner_kind == "v2")
    assert snapshot["memory_ownership_guard_installed"] is True
    assert snapshot["source_sha256"] == expected_sha256


def test_memory_guard_admits_owned_v2_runner_before_profile() -> None:
    assert profile_cap._install_on_class(
        FakeRunner,
        256,
        runner_kind="v2",
    )
    assert profile_cap._install_memory_ownership_guard(
        FakeWorker,
        runner_cls=FakeRunner,
        runner_kind="v2",
        cap=256,
    )
    runner = FakeRunner()
    worker = FakeWorker(model_runner=runner, use_v2_model_runner=True)

    assert worker.determine_available_memory() == "profiled"
    assert runner.observations == [(256, 4096)]
    assert worker.memory_profile_calls == 1
    assert (
        profile_cap.startup_profile_cap_snapshot()[
            "memory_ownership_guard_calls"
        ]
        == 1
    )


def test_memory_guard_rejects_worker_runner_kind_mismatch_before_profile() -> None:
    profile_cap._install_on_class(FakeRunner, 256, runner_kind="v2")
    profile_cap._install_memory_ownership_guard(
        FakeWorker,
        runner_cls=FakeRunner,
        runner_kind="v2",
        cap=256,
    )
    runner = FakeRunner()
    worker = FakeWorker(model_runner=runner, use_v2_model_runner=False)

    with pytest.raises(RuntimeError, match="disagrees with Worker"):
        worker.determine_available_memory()
    assert runner.observations == []
    assert worker.memory_profile_calls == 0


def test_memory_guard_rejects_unexpected_runner_type_before_profile() -> None:
    profile_cap._install_on_class(FakeRunner, 256, runner_kind="v2")
    profile_cap._install_memory_ownership_guard(
        FakeWorker,
        runner_cls=FakeRunner,
        runner_kind="v2",
        cap=256,
    )
    runner = OtherFakeRunner()
    worker = FakeWorker(model_runner=runner, use_v2_model_runner=True)

    with pytest.raises(RuntimeError, match="unexpected model runner type"):
        worker.determine_available_memory()
    assert runner.observations == []
    assert worker.memory_profile_calls == 0


def test_compile_warmup_cap_bounds_hidden_kernel_budget_and_restores() -> None:
    assert profile_cap._install_on_worker_class(FakeWorker, 32)
    worker = FakeWorker()

    assert worker._compile_or_warm_up_model_impl() == "warmed"
    assert worker.observations == [(32, 32)]
    assert worker.dummy_run_sizes == [32]
    assert worker.scheduler_config.max_num_batched_tokens == 4096
    assert worker.scheduler_config.max_num_scheduled_tokens == 4096
    snapshot = profile_cap.startup_profile_cap_snapshot()
    assert snapshot["compile_warmup_installed"] is True
    assert snapshot["compile_warmup_calls"] == 1
    assert snapshot["last_worker_runtime_batched_tokens"] == 4096
    assert snapshot["last_worker_runtime_scheduled_tokens"] == 4096


def test_compile_warmup_cap_restores_none_scheduled_budget() -> None:
    assert profile_cap._install_on_worker_class(FakeWorker, 32)
    worker = FakeWorker(runtime_scheduled=None)

    assert worker._compile_or_warm_up_model_impl() == "warmed"
    assert worker.observations == [(32, 32)]
    assert worker.dummy_run_sizes == [32]
    assert worker.scheduler_config.max_num_batched_tokens == 4096
    assert worker.scheduler_config.max_num_scheduled_tokens is None


def test_compile_warmup_exception_restores_both_runtime_budgets() -> None:
    assert profile_cap._install_on_worker_class(FakeWorker, 32)
    worker = FakeWorker(should_fail=True)

    with pytest.raises(RuntimeError, match="synthetic compile warmup failure"):
        worker._compile_or_warm_up_model_impl()

    assert worker.observations == [(32, 32)]
    assert worker.dummy_run_sizes == [32]
    assert worker.scheduler_config.max_num_batched_tokens == 4096
    assert worker.scheduler_config.max_num_scheduled_tokens == 4096


def test_single_range_rejects_singleton_profile_before_cache_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(profile_cap._SINGLE_RANGE_ENV, "1")
    profile_cap._install_on_class(FakeRunner, 1)
    runner = FakeRunner()

    with pytest.raises(RuntimeError, match="singleton Q1"):
        runner.profile_run()

    # Torch specializes dynamic integers with value 0 or 1.  Reaching the
    # original profile here would let Q1 produce a 13-argument standalone
    # artifact for the sole Q1-Q4096 range, while Q2+ requires the dynamic
    # shape scalar and therefore 14 arguments.
    assert runner.observations == []
    assert runner.max_num_tokens == 4096


def test_single_range_q2_profile_preserves_symbolic_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(profile_cap._SINGLE_RANGE_ENV, "1")
    profile_cap._install_on_class(FakeRunner, 2)
    runner = FakeRunner()

    assert runner.profile_run() == "profiled"
    assert runner.observations == [(2, 4096)]
    assert runner.max_num_tokens == 4096


def test_single_range_q40_capture_contract_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(profile_cap._SINGLE_RANGE_ENV, "1")
    monkeypatch.setenv(
        "VLLM_SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE",
        "40",
    )
    profile_cap._install_on_class(FakeRunner, 2)
    runner = FakeRunner(
        compilation_config=FakeCompilationConfig(
            capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40],
            max_capture_size=40,
        )
    )

    assert runner.profile_run() == "profiled"
    assert runner.observations == [(2, 4096)]
    assert runner.max_num_tokens == 4096


def test_exception_still_restores_runtime_width() -> None:
    profile_cap._install_on_class(FakeRunner, 256)
    runner = FakeRunner(should_fail=True)

    with pytest.raises(RuntimeError, match="synthetic profile failure"):
        runner.profile_run()

    assert runner.observations == [(256, 4096)]
    assert runner.max_num_tokens == 4096


def test_profile_at_or_below_cap_remains_unchanged() -> None:
    profile_cap._install_on_class(FakeRunner, 256)
    runner = FakeRunner(runtime_max=128, scheduler_max=128)

    assert runner.profile_run() == "profiled"
    assert runner.observations == [(128, 128)]
    assert profile_cap.startup_profile_cap_snapshot()["capped_calls"] == 0


def test_cap_requires_explicit_kv_cache_size() -> None:
    profile_cap._install_on_class(FakeRunner, 256)
    runner = FakeRunner(cache_bytes=None)

    with pytest.raises(RuntimeError, match="explicit positive"):
        runner.profile_run()

    assert runner.observations == []
    assert runner.max_num_tokens == 4096


def test_installation_is_idempotent_but_rejects_cap_change() -> None:
    assert profile_cap._install_on_class(FakeRunner, 256)
    assert not profile_cap._install_on_class(FakeRunner, 256)
    with pytest.raises(RuntimeError, match="already installed"):
        profile_cap._install_on_class(FakeRunner, 128)


@pytest.mark.parametrize("value", ["", "zero", "0", "4097", "-1"])
def test_cap_parser_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(profile_cap._CAP_ENV, value)
    with pytest.raises(RuntimeError):
        profile_cap._parse_cap()
