from __future__ import annotations

from types import SimpleNamespace

import pytest

import spark_nf3_workspace_reserve as workspace_reserve


class FakeBuffer:
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes

    def numel(self) -> int:
        return self.size_bytes

    def element_size(self) -> int:
        return 1


class FakeWorkspaceManager:
    def __init__(
        self,
        size_bytes: int,
        events: list[tuple[str, int]],
        *,
        refuse_growth: bool = False,
    ) -> None:
        self.events = events
        self.refuse_growth = refuse_growth
        self._num_ubatches = 1
        self._current_workspaces = [FakeBuffer(size_bytes)]
        self.locked = False

    @property
    def size_bytes(self) -> int:
        return self._current_workspaces[0].size_bytes

    @staticmethod
    def _workspace_size_bytes(workspace: FakeBuffer) -> int:
        return workspace.size_bytes

    def _ensure_workspace_size(self, required_bytes: int) -> FakeBuffer:
        self.events.append(("reserve", required_bytes))
        if not self.refuse_growth and self.size_bytes < required_bytes:
            self._current_workspaces[0] = FakeBuffer(required_bytes)
        return self._current_workspaces[0]

    def is_locked(self) -> bool:
        return self.locked


class FakeRunner:
    def __init__(
        self,
        workspace_module: SimpleNamespace | None = None,
        *,
        replace_during_capture: bool = False,
    ) -> None:
        self.workspace_module = workspace_module
        self.replace_during_capture = replace_during_capture
        self.captured_workspace: FakeBuffer | None = None

    def capture_model(self) -> int:
        assert self.workspace_module is not None
        manager = self.workspace_module.current_workspace_manager()
        self.captured_workspace = manager._current_workspaces[0]
        manager.events.append(("capture", manager.size_bytes))
        if self.replace_during_capture:
            manager._current_workspaces[0] = FakeBuffer(manager.size_bytes)
        self.workspace_module.lock_workspace()
        return 123


def _fake_workspace(
    *,
    size_bytes: int,
    refuse_growth: bool = False,
) -> tuple[SimpleNamespace, FakeWorkspaceManager, list[tuple[str, int]]]:
    events: list[tuple[str, int]] = []
    manager = FakeWorkspaceManager(
        size_bytes,
        events,
        refuse_growth=refuse_growth,
    )

    def current_workspace_manager() -> FakeWorkspaceManager:
        return manager

    def lock_workspace() -> None:
        events.append(("lock", manager.size_bytes))
        manager.locked = True

    workspace_module = SimpleNamespace(
        WorkspaceManager=FakeWorkspaceManager,
        current_workspace_manager=current_workspace_manager,
        lock_workspace=lock_workspace,
    )
    return workspace_module, manager, events


@pytest.fixture(autouse=True)
def reset_patch() -> None:
    workspace_reserve._reset_for_tests()
    yield
    workspace_reserve._reset_for_tests()


def test_reserve_precedes_capture_and_pointer_stays_stable_until_lock() -> None:
    workspace, manager, events = _fake_workspace(
        size_bytes=544 * 1024**2
    )
    assert workspace_reserve._install_on_class(
        FakeRunner,
        workspace,
        reserve_bytes=workspace_reserve._REFERENCE_RESERVE_BYTES,
        profile="reference-four-spark",
    )
    runner = FakeRunner(workspace)

    assert runner.capture_model() == 123

    assert events == [
        ("reserve", 768 * 1024**2),
        ("capture", 768 * 1024**2),
        ("lock", 768 * 1024**2),
    ]
    assert runner.captured_workspace is manager._current_workspaces[0]
    assert manager.locked
    snapshot = workspace_reserve.workspace_reserve_snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["last_workspace_bytes"] == 768 * 1024**2
    assert snapshot["owned"]


def test_reserve_never_shrinks_a_larger_future_workspace() -> None:
    workspace, manager, events = _fake_workspace(
        size_bytes=896 * 1024**2
    )
    original_workspace = manager._current_workspaces[0]
    workspace_reserve._install_on_class(
        FakeRunner,
        workspace,
        reserve_bytes=workspace_reserve._REFERENCE_RESERVE_BYTES,
        profile="reference-four-spark-adaptive-2-4",
    )

    FakeRunner(workspace).capture_model()

    assert events == [
        ("reserve", 768 * 1024**2),
        ("capture", 896 * 1024**2),
        ("lock", 896 * 1024**2),
    ]
    assert manager._current_workspaces[0] is original_workspace


def test_failed_reservation_never_starts_capture_or_lock() -> None:
    workspace, manager, events = _fake_workspace(
        size_bytes=544 * 1024**2,
        refuse_growth=True,
    )
    workspace_reserve._install_on_class(
        FakeRunner,
        workspace,
        reserve_bytes=workspace_reserve._REFERENCE_RESERVE_BYTES,
        profile="reference-four-spark",
    )

    with pytest.raises(RuntimeError, match="reservation returned only"):
        FakeRunner(workspace).capture_model()

    assert events == [("reserve", 768 * 1024**2)]
    assert not manager.locked


def test_buffer_replacement_during_capture_fails_closed() -> None:
    workspace, _, events = _fake_workspace(size_bytes=544 * 1024**2)
    workspace_reserve._install_on_class(
        FakeRunner,
        workspace,
        reserve_bytes=workspace_reserve._REFERENCE_RESERVE_BYTES,
        profile="reference-four-spark",
    )

    with pytest.raises(RuntimeError, match="storage changed during"):
        FakeRunner(workspace, replace_during_capture=True).capture_model()

    assert [name for name, _ in events] == ["reserve", "capture", "lock"]


def test_multiple_ubatch_lanes_fail_before_capture() -> None:
    workspace, manager, events = _fake_workspace(size_bytes=544 * 1024**2)
    manager._num_ubatches = 2
    manager._current_workspaces.append(FakeBuffer(manager.size_bytes))
    workspace_reserve._install_on_class(
        FakeRunner,
        workspace,
        reserve_bytes=workspace_reserve._REFERENCE_RESERVE_BYTES,
        profile="reference-four-spark",
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        FakeRunner(workspace).capture_model()

    assert events == []


@pytest.mark.parametrize(
    "profile",
    (
        "reference-four-spark",
        "reference-four-spark-adaptive-2-4",
        "reference-four-spark-adaptive-2-4-c8",
    ),
)
def test_exact_reference_profiles_admit_768_mib(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.setenv(workspace_reserve._PROFILE_ENV, profile)
    monkeypatch.setenv(
        workspace_reserve._RESERVE_ENV,
        str(workspace_reserve._REFERENCE_RESERVE_BYTES),
    )
    monkeypatch.setenv(workspace_reserve._V2_RUNNER_ENV, "1")

    assert workspace_reserve._parse_configuration() == (
        profile,
        768 * 1024**2,
    )


@pytest.mark.parametrize(
    ("profile", "reserve"),
    (
        ("dcp4-compat", 768 * 1024**2),
        ("reference-four-spark", 639 * 1024**2),
        ("reference-four-spark", 0),
    ),
)
def test_configuration_fails_closed_outside_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    reserve: int,
) -> None:
    monkeypatch.setenv(workspace_reserve._PROFILE_ENV, profile)
    monkeypatch.setenv(workspace_reserve._RESERVE_ENV, str(reserve))
    monkeypatch.setenv(workspace_reserve._V2_RUNNER_ENV, "1")

    with pytest.raises(RuntimeError):
        workspace_reserve._parse_configuration()


def test_source_attestation_rejects_capture_model_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workspace_reserve,
        "_EXPECTED_CAPTURE_MODEL_SHA256",
        "0" * 64,
    )
    bindings = workspace_reserve._RuntimeBindings(
        version=workspace_reserve._EXPECTED_VERSION,
        model_runner_module=SimpleNamespace(),
        runner_cls=FakeRunner,
        workspace_module=SimpleNamespace(),
        manager_cls=FakeWorkspaceManager,
    )

    with pytest.raises(RuntimeError, match="capture_model source attestation"):
        workspace_reserve._attest(bindings)
