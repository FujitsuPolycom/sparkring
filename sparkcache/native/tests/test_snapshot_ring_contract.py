from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "spark_cache_snapshot.h"
CUDA = ROOT / "src" / "spark_cache_snapshot.cu"
CMAKE = ROOT / "CMakeLists.txt"
BINDING = ROOT / "python" / "spark_cache_snapshot_native.py"
SPEC = importlib.util.spec_from_file_location(
    "spark_cache_snapshot_native", BINDING
)
assert SPEC is not None and SPEC.loader is not None
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def test_snapshot_ctypes_sizes_are_fixed() -> None:
    assert ctypes.sizeof(native.SnapshotConfig) == 48
    assert ctypes.sizeof(native.SnapshotSource) == 32
    assert ctypes.sizeof(native.SnapshotSubmission) == 32
    assert ctypes.sizeof(native.SnapshotTicket) == 16
    assert ctypes.sizeof(native.SnapshotReadyView) == 104
    assert ctypes.sizeof(native.SnapshotStats) == 96
    assert ctypes.sizeof(native.SnapshotAbiInfo) == 64


def test_header_contract_is_fail_open_and_bounded() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "#define SPARK_CACHE_SNAPSHOT_ABI_VERSION 1u" in text
    assert "#define SPARK_CACHE_SNAPSHOT_MIN_SLOTS 2u" in text
    assert "#define SPARK_CACHE_SNAPSHOT_MAX_SLOTS 3u" in text
    assert "SPARK_CACHE_SNAPSHOT_WOULD_BLOCK" in text
    assert "spark_cache_snapshot_abandon_context" in text
    assert "never inference errors" in text


def test_cuda_contract_uses_external_stream_and_completion_event() -> None:
    text = CUDA.read_text(encoding="utf-8")
    submit = text.index("spark_cache_snapshot_try_submit")
    poll = text.index("spark_cache_snapshot_poll")
    body = text[submit:poll]
    assert "gather_snapshot_kernel<<<" in body
    assert "record_snapshot_completion_event(slot.complete, stream)" in body
    assert "cudaStreamSynchronize" not in body
    assert "cudaDeviceSynchronize" not in body
    assert "cudaEventQuery" in text


def test_abandoned_slots_are_reaped_without_blocking_and_shutdown_is_checked() -> None:
    text = CUDA.read_text(encoding="utf-8")
    reaper = text.index("reap_abandoned_locked")
    shutdown = text.index("shutdown_locked")
    reaper_body = text[reaper:shutdown]
    assert "cudaEventQuery" in reaper_body
    assert "cudaEventSynchronize" not in reaper_body
    assert "ring.reap_discarded" in reaper_body

    shutdown_end = text.index("fill_view_state")
    shutdown_body = text[shutdown:shutdown_end]
    assert "ring.has_writing()" in shutdown_body
    assert "every WRITING view to be released" in shutdown_body
    assert "cudaEventSynchronize" in shutdown_body
    assert "ring.reap_discarded" in shutdown_body


def test_post_launch_failure_is_quarantined_until_stream_drain() -> None:
    text = CUDA.read_text(encoding="utf-8")
    quarantine = text.index("quarantine_post_launch_failure_locked")
    quarantine_end = text.index("bool ticket_matches", quarantine)
    body = text[quarantine:quarantine_end]
    assert "requires_stream_drain = true" in body
    assert "quarantined_stream = stream" in body
    assert "snapshot->ring.abandon(" in body
    assert "drain_quarantined_slot_locked" in body
    assert "ring.complete" not in body

    drain = text.index("drain_quarantined_slot_locked")
    reap = text.index("reap_abandoned_locked")
    drain_body = text[drain:reap]
    assert "synchronize_quarantined_stream" in drain_body
    assert "ring.reap_discarded" in drain_body
    assert drain_body.index("synchronize_quarantined_stream") < (
        drain_body.index("ring.reap_discarded")
    )
    assert "requires_stream_drain = false" in drain_body


def test_post_launch_failure_has_a_compile_time_regression_seam() -> None:
    text = CUDA.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    assert "record_snapshot_completion_event" in text
    assert "SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE" in text
    assert "SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE" in text
    assert "SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE" in cmake
    assert "SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE" in cmake


def test_shutdown_drains_quarantine_before_native_release() -> None:
    text = CUDA.read_text(encoding="utf-8")
    shutdown = text.index("shutdown_locked")
    fill = text.index("fill_view_state")
    body = text[shutdown:fill]
    assert "requires_stream_drain" in body
    assert "drain_quarantined_slot_locked" in body
    assert body.index("drain_quarantined_slot_locked") < body.index(
        "shutdown_complete = true"
    )


def test_poll_never_trusts_stale_event_for_quarantined_ticket() -> None:
    text = CUDA.read_text(encoding="utf-8")
    poll = text.index("SparkCacheSnapshotStatus poll_locked")
    kernel = text.index("__global__ void gather_snapshot_kernel", poll)
    body = text[poll:kernel]
    drain = body.index("drain_quarantined_slot_locked")
    query = body.index("cudaEventQuery")
    assert "requires_stream_drain" in body
    assert drain < query


def test_ready_memoryview_is_read_only() -> None:
    owner = (ctypes.c_ubyte * 8)(*range(8))
    view = native.SnapshotReadyView(
        host_address=ctypes.addressof(owner),
        capacity_bytes=8,
        used_bytes=8,
    )
    payload = native.ready_memoryview(view)
    assert payload.readonly
    assert bytes(payload) == bytes(range(8))
    payload.release()


def test_library_loader_rejects_unpinned_hash_before_dlopen(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "not-a-library.so"
    candidate.write_bytes(b"not a library")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    wrong = "0" * 64 if actual != "0" * 64 else "1" * 64
    with pytest.raises(native.NativeSnapshotError, match="SHA-256 mismatch"):
        native.load_library(candidate, expected_sha256=wrong)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows denies replacing this deliberately retained open inode",
)
def test_attested_dlopen_uses_open_inode_across_atomic_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "snapshot.so"
    replacement = tmp_path / "replacement.so"
    original = b"attested original library bytes"
    candidate.write_bytes(original)
    replacement.write_bytes(b"unattested replacement bytes")
    expected = hashlib.sha256(original).hexdigest()
    observed: dict[str, bytes] = {}

    monkeypatch.setattr(
        native, "_secure_fd_load_path", lambda fd: f"fd://{fd}"
    )

    class FakeLibrary:
        pass

    def fake_dlopen(load_path: str, *, mode: int) -> FakeLibrary:
        del mode
        fd = int(load_path.removeprefix("fd://"))
        os.replace(replacement, candidate)
        duplicate = os.dup(fd)
        with os.fdopen(duplicate, "rb", closefd=True) as source:
            source.seek(0)
            observed["loaded"] = source.read()
        return FakeLibrary()

    library = native._dlopen_attested(
        candidate,
        expected_sha256=expected,
        cdll_factory=fake_dlopen,
    )
    try:
        assert observed["loaded"] == original
        assert candidate.read_bytes() == b"unattested replacement bytes"
    finally:
        library._spark_cache_snapshot_attested_file.close()


def test_attested_fd_lifetime_matches_library_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "snapshot.so"
    payload = b"one immutable inode"
    candidate.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        native, "_secure_fd_load_path", lambda fd: f"fd://{fd}"
    )

    class FakeLibrary:
        pass

    library = native._dlopen_attested(
        candidate,
        expected_sha256=expected,
        cdll_factory=lambda _path, mode: FakeLibrary(),
    )
    retained_fd = library._spark_cache_snapshot_attested_file.fileno()
    assert os.fstat(retained_fd).st_size == len(payload)
    del library
    gc.collect()
    with pytest.raises(OSError):
        os.fstat(retained_fd)


def test_secure_loader_fails_closed_without_linux_procfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "snapshot.so"
    payload = b"correctly hashed but unsupported platform"
    candidate.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(native.sys, "platform", "win32")
    with pytest.raises(
        native.NativeSnapshotError,
        match="requires Linux /proc/self/fd",
    ):
        native._dlopen_attested(
            candidate,
            expected_sha256=expected,
            cdll_factory=lambda _path, mode: object(),
        )


def test_slot_array_is_strict_uint32() -> None:
    values = native.slot_array([9, 3, 17])
    assert list(values) == [9, 3, 17]
    with pytest.raises(ValueError):
        native.slot_array([])
    with pytest.raises(ValueError):
        native.slot_array([-1])
