from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_context_cache_codec import LayerPlan
from spark_context_cache_native_placement import (
    ArenaMode,
    NativePlacementAdapter,
    NativePlacementCallError,
    NativePlacementContractError,
    NativePlacementLibrary,
    RestoreState,
    build_destination_descriptors,
)
import spark_cache_native as native


class MockFunction:
    def __init__(self, implementation=None):
        self.implementation = implementation or (lambda *args: 0)
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


class MockLibrary:
    SYMBOLS = (
        "spark_cache_parse_verified_v1_chunk",
        "spark_cache_placement_create",
        "spark_cache_placement_destroy",
        "spark_cache_placement_configure_destinations",
        "spark_cache_placement_begin_restore",
        "spark_cache_placement_acquire_arena_view",
        "spark_cache_placement_submit_direct_slab",
        "spark_cache_placement_submit_transposed_slab",
        "spark_cache_placement_finish_restore",
        "spark_cache_placement_abort_restore",
        "spark_cache_placement_copy_last_error",
        "spark_cache_placement_runtime_last_error",
        "spark_cache_placement_status_string",
    )

    def __init__(self):
        for symbol in self.SYMBOLS:
            setattr(self, symbol, MockFunction())
        self.spark_cache_placement_create.implementation = self._create
        self.spark_cache_placement_acquire_arena_view.implementation = self._acquire
        self.spark_cache_placement_copy_last_error.implementation = self._copy_error
        self.spark_cache_placement_runtime_last_error.implementation = lambda: (
            b"mock runtime failure"
        )
        self.spark_cache_placement_status_string.implementation = lambda _status: (
            b"MOCK_ERROR"
        )

    @staticmethod
    def _create(_config, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            0xCACE
        )
        return 0

    @staticmethod
    def _acquire(_placement, arena_index, output):
        view = ctypes.cast(output, ctypes.POINTER(native.ArenaView))[0]
        view.host_address = 0xA000
        view.device_address = 0xB000
        view.capacity_bytes = 4096
        view.arena_index = arena_index
        view.arena_mode = native.ARENA_MAPPED_HOST
        view.flags = 1
        return 0

    @staticmethod
    def _copy_error(_placement, output, capacity):
        message = b"mock native failure\0"
        ctypes.memmove(output, message, min(len(message), capacity))
        return 0


class FakeTensor:
    def __init__(
        self,
        *,
        pointer: int,
        shape=(2, 4, 3),
        strides=(12, 3, 1),
        element_size=2,
        device_type="cuda",
        contiguous=True,
    ):
        self.shape = shape
        self._strides = strides
        self._element_size = element_size
        self.device = SimpleNamespace(type=device_type)
        self._pointer = pointer
        self._contiguous = contiguous

    def stride(self):
        return self._strides

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._pointer

    def is_contiguous(self):
        return self._contiguous


def _attested_mock(tmp_path: Path, mock: MockLibrary | None = None):
    artifact = tmp_path / "libspark_cache_placement.so"
    artifact.write_bytes(b"native-test-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    loaded = mock or MockLibrary()
    library = NativePlacementLibrary.load(
        artifact,
        expected_sha256=digest,
        binding_loader=lambda _path: (
            loaded,
            SimpleNamespace(abi_version=native.ABI_VERSION),
        ),
    )
    return library, loaded


def _configured_adapter(tmp_path: Path, mock: MockLibrary | None = None):
    library, loaded = _attested_mock(tmp_path, mock)
    adapter = NativePlacementAdapter.create(
        library,
        arena_mode=ArenaMode.MAPPED_HOST,
        arena_bytes=4096,
        max_destinations=4,
        max_slots=32,
        max_chunks_per_slab=8,
        device_ordinal=0,
    )
    adapter.configure(
        (LayerPlan("target", "target_ckv", 6),),
        {"target": FakeTensor(pointer=0x1000)},
    )
    return adapter, loaded


def test_hash_mismatch_refuses_to_load_before_cdll(tmp_path):
    artifact = tmp_path / "placement.so"
    artifact.write_bytes(b"wrong bytes")
    called = False

    def loader(_path):
        nonlocal called
        called = True
        return MockLibrary()

    with pytest.raises(NativePlacementContractError, match="SHA-256 mismatch"):
        NativePlacementLibrary.load(
            artifact,
            expected_sha256="0" * 64,
            binding_loader=loader,
        )
    assert called is False


def test_canonical_binding_rejection_fails_closed(tmp_path):
    artifact = tmp_path / "placement.so"
    artifact.write_bytes(b"native-test-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    def reject(_path):
        raise native.NativePlacementError("ctypes ABI size mismatch")

    with pytest.raises(NativePlacementContractError, match="ABI size mismatch"):
        NativePlacementLibrary.load(
            artifact,
            expected_sha256=digest,
            binding_loader=reject,
        )


def test_destination_descriptors_follow_codec_layer_major_order():
    plans = (
        LayerPlan("target.0", "target_ckv", 6),
        LayerPlan("index.0", "sparse_indexer", 6),
        LayerPlan("target.1", "target_ckv", 6),
    )
    descriptors = build_destination_descriptors(
        plans,
        {
            "target.0": FakeTensor(pointer=0x1000),
            "index.0": FakeTensor(pointer=0x2000),
            "target.1": FakeTensor(pointer=0x3000),
        },
    )

    assert [
        (
            item.destination_base,
            item.destination_rows,
            item.destination_row_stride_bytes,
            item.bytes_per_token,
            item.record_kind,
            item.source_layer_ordinal,
        )
        for item in descriptors
    ] == [
        (0x1000, 8, 6, 6, 0, 0),
        (0x2000, 8, 6, 6, 1, 0),
        (0x3000, 8, 6, 6, 0, 1),
    ]


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (FakeTensor(pointer=0x1000, device_type="cpu"), "CUDA tensor"),
        (FakeTensor(pointer=0x1000, contiguous=False), "contiguous"),
        (
            FakeTensor(pointer=0x1000, shape=(2, 4, 4), strides=(16, 4, 1)),
            "bytes-per-token",
        ),
        (FakeTensor(pointer=0), "data pointer"),
    ],
)
def test_destination_descriptor_rejects_unsafe_tensor_contracts(tensor, message):
    with pytest.raises(NativePlacementContractError, match=message):
        build_destination_descriptors(
            (LayerPlan("target", "target_ckv", 6),),
            {"target": tensor},
        )


def test_destination_descriptor_requires_exact_registered_tensor_set():
    plans = (LayerPlan("target", "target_ckv", 6),)
    with pytest.raises(NativePlacementContractError, match="registered tensor set"):
        build_destination_descriptors(
            plans,
            {
                "target": FakeTensor(pointer=0x1000),
                "unplanned": FakeTensor(pointer=0x2000),
            },
        )


def test_request_stays_parked_until_native_finish_succeeds(tmp_path):
    adapter, mock = _configured_adapter(tmp_path)

    restore = adapter.begin_parked_restore("request-7", (1, 3, 5))
    assert restore.state is RestoreState.PARKED
    assert restore.can_resume is False
    assert restore.needs_recompute is False
    arena = restore.acquire_arena(0)
    assert arena.host_address == 0xA000
    assert arena.capacity_bytes == 4096

    stats = restore.finish()

    assert restore.state is RestoreState.FINISHED
    assert restore.can_resume is True
    assert restore.needs_recompute is False
    assert stats.slot_uploads == 0
    assert len(mock.spark_cache_placement_finish_restore.calls) == 1
    adapter.close()


def test_native_finish_failure_aborts_and_never_releases_parked_request(tmp_path):
    mock = MockLibrary()
    mock.spark_cache_placement_finish_restore.implementation = lambda *_args: 5
    adapter, loaded = _configured_adapter(tmp_path, mock)
    restore = adapter.begin_parked_restore("request-8", (2, 4))

    with pytest.raises(NativePlacementCallError, match="finish restore"):
        restore.finish()

    assert restore.state is RestoreState.ABORTED
    assert restore.can_resume is False
    assert restore.needs_recompute is True
    assert len(loaded.spark_cache_placement_abort_restore.calls) == 1
    adapter.close()


def test_context_exit_aborts_unfinished_restore_and_allows_next_request(tmp_path):
    adapter, loaded = _configured_adapter(tmp_path)

    with adapter.begin_parked_restore("request-9", (1, 2)) as restore:
        assert restore.state is RestoreState.PARKED

    assert restore.state is RestoreState.ABORTED
    assert restore.needs_recompute is True
    assert len(loaded.spark_cache_placement_abort_restore.calls) == 1

    second = adapter.begin_parked_restore("request-10", (3, 4))
    second.abort()
    adapter.close()


def test_only_one_parked_restore_may_own_native_placement(tmp_path):
    adapter, _mock = _configured_adapter(tmp_path)
    restore = adapter.begin_parked_restore("request-11", (1,))

    with pytest.raises(NativePlacementContractError, match="already active"):
        adapter.begin_parked_restore("request-12", (2,))

    restore.abort()
    adapter.close()
