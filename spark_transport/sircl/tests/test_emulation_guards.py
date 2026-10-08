"""The GPU emulation's sizing of the one shared GPU, offline: hardware queues and resident grids."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparkring_sircl.testing import gpu_emulation as ge  # noqa: E402


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("SIRCL_LARGE_BLOCKS", raising=False)
    return monkeypatch


def test_a_multi_rank_group_asks_for_32_hardware_queues_before_the_context_exists(clean):
    clean.setattr(torch.cuda, "is_initialized", lambda: False)
    assert ge.share_hardware_queues(1) is None and "CUDA_DEVICE_MAX_CONNECTIONS" not in ge.os.environ
    assert ge.share_hardware_queues(8) == "CUDA_DEVICE_MAX_CONNECTIONS=32 (set by the emulation)"
    assert ge.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "32"
    assert ge.share_hardware_queues(8) is None            # already 32
    clean.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "16")
    assert ge.share_hardware_queues(2) == "CUDA_DEVICE_MAX_CONNECTIONS=16 (set by the caller, below 32)"
    assert ge.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "16"


def test_a_context_created_first_is_reported_not_changed(clean):
    clean.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.warns(RuntimeWarning, match="the ranks' streams may share hardware queues"):
        text = ge.share_hardware_queues(8)
    assert "CUDA_DEVICE_MAX_CONNECTIONS unset (8)" in text and "CUDA_DEVICE_MAX_CONNECTIONS" not in ge.os.environ
    clean.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    assert ge.share_hardware_queues(8) is None


def test_every_ranks_grid_stays_resident_at_one_block_per_multiprocessor(clean):
    # An RTX 5090 has 170 multiprocessors: eight ranks get 16 blocks each, four and fewer keep the default 32.
    assert [ge.resident_grid_cap(world, 170, 32) for world in (2, 3, 4, 6, 8, 16)] == [None, None, None, 16, 16, 8]
    assert ge.resident_grid_cap(8, 48, 32) == 4
    clean.setenv("SIRCL_LARGE_BLOCKS", "32")
    assert ge.resident_grid_cap(8, 170, 32) is None       # the caller's cap stands
