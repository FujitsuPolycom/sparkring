from __future__ import annotations

from types import SimpleNamespace

import pytest

import spark_tp4_capability as capability


def _record(rank: int) -> dict:
    return {
        "rank": rank,
        "adapter_abi": capability.ADAPTER_ABI,
        "native_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "shared": {"mode": "custom", "rails": "dual"},
        "errors": (),
    }


def test_vote_rejects_rank_specific_capability_failure() -> None:
    records = [_record(rank) for rank in range(4)]
    records[2]["errors"] = ("native library is missing",)

    with pytest.raises(RuntimeError, match="rank 2: native library is missing"):
        capability.validate_capabilities(records)


def test_vote_rejects_shared_protocol_mismatch() -> None:
    records = [_record(rank) for rank in range(4)]
    records[3]["shared"] = {"mode": "custom", "rails": "single"}

    with pytest.raises(RuntimeError, match="rank 3: shared disagrees"):
        capability.validate_capabilities(records)


def test_successful_vote_is_cached_on_communicator(monkeypatch) -> None:
    communicator = SimpleNamespace(rank_in_group=0, world_size=4)
    calls = []
    monkeypatch.setattr(capability, "local_capability", lambda rank: _record(rank))

    def exchange(_communicator, local):
        calls.append(local)
        return [_record(rank) for rank in range(4)]

    capability.ensure_capability_vote(communicator, exchange=exchange)
    capability.ensure_capability_vote(communicator, exchange=exchange)

    assert len(calls) == 1
    assert communicator._sparkring_sircl_capability_voted is True
