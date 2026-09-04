from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import spark_tp4_capability as capability


def _record(rank: int) -> dict:
    return {
        "rank": rank,
        "adapter_abi": capability.ADAPTER_ABI,
        "native_abi_version": capability.NATIVE_ABI_VERSION,
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


def test_vote_rejects_native_abi_mismatch() -> None:
    records = [_record(rank) for rank in range(4)]
    records[1]["native_abi_version"] = capability.NATIVE_ABI_VERSION + 1

    with pytest.raises(RuntimeError, match="rank 1: native_abi_version disagrees"):
        capability.validate_capabilities(records)


def test_shared_record_covers_ports_timeouts_and_admission(monkeypatch) -> None:
    environment = {
        "SPARK_TP4_CONTROL_PORT0": "12000",
        "SPARK_TP4_GRAPH_CONTROL_PORT0": "12010",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1": "12021",
        "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "17",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS": "23",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
    }
    errors = []
    with monkeypatch.context() as context:
        context.setattr(os, "environ", environment)
        shared = capability._shared_capability(errors)

    assert errors == []
    assert shared["control_ports"]["eager"][0] == 12000
    assert shared["control_ports"]["graph"][0] == 12010
    assert shared["control_ports"]["bidirectional_secondary"][1] == 12021
    assert shared["timeouts"] == {
        "control_connect_seconds": 17,
        "bidirectional_seconds": 23,
    }
    assert shared["admission"]["collective_max_query_rows"] == 40
    assert shared["operation_slots"] == 2
    assert shared["rail_count"] == 2


def test_local_record_reports_native_cuda_and_rdma_capability(monkeypatch) -> None:
    class NativeAbi:
        def __call__(self):
            return capability.NATIVE_ABI_VERSION

    symbols = {
        name: NativeAbi() if name == "spark_tp4_get_abi_version" else object()
        for name in capability.REQUIRED_SYMBOLS
    }
    monkeypatch.setenv("SPARK_TP4_LIBRARY", "/fixture/native.so")
    monkeypatch.setenv("SPARKRING_SIRCL_MANIFEST_PATH", "/fixture/manifest.json")
    monkeypatch.setenv("SPARK_TP4_DEVICE0", "rdma0")
    monkeypatch.setenv("SPARK_TP4_DEVICE1", "rdma1")
    monkeypatch.setenv("SPARK_TP4_GID0", "3")
    monkeypatch.setenv("SPARK_TP4_GID1", "3")
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(capability, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(capability.ctypes, "CDLL", lambda _path: SimpleNamespace(**symbols))
    monkeypatch.setattr(capability, "_device_gid_available", lambda _device, _gid: True)
    monkeypatch.setattr(
        capability,
        "_cuda_capability",
        lambda errors: {"available": True, "device_count": 1},
    )

    record = capability.local_capability(0)

    assert record["errors"] == ()
    assert record["native_abi_version"] == capability.NATIVE_ABI_VERSION
    assert record["local"]["cuda"] == {"available": True, "device_count": 1}
    assert all(item["available"] for item in record["local"]["rdma"])


def test_local_record_converts_probe_exceptions_to_rank_errors(monkeypatch) -> None:
    monkeypatch.setenv("SPARK_TP4_LIBRARY", "/fixture/native.so")
    monkeypatch.setenv("SPARKRING_SIRCL_MANIFEST_PATH", "/fixture/manifest.json")
    monkeypatch.setenv("SPARK_TP4_DEVICE0", "rdma0")
    monkeypatch.setenv("SPARK_TP4_DEVICE1", "rdma1")
    monkeypatch.setenv("SPARK_TP4_GID0", "3")
    monkeypatch.setenv("SPARK_TP4_GID1", "3")
    monkeypatch.setattr(Path, "is_file", lambda _path: True)

    def unreadable(_path):
        raise OSError("fixture unreadable")

    def unavailable(_device, _gid):
        raise OSError("fixture sysfs failure")

    monkeypatch.setattr(capability, "_sha256", unreadable)
    monkeypatch.setattr(capability, "_device_gid_available", unavailable)
    monkeypatch.setattr(
        capability,
        "_cuda_capability",
        lambda errors: {"available": True, "device_count": 1},
    )

    record = capability.local_capability(2)

    assert record["rank"] == 2
    assert any("native library cannot be read" in error for error in record["errors"])
    assert any("overlay manifest cannot be read" in error for error in record["errors"])
    assert sum("RDMA device/GID probe failed" in error for error in record["errors"]) == 2


def test_gid_probe_rejects_zero_gid_and_accepts_configured_address(tmp_path) -> None:
    gid_path = tmp_path / "rdma0" / "ports" / "1" / "gids" / "3"
    gid_path.parent.mkdir(parents=True)
    gid_path.write_text("0000:0000:0000:0000:0000:0000:0000:0000\n")

    assert not capability._device_gid_available(
        "rdma0", "3", sysfs_root=tmp_path
    )

    gid_path.write_text("fe80:0000:0000:0000:0000:0000:0000:0001\n")

    assert capability._device_gid_available("rdma0", "3", sysfs_root=tmp_path)


def test_successful_vote_is_cached_on_communicator(monkeypatch, capsys) -> None:
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
    assert capsys.readouterr().out == (
        "SIRCL capability vote accepted: physical_ranks=4\n"
    )
