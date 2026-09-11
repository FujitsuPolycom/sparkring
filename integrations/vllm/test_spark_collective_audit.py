from __future__ import annotations

import pytest

import spark_collective_audit as audit


@pytest.fixture(autouse=True)
def reset(monkeypatch: pytest.MonkeyPatch):
    audit._reset_for_tests()
    monkeypatch.delenv("SPARK_TP4_GRAPH_STATUS_PATH", raising=False)
    yield
    audit._reset_for_tests()


def test_disabled_audit_has_zero_overhead_state() -> None:
    audit.record_stock("all_reduce", capturing=False, reason="ineligible")
    assert audit.stock_collective_snapshot() == {
        "capture": {},
        "eager": {},
        "capture_total": 0,
        "eager_total": 0,
    }


def test_records_bounded_phase_family_reason_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json"
    )
    audit.record_stock(
        "all_reduce", capturing=True, reason="ineligible_signature"
    )
    audit.record_stock(
        "all_reduce", capturing=True, reason="ineligible_signature"
    )
    audit.record_stock(
        "pynccl_all_gather", capturing=False, reason="capture_unsupported"
    )

    assert audit.stock_collective_snapshot() == {
        "capture": {"all_reduce:ineligible_signature": 2},
        "eager": {"pynccl_all_gather:capture_unsupported": 1},
        "capture_total": 2,
        "eager_total": 1,
    }


def test_records_structured_signature_evidence_without_changing_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json"
    )
    signature = audit.StockCollectiveSignature(
        shape=(160, 6144),
        dtype="torch.bfloat16",
        is_cuda=True,
        contiguous=True,
        world_size=4,
        unique_name="tp:0",
    )

    for _ in range(2):
        audit.record_stock(
            "all_reduce",
            capturing=False,
            reason="ineligible_signature",
            signature=signature,
        )

    snapshot = audit.stock_collective_snapshot()
    assert snapshot["eager"] == {"all_reduce:ineligible_signature": 2}
    assert snapshot["eager_total"] == 2
    assert snapshot["signatures"] == {
        "capture": [],
        "eager": [
            {
                "family": "all_reduce",
                "reason": "ineligible_signature",
                "count": 2,
                "shape": [160, 6144],
                "dtype": "torch.bfloat16",
                "is_cuda": True,
                "contiguous": True,
                "world_size": 4,
                "unique_name": "tp:0",
            }
        ],
    }
    assert snapshot["signature_limit_per_phase"] == 512
    assert snapshot["signature_dropped_calls"] == {
        "capture": 0,
        "eager": 0,
    }
    assert snapshot["signature_dropped_calls_by_family_reason"] == {
        "capture": {},
        "eager": {},
    }


def test_signature_evidence_has_fixed_cardinality_and_counts_dropped_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json"
    )
    monkeypatch.setattr(audit, "_MAX_SIGNATURES_PER_PHASE", 2)

    for rows in (7, 8, 9, 9):
        audit.record_stock(
            "all_reduce",
            capturing=True,
            reason="ineligible_signature",
            signature=audit.StockCollectiveSignature(
                shape=(rows, 6144),
                dtype="torch.bfloat16",
                is_cuda=True,
                contiguous=True,
                world_size=4,
                unique_name="tp:0",
            ),
        )

    snapshot = audit.stock_collective_snapshot()
    assert snapshot["capture_total"] == 4
    assert len(snapshot["signatures"]["capture"]) == 2
    assert [
        item["shape"] for item in snapshot["signatures"]["capture"]
    ] == [[7, 6144], [8, 6144]]
    assert snapshot["signature_dropped_calls"] == {
        "capture": 2,
        "eager": 0,
    }
    assert snapshot["signature_dropped_calls_by_family_reason"] == {
        "capture": {"all_reduce:ineligible_signature": 2},
        "eager": {},
    }


def test_default_capacity_preserves_512_distinct_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json"
    )

    for rows in range(1, 513):
        audit.record_stock(
            "dcp_query_all_gather",
            capturing=True,
            reason="ineligible_signature",
            signature=audit.StockCollectiveSignature(
                shape=(rows, 16, 576),
                dtype="torch.bfloat16",
                is_cuda=True,
                contiguous=True,
                world_size=4,
                unique_name="dcp:0",
            ),
        )

    snapshot = audit.stock_collective_snapshot()
    assert len(snapshot["signatures"]["capture"]) == 512
    assert snapshot["signature_limit_per_phase"] == 512
    assert snapshot["signature_dropped_calls"] == {
        "capture": 0,
        "eager": 0,
    }
    assert snapshot["signature_dropped_calls_by_family_reason"] == {
        "capture": {},
        "eager": {},
    }


def test_signature_overflow_is_attributed_without_losing_aggregate_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json")
    monkeypatch.setattr(audit, "_MAX_SIGNATURES_PER_PHASE", 2)

    calls = (
        ("all_reduce", "ineligible_signature", 1),
        ("dcp_owner_topk_all_gather", "ineligible_signature", 2),
        ("pynccl_all_gather", "ineligible_signature", 3),
        ("pynccl_all_gather", "ineligible_signature", 3),
        ("all_reduce", "unsupported_dtype", 4),
    )
    for family, reason, rows in calls:
        audit.record_stock(
            family,
            capturing=False,
            reason=reason,
            signature=audit.StockCollectiveSignature(
                shape=(rows,),
                dtype="torch.uint8",
                is_cuda=True,
                contiguous=True,
                world_size=4,
                unique_name="",
            ),
        )

    snapshot = audit.stock_collective_snapshot()
    assert snapshot["eager"] == {
        "all_reduce:ineligible_signature": 1,
        "all_reduce:unsupported_dtype": 1,
        "dcp_owner_topk_all_gather:ineligible_signature": 1,
        "pynccl_all_gather:ineligible_signature": 2,
    }
    assert snapshot["eager_total"] == 5
    assert snapshot["signature_dropped_calls"] == {
        "capture": 0,
        "eager": 3,
    }
    assert snapshot["signature_dropped_calls_by_family_reason"] == {
        "capture": {},
        "eager": {
            "all_reduce:unsupported_dtype": 1,
            "pynccl_all_gather:ineligible_signature": 2,
        },
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 512),
        ("1", 1),
        ("1024", 1024),
        ("2048", 2048),
    ],
)
def test_signature_limit_configuration(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: int,
) -> None:
    if raw is None:
        monkeypatch.delenv(
            "SPARK_COLLECTIVE_AUDIT_SIGNATURE_LIMIT_PER_PHASE",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "SPARK_COLLECTIVE_AUDIT_SIGNATURE_LIMIT_PER_PHASE",
            raw,
        )

    assert audit._signature_limit_from_env() == expected


@pytest.mark.parametrize("raw", ["", "0", "-1", "2049", "x", "1.5"])
def test_signature_limit_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(
        "SPARK_COLLECTIVE_AUDIT_SIGNATURE_LIMIT_PER_PHASE",
        raw,
    )

    with pytest.raises(RuntimeError, match="must be an integer between"):
        audit._signature_limit_from_env()


@pytest.mark.parametrize(
    ("shape", "dtype", "dim", "expected"),
    [
        ((3, 16, 576), "torch.bfloat16", 1, "dcp_query_all_gather"),
        ((3, 16, 576), "torch.bfloat16", -2, "dcp_query_all_gather"),
        ((3, 64), "torch.float32", 0, "dcp_lse_all_gather"),
        ((3, 64), "torch.float32", -2, "dcp_lse_all_gather"),
        ((3, 64), "torch.float32", 1, "dcp_all_gather"),
        ((3, 16, 576), "torch.bfloat16", 0, "dcp_all_gather"),
        ((3, 64), "torch.bfloat16", 0, "dcp_all_gather"),
        ((3, 32), "torch.float32", 0, "dcp_all_gather"),
    ],
)
def test_classifies_exact_dcp_all_gather_abis(
    shape: tuple[int, ...],
    dtype: str,
    dim: int,
    expected: str,
) -> None:
    signature = audit.StockCollectiveSignature(
        shape=shape,
        dtype=dtype,
        is_cuda=True,
        contiguous=True,
        world_size=4,
        unique_name="dcp:0",
    )

    assert (
        audit.classify_stock_family(
            "group_all_gather",
            signature,
            dim=dim,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_cuda", False),
        ("contiguous", False),
        ("world_size", 2),
        ("unique_name", "tp:0"),
    ],
)
def test_non_exact_dcp_query_signature_remains_generic(
    field: str,
    value: object,
) -> None:
    fields = {
        "shape": (3, 16, 576),
        "dtype": "torch.bfloat16",
        "is_cuda": True,
        "contiguous": True,
        "world_size": 4,
        "unique_name": "dcp:0",
    }
    fields[field] = value
    signature = audit.StockCollectiveSignature(**fields)

    expected = (
        "group_all_gather"
        if field == "unique_name"
        else "dcp_all_gather"
    )
    assert (
        audit.classify_stock_family(
            "group_all_gather",
            signature,
            dim=1,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", tuple(range(9)), "shape rank"),
        ("dtype", "x" * 65, "dtype"),
        ("unique_name", "x" * 129, "unique_name"),
    ],
)
def test_signature_rejects_unbounded_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    fields = {
        "shape": (1, 6144),
        "dtype": "torch.bfloat16",
        "is_cuda": True,
        "contiguous": True,
        "world_size": 4,
        "unique_name": "tp:0",
    }
    fields[field] = value

    with pytest.raises(ValueError, match=message):
        audit.StockCollectiveSignature(**fields)


def test_rejects_empty_unbounded_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SPARK_TP4_GRAPH_STATUS_PATH", "/tmp/status-rank0.json"
    )
    with pytest.raises(ValueError, match="nonempty"):
        audit.record_stock("", capturing=False, reason="x")
    with pytest.raises(ValueError, match="nonempty"):
        audit.record_stock("x", capturing=False, reason="")


def test_armed_audit_starts_status_reporter_from_recording_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An armed audit must produce a status file regardless of model
    width or graph eligibility, so recording a stock collective must
    start the reporter even when no graph session was ever prepared."""
    import spark_graph_status_reporter as reporter
    import spark_tp4_backend as backend

    status_path = tmp_path / "status-rank0.json"
    monkeypatch.setenv("SPARK_TP4_GRAPH_STATUS_PATH", str(status_path))
    started: list[int] = []
    monkeypatch.setattr(
        reporter,
        "ensure_status_reporter",
        lambda *, rank, interval_seconds=0.25: started.append(rank),
    )
    backend._record_stock_path(
        capturing=False, reason="ineligible_signature"
    )
    assert started == [0]
