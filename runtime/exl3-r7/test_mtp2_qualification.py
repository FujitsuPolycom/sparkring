from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("mtp2_qualification.py")
SPEC = importlib.util.spec_from_file_location("mtp2_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def metrics(drafts: int, drafted: int, accepted: int, pos0: int, pos1: int) -> str:
    return "\n".join(
        (
            f'vllm:spec_decode_num_drafts_total{{model_name="m",engine="0"}} {drafts}',
            f'vllm:spec_decode_num_draft_tokens_total{{model_name="m",engine="0"}} {drafted}',
            f'vllm:spec_decode_num_accepted_tokens_total{{model_name="m",engine="0"}} {accepted}',
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="m",engine="0",position="0"}} {pos0}',
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="m",engine="0",position="1"}} {pos1}',
        )
    )


def test_mtp2_metric_delta_passes() -> None:
    before = module.parse_spec_metrics(metrics(10, 20, 15, 9, 6), "m")
    after = module.parse_spec_metrics(metrics(30, 60, 47, 28, 19), "m")
    result = module.validate_mtp2_metrics(module.metric_delta(before, after))
    assert result["drafts"] == 20
    assert result["draft_tokens"] == 40
    assert result["accepted_tokens"] == 32
    assert result["accepted_tokens_per_position"] == [19, 13]


def test_mtp2_metric_delta_rejects_missing_second_position() -> None:
    before = module.parse_spec_metrics(metrics(0, 0, 0, 0, 0), "m")
    after = module.parse_spec_metrics(metrics(10, 20, 8, 8, 0), "m")
    with pytest.raises(module.QualificationError, match="both draft positions"):
        module.validate_mtp2_metrics(module.metric_delta(before, after))


def test_metric_parser_filters_model_and_engine() -> None:
    text = metrics(10, 20, 15, 9, 6) + "\n" + metrics(30, 60, 47, 28, 19).replace(
        'model_name="m"', 'model_name="other"'
    )
    result = module.parse_spec_metrics(text, "m")
    assert result["totals"][module.SPEC_COUNTERS[0]] == 10
    assert result["positions"] == {0: 9, 1: 6}


def status(
    rank: int,
    published: int,
    *,
    stock_q: int | None = None,
    stock_phase: str = "capture",
    stock_family: str = "all_reduce",
    stock_width: int = 6144,
    dropped_capture: int = 0,
    dropped_eager: int = 0,
    all_reduce_captured_nodes: int = 24,
    vocabulary_captured_nodes: int = 16,
) -> dict:
    session = {
        "capture_configured": True,
        "captured_nodes": all_reduce_captured_nodes,
        "published_sequence": published,
        "consumed_sequence": published,
        "completed_sequence": published,
        "overflow_sequence": 0,
        "polling_enabled": True,
        "host_native_atomics": True,
        "submit_affinity_verified": True,
        "progress_affinity_verified": True,
        "fatal": False,
    }
    signatures = {"capture": [], "eager": []}
    if stock_q is not None:
        signatures[stock_phase].append(
            {
                "family": stock_family,
                "reason": "ineligible_signature",
                "shape": [stock_q, stock_width],
                "dtype": "torch.bfloat16",
                "count": 1,
            }
        )
    return {
        "schema_version": 3,
        "rank": rank,
        "pid": 1000 + rank,
        "snapshot_start_unix_ns": published * 10,
        "snapshot_end_unix_ns": published * 10 + 1,
        "snapshot": {
            "all_reduce": {"sessions": {str(rank): session}},
            "vocabulary": {
                "sessions": {
                    str(rank): {
                        **session,
                        "captured_nodes": vocabulary_captured_nodes,
                    }
                }
            },
            "stock_collectives": {
                "signature_dropped_calls": {
                    "capture": dropped_capture,
                    "eager": dropped_eager,
                },
                "signatures": signatures,
            },
        },
    }


def write_status(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_transport_audit_passes(tmp_path: Path) -> None:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(tmp_path / f"after-{rank}.json", status(rank, 20))
        for rank in range(4)
    ]
    args = type("Args", (), {"before_status": before, "after_status": after})()
    report = module.run_transport(args)
    assert report["status"] == "pass"
    assert report["maximum_query_rows"] == 24
    assert report["ranks"][0]["all_reduce"]["minimum_captured_nodes"] == 24
    assert report["ranks"][0]["vocabulary"]["minimum_captured_nodes"] == 16


@pytest.mark.parametrize(
    ("status_kwargs", "family"),
    (
        ({"all_reduce_captured_nodes": 23}, "all_reduce"),
        ({"vocabulary_captured_nodes": 15}, "vocabulary"),
    ),
)
def test_transport_audit_rejects_incomplete_family_census(
    tmp_path: Path,
    status_kwargs: dict[str, int],
    family: str,
) -> None:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(
            tmp_path / f"after-{rank}.json",
            status(rank, 20, **status_kwargs),
        )
        for rank in range(4)
    ]
    args = type("Args", (), {"before_status": before, "after_status": after})()
    with pytest.raises(module.QualificationError, match=family):
        module.run_transport(args)


def test_transport_audit_rejects_required_stock_capture(tmp_path: Path) -> None:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(
            tmp_path / f"after-{rank}.json",
            status(rank, 20, stock_q=24 if rank == 2 else None),
        )
        for rank in range(4)
    ]
    args = type("Args", (), {"before_status": before, "after_status": after})()
    with pytest.raises(module.QualificationError, match="Q1-Q24"):
        module.run_transport(args)


@pytest.mark.parametrize(
    "status_kwargs",
    (
        {
            "stock_q": 24,
            "stock_phase": "eager",
            "stock_family": "misclassified",
            "stock_width": 6144,
        },
        {
            "stock_q": 24,
            "stock_phase": "eager",
            "stock_family": "misclassified",
            "stock_width": 38720,
        },
    ),
)
def test_transport_audit_rejects_required_stock_shape_in_eager_phase(
    tmp_path: Path,
    status_kwargs: dict[str, object],
) -> None:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(
            tmp_path / f"after-{rank}.json",
            status(rank, 20, **status_kwargs),
        )
        for rank in range(4)
    ]
    args = type("Args", (), {"before_status": before, "after_status": after})()
    with pytest.raises(module.QualificationError, match="Q1-Q24"):
        module.run_transport(args)


@pytest.mark.parametrize("dropped_phase", ("capture", "eager"))
def test_transport_audit_rejects_dropped_signature_calls(
    tmp_path: Path,
    dropped_phase: str,
) -> None:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(
            tmp_path / f"after-{rank}.json",
            status(rank, 20, **{f"dropped_{dropped_phase}": 1}),
        )
        for rank in range(4)
    ]
    args = type("Args", (), {"before_status": before, "after_status": after})()
    with pytest.raises(module.QualificationError, match="dropped calls"):
        module.run_transport(args)
