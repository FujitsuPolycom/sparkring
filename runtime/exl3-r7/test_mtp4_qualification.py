from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("mtp4_qualification.py")
SPEC = importlib.util.spec_from_file_location("mtp4_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def metrics(
    drafts: int,
    drafted: int,
    accepted: int,
    positions: tuple[int, int, int, int],
) -> str:
    lines = [
        f'vllm:spec_decode_num_drafts_total{{model_name="m",engine="0"}} {drafts}',
        f'vllm:spec_decode_num_draft_tokens_total{{model_name="m",engine="0"}} {drafted}',
        f'vllm:spec_decode_num_accepted_tokens_total{{model_name="m",engine="0"}} {accepted}',
    ]
    lines.extend(
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="m",engine="0",position="{position}"}} {value}'
        for position, value in enumerate(positions)
    )
    return "\n".join(lines)


def test_mtp4_metric_delta_passes_and_reports_all_positions() -> None:
    before = module.parse_spec_metrics(metrics(10, 40, 28, (10, 8, 6, 4)), "m")
    after = module.parse_spec_metrics(
        metrics(30, 120, 84, (30, 24, 18, 12)), "m"
    )
    result = module.validate_mtp4_metrics(module.metric_delta(before, after))
    assert result == {
        "drafts": 20,
        "draft_tokens": 80,
        "accepted_tokens": 56,
        "acceptance_rate": 0.7,
        "accepted_tokens_per_position": [20, 16, 12, 8],
    }


def test_mtp4_metric_delta_rejects_missing_fourth_position() -> None:
    before = module.parse_spec_metrics(metrics(0, 0, 0, (0, 0, 0, 0)), "m")
    after = module.parse_spec_metrics(metrics(10, 40, 21, (8, 7, 6, 0)), "m")
    with pytest.raises(module.QualificationError, match="all four draft positions"):
        module.validate_mtp4_metrics(module.metric_delta(before, after))


@pytest.mark.parametrize(
    ("drafted", "accepted", "positions", "message"),
    (
        (41, 20, (10, 6, 3, 1), "invalid MTP4 totals"),
        (40, 22, (10, 6, 3, 2), "invalid MTP4 position counters"),
        (40, 20, (9, 10, 1, 0), "all four draft positions"),
    ),
)
def test_mtp4_metric_delta_fails_closed(
    drafted: int,
    accepted: int,
    positions: tuple[int, int, int, int],
    message: str,
) -> None:
    before = module.parse_spec_metrics(metrics(0, 0, 0, (0, 0, 0, 0)), "m")
    after = module.parse_spec_metrics(metrics(10, drafted, accepted, positions), "m")
    with pytest.raises(module.QualificationError, match=message):
        module.validate_mtp4_metrics(module.metric_delta(before, after))


def status(
    rank: int,
    published: int,
    *,
    stock_q: int | None = None,
    stock_phase: str = "capture",
    stock_width: int = 6144,
    dropped_capture: int = 0,
    dropped_eager: int = 0,
    all_reduce_captured_nodes: int = 40,
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
                "family": "test",
                "reason": "ineligible_signature",
                "shape": [stock_q, stock_width],
                "dtype": "torch.bfloat16",
                "count": 1,
            }
        )
    capture_counts = {
        "dcp_combine:original": 1,
        "dcp_lse_all_gather:ineligible_signature": 1,
        "dcp_query_all_gather:ineligible_signature": 1,
        "dcp_owner_topk_all_gather:graph_capture_unsupported": 1,
    }
    eager_counts = {
        "dcp_combine:original": published,
        "dcp_lse_all_gather:ineligible_signature": published,
        "dcp_query_all_gather:ineligible_signature": published,
        "dcp_owner_topk_all_gather:ineligible_signature": published,
    }
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
            "dcp": {
                "family_selection": {
                    "mode": "stock",
                    "query_enabled": False,
                    "combine_enabled": False,
                },
                "sessions": {},
            },
            "indexer": {"sessions": {}},
            "stock_collectives": {
                "capture": capture_counts,
                "eager": eager_counts,
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


def transport_args(tmp_path: Path, **after_kwargs: object) -> object:
    before = [
        write_status(tmp_path / f"before-{rank}.json", status(rank, 10))
        for rank in range(4)
    ]
    after = [
        write_status(
            tmp_path / f"after-{rank}.json",
            status(rank, 20, **after_kwargs),
        )
        for rank in range(4)
    ]
    return type("Args", (), {"before_status": before, "after_status": after})()


def test_transport_audit_requires_q1_q40_and_depth_independent_vocab_census(
    tmp_path: Path,
) -> None:
    report = module.run_transport(transport_args(tmp_path))
    assert report["schema"] == "sparkring-r7-mtp4-transport-audit/v1"
    assert report["fixed_mtp_depth"] == 4
    assert report["maximum_query_rows"] == 40
    assert report["ranks"][0]["required_query_rows"] == list(range(1, 41))
    assert report["ranks"][0]["all_reduce"]["minimum_captured_nodes"] == 40
    assert report["ranks"][0]["vocabulary"]["minimum_captured_nodes"] == 16
    assert report["ranks"][0]["dcp_indexer"]["mode"] == "stock"
    assert report["ranks"][0]["dcp_indexer"]["eager_deltas"] == {
        "dcp_combine:original": 10,
        "dcp_lse_all_gather:ineligible_signature": 10,
        "dcp_query_all_gather:ineligible_signature": 10,
        "dcp_owner_topk_all_gather:ineligible_signature": 10,
    }


@pytest.mark.parametrize(
    ("status_kwargs", "family"),
    (
        ({"all_reduce_captured_nodes": 39}, "all_reduce"),
        ({"vocabulary_captured_nodes": 15}, "vocabulary"),
    ),
)
def test_transport_audit_rejects_incomplete_native_census(
    tmp_path: Path,
    status_kwargs: dict[str, int],
    family: str,
) -> None:
    with pytest.raises(module.QualificationError, match=family):
        module.run_transport(transport_args(tmp_path, **status_kwargs))


@pytest.mark.parametrize("stock_phase", ("capture", "eager"))
@pytest.mark.parametrize("stock_width", (6144, 38720))
def test_transport_audit_rejects_stock_q40_for_both_tp_families(
    tmp_path: Path,
    stock_phase: str,
    stock_width: int,
) -> None:
    with pytest.raises(module.QualificationError, match="Q1-Q40"):
        module.run_transport(
            transport_args(
                tmp_path,
                stock_q=40,
                stock_phase=stock_phase,
                stock_width=stock_width,
            )
        )


def test_transport_audit_ignores_stock_q41_outside_declared_contract(
    tmp_path: Path,
) -> None:
    report = module.run_transport(transport_args(tmp_path, stock_q=41))
    assert report["status"] == "pass"


def test_transport_audit_rejects_custom_indexer_session(tmp_path: Path) -> None:
    args = transport_args(tmp_path)
    payload = json.loads(Path(args.after_status[2]).read_text(encoding="utf-8"))
    payload["snapshot"]["indexer"]["sessions"] = {"2": {"captured_nodes": 1}}
    Path(args.after_status[2]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.QualificationError, match="indexer must remain stock"):
        module.run_transport(args)


def test_transport_audit_rejects_nonadvancing_stock_dcp_counter(
    tmp_path: Path,
) -> None:
    args = transport_args(tmp_path)
    payload = json.loads(Path(args.after_status[1]).read_text(encoding="utf-8"))
    payload["snapshot"]["stock_collectives"]["eager"][
        "dcp_query_all_gather:ineligible_signature"
    ] = 10
    Path(args.after_status[1]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.QualificationError, match="did not advance"):
        module.run_transport(args)


@pytest.mark.parametrize("dropped_phase", ("capture", "eager"))
def test_transport_audit_rejects_dropped_signature_calls(
    tmp_path: Path,
    dropped_phase: str,
) -> None:
    with pytest.raises(module.QualificationError, match="dropped calls"):
        module.run_transport(
            transport_args(tmp_path, **{f"dropped_{dropped_phase}": 1})
        )


def test_cli_declares_mtp4_mode_and_requires_baseline() -> None:
    args = module.parse_args(
        [
            "qualify-mtp4",
            "--baseline",
            "mtp0.json",
            "--output",
            "mtp4.json",
        ]
    )
    assert args.mode == "qualify-mtp4"
    assert args.baseline == "mtp0.json"
