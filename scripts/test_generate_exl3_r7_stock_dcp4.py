"""Tests for the public stock-DCP4 R7 baseline generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_exl3_r7_candidate as gen  # noqa: E402
import generate_exl3_r7_stock_dcp4 as stock_gen  # noqa: E402
import prepare_exl3_r7_mtp2 as mtp2  # noqa: E402


def _inputs() -> tuple[dict, dict, dict]:
    return (
        json.loads(gen.TEMPLATE_PATH.read_text(encoding="utf-8")),
        json.loads(gen.PINS_PATH.read_text(encoding="utf-8")),
        json.loads(gen.RECIPE_PATH.read_text(encoding="utf-8")),
    )


def test_stock_profile_passes_mtp2_stock_control_validation() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    mtp2.validate_stock_control(stock)


def test_stock_profile_is_mtp_off() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    assert stock["environment"]["VLLM_SPARK_MTP_TOKENS"] == "0"
    args = stock["extra_vllm_args"]
    assert "--speculative-config" not in args


def test_stock_profile_uses_hybrid_transport() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    assert "/opt/sparkring/nccl/libnccl.so.2" in stock["environment"]["LD_PRELOAD"]
    assert stock["environment"]["NCCL_IB_DISABLE"] == "0"
    assert stock["environment"]["NCCL_NET"] == "IB"


def test_stock_profile_preserves_online_k6() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    assert stock["environment"]["ONLINE_QUANT"] == "exl3-b6"
    assert stock["environment"]["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == "6"


def test_stock_profile_uses_fp8_ds_mla_and_dcp4() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    args = stock["extra_vllm_args"]
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8_ds_mla"
    assert args[args.index("--dcp-comm-backend") + 1] == "ag_rs"
    assert args[args.index("--dcp-kv-cache-interleave-size") + 1] == "1"


def test_stock_profile_captures_q24() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    args = stock["extra_vllm_args"]
    compilation = json.loads(args[args.index("--compilation-config") + 1])
    assert 24 in compilation["cudagraph_capture_sizes"]
    assert compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"


def test_stock_profile_has_correct_identity() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    identity = stock["identity"]
    assert identity["model_repository"] == (
        "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
    )
    assert identity["model_revision"] == (
        "9ab9579774cc432df91567a36f6e9e863e0d4c9f"
    )
    assert identity["model_config_sha256"] == (
        "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126"
    )
    assert identity["model_index_sha256"] == (
        "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd"
    )
    assert stock["profile_id"] == "glm52-exl3-r7-3.5bpw"


def test_stock_profile_matches_qualified_diagnostic_contract() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    assert "VLLM_SPARK_R7_NONFINITE_TRACE" not in stock["environment"]
    assert stock["environment"]["SPARK_TP4_DCP_COLLECTIVE_AUDIT"] == "1"
    assert stock["environment"]["SPARK_TP4_GRAPH_STATUS_PATH"] == (
        "/cache/jit/sparkring-r7-dcp4-stock-graph-status.json"
    )
    assert not any(
        volume["container"] == gen.DCP_AUDIT_CONTAINER_PATH
        for volume in stock["extra_volumes"]
    )
    assert (
        f"{gen.DCP_AUDIT_SHA256}  {gen.DCP_AUDIT_CONTAINER_PATH}"
        in stock["attestation_hook"][2]
    )


def test_stock_profile_validation_rejects_mtp_on() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    stock["environment"]["VLLM_SPARK_MTP_TOKENS"] = "2"
    with pytest.raises(stock_gen.StockProfileError, match="MTP-off"):
        stock_gen.validate_stock_profile(stock)


def test_stock_profile_validation_rejects_missing_dcp() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    args = stock["extra_vllm_args"]
    idx = args.index("--dcp-comm-backend")
    del args[idx:idx + 2]
    with pytest.raises(stock_gen.StockProfileError, match="ag_rs"):
        stock_gen.validate_stock_profile(stock)


def test_stock_profile_validation_rejects_eager() -> None:
    template, pins, recipe = _inputs()
    stock = stock_gen.derive_stock_profile(template, pins, recipe)
    stock["extra_vllm_args"].append("--enforce-eager")
    with pytest.raises(stock_gen.StockProfileError, match="CUDA graphs"):
        stock_gen.validate_stock_profile(stock)


def test_cli_emits_stock_profile(tmp_path: Path) -> None:
    output = tmp_path / "stock.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_exl3_r7_stock_dcp4.py"),
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    stock = json.loads(output.read_text())
    mtp2.validate_stock_control(stock)
