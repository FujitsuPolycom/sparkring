from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _module():
    spec = importlib.util.spec_from_file_location(
        "gb10_parser_replay", HERE / "parser_replay.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_names_nine_semantically_rebased_parser_files() -> None:
    contract = json.loads((HERE / "runtime-contract.json").read_text(encoding="utf-8"))
    parser_files = [
        value
        for value in contract["runtime_patch"]["files"]
        if value["path"].startswith("vllm/parser/")
    ]
    assert len(parser_files) == 9
    streaming = next(
        value
        for value in parser_files
        if value["path"].endswith("streaming_parser_engine.py")
    )
    assert streaming["preimage_sha256"] == (
        "2eace718fc728b46676cd5d01eee2c893aec396a4551f05c529e83a120d07715"
    )
    assert streaming["result_sha256"] == (
        "da5e9db173979dfa9bea0c3a97c26997f9dda31b5b1c40d6d6cc8a622fc467f4"
    )


def test_replay_covers_safe_positive_and_negative_recovery() -> None:
    module = _module()
    cases = module.replay_cases()
    assert set(cases) == {
        "valid_wrapper_required",
        "missing_wrapper_required",
        "corrupted_wrapper_required",
        "reasoning_boundary_required",
        "valid_wrapper_auto",
        "undeclared_tool",
        "tool_choice_none",
        "truncated_invoke",
    }
    assert cases["missing_wrapper_required"]["expected"] == "tool_call"
    assert cases["missing_wrapper_required"]["tool_choice"] == "required"
    assert cases["corrupted_wrapper_required"]["expected"] == "tool_call"
    assert cases["reasoning_boundary_required"]["thinking"] is True
    assert cases["valid_wrapper_auto"]["tool_choice"] == "auto"
    assert cases["undeclared_tool"]["expected"] == "content"
    assert cases["tool_choice_none"]["tool_choice"] == "none"
    assert cases["truncated_invoke"]["expected"] == "content"


def test_replay_uses_noyb_emit_result_contract() -> None:
    module = _module()
    text = module._invoke()
    assert 'invoke name="emit_result"' in text
    assert 'parameter name="value" string="false">9001' in text
