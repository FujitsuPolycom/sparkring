"""``SPARK_TP4_ENABLED``: the switch that turns every four-rank startup hook off.

A launcher that serves on SIRCL's ring sessions sets ``SPARK_TP4_ENABLED=0``
so that no four-rank hook installs, whatever the profile's own four-rank
variables say. Unset or ``1`` leaves each hook to its own variable; any
other value stops startup.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SITECUSTOMIZE = Path(__file__).with_name("sitecustomize.py")

FOUR_RANK_HOOKS = (
    ("VLLM_SPARK_TP4_MODE", "custom", "spark_tp4_backend"),
    ("SPARK_TP4_HEALTH_GATE", "1", "spark_tp4_health_gate"),
    ("VLLM_SPARK_TP4_VOCAB_MODE", "custom", "spark_tp4_vocab_allgather_backend"),
    ("SPARK_TP4_DCP_COLLECTIVE_AUDIT", "1", "spark_dcp_collective_audit"),
)
TIMING_HOOK = ("SPARK_CUDAGRAPH_REPLAY_TIMING", "1", "spark_cudagraph_replay_timing")
FLAGS = tuple(flag for flag, _value, _module in (*FOUR_RANK_HOOKS, TIMING_HOOK)) + ("SPARK_TP4_ENABLED",)


def _counting_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.calls = 0  # type: ignore[attr-defined]

    def install() -> bool:
        module.calls += 1  # type: ignore[attr-defined]
        return True

    module.install = install  # type: ignore[attr-defined]
    return module


def _enable_every_hook(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    modules = {}
    for flag, value, module_name in (*FOUR_RANK_HOOKS, TIMING_HOOK):
        monkeypatch.setenv(flag, value)
        modules[module_name] = _counting_module(module_name)
        monkeypatch.setitem(sys.modules, module_name, modules[module_name])
    return modules


def test_zero_turns_every_four_rank_hook_off(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = _enable_every_hook(monkeypatch)
    monkeypatch.setenv("SPARK_TP4_ENABLED", "0")

    runpy.run_path(SITECUSTOMIZE, run_name="test_four_rank_switch_off")

    assert all(modules[name].calls == 0 for _flag, _value, name in FOUR_RANK_HOOKS)  # type: ignore[attr-defined]
    # Replay timing is not a four-rank hook; its own variable still decides.
    assert modules[TIMING_HOOK[2]].calls == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", [None, "", "1"])
def test_unset_or_one_leaves_each_hook_to_its_variable(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    modules = _enable_every_hook(monkeypatch)
    if value is not None:
        monkeypatch.setenv("SPARK_TP4_ENABLED", value)

    runpy.run_path(SITECUSTOMIZE, run_name="test_four_rank_switch_on")

    assert all(module.calls == 1 for module in modules.values())  # type: ignore[attr-defined]


def test_another_value_stops_startup(tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_bytes(SITECUSTOMIZE.read_bytes())
    environment = {k: v for k, v in os.environ.items() if k not in FLAGS}
    environment.update(PYTHONPATH=str(tmp_path), SPARK_TP4_ENABLED="off")
    result = subprocess.run(
        [sys.executable, "-c", "print('STARTUP_CONTINUED')"],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 78, result.stderr
    assert "STARTUP_CONTINUED" not in result.stdout
    assert "SPARK_TP4_ENABLED must be 0 or 1" in result.stderr


def test_zero_keeps_broken_four_rank_modules_unimported_at_startup(tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_bytes(SITECUSTOMIZE.read_bytes())
    for _flag, _value, module_name in FOUR_RANK_HOOKS:
        (tmp_path / f"{module_name}.py").write_text(
            "raise ImportError('a disabled four-rank hook must not be imported')\n", encoding="utf-8",
        )
    environment = {k: v for k, v in os.environ.items() if k not in FLAGS}
    environment.update(PYTHONPATH=str(tmp_path), SPARK_TP4_ENABLED="0",
                       **{flag: value for flag, value, _module in FOUR_RANK_HOOKS})
    result = subprocess.run(
        [sys.executable, "-c", "print('STARTUP_CONTINUED')"],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "STARTUP_CONTINUED"
