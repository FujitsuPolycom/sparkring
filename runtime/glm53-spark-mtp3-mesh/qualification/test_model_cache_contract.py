"""Offline request construction and numeric evidence checks for cache probes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("cache_probe_contract", Path(__file__).with_name("model_cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


@pytest.mark.parametrize("kind,phase", [("semantic", "semantic"), ("persistent", "before-restart")])
def test_generated_prompt_uses_requested_expected_text(kind, phase, tmp_path, monkeypatch, capsys):
    expected = "CUSTOM_CANARY"
    monkeypatch.setattr(sys, "argv", ["model_cache", "--endpoint", "http://unused", "--model", "fixture",
                                    "--kind", kind, "--phase", phase, "--expected-text", expected,
                                    "--output", str(tmp_path / "output")])
    monkeypatch.setattr(cache, "fetch", lambda *args: pytest.fail("Network access forbidden"))
    cache.main()
    result = json.loads(capsys.readouterr().out)
    suffix = f"Respond with exactly {expected} and no other text."
    prompt = suffix if kind == "semantic" else "benchmark " * 8192 + "\n" + suffix
    assert result["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert result["executed"] is False
    assert not (tmp_path / "output").exists()


def test_metric_overflow_is_not_serialized_as_infinity():
    assert cache.cache_metrics("prefix_hit_total 1e999\nexternal_hit_total -1e999\nprefix_valid 12.5\n") == {"prefix_valid": 12.5}


@pytest.mark.parametrize("value", ["-", ".", "1e", "++2"])
def test_malformed_metric_numbers_are_ignored(value):
    assert cache.cache_metrics(f"prefix_hits {value}\n") == {}
