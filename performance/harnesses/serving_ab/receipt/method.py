"""Named measurement methods, and the check that a benchmark result was measured by one.

A method fixes what a decode measurement sends: the benchmark tool and version, sampling temperature,
output-token limit, context lengths, stream counts, cell duration, prefill mode and whether the coding-peak
probe runs. Two arms compare only when their results were measured the same way. llm-inference-bench
(``llm_decode_bench.py``) records these settings in its result JSON (``metadata``: ``version``,
``temperature``, ``max_tokens``, ``context_lengths``, ``concurrency_levels``, ``duration_per_test``,
``prefill_mode``, ``decode_warmup_seconds``, ``ignore_eos``; the top-level ``coding_peak`` object, with its
temperature, when that probe ran), so :func:`recorded` reads the method from the result itself rather than
from how the run was started.

:data:`METHODS`:

``owner-temp1-matrix``
    llm-inference-bench 0.7.6 at temperature 1, contexts 8k, 16k, 32k, 64k and 128k, 1, 2, 4 and 8
    streams, 20 s cells after a 5 s decode warm-up, at most 2,048 output tokens, the integrated prefill scout
    and the coding-peak probe at temperature 1: the matrix the deployment owner publishes (its result files
    record these values).
``serving-ab-temp0``
    The serving A/B runner's decode method (``measure.py``): temperature 0, at most 1,024 output tokens,
    30 s cells after a 10 s warm-up, prefill skipped (time to first token is measured separately); its
    contexts and stream counts come from the metric set and are not fixed here.

A field set to ``None`` in a method is not checked; :func:`check` lists every other field whose recorded
value differs. A result whose ``temperature`` is ``None`` used the server's default temperature.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

FIELDS = ("tool", "version", "temperature", "max_tokens", "context_lengths", "concurrency_levels",
          "duration_per_test", "prefill_mode", "decode_warmup_seconds", "coding_peak", "coding_peak_temperature",
          "ignore_eos")

METHODS: dict[str, dict] = {
    "owner-temp1-matrix": {
        "description": "llm-inference-bench 0.7.6, temperature 1, contexts 8k-128k, 1/2/4/8 streams, 20 s cells "
                       "after a 5 s warm-up, at most 2,048 tokens, integrated prefill scout, coding peak at temperature 1",
        "tool": "llm-inference-bench", "version": "0.7.6", "temperature": 1.0, "max_tokens": 2048,
        "context_lengths": [8192, 16384, 32768, 65536, 131072], "concurrency_levels": [1, 2, 4, 8],
        "duration_per_test": 20.0, "prefill_mode": "integrated_decode_scout", "decode_warmup_seconds": 5.0,
        "coding_peak": True, "coding_peak_temperature": 1.0, "ignore_eos": None,
        "bench_args": ["--contexts", "8k,16k,32k,64k,128k", "--concurrency", "1,2,4,8", "--duration", "20",
                       "--max-tokens", "2048", "--temperature", "1", "--decode-warmup-seconds", "5",
                       "--cell-warmup-timeout-seconds", "900", "--coding-peak", "--coding-peak-temperature", "1"],
    },
    "serving-ab-temp0": {
        "description": "serving A/B runner decode: temperature 0, at most 1,024 tokens, 30 s cells after 10 s, "
                       "prefill skipped; contexts and streams from the metric set",
        "tool": "llm-inference-bench", "version": None, "temperature": 0.0, "max_tokens": 1024,
        "context_lengths": None, "concurrency_levels": None, "duration_per_test": 30.0, "prefill_mode": "skipped",
        "decode_warmup_seconds": 10.0, "coding_peak": False, "coding_peak_temperature": None, "ignore_eos": None,
        "bench_args": ["--skip-prefill", "--token-targeting", "exact", "--temperature", "0", "--max-tokens", "1024",
                       "--decode-warmup-seconds", "10", "--cell-warmup-timeout-seconds", "900"],
    },
}


def recorded(result: Mapping) -> dict:
    """The method fields a llm-inference-bench result records (``None`` where it records nothing)."""
    meta = result.get("metadata") or {}
    peak = result.get("coding_peak")
    out = {
        "tool": "llm-inference-bench" if "version" in meta else None,
        "version": meta.get("version"),
        "temperature": None if meta.get("temperature") is None else float(meta["temperature"]),
        "max_tokens": meta.get("max_tokens"),
        "context_lengths": list(meta.get("context_lengths") or []) or None,
        "concurrency_levels": list(meta.get("concurrency_levels") or []) or None,
        "duration_per_test": None if meta.get("duration_per_test") is None else float(meta["duration_per_test"]),
        "prefill_mode": meta.get("prefill_mode"),
        "decode_warmup_seconds": meta.get("decode_warmup_seconds"),
        "coding_peak": bool(isinstance(peak, Mapping) and (peak.get("runs_ok") or peak.get("samples"))),
        "coding_peak_temperature": (None if not isinstance(peak, Mapping) or peak.get("temperature") is None
                                    else float(peak["temperature"])),
        "ignore_eos": meta.get("ignore_eos"),
    }
    return out


def check(result: Mapping, name: str) -> list[str]:
    """Every field of method ``name`` that ``result`` records differently (empty: measured by it)."""
    expected = METHODS[name]
    actual = recorded(result)
    problems = []
    for field in FIELDS:
        want = expected.get(field)
        if want is None:
            continue
        have = actual.get(field)
        if isinstance(want, float) and isinstance(have, (int, float)):
            same = abs(float(have) - want) < 1e-9
        else:
            same = have == want
        if not same:
            shown = ("the server's default" if field in ("temperature", "coding_peak_temperature") and have is None
                     else json.dumps(have))
            problems.append(f"{field}: method {name} has {json.dumps(want)}, the result records {shown}")
    return problems


def identify(result: Mapping) -> str | None:
    """The name of the first method whose every fixed field ``result`` records, or None."""
    return next((name for name in METHODS if not check(result, name)), None)


def signature(result: Mapping) -> dict:
    """The recorded method fields that two compared results must share (all of :data:`FIELDS`)."""
    return recorded(result)


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_check(path, result: Mapping, name: str | None) -> tuple[str, bool]:
    fields = recorded(result)
    lines = [f"{path}: " + ", ".join(f"{k} {json.dumps(v)}" for k, v in fields.items())]
    if name is None:
        found = identify(result)
        lines.append(f"method: {found or 'none of ' + ', '.join(METHODS)}")
        return "\n".join(lines), True
    problems = check(result, name)
    lines.append(f"method {name}: {'matches' if not problems else 'DIFFERS'}")
    lines += [f"  {p}" for p in problems]
    return "\n".join(lines), not problems
