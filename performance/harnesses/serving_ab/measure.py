"""One start's client measurements, run from the operator's machine against the rank-0 API.

Metric sets:

- ``warmup``: fingerprints, prompt logprobs, decode at contexts 0 and 32k with 1 and 8 streams for 10 s,
  TTFT at 8k and 32k once. Its results are kept and not reported.
- ``phase1``: fingerprints, prompt logprobs, decode at contexts 0 and 32k with 1, 2, 4 and 8 streams
  (30 s cells after 10 s of warm-up), TTFT at 8k and 32k, three samples each with a unique prefix.
- ``phase2``: ``phase1`` with TTFT at 2k, 8k, 32k and 128k.

The decode step's measurement method is a named method of :mod:`.receipt.method`:

- ``serving-ab-temp0`` (the default): temperature 0, at most 1,024 output tokens, prefill skipped, the metric
  set's contexts and stream counts;
- ``owner-temp1-matrix``: the deployment owner's matrix (temperature 1, contexts 8k-128k, 1/2/4/8 streams,
  20 s cells after a 5 s warm-up, at most 2,048 tokens, integrated prefill scout, coding peak at temperature
  1), whatever the metric set; the benchmark checkout (``--bench-dir``) must be llm-inference-bench 0.7.6 for
  its result to match the method.

``temperature`` and ``max_tokens`` override the method's values; the method recorded is then the one whose
every fixed setting the run still has, or ``custom``. :func:`measure` returns the method's name and settings
with the step exit codes (the runner writes them to ``measure.json``) and writes ``method.json``. Every
decode cell starts from an idle server (:mod:`.bench_run`).

Decode runs llm-inference-bench's llm_decode_bench.py (``--bench-dir``) through :mod:`.bench_run`; TTFT
runs performance/harnesses/validation/prefill_probe.py; prompt logprobs run
performance/records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance.harnesses.serving_ab.receipt import method as methods  # noqa: E402

PREFILL = ROOT / "performance/harnesses/validation/prefill_probe.py"
LOGPROBS = ROOT / "performance/records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py"
DEFAULT_METHOD = "serving-ab-temp0"
METRICS = {
    "warmup": {"concurrency": "1,8", "duration": "10", "ttft": "8k,32k", "repeats": "1"},
    "phase1": {"concurrency": "1,2,4,8", "duration": "30", "ttft": "8k,32k", "repeats": "3"},
    # Up to 16 streams at contexts 0 and 16k, for a profile served with --max-num-seqs 16.
    "c16": {"concurrency": "1,2,4,8,16", "duration": "30", "ttft": "8k,16k,32k", "repeats": "3", "contexts": "0,16k"},
    # No measurement: the start only becomes ready and passes its checks (for a profile capture).
    "none": {},
    # phase1 plus decode at 16k context and TTFT at 16k, for comparisons recorded at 16K.
    "phase1-16k": {"concurrency": "1,2,4,8", "duration": "30", "ttft": "8k,16k,32k", "repeats": "3",
                   "contexts": "0,16k,32k"},
    "phase2": {"concurrency": "1,2,4,8", "duration": "30", "ttft": "2k,8k,32k,128k", "repeats": "3"},
}


def _contexts(text: str) -> list[int]:
    return [int(float(c[:-1]) * 1024) if c.endswith("k") else int(c) for c in text.split(",")]


def decode_method(metrics: str, method: str = DEFAULT_METHOD, *, temperature: float | None = None,
                  max_tokens: int | None = None) -> dict:
    """The decode step's benchmark arguments and the method settings they give (see the module docstring)."""
    m = METRICS[metrics]
    profile = methods.METHODS[method]
    temperature = profile["temperature"] if temperature is None else float(temperature)
    max_tokens = profile["max_tokens"] if max_tokens is None else int(max_tokens)
    if method == DEFAULT_METHOD:
        contexts, concurrency, duration = m.get("contexts", "0,32k"), m["concurrency"], m["duration"]
        args = ["--skip-prefill", "--contexts", contexts, "--concurrency", concurrency, "--token-targeting", "exact",
                "--temperature", f"{temperature:g}", "--max-tokens", str(max_tokens), "--duration", duration,
                "--decode-warmup-seconds", "10", "--cell-warmup-timeout-seconds", "900"]
        settings = {"temperature": temperature, "max_tokens": max_tokens, "context_lengths": _contexts(contexts),
                    "concurrency_levels": [int(c) for c in concurrency.split(",")],
                    "duration_per_test": float(duration), "prefill_mode": "skipped", "decode_warmup_seconds": 10.0,
                    "coding_peak": False}
    else:
        args = list(profile["bench_args"])
        args[args.index("--temperature") + 1] = f"{temperature:g}"
        args[args.index("--max-tokens") + 1] = str(max_tokens)
        settings = {field: profile[field] for field in methods.FIELDS if profile.get(field) is not None
                    and field not in ("tool", "version")}
        settings.update(temperature=temperature, max_tokens=max_tokens)
    named = [name for name, fixed in methods.METHODS.items()
             if all(fixed.get(f) is None or f in ("tool", "version") or settings.get(f) == fixed[f]
                    for f in methods.FIELDS)]
    return {"method": method if method in named else "custom", "requested_method": method, "settings": settings,
            "bench_args": args}


def measure(out: Path, *, host: str, port: int, model: str, context_limit: int, bench_dir: str, metrics: str,
            python: str = sys.executable, method: str = DEFAULT_METHOD, temperature: float | None = None,
            max_tokens: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = METRICS[metrics]
    if not m:
        return {}
    decode = decode_method(metrics, method, temperature=temperature, max_tokens=max_tokens)
    (out / "method.json").write_text(json.dumps({**decode, "metrics": metrics}, indent=1), encoding="utf-8")
    base = f"http://{host}:{port}"
    steps = [
        ("fingerprint", [python, "-m", "performance.harnesses.serving_ab.fingerprint", base + "/v1",
                         str(out / "fingerprint.json")]),
        ("logprobs", [python, str(LOGPROBS), base + "/v1", str(out / "logprobs.json")]),
        ("decode", [python, "-m", "performance.harnesses.serving_ab.bench_run", bench_dir, "--host", host,
                    "--port", str(port), "--model", model, "--no-hw-monitor", "--display-mode", "plain",
                    "--no-resume", *decode["bench_args"], "--output", str(out / "decode.json")]),
        ("prefill", [python, str(PREFILL), "--endpoint", base, "--model", model, "--contexts", m["ttft"],
                     "--context-limit", str(context_limit), "--repeats", m["repeats"], "--temperature", "0",
                     "--output", str(out / "prefill.jsonl")]),
    ]
    codes = {"method": decode["method"], "method_settings": decode["settings"]}
    for name, command in steps:
        started = time.time()
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            codes[name] = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=ROOT).returncode
        codes[f"{name}_seconds"] = round(time.time() - started, 1)
    return codes
