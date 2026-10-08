"""One start's client measurements, run from the operator's machine against the rank-0 API.

Metric sets:

- ``warmup``: fingerprints, prompt logprobs, decode at contexts 0 and 32k with 1 and 8 streams for 10 s,
  TTFT at 8k and 32k once. Its results are kept and not reported.
- ``phase1``: fingerprints, prompt logprobs, decode at contexts 0 and 32k with 1, 2, 4 and 8 streams
  (temperature 0, at most 1,024 tokens, 30 s cells after 10 s of warm-up), TTFT at 8k and 32k, three
  samples each with a unique prefix.
- ``phase2``: ``phase1`` with TTFT at 2k, 8k, 32k and 128k.

Decode runs llm-inference-bench's llm_decode_bench.py (``--bench-dir``) through :mod:`.bench_run`; TTFT
runs performance/harnesses/validation/prefill_probe.py; prompt logprobs run
performance/records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFILL = ROOT / "performance/harnesses/validation/prefill_probe.py"
LOGPROBS = ROOT / "performance/records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py"
METRICS = {
    "warmup": {"concurrency": "1,8", "duration": "10", "ttft": "8k,32k", "repeats": "1"},
    "phase1": {"concurrency": "1,2,4,8", "duration": "30", "ttft": "8k,32k", "repeats": "3"},
    "phase2": {"concurrency": "1,2,4,8", "duration": "30", "ttft": "2k,8k,32k,128k", "repeats": "3"},
}


def measure(out: Path, *, host: str, port: int, model: str, context_limit: int, bench_dir: str, metrics: str,
            python: str = sys.executable) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = METRICS[metrics]
    base = f"http://{host}:{port}"
    steps = [
        ("fingerprint", [python, "-m", "performance.harnesses.serving_ab.fingerprint", base + "/v1",
                         str(out / "fingerprint.json")]),
        ("logprobs", [python, str(LOGPROBS), base + "/v1", str(out / "logprobs.json")]),
        ("decode", [python, "-m", "performance.harnesses.serving_ab.bench_run", bench_dir, "--host", host,
                    "--port", str(port), "--model", model, "--no-hw-monitor", "--display-mode", "plain",
                    "--no-resume", "--skip-prefill", "--contexts", "0,32k", "--concurrency", m["concurrency"],
                    "--token-targeting", "exact", "--temperature", "0", "--max-tokens", "1024",
                    "--duration", m["duration"], "--decode-warmup-seconds", "10",
                    "--cell-warmup-timeout-seconds", "900", "--output", str(out / "decode.json")]),
        ("prefill", [python, str(PREFILL), "--endpoint", base, "--model", model, "--contexts", m["ttft"],
                     "--context-limit", str(context_limit), "--repeats", m["repeats"], "--temperature", "0",
                     "--output", str(out / "prefill.jsonl")]),
    ]
    codes = {}
    for name, command in steps:
        started = time.time()
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            codes[name] = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=ROOT).returncode
        codes[f"{name}_seconds"] = round(time.time() - started, 1)
    return codes
