"""Find and parse the lines of a vLLM container log that state what the server resolved at startup.

:func:`scan` reads a log (``docker logs`` output, with or without ``--timestamps``) and returns every line
it recognises, raw and parsed:

- ``non-default args: {...}``: the explicit arguments, a ``dict`` repr (:mod:`.pyrepr`); only the API
  server (rank 0) logs it;
- ``Initializing a V1 LLM engine (vVERSION) with config: k=v, ...``: the engine's resolved configuration
  (``VllmConfig.__str__``): target KV dtype, quantization, load format, parallel sizes, the speculative
  method, the resolved ``compilation_config`` and ``kernel_config``; rank 0 only;
- single-purpose lines any rank logs: the resolved attention or KV block size, the Engram configuration,
  enabled custom fusions, the MoE and GEMM kernels chosen per quantization, the chunked-prefill token
  budget, the KV cache size in tokens, variables vLLM sets itself, the format of each LM head (target, then
  draft, in log order), SIRCL receipt lines,
  all-reduce backends, the async-scheduling warnings and SparkRing's startup audit lines.

When a line occurs more than once (the API server and the engine both log some), the last occurrence of
the last server start wins, and ``occurrences`` counts them. Lines cut short by a capture script are parsed
as far as they go and flagged ``truncated``.
"""

from __future__ import annotations

import re

from . import pyrepr

NON_DEFAULT = re.compile(r"non-default args: (\{.*)$")
ENGINE = re.compile(r"Initializing a V1 LLM engine \(v([^)]*)\) with config: (.*)$")
# One pattern per single-purpose line: name -> (pattern, converter of the match).
SINGLE = {
    "attention_block_size": (re.compile(r"Setting attention block size to (\d+) tokens"), lambda m: int(m[1])),
    "kv_block_size": (re.compile(r"Setting kv cache block size to (\d+) for (\S+) backend"),
                      lambda m: {"tokens": int(m[1]), "backend": m[2]}),
    "kv_lcm_block_size": (re.compile(r"kv lcm block sizes (\d+)"), lambda m: int(m[1])),
    "engram": (re.compile(r"Resolved Engram configuration: (EngramConfig\(.*\))\s*$"),
               lambda m: pyrepr.value_or_text(m[1])),
    "custom_fusions": (re.compile(r"Enabled custom fusions: (.*?)\s*$"),
                       lambda m: [x.strip() for x in m[1].split(",") if x.strip()]),
    "chunked_prefill_tokens": (re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)"),
                               lambda m: int(m[1])),
    "kv_cache_tokens": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda m: int(m[1].replace(",", ""))),
    "model_runner_v2": (re.compile(r"Using V2 Model Runner"), lambda m: True),
    "breakable_cudagraph": (re.compile(r"Breakable CUDA graph enabled"), lambda m: True),
    "max_model_len": (re.compile(r"Using max model len (\d+)"), lambda m: int(m[1])),
    "architecture": (re.compile(r"Resolved architecture: (\S+)"), lambda m: m[1]),
    "kv_cache_layout": (re.compile(r"Using (\S+) KV cache layout"), lambda m: m[1]),
}
# Lines that occur once per subject: name -> (pattern, key of the subject, value).
KEYED = {
    "moe_kernel": (re.compile(r"Using '([^']+)' (\S+) MoE backend"), lambda m: m[2], lambda m: m[1]),
    "gemm_kernel": (re.compile(r"Using (\S+) for (\S+) GEMM"), lambda m: m[2], lambda m: m[1]),
    "auto_env": (re.compile(r"Auto-enabling ([A-Z0-9_]+)=(\S+?)\.?(?:\s|$)"), lambda m: m[1], lambda m: m[2]),
    "allreduce_backends": (re.compile(r"Using (\[[^\]]*\]) all-reduce backends .* for group '([^']+)'"),
                           lambda m: m[2], lambda m: pyrepr.value_or_text(m[1])),
    "startup_audit": (re.compile(r"SPARKRING AUDIT \[([A-Z0-9_]+)\] (.*?)\s*$"), lambda m: m[1], lambda m: m[2]),
}
# Lines that occur once per model part (target and draft), kept in log order.
LISTED = {"lm_heads": (re.compile(r"Quantizing LM head shards to (\S+) with (\S+) activations"),
                       lambda m: {"format": m[1], "activations": m[2]})}
RECEIPT = re.compile(r"SIRCL receipt group=(\S+) (.*?)\s*$")
ASYNC_OFF = re.compile(r"Async scheduling (?:not supported with|will be disabled|is not compatible|is disabled)")


def _message(line: str) -> str:
    """The line without a ``docker logs --timestamps`` prefix (kept for matching only)."""
    return re.sub(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", "", line.rstrip("\r\n"))


def _receipt_fields(text: str) -> dict:
    fields = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def scan(text: str) -> dict:
    """Every recognised line of ``text`` (a container log), raw and parsed. See the module docstring."""
    out: dict = {"raw": {}, "occurrences": {}}

    def keep(name: str, line: str, value, truncated: bool = False):
        out[name] = value
        out["raw"][name] = line
        out["occurrences"][name] = out["occurrences"].get(name, 0) + 1
        if truncated:
            out.setdefault("truncated", []).append(name)
        elif name in out.get("truncated", []):
            out["truncated"].remove(name)

    for line in text.splitlines():
        message = _message(line)
        match = NON_DEFAULT.search(message)
        if match:
            value, truncated = pyrepr.parse(match[1], partial=True)
            keep("non_default_args", message, value if isinstance(value, dict) else {}, truncated)
            continue
        match = ENGINE.search(message)
        if match:
            config = {key: pyrepr.value_or_text(value) for key, value in pyrepr.split_assignments(match[2])}
            keep("engine_config", message, {"version": match[1], **config},
                 truncated=not message.rstrip().endswith(")"))
            continue
        match = RECEIPT.search(message)
        if match:
            receipts = out.setdefault("sircl_receipt_lines", {})
            receipts[match[1]] = _receipt_fields(match[2])
            out["raw"].setdefault("sircl_receipt_lines", {})[match[1]] = message
            continue
        if ASYNC_OFF.search(message):
            keep("async_scheduling_disabled", message, True)
            continue
        listed = next(((name, pattern.search(message), convert) for name, (pattern, convert) in LISTED.items()
                       if pattern.search(message)), None)
        if listed:
            name, match, convert = listed
            out.setdefault(name, []).append(convert(match))
            out["raw"].setdefault(name, []).append(message)
            continue
        for name, (pattern, convert) in SINGLE.items():
            match = pattern.search(message)
            if match:
                keep(name, message, convert(match))
                break
        else:
            for name, (pattern, subject, value) in KEYED.items():
                match = pattern.search(message)
                if match:
                    out.setdefault(name, {})[subject(match)] = value(match)
                    out["raw"].setdefault(name, {})[subject(match)] = message
                    break
    return out


GREP_PATTERN = "|".join([
    "non-default args", "with config:", "Setting attention block size", "Setting kv cache block size",
    "kv lcm block sizes", "Resolved Engram configuration", "Enabled custom fusions",
    "Chunked prefill is enabled", "GPU KV cache size", "Quantizing LM head", "Using V2 Model Runner",
    "Breakable CUDA graph enabled", "Using max model len", "Resolved architecture", "KV cache layout",
    "MoE backend", "GEMM", "Auto-enabling", "all-reduce backends", "SPARKRING AUDIT", "SIRCL receipt group",
    "Async scheduling",
])
"""An extended regular expression (``grep -E``) selecting the lines :func:`scan` recognises, so a remote
capture sends only those lines instead of a whole log."""
