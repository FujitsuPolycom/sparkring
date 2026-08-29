# Sanitized benchmark receipts

Most JSON files are sanitized copies of accepted `llm_decode_bench.py`
receipts used by the DeepSeek, GLM-5.2, and Qwen result tables. The GLM-5.3
directory contains a functional semantic-canary receipt and does not contribute
to a throughput table.

| Directory | Contents |
|---|---|
| [`deepseek-v4-flash/temp1/`](deepseek-v4-flash/temp1/) | 28 two-Spark TP2/DCP1 sustained-decode and Coding Peak receipts |
| [`deepseek-v4-flash/temp1/20260823-tp2/`](deepseek-v4-flash/temp1/20260823-tp2/) | 31 additional TP2 receipts contributing to the N=5/N=3 pair matrix, including prefill |
| [`deepseek-v4-flash/temp1/20260823-tp4/`](deepseek-v4-flash/temp1/20260823-tp4/) | 31 TP4 receipts contributing to the N=5/N=3 cycle matrix, including prefill and Coding Peak |
| [`glm-3.5bpw/temp1/`](glm-3.5bpw/temp1/) | 10 four-Spark TP4/DCP4 sustained-decode and Coding Peak receipts |
| [`glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/`](glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/) | Sanitized post-restore semantic canary for the TP4/DCP1 SparkCache validation |
| [`qwen38-27b/temp1/`](qwen38-27b/temp1/) | 13 two-Spark and 16 four-Spark accepted prefill, decode, and Coding Peak receipts |

Every receipt records `temperature: 1.0`. DeepSeek used effective top-p 1.0;
GLM used the checkpoint's effective top-p 0.95. Top-p was not overridden on
the benchmark command line. Qwen used effective top-p 0.95 and top-k 20 from
the pinned checkpoint's `generation_config.json`; neither was overridden by
the benchmark command.

## Recorded commands

The receipts retain the measured argument arrays in `public_replay_command`.
Replace `<rank-0-endpoint>` and `<output-directory>` before running one. Remote
hardware monitoring is disabled because SSH targets are private site data.

These runs used `llm_decode_bench.py` version 0.4.31. The measured script
SHA-256 is
`07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851`.
That exact script is not published in this repository, so these files document
the commands and evidence but are not turnkey replay scripts.

The original run also recorded endpoint bindings, the client hostname, SSH
monitor targets, local GPU diagnostics, event text, and hardware summaries.
Those fields are removed. Hardware summaries from these runs include a
pre-measurement or cancellation tail and are not suitable for precise thermal
or power claims.

Each sanitized file records the SHA-256 digest of its unsanitized source. That
digest identifies the private source receipt without publishing site details.
JIT/server-log-rejected invocations are absent. Valid rows from a mixed receipt
are retained only when that row independently passed request-error, timeout,
alignment, capacity, and aggregate-validity checks.

## Prefill boundary

The result tables retain prefill TTFT because it ends at the first token and
does not measure sustained sampled decode.
