# Sanitized benchmark receipts

These JSON files are sanitized copies of the accepted temperature-1
`llm_decode_bench.py` receipts used by the normalized DeepSeek and GLM result
tables. They preserve the workload arguments, timing policy, token accounting,
per-cell observations, summary table, and methodology.

| Directory | Contents |
|---|---|
| [`deepseek-v4-flash/temp1/`](deepseek-v4-flash/temp1/) | 28 earlier two-Spark TP2/DCP1 sustained-decode and Coding Peak receipts |
| [`deepseek-v4-flash/temp1/20260823-tp2/`](deepseek-v4-flash/temp1/20260823-tp2/) | 31 additional TP2 receipts contributing to the N=5/N=3 pair matrix, including temperature-1 prefill |
| [`deepseek-v4-flash/temp1/20260823-tp4/`](deepseek-v4-flash/temp1/20260823-tp4/) | 31 TP4 receipts contributing to the N=5/N=3 cycle matrix, including prefill and Coding Peak |
| [`glm-3.5bpw/temp1/`](glm-3.5bpw/temp1/) | 10 four-Spark TP4/DCP4 sustained-decode and Coding Peak receipts |

Every receipt records `temperature: 1.0`. DeepSeek used effective top-p 1.0;
GLM used the checkpoint's effective top-p 0.95. Top-p was not overridden on
the benchmark command line.

## Replay commands

`public_replay_command` is an argument array that can be joined or passed to a
process launcher. Replace `<rank-0-endpoint>` and `<output-directory>`. It
retains the measured workload and timing arguments but disables remote hardware
monitoring because the original SSH targets are private site configuration.

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

The readable result tables retain prefill TTFT measurements because the metric
ends at the first token and does not measure sustained sampled decode. The
20260823 TP2/TP4 bundles contain new prefill-only receipts whose harness
envelopes record temperature 1.0. Older prefill envelopes that recorded
temperature 0 remain excluded from this temperature-1-only bundle.
