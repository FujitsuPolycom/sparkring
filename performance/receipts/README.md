# Sanitized benchmark receipts

These JSON files are sanitized copies of the accepted temperature-1
`llm_decode_bench.py` receipts used by the normalized DeepSeek and GLM result
tables. They preserve the workload arguments, timing policy, token accounting,
per-cell observations, summary table, and methodology.

| Directory | Contents |
|---|---|
| [`deepseek-v4-flash/temp1/`](deepseek-v4-flash/temp1/) | 28 two-Spark TP2/DCP1 sustained-decode repetitions and Coding Peak receipts |
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

## Prefill boundary

The readable result tables retain prefill TTFT measurements because the metric
ends at the first token and does not measure sustained sampled decode. Their
prefill-only harness envelopes recorded temperature 0, so the raw prefill
receipts are excluded from this temperature-1-only public bundle. Publish new
prefill receipts only after rerunning them with a harness envelope that records
temperature 1.0.
