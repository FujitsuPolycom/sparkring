# Performance evidence

`performance/` contains reproducible measurement programs and bounded evidence
records for the supported GLM-5.2 EXL3 3.5-bpw and DeepSeek-V4-Flash-0731
serving configurations. It does not establish a general hardware, model, or
production-serving claim.

## Layout

| Path | Purpose |
|---|---|
| `harnesses/bench/` | Python programs and tests for roofline, collective-attribution, and expert-bitwidth accounting |
| `harnesses/q2r_phase_timing/` | CUDA-event phase timing and the optional Q-to-route probe bridge |
| `harnesses/vllm/` | vLLM timing, payload-planning, flight-recording, and prefill-capacity research |
| `harnesses/moe_round_floor/` | Routed-expert timing, route reuse, and capture diagnostics |
| `harnesses/transport/` | Model-loop replay and direct-link payload sweeps |
| `methodology/` | Measurement definitions, attribution rules, and CUDA-graph correctness requirements |
| `records/glm-3.5bpw/` | GLM-5.2 EXL3 R7 evidence records |
| `records/deepseek-v4-flash/` | DeepSeek-V4-Flash-0731 evidence records |
| `records/transport/` | Transport evidence records independent of a model result |

Run harness tests offline from the repository root:

```bash
python -m pytest performance/harnesses/bench -q
```

The tests validate program logic and record-processing boundaries. They do not
measure a GPU, fabric, image, or serving stack.

## Methodology

A harness must produce machine-readable output that preserves the input
configuration and raw observations needed to recompute its reported summary.
Use the methodology documents to define timing scope, collective attribution,
and CUDA-graph admissibility before comparing configurations.

A measurement compares only runs with the same relevant model revision, image
identity, topology, rank count, tensor-parallel and decode-context-parallel
degrees, request shape, concurrency, cache state, and timing method. If a
variable differs, identify it as the comparison variable or report the runs
separately.

Treat warm-up, graph capture, request errors, transport-counter changes, and
missing ranks as gates. A rejected or incomplete capture is not a zero-valued
observation and must not be averaged into a result.

## Evidence record format

Every record and any prose that reports its values must contain these labeled
sections:

1. **Conditions** — supported model configuration, immutable image or artifact
   identity, hardware and topology, rank layout, harness revision, input
   shape, concurrency, cache state, repetitions, and all changed variables.
2. **Measurement** — metric definition, clock or event source, aggregation,
   warm-up policy, gate outcomes, raw-record location, and uncertainty or
   variability calculation.
3. **Result** — observed values with units and the precise population or run
   window they summarize.
4. **Conclusion** — the narrow claim warranted by the result, including the
   compared configurations when applicable.
5. **Limitations** — unmeasured behavior, failed or omitted gates, sampling
   limits, topology limits, and reasons the result cannot be generalized.

A record without all five sections is research material, not a qualified
performance claim. Do not describe a number from one model configuration as a
result for the other model configuration.

## Status and retention

Evidence is qualified only for the conditions written in its record. Preserve
raw machine-readable records when changing summaries so reviewers can verify
aggregation. A later run may supersede a claim only when it identifies the
same metric and conditions or explicitly states each changed condition.
