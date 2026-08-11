# Evidence-analysis checklist: matched 16K sustained decode

`scripts/compare_benchmark_evidence.py` compares two offline
`llm_decode_bench.py` v0.4.31 JSON documents. The implemented scope is one
16,384-token context at C1, C2, C4, and C8 under duration-based decode. The
tool does not contact a cluster, launch a model, or run a benchmark.

The comparator is an implemented evidence-analysis utility. A successful
report means that the two input documents are structurally valid and directly
comparable within the fields they record. It does not promote either serving
configuration or turn two independently collected runs into a sealed A/B.

## Comparison contract

Both documents must meet every condition below before the tool emits throughput
deltas.

### Benchmark type

- `metadata.version` identifies `llm_decode_bench.py` v0.4.31.
- `metadata.decode_mode` and every result `benchmark_mode` are `duration`.
- `duration_per_test` is at least 10 seconds.
- `max_tokens` is at least 256.
- Bounded gates and indeterminate documents are rejected, including a pair of
  two bounded documents.

### Runtime and workload identity

The following recorded fields must be present, valid, and equal between the
baseline and candidate:

| Field group | Compared values |
|---|---|
| Runtime identity | engine, model, harness version, primary decode layer, decode mode |
| Shape | context lengths, concurrency levels, DCP size, KV token budget |
| Measurement | cell duration, maximum output tokens, temperature, ignore-EOS behavior |
| Warmup | warmup duration, warmup context, warmup concurrency, cell timeout |
| Prompt mix | unique-context percentage, shared-context percentage, prefill-skip behavior |

Unique- and shared-context percentages must each be within 0–100 and sum to
100. Numeric metadata must be finite. Boolean metadata must be actual JSON
booleans rather than numeric substitutes.

### Coverage and cell validity

Each document must contain exactly one 16,384-token context and exactly one
result for C1, C2, C4, and C8. Metadata, results, and an optional
`summary_table` must agree. Missing, duplicated, unexpected, or multi-context
cells fail closed.

Every result must satisfy:

- `aggregate_tps` is finite and positive;
- `num_errors == 0`;
- `effective_concurrency` equals the requested concurrency;
- `measurement_seconds` is finite and positive;
- `underfilled`, `warmup_timed_out`, and `capacity_limited` are false;
- `benchmark_mode == "duration"`.

Missing fields are invalid. The comparator never invents defaults. If a
`summary_table` is present, it is checked against `results`; it cannot override
the result records.

### Distinct inputs

Semantically identical benchmark documents are rejected as an invalid comparison.
The CLI records the SHA-256 digest of both input files in its report so the
comparison is traceable to exact evidence.

## Reported result

After every gate passes, the tool emits, for each concurrency:

- baseline and candidate aggregate tokens per second;
- absolute candidate-minus-baseline delta;
- percentage delta relative to the baseline.

No numeric deltas are emitted for type, metadata, settings, coverage, cell
validity, or non-finite derived-value failures. A negative percentage is a
throughput regression signal, not a statistical conclusion.

Every report carries this scope statement:

> Cross-document comparison with matched sustained-16K settings. Not a sealed
> A/B unless both documents were produced under controlled conditions with
> identical configuration except for the declared experimental variable.

## Conditions outside the JSON schema

The v0.4.31 benchmark document does not prove image digest, model revision,
launch arguments, transport path, hardware identity, competing traffic,
thermal state, or run ordering. Preserve those facts in a separate launch and
telemetry receipt. Declare one experimental variable before collecting the
pair; the comparator cannot infer which external variable changed.

For cache comparisons, name the mechanism precisely:

| Cache mechanism | Implementation |
|---|---|
| Native automatic prefix cache | vLLM prefix-caching configuration |
| LMCache | LMCache connector and one local cache service per rank |
| SparkCache | `sparkcache/` and `SPARK_CONTEXT_CACHE_ENABLE` |

Changing more than one cache mechanism produces a combined cache comparison,
not evidence attributable to one implementation.

## Limitations

- One measurement per cell is a point observation, not a confidence interval.
- The tool does not compare prefill scouts, coding-peak runs, bounded gates, or
  multi-context matrices.
- A sealed A/B requires a controlled collection protocol, stable external
  receipts, and an ordering strategy such as alternating runs.

## Usage

```bash
python scripts/compare_benchmark_evidence.py \
    --baseline evidence/baseline-16k.json \
    --candidate evidence/candidate-16k.json
```

| Exit code | Meaning |
|---:|---|
| 0 | Both documents are valid and comparable; deltas were emitted |
| 2 | Document type, matched settings, or distinct-input requirement failed |
| 3 | An input file is missing or is not valid JSON |
| 4 | Metadata, coverage, result cells, or derived deltas are invalid |
