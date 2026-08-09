# Evidence-analysis checklist: cache-off vs cache-on 16K benchmark comparison

This checklist governs the offline comparison of matched cache-off
versus cache-on sustained 16K C1/C2/C4/C8 benchmark JSON. It is
consumed alongside `scripts/compare_benchmark_evidence.py`, which
automates the mechanical checks below.

## Scope

This checklist applies to **sustained-decode 16K matrix** evidence
produced by `llm_decode_bench.py` v0.4.31 with 25-second duration
cells, 1,024-token maximum, temperature 0, 0% unique / 100% shared
contexts, DCP4, 562,688-token auto-detected KV budget, zero-second
decode warmup, prefill skipped, and 600-second cell-warmup timeout.

**Never mix bounded 128-token gate figures with sustained 25-second
matrix figures.** The comparison tool classifies each document as
`sustained_matrix` or `bounded_gate` and refuses cross-type
comparisons.

## Pre-comparison requirements

Before claiming any delta between a cache-off baseline and a cache-on
candidate:

### 1. Document type must match

Both documents must be classified `sustained_matrix` by the tool.
A `bounded_gate` document (max_tokens < 256 or duration < 10s) cannot
be compared against a `sustained_matrix` document.

### 2. Workload settings must be identical

Every setting in the tool's `MATCHED_SETTINGS` list must match
exactly between the two documents:

| Setting | Why it matters |
|---|---|
| Harness name and version | Different harnesses measure differently |
| Context length (tokens) | 16K vs 8K/32K changes the decode path |
| Concurrencies list | C1/C2/C4/C8 vs C1/C2/C8 changes the comparison |
| Duration per cell (seconds) | 25s vs 15s changes sustained measurement |
| Decode warmup (seconds) | Affects first-cell stability |
| Max output tokens | 1024 vs 128 changes the measurement window |
| Temperature | 0 (greedy) vs >0 changes output and throughput |
| Unique context percent | 0% unique vs 100% changes cache behavior |
| Shared context percent | 100% shared vs 0% changes cache behavior |
| DCP size | DCP4 vs DCP1 changes the collective path |
| KV budget (tokens) | 562,688 auto-detected vs manual changes capacity |
| ignore_eos | True vs False changes when streams end |
| skip_prefill | True vs False changes whether prefill is measured |
| Cell warmup timeout (seconds) | 600 vs 300 vs 60 changes readiness-limited suppression |

If any setting differs, the comparison reports `settings_mismatch` and
**no delta is claimed**. This is fail-closed.

### 3. Cell validity must be confirmed

Each cell must have:
- exact effective concurrency (requested = observed)
- zero request errors
- `all_cells_valid: true` in the document

The `--strict` flag enforces this at exit time.

### 4. The only variable that changed is cache state

The comparison is valid only if the **sole difference** between the
two runs is the cache variable (cache-off vs cache-on). This means:

- same model revision and image digest
- same serving configuration (TP, DCP, MTP, KV, batching, graphs)
- same hardware (four directly cabled DGX Sparks)
- same network path (direct-cable ring, no switch)
- same time of day or equivalent thermal/energy state
- same operator and launch procedure

The tool cannot verify this; it is the operator's responsibility.

## Delta reporting

When all checks pass, the tool reports per-concurrency:
- `baseline_tps`: cache-off aggregate tok/s
- `candidate_tps`: cache-on aggregate tok/s
- `delta`: candidate - baseline (absolute)
- `delta_percent`: ((candidate - baseline) / baseline) * 100
- `status`: `compared`

A negative delta_percent means cache-on is **slower** than cache-off
at that concurrency. This is a throughput regression signal.

## Claim labels

Any reported delta must carry the label:

> Cross-document comparison with matched sustained-16K settings. Not a
> sealed A/B unless both documents were produced under controlled
> conditions with identical configuration except for the cache variable.

The tool includes this in `claim_note`.

## Distinguishing cache layers

Three distinct cache mechanisms exist in this repository. They must not
be conflated:

| Cache layer | Implementation | Controlled by |
|---|---|---|
| **Native APC** | vLLM `--enable-prefix-caching` | vLLM engine state |
| **LMCache** | `LMCache` integration, one local server per rank | LMCache server process |
| **SparkCache** | `sparkcache/` implementation | `SPARK_CONTEXT_CACHE_ENABLE` |

A "cache-off" baseline must specify **which** cache(s) were disabled.
A "cache-on" candidate must specify **which** cache(s) were enabled.
Comparing "LMCache-on + APC-on" against "LMCache-off + APC-off" is a
combined cache comparison, not an LMCache-only comparison.

## What this checklist does NOT cover

- **Sealed A/B claims**: the tool verifies settings match; it cannot
  verify controlled conditions. A sealed A/B requires operator
  protocol (same run session, alternating order, etc.).
- **Statistical significance**: single-cell measurements are not
  distributions. A delta_percent from one cell is a point observation,
  not a confidence interval.
- **Prefill comparisons**: this checklist is for sustained-decode
  only. Prefill scout comparisons have their own methodology
  (single-sample, documented in RESULTS.md).
- **128-token gate comparisons**: bounded gate figures use different
  max_tokens, duration, and concurrency shapes. They are never
  compared against sustained matrix figures.

## Offline tool usage

```bash
python scripts/compare_benchmark_evidence.py \
    --baseline evidence/cache-off-16k.json \
    --candidate evidence/cache-on-16k.json \
    --strict
```

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | Compared successfully, all cells valid |
| 2 | Type mismatch or settings mismatch — no delta claimed |
| 3 | Configuration error (missing file, invalid JSON) |
| 4 | Invalid cells or missing throughput (strict mode only) |
