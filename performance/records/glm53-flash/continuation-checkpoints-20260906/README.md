# Fixed-8K continuation checkpoint evidence

Status: **research-only**. The recorded runtime completes cold long prompts
without splitting the final continuation solely to export recurrent checkpoints.
This record describes the exact image below; it does not qualify the rebuilt
source composition that also adds request attribution.

## Conditions

Four DGX Spark GB10 GPUs, TP4/DCP4/PP1, native MTP3, 8,192 maximum batched
tokens, 16 sequences, and 24 GiB KV cache per rank. The model is
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. The image config is
`sha256:489d1975619e9083d14f14bcd1c6cbb4a96c41e3ab3978af048dc3ed2bb452a8`.
Both `SPARK_GDN_PREFILL_CHECKPOINTS=2` and
`SPARK_GDN_CONTINUATION_CHECKPOINTS=1` are enabled. The native virtual mesh,
graph-only decode geometry, linked posting, B12X compute patches, and SparkCache
remain in the composition; periodic full capture is off.

Requests have concurrency one, temperature zero, one output token, and a unique
prefix. Exact prompt usage and zero cached tokens are checked for every sample.
There is no restarted paired control. Four scheduler/allocator/helper files
changed, with a separate cache root and regenerated ownership/image inventories.
Component probes used a source-identical child before a metadata-only
deterministic rebuild. Full conditions are in [the record](continuation-evidence.json).

## Measurement

TTFT spans submission through the first nonempty streamed content or reasoning
delta, including client/network overhead. The historical Windows harness uses
`time.monotonic` backed by GetTickCount64, with about 15.625 ms granularity.
Reported decimal precision is not clock accuracy. Startup/API warmup and idle
gauges passed; one shape-warmup row per 8K–64K size is excluded. Prior semantic
and prefill requests warmed 128K. Each size has three measured samples.

The table uses prompt tokens divided by median TTFT; min/max describe those
three TTFTs. Raw rows are [8K–64K](continuation-prefill.json) and
[128K](continuation-prefill-128k.json). [Provenance](provenance.json) identifies
the retained historical harness and the portable public copy; those hashes are
not signed execution-time attestations.

## Result

| Prompt tokens | Median TTFT | Min–max TTFT | Prompt tok/s |
|---:|---:|---:|---:|
| 8,192 | 2.797 s | 2.782–2.813 s | 2,929 |
| 16,384 | 5.672 s | 5.656–5.672 s | 2,889 |
| 32,768 | 11.328 s | 11.297–11.344 s | 2,893 |
| 65,536 | 22.719 s | 22.718–22.750 s | 2,885 |
| 131,072 | 45.875 s | 45.782–45.875 s | 2,857 |

Six [recurrence cases](continuation-recurrence.json) and three
[convolution cases](continuation-convolution.json) passed on GPU. The recorded
model checks include 15 regular semantic/cache cases, four concurrent cold
requests, and 25 exact answers across long-context attempts. All four workers
recorded an 8,192→16,384 continuation with checkpoint ends 14,336 and 15,360
across 34 GDN layers.

## Conclusion

The fixed-8K continuation implementation completed these cases at about
2,857–2,929 prompt tokens/s. These candidate-only observations establish no
isolated percentage speedup. The scheduler keeps the 8K chunk ceiling; it
exports intermediate state within an eligible final chunk rather than adding
checkpoint-driven model passes.

## Limitations

Three samples and synthetic fact retrieval do not establish general quality,
decode performance, concurrency performance, arbitrary-length correctness, or
other chunk sizes. CPU tests exercise a 6K schedule, but only fixed 8K has this
serving evidence. Historical screenshots used different harnesses/compositions.

Two 128K cache-reuse expectations missed and recomputed safely, including one
after worker publication. Later repeated/extended checks reused 129,024 tokens.
Scheduler visibility was not traced; this is not attributed as a continuation
regression. The rank-three launch memory gate passed on recheck without a reboot
or threshold change. The attribution-containing rebuild requires its own GPU,
startup, and serving validation.

## Reproduction

The [portable harness](../../../harnesses/vllm/prefill_checks/continuation_serve_checks.py)
retains the historical timing and request logic. Set `BENCH_API_BASE`,
`BENCH_MODEL`, `BENCH_RANK0_SSH`, and `BENCH_RANK0_CONTAINER` for an explicitly
authorized idle deployment, then run:

```bash
python performance/harnesses/vllm/prefill_checks/continuation_serve_checks.py prefill \
  --sizes 8192,16384,32768,65536,131072 --samples 3 --output /path/to/absent-results.json
```

This command sends inference requests and reads rank-zero Docker health over
SSH. Establish all-rank startup/health separately. The published harness has
offline gate coverage but has not been rerun on a GPU.
