# TP4 continuation coalescing and mHC prefill measurements

Status: **research-only**. These observations describe the source assembly
identified in `reference-measurements.json`. They do not qualify an image
rebuilt from the SparkRing source-image recipe.

## Conditions

Four NVIDIA DGX Sparks used a hardware-forwarded Ethernet ring, TP4, native
MTP depth three, an 8,192-token scheduler budget, 512-token cache blocks, and
24 GiB KV allocation per rank. The request context limit was 1,048,576 tokens;
the measured prompts were 8K, 16K and 32K. TP4 mesh collectives and dual-domain
NCCL were retained. SparkCache, compact index cache and DCP4-only owner top-k
exchange were disabled. CKV gathering was active at DCP2 and bypassed at DCP1.

The target was `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. vLLM revision was
`8f8ea47be212bbdd91b2172d5958ea2aae2b0e50`; B12X revision was
`0b6d61c37c87ae49d2f9d20d38b9da023146e243`. The JSON records the exact local
image config identity. That identity is not a public registry pull reference.

Within each DCP setting, the comparison toggled continuation coalescing and
mHC token sharding together. It does not isolate their individual effects.
The order was DCP2 disabled, DCP2 enabled, DCP1 disabled, DCP1 enabled. Separate
launches retained the same source/image identity and other relevant settings.

## Measurement

Each prefill shape has three cold samples after excluded warmups and
cold/repeated/extended exact-answer checks. The client used Python
`time.perf_counter` for time to first token (TTFT); each request permitted one
output token. Throughput is prompt tokens divided by median TTFT. All 36 raw
prefill timings, minimum/maximum/median values, and cold-cache gates are in
`reference-measurements.json`.

The JSON identifies the prefill client's source SHA-256 and the frozen
llm-decode-bench 0.4.34 source SHA-256. These are client identities, separate
from the serving engine and image identities above.

Decode used one 10-second observation per C1/C4 and 8K/32K cell, temperature 1,
and a 2,048-token output cap. Normalization divides aggregate emitted tokens/s
by measured MTP acceptance length. Raw throughput, acceptance, counters and
observation durations are retained in the JSON. Counter agreement is checked
within the benchmark's measurement-boundary tolerance.

All four configurations passed source, container-generation, expected feature
activation/absence and bounded serving checks. The original receipt hashes
are recorded in the sanitized JSON; private host and request data are omitted.

## Result

Median prefill throughput, in prompt tokens/s:

| DCP | Coalescing and mHC | 8K | 16K | 32K |
|---:|---|---:|---:|---:|
| 2 | Disabled | 1,938.7 | 2,345.5 | 2,619.0 |
| 2 | Enabled | 3,493.9 | 3,461.8 | 3,445.9 |
| 1 | Disabled | 2,263.0 | 2,615.7 | 2,828.4 |
| 1 | Enabled | 3,616.3 | 3,615.1 | 3,586.1 |

MTP-normalized decode steps/s, aggregated across the stated concurrency:

| DCP | Coalescing and mHC | Context | C1 | C4 |
|---:|---|---:|---:|---:|
| 2 | Disabled | 8K | 19.73 | 45.41 |
| 2 | Enabled | 8K | 20.05 | 46.62 |
| 2 | Disabled | 32K | 19.86 | 45.01 |
| 2 | Enabled | 32K | 19.80 | 44.47 |
| 1 | Disabled | 8K | 21.17 | 47.93 |
| 1 | Enabled | 8K | 21.40 | 48.00 |
| 1 | Disabled | 32K | 21.23 | 46.21 |
| 1 | Enabled | 32K | 21.17 | 49.93 |

## Conclusion

Enabling the two prefill features improved the measured prefill throughput
within both DCP settings. DCP1 produced the highest observed prefill throughput
under these conditions. Most normalized decode observations were close to
their controls; the DCP1 32K/C4 increase requires repetition before a decode
improvement can be claimed.

## Limitations

The launches were sequential, not interleaved trials. Three prefill samples
describe observed spread, not a confidence interval across restarts. Decode
has one short observation per cell. The DCP2 enabled 8K/C1 boundary reports
575 server tokens and 571 client tokens, within the accepted tolerance.

Fixed-input generated-token comparisons passed the preselected patches-on/off
bound at both DCP settings. A separate DCP2 enabled cached-versus-cold check
exceeded its diagnostic logprob bound: maximum difference 0.127015, despite
identical answer tokens. The disabled DCP2 difference was 0.018938. The passing
matched patch comparisons do not establish cache-invariant probabilities.

The observations do not qualify broad accuracy, full-vocabulary numerical
equivalence, 1M-context requests, mixed scheduling, sustained high concurrency,
other model families, or a newly built image. A source-recipe rebuild needs
its own activation, correctness and timing evidence.
