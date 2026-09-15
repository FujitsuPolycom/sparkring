# Qwen TP2 complete-checkpoint findings

Status: **Experimental**. These measurements use a source-pinned maintainer
deployment with complete request-boundary SparkCache support. That source branch
and image are not published. The [published quickstart](../../../profiles/qwen38-flash-next-tp2/README.md)
uses aligned caching and is not the measured composition below.

## Conditions and result

Two GB10 nodes; original Qwen3.8-Flash-Next NVFP4 revision
`ada4da32a583a78aa47299f45a70603c950490b8`, not QAD; TP2/DCP1, managed B12X,
MTP3, native 262144 context without YaRN, 16 sequences, 8192 batch, 24 GiB FP8
KV per rank, image 3/video 1. One DAC carries both Socket Direct functions.
The reported logical KV pool is 2954103 tokens, not doubled across TP ranks.

The matched full matrix uses 17-second steady decode windows, temperature 1,
2048 maximum output tokens and identical input/sampling payloads apart from
cache salts. All 32 scored cells across the two arms passed validity checks.

| Context | Cache-on prefill tok/s | C1 decode tok/s | C8 decode tok/s |
|---:|---:|---:|---:|
| 8K | 3102 | 45.79 | 163.90 |
| 16K | 3093 | 48.09 | 182.92 |
| 32K | 2983 | 42.92 | 174.03 |
| 64K | 2807 | 44.45 | 182.25 |
| 128K | 2506 | — | — |

The historical baseline is the `measurements` section of the
[R37 native-cache record](r37-tp2.json). Its text tests used image 1/video 0;
the matched cache-on/cache-off runs here both use image 3/video 1.
Prefill is 0–0.61% below the matched cache-disabled control and 0.71–1.29%
below that historical baseline. Matched C1 normalized decode is 0.40–1.30% lower;
other concurrency results are mixed. The operator accepted this trade-off.
Strict parity is not claimed. All C1/C2/C4/C8 cells, acceptance lengths, control
values, source identities and limitations are in [the measurement record](r37-boundary-tp2.json).
One full pair is not a formal confidence bound; 128K lacks computed-token
corroboration, and the historical sample is not a matched-seed control.

Decode tok/s is aggregate output throughput across the concurrent requests.
Normalized decode divides that rate by effective acceptance length: emitted
tokens per inferred request step, including target-sampled tokens and
non-speculative steps. Its units are inferred request-step equivalents/s,
pooled across requests—not CUDA kernel time or batch-forward frequency.
The measurement record defines the counter-based calculation and every field's units.

## Persistent state and memory

Fresh-salt requests had zero cache credit before a process restart. Identical
request hashes then restored the counts below, with correct answers:

| Request | Prompt / restored tokens | Cold / restored seconds |
|---|---:|---:|
| Text | 5918 / 5918 | 2.516 / 0.937 |
| Text | 50768 / 50768 | 18.234 / 1.562 |
| Three images plus one video | 13677 / 11392 | 10.172 / 1.657 |

Text bundles contain attention, recurrent, MTP and auxiliary continuation state.
Media retains aligned persistence. Changed inputs, a missing rank manifest,
one-rank corruption/recomputation, cancellation/drain and C16 short-answer checks
passed under the recorded conditions. The GPU-free suite passed 1342 tests with
eight skips; Ruff and 82 private diagnostic tests passed.

The explicit host buffers total about 2.75 GiB per rank plus temporary payload
assembly; disk capacity is a separate 4 GiB per rank. A 257468-token cold retrieval
passed all three keys. Minimum host-available memory was 15.64/18.23 GiB, with
less than 1 MiB swap used. This is not a worst-case concurrent-media memory test.

## Tuning decisions and release boundary

The 2 MiB collective cutoff and 8192 batch were retained. Batch 11392 lost
prefill; OMP1 and restricted CPU placement gave no consistent overall win.
The reply-only control did not justify bypassing cache completion acknowledgements.
Recurrent-state kernel fusion and uniform-worklist omission passed component
checks but did not produce consistent full-model gains; neither is retained.

A pull-and-run boundary profile still requires its exact source/image publication,
launcher admission, and packaging verification. Preserve the published aligned
image and evidence. Do not attach these numbers to it or change immutable receipts.
TP4, DCP2, QAD, dynamic speculation, arbitrary video quality and prolonged
store-pressure stability are not qualified by this record.
