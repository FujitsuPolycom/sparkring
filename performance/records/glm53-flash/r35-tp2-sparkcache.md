# GLM-5.3-Flash R35 TP2 measurements

GLM-5.3-Flash NVFP4-Spark passed bounded serving, cache and decode checks on two
GB10 nodes using local image
`sha256:7b698d4299aaaebb359e287d75c7f18275311b6a6d56322d9767b0b4f35cd60b`.
The [measurement record](r35-tp2-sparkcache.json) contains the source artifact
hash, configuration and per-cell values. This is not a long-duration stability
or completed 1M-request qualification. The measured image is available through
the [R35 publication record](../../../runtime/images/sparkring-r35/publication.json).

The configuration uses TP2/DCP1, MTP3, FP8 KV, SparkCache, continuation-prefill
coalescing and mHC prefill sharding. It permits 1M context, with a computed 1.1M
KV pool, 7.5 GiB KV per rank, eight sequences and two OpenMP threads.

## Decode

Each cell is one 60-second measurement with llm-decode-bench 0.6.2, temperature
1 and an 8K output limit. Values are aggregate output tokens/s. Short requests
have no added context padding; they are not empty prompts.

| Context | One request | Four concurrent | Eight concurrent |
|---|---:|---:|---:|
| Short | 31.73 | 69.63 | 96.15 |
| 16K | 32.26 | 69.71 | 103.34 |

All six cells had zero API errors and queue fraction, matched the requested
concurrency, and passed warmup, capacity and loop checks. These are single
measurements without a matched R33 baseline. Verifier counters in the JSON
record describe aggregate request steps, not whole GPU batch forward passes.

## Cache and feature checks

Three exact arithmetic checks passed. An 8K lookup passed cold and warm; after
both model processes restarted, both workers restored its snapshot and the API
reported 8,192 cached tokens with zero cache tokens recreated. Four concurrent
requests sharing the cached data but asking different arithmetic questions all
returned their distinct correct answers and reused 8,192 tokens.

Logs show RoCEnante all-reduce/all-gather, B12X KDA, coalesced checkpoints at
6,144 and 7,936 within an 8,192-token span, and mHC on both ranks: 8,192 input
rows, 4,096 owner rows, 90 reduce-scatter calls, 90 all-gather calls and no
auxiliary gathers.

## R35 compatibility

The tested R35 arguments omit `--gdn-decode-kernel b12x` and retain
`--kda-prefill-backend b12x`. Under R35, the former option selects GDN prefill
metadata that rejects GLM's geometry. The published R33 launcher is unchanged;
its arguments cannot be assumed compatible merely by replacing its image.
The [R35 image recipe](../../../runtime/images/sparkring-r35/README.md) records
the pinned composition.

## Retained failed fixture

A changed-tail fixture that retained a conflicting answer instruction produced
one clarification response and one output-budget exhaustion among four requests.
The data-only shared prefix with a single arithmetic question per suffix passed
all four strict checks. That was a prompt correction, not an image fix.
