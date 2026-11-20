# DeepSeek V4.1 row-owned prefill on four DGX Sparks

**Research-only.** The opt-in prefill path served DeepSeek-V4.1-Flash-1017
on one four-Spark cycle. One cold request at each length measured 5,321,
5,081 and 4,661 prompt tokens/s at 16K, 64K and 128K.

## Conditions

| Setting | Value |
|---|---|
| Hardware and topology | Four GB10 GPUs on a direct cycle, one rank per Spark |
| Runtime | SparkRing installer image `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`, B12X loader, attention and MoE |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Parallelism and transport | TP4, DCP1, SparkRing patched NCCL over RoCE |
| Serving | Eager execution, 262,144-token context limit, 8,192-token batches, 16 sequences, 0.83 GPU memory utilization, DSpark5 with adaptive verification off |
| Prefill plugin | [Row-owned prefill](../../../integrations/vllm/ds41_row_shard/README.md), minimum 512 rows per rank, Engram on disk with a replicated projection |

The mounted plugin files were identified by SHA-256.

| File | SHA-256 |
|---|---|
| `__init__.py` | `7a1be35f46e0bd55cf8d7e745d1f5d645ba5c016414948488d175c37cf4fe447` |
| `core.py` | `e804059c1b77d7dbd4412bdb4e3c0873fda4428b5ec743735de3b60b3998ee01` |
| `source_edit.py` | `51f0ec4c9389a80c96a538e219c52bd620199723ed582ee29e98be9f65c3ffe3` |

## Measurement

A loopback HTTP client tokenized distinct synthetic inventory prompts, then
sent exactly 16,384, 65,536 or 131,072 token IDs to the streaming completions
endpoint. Each timed request used temperature zero, concurrency one and one
output token. The server reported zero cached tokens for all three requests.
The client measured elapsed monotonic time from submission to the first
nonempty text delta. Throughput divides server-reported prompt tokens by
that time.

Engine warmup and the short, 8,192-token and continuation checks completed
before the timed requests. The [raw observations](row-owned-prefill-ring-20261119.jsonl)
retain per-request timing, cache use and per-rank collective counts.

## Result

| Prompt tokens | Client TTFT | Prompt tokens/s |
|---:|---:|---:|
| 16,384 | 3.0794 s | 5,321 |
| 65,536 | 12.8977 s | 5,081 |
| 131,072 | 28.1192 s | 4,661 |

All four ranks logged 2,048 owned rows per 8,192-token chunk, with two
Engram reduce-scatters per forward. The 8,192-token cold request returned
output and its 8,200-token continuation reused 7,936 cached tokens.

## Conclusion

The installed model exercised the row-owned attention output, feed-forward
output and Engram lookup reductions on the four-Spark cycle. One cold
sample completed at each of the three prompt lengths.

## Limitations

Each length has one observation, so run-to-run variability is unknown.
The run used one configuration and covers eager text serving on this
cycle. Matched baseline runs, graph-mode serving, sustained decode,
numerical quality, and chain topology are outside this evidence.
