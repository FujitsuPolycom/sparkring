# Swift 1.5 Qwen3.8-Flash-Next NVFP4 on the two-Spark installer profile

Status: **implemented; functional checks passed; measured on one pair; not
serving-qualified**.

`sudo sparkring install --profile swift15-qwen38-flash-next-tp2` installed
revision `3ff0520224f264a2d0ac4ab56ece8f2f13aadb38` of
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` on one directly cabled Spark pair
and replaced the pair's running `qwen38-flash-next-tp2` deployment. Every
functional check passed, and a 256-request correctness screen returned no
degenerate or failed response.

## Conditions

- **Profile and image:** `swift15-qwen38-flash-next-tp2` on installer image
  `dev-20260927-h2dstaging-cuda1342-nccl2323-status031`. The installed package,
  `sparkring_0.1.0~dev.1790521666+gitbbbd02e538ee`, was built from commit
  `bbbd02e538ee`. That commit's profile, checkpoint pin and launcher files
  equal those of the commit that adds this record, except the generated
  Compose deployment labels.
- **Serving settings:** as the [profile guide](../../../profiles/swift15-qwen38-flash-next-tp2/README.md)
  lists: TP2, 16 sequences, 262,144-token limit, 8,192-token batches, 24 GiB
  FP8 KV cache per rank (2,877,721 tokens), vLLM prefix cache on, SparkCache
  off. The routed experts ran on the B12X NVFP4 MoE backend, the BF16 MTP
  draft experts on FlashInfer CUTLASS, and the BF16 PLE n-gram table was read
  from the checkpoint files each step
  ([startup log excerpt](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/startup-log-excerpt.txt)).
- **Installation:** neither Spark held the checkpoint. Node A downloaded its
  95 required files (173.7 GiB) from Hugging Face in 1,910 s; Node B received
  them over the fabric in 167 s. The previous deployment stopped in 16 s; the
  API was ready 545 s after the model started, including the first start's
  kernel compilation
  ([installer phases](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/install-phases.txt)).
- **Clients:** the correctness programs ran on Node A against its local API;
  the throughput matrix ran on a separate machine on Node A's network.

## Measurement

- **Functional checks:** [`functional_qwen.py`](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/programs/functional_qwen.py)
  asks for counting, arithmetic and code with thinking off, an automatic and a
  forced tool call, and a description of a generated two-color image, then
  asks the arithmetic question with thinking on and requires reasoning text.
  Each check passes or fails on the reply's content.
- **Correctness screen:** [`stress.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/stress.py),
  8 rounds of 32 requests (24 short questions with known answers and 8
  questions about a code hidden in about 6K tokens) through 16 threads with
  thinking off. A response is degenerate when one word repeats 8 or more times
  in a row; other responses that miss the expected answer count as wrong.
- **Prefix cache:** one prompt of 20,075 tokens sent twice; vLLM's usage
  report gives the cached prompt tokens of the second request.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting,
  1, 8 and 16 concurrent streams, no added context and 64K context, 20 s per
  cell after a 5 s warm-up at 64K context, up to 2,048 output tokens with end
  of sequence ignored. Decode is the aggregate output rate from stream usage.
  Prefill is a cold prompt's length divided by its time to first token: the
  64K decode cell's scout request, plus scout-only prompts of 8K and 128K.
  Tokens per step and steps per second come from vLLM's speculative-decoding
  counters. Each cell ran once
  ([raw matrix](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/tp2-matrix.json)).
- **Comparison:** the `qwen38-flash-next-tp2` matrix on the same pair in the
  [Qwen step-5500 record](dev-20260925-qwendecode-qwen-step5500-20260926.md)
  ([raw](dev-20260925-qwendecode-qwen-step5500-20260926/tp2-matrix-installer.json)),
  measured with the same benchmark settings except 17 s cells, on image
  `dev-20260925-qwendecode-cuda1342-nccl2323-status031`.

## Result

**Correctness.** All 7 functional checks passed
([output](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/functional.txt)).
The screen returned 256 responses: 0 degenerate, 0 failed and 2 wrong, both to
question `a7` (the remainder of 1000 divided by 7, answered 5 instead of 6)
([summary](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/stress.json)).
The second prefix-cache request reused 17,088 of its 20,075 prompt tokens
([output](dev-20260927-h2dstaging-swift15-qwen38-tp2-20260927/prefix-check.txt)).
Prefill steps of 2,848 to 6,034 rows logged hyper-connection token-row
ownership (`QWEN_HC_PREFILL mode=shard`).

**Throughput**, aggregate tokens/s across streams:

| Context | Prefill | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| 0 | — | 46.9 | 183.9 | 259.3 |
| 64K | 3,634 | 40.4 | 147.0 | 206.0 |

Scout-only prefill was 3,667 tokens/s at 8K and 3,341 at 128K.

**Comparison with `qwen38-flash-next-tp2`**, no added context:

| Streams | Decode, Swift / Qwen (tok/s) | Tokens per step, Swift / Qwen | Steps per second, Swift / Qwen |
|---:|---|---|---|
| 1 | 46.9 / 56.2 | 2.12 / 2.39 | 22.2 / 23.5 |
| 8 | 183.9 / 196.6 | 2.26 / 2.34 | 81.3 / 84.2 |
| 16 | 259.3 / 283.3 | 2.41 / 2.34 | 107.5 / 121.1 |

At 64K, Qwen prefilled 3,666 tokens/s and decoded 43.3 / 162.4 / 233.1 at 1,
8 and 16 streams.

## Conclusion

On one Spark pair, the one-command installer installs
`swift15-qwen38-flash-next-tp2` from an empty checkpoint directory and serves
it with the Qwen3.8-Flash-Next architecture features active; the functional
checks and the correctness screen pass. At temperature 1.0 without added
context, Swift decodes 17%, 6% and 8% slower than `qwen38-flash-next-tp2` at
1, 8 and 16 streams, and prefills 64K within 1%. At one stream the gap is
mostly fewer accepted draft tokens per step; at 16 streams it is mostly a
lower step rate.

## Limitations

- Each cell ran once. At temperature 1.0, draft acceptance and therefore
  decode rates vary between runs; the Qwen record observed up to 20%.
- The Qwen matrix ran on the parent image with 17 s cells, so the comparison
  crosses one image change and one cell-length change.
- The per-step cost of reading PLE rows from NVMe, the BF16 draft experts and
  the BF16 shared experts is not separated.
- The functional checks and the screen test correctness, not the quality of
  this NVFP4 checkpoint against its BF16 source.
- The four-Spark profile `swift15-qwen38-flash-next-tp4` is not covered.
