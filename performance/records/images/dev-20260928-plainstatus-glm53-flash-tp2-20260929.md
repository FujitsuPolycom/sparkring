# GLM-5.3-Flash NVFP4-Spark on two Sparks with 10 GiB of KV cache per Spark

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one pair; single-run timing; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp2 --yes --json`, from a Git bundle on Node A, ref `feat/glm-kv-capacity`, installed source commit `cb4cd8e4af43`, which was not published; its installer files and `glm53-flash-nvfp4-spark-tp2` profile equal those of the commit that adds this record. It installed on one directly cabled Spark pair, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` as `GLM-5.3-Flash-NVFP4-Spark-TP2`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp2` on installer image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`, selected by [`release.json`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json); API port 8000.
- **Serving settings:** the profile's [`config.json`](../../../profiles/glm53-flash-nvfp4-spark-tp2/config.json) (SHA-256 `c280885d4ad25bdf4b2df32fb03f85a02615fdae8f3a0a3c37de81b47b349432`): a 1,048,576-token context window, 10 GiB of KV cache per Spark (`--kv-cache-memory-bytes 10737418240`) in 2,048-token pages, 8 requests at once, and up to 8 images and 1 video per request. The [GLM memory record](../glm53-flash/installer-memory-20260929.md) measures these settings under image, video and text load.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 181.1 s ([installer phases](dev-20260928-plainstatus-glm53-flash-tp2-20260929/install-phases.txt)).
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) as of main commit `59b253fd`.

## Measurement

- **KV capacity:** the rank-0 engine's startup log line `GPU KV cache size: N tokens`.
- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**KV capacity.** The engine reported `GPU KV cache size: 1,530,566 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.46x`.

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-plainstatus-glm53-flash-tp2-20260929/functional.txt)). The screen returned 256 responses in 185.5 s: 0 degenerate, 0 failed and 1 wrong ([summary](dev-20260928-plainstatus-glm53-flash-tp2-20260929/stress.json)). The wrong response answered question `a4` (x + 2x + 3x = 48) with 6 in round 1; the other seven answers to that question were 8.

**Throughput** ([matrix](dev-20260928-plainstatus-glm53-flash-tp2-20260929/tp2-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 38.0 / 107.7 / 98.3 | 15.0 / 40.5 / 40.3 | 2.54 / 2.66 / 2.44 | 2,131 / 2,507 / 2,481 |

Benchmark request errors: 0. With 8 requests at once, the 16-stream cell queues half of its streams.

## Conclusion

On one directly cabled Spark pair, `install.sh` installed `glm53-flash-nvfp4-spark-tp2` with a 1,048,576-token context window and 10 GiB of KV cache per Spark, which held 1,530,566 tokens. The deployment served `GLM-5.3-Flash-NVFP4-Spark-TP2`: all 7 functional checks passed, and a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one directly cabled Spark pair. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- No request used the full 1,048,576-token context window; the longest prompt was 128K tokens.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint. A temperature-0 answer can change between rounds because batch composition changes the arithmetic order.
