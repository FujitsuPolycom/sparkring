# DeepSeek-V4.1-Flash-1017 on four Sparks with the installer image (`--checkpoint 1017`)

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`sudo sparkring install --profile deepseek-v41-flash-tp4 --checkpoint 1017` installed `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision `1bae0e108899` on one four-Spark ring, which then served it as `DeepSeek-V4.1-Flash-1017-TP4`. The acceptance harness then measured that deployment without installing again.

## Conditions

- **Profile and image:** `deepseek-v41-flash-tp4`, checkpoint `1017`, on installer image `dev-20261013-dcp2prefill-status038-sircl-tp6` (image ID `sha256:3cc9a81fd344…`, as the [installer image lock](../../../runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/installer-image.json) records), the default lock of the installed source commit (no `--image`); SIRCL transport on the cycle of four; API port 8015.
- **Installation:** package built from source commit `b1bd448691ce` with `scripts/build_deb.py` on Node A; `sudo sparkring install --profile deepseek-v41-flash-tp4 --checkpoint 1017 --model-path DIR --max-images 16 --yes`, where `DIR` held an identical copy of the revision on each Spark (every file checked against the pin manifest by SHA-256, then hard-linked; nothing downloaded). `--max-images 16` replaced the profile's 4 images per prompt. This was the revision's first start on this image: Node 0's API readiness step took 1,023.0 s, most of it compiling into the revision's own cache directory; vLLM reported a KV cache of 11,290,009 tokens.
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `b1bd448691ce`.
- **Dashboard:** quiet mode was on while the harness measured, so the SparkRing dashboard sent no request to the model.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.7.6 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-1017-20261017/functional.txt)). The screen returned 256 responses in 83.6 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-1017-20261017/stress.json)).

**Throughput** (matrices: [run 1](dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-1017-20261017/tp4-matrix-run1.json), [run 2](dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-1017-20261017/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 61.2 / 194.9 / 275.2 | 26.1 / 84.8 / 119.0 | 2.34 / 2.31 / 2.34 | 4,248 / 4,279 / 4,057 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 61.2 / 195 / 275 tok/s, prefill 64K 4,279 tok/s.

**Against the original checkpoint.** The [record of the default checkpoint](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004.md) (revision `dba1be0a40aa`, same harness and llm-inference-bench 0.7.6, image `dev-20261004-kraken-cuda1342-nccl2323-status034`) measured decode 59.9 / 179.3 / 258.3 tok/s and 64K prefill 3,798 tok/s after its first start, and 56.5–62.1 / 198.1–198.2 / 269.7–273.4 tok/s and 4,260–4,436 tok/s on a second installation. The values above fall within or above those ranges. The images differ, so this does not isolate the checkpoint.

## Conclusion

On one four-Spark ring, `sparkring install --checkpoint 1017` installed `deepseek-v41-flash-tp4` with DeepSeek's 1017 weights, which served `DeepSeek-V4.1-Flash-1017-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint; no task benchmark compared the two revisions.
- `--max-images 16` differs from the profile's default; the other serving settings are the profile's.
- The 1017 release also changed DeepSeek's reference prompt encoder (`encoding/encoding.py`): a user message that holds only tool results no longer starts a new turn. The image's vLLM encoder (`vllm/tokenizers/deepseek_v41_encoding.py`) predates it. With tool definitions in the request the two render the same prompt, because the encoder then keeps earlier reasoning; they differ for tool results sent without tool definitions with thinking on. This record does not cover that case.
