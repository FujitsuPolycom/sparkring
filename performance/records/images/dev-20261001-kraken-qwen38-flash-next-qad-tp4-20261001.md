# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; single-run timing; not serving-qualified**.

`install.sh --profile qwen38-flash-next-qad-tp4 --yes --json --image-lock LOCK`, from the published one-line command at `main`, installed source commit `cb3716018d2d` on one four-Spark ring, which then served `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision `60215d26cf5e` as `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`.

## Conditions

- **Profile and image:** `qwen38-flash-next-qad-tp4` on installer image `dev-20261001-kraken-cuda1342-nccl2323-status034`, selected by an explicit development image lock (`--image-lock`) with the image ID and receipts that the [installer image lock](../../../runtime/releases/dev-20261001-kraken-cuda1342-nccl2323-status034/installer-image.json) records; the image was preloaded on Node A and relayed to the other Sparks; API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 404.5 s ([installer phases](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261001/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `5925dfa71d34`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261001/functional.txt)). The screen returned 256 responses in 85.2 s: 0 degenerate, 0 failed and 1 wrong, to questions `a7` ([summary](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261001/stress.json)).

**Throughput** ([matrix](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261001/tp4-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 80.9 / 280.9 / 416.9 | 33.4 / 118.8 / 177.7 | 2.42 / 2.36 / 2.35 | 4,810 / 4,564 / 4,094 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 80.9 / 281 / 417 tok/s, prefill 64K 4,564 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `qwen38-flash-next-qad-tp4`, which served `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
