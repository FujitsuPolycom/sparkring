# Swift 1.5 Qwen3.8-Flash-Next NVFP4 on four Sparks with the installer image

Status: **research-only; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile swift15-qwen38-flash-next-tp4 --yes --json --image-lock LOCK`, from the published one-line command at `main`, installed source commit `2e7e63a75627` on one four-Spark ring, which then served `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` revision `3ff0520224f2` as `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP4`.

## Conditions

- **Profile and image:** `swift15-qwen38-flash-next-tp4` on installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`, selected by an explicit development image lock (`--image-lock`) with the image ID and receipts that the [installer image lock](../../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json) records; the image was loaded on every Spark before the installation; API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. An installation with the same profile and image lock, for the [decode matrices](dev-20261004-kraken-matrix-20261004.md), had started this deployment 2 h 22 min earlier, and it was still serving, so Node 0's API readiness step took 0.5 s ([installer phases](dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `9bcc9ca18b19`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.7.6 (the acceptance records of images up to `dev-20261001-kraken-cuda1342-nccl2323-status034` used 0.6.2) at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004/functional.txt)). The screen returned 256 responses in 84.3 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004/stress.json)).

**Throughput** (matrices: [run 1](dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004/tp4-matrix-run1.json), [run 2](dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 74.1 / 276.5 / 401.7 | 32.6 / 119.4 / 172.9 | 2.28 / 2.32 / 2.32 | 4,788 / 4,582 / 3,825 |

Benchmark request errors: 1, in run 1's 1-stream cell, which is invalid: the benchmark reported "scheduler did not drain after C=1 ctx=0 within 300s: running=1", because one request kept generating for about 7 minutes after the client stopped reading it. The 1-stream values are run 2's alone; run 2 completed every cell without an error.

## Conclusion

On one four-Spark ring, `install.sh` installed `swift15-qwen38-flash-next-tp4`, which served `Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The 1-stream decode figures come from one run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
