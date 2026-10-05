# GLM-5.3-Flash NVFP4-Spark on two Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on two Sparks of a four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp2 --yes --json --image-lock LOCK --on 2,3`, from the published one-line command at `main`, installed source commit `2e7e63a75627` on Sparks 2 and 3 of a four-Spark ring, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` as `GLM-5.3-Flash-NVFP4-Spark-TP2`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp2` on installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`, selected by an explicit development image lock (`--image-lock`) with the image ID and receipts that the [installer image lock](../../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json) records; the image was loaded on every Spark before the installation; API port 8000.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 592.4 s ([installer phases](dev-20261004-kraken-glm53-flash-nvfp4-spark-tp2-20261004/install-phases.txt)).
- **Cluster:** Sparks 2 and 3 of a four-Spark ring, selected with `--on 2,3`; the installer ran them as a pair (`direct-pair-2`) over the ring cable that joins them.
- **Other two Sparks:** Sparks 0 and 1 served `mimo-v26-flash-mopd-tp2`, which received no requests during this record's checks, screen and benchmark.
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `9bcc9ca18b19`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.7.6 (the acceptance records of images up to `dev-20261001-kraken-cuda1342-nccl2323-status034` used 0.6.2) at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261004-kraken-glm53-flash-nvfp4-spark-tp2-20261004/functional.txt)). The screen returned 256 responses in 194.8 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20261004-kraken-glm53-flash-nvfp4-spark-tp2-20261004/stress.json)).

**Throughput** (matrices: [run 1](dev-20261004-kraken-glm53-flash-nvfp4-spark-tp2-20261004/tp2-matrix-run1.json), [run 2](dev-20261004-kraken-glm53-flash-nvfp4-spark-tp2-20261004/tp2-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 35.7 / 109.6 / 99.0 | 14.2 / 41.0 / 40.5 | 2.52 / 2.67 / 2.45 | 2,102 / 2,403 / 2,415 |

Benchmark request errors: 0.

## Conclusion

On Sparks 2 and 3 of a four-Spark ring, `install.sh` installed `glm53-flash-nvfp4-spark-tp2`, which served `GLM-5.3-Flash-NVFP4-Spark-TP2`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on two Sparks of a four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
