# MiMo-V2.6-Flash-MOPD on two Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on two Sparks of a four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile mimo-v26-flash-mopd-tp2 --yes --json --image-lock LOCK --on 0,1`, from the published one-line command at `main`, installed source commit `2e7e63a75627` on Sparks 0 and 1 of a four-Spark ring, which then served `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` revision `2479e2d0029e` as `MiMo-V2.6-Flash-MOPD-TP2`.

## Conditions

- **Profile and image:** `mimo-v26-flash-mopd-tp2` on installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`, selected by an explicit development image lock (`--image-lock`) with the image ID and receipts that the [installer image lock](../../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json) records; the image was loaded on every Spark before the installation; API port 8020.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 462 s ([installer phases](dev-20261004-kraken-mimo-v26-flash-mopd-tp2-20261004/install-phases.txt)).
- **Cluster:** Sparks 0 and 1 of a four-Spark ring, selected with `--on 0,1`; the installer ran them as a pair (`direct-pair-2`) over the ring cable that joins them.
- **Other two Sparks:** the installation of `glm53-flash-nvfp4-spark-tp2` on Sparks 2 and 3, including its checkpoint verification and model start, began 46 s before this record's correctness screen ended and continued through both of this record's benchmark runs.
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `9bcc9ca18b19`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.7.6 (the acceptance records of images up to `dev-20261001-kraken-cuda1342-nccl2323-status034` used 0.6.2) at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261004-kraken-mimo-v26-flash-mopd-tp2-20261004/functional.txt)). The screen returned 256 responses in 102.8 s: 0 degenerate, 0 failed and 8 wrong, to questions `a4` ([summary](dev-20261004-kraken-mimo-v26-flash-mopd-tp2-20261004/stress.json)).

**Throughput** (matrices: [run 1](dev-20261004-kraken-mimo-v26-flash-mopd-tp2-20261004/tp2-matrix-run1.json), [run 2](dev-20261004-kraken-mimo-v26-flash-mopd-tp2-20261004/tp2-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 29.4 / 114.7 / 205.1 | 13.0 / 40.5 / 65.4 | 2.26 / 2.83 / 3.13 | 3,654 / 2,801 / 2,162 |

Benchmark request errors: 0.

**Wrong answers.** All 8 wrong responses are question `a4` ("If x + 2x + 3x = 48, what is x?"), answered `16` every time, as on image `dev-20261001-kraken-cuda1342-nccl2323-status034` ([record](dev-20261001-kraken-mimo-v26-flash-mopd-tp2-20261001.md)).

## Conclusion

On Sparks 0 and 1 of a four-Spark ring, `install.sh` installed `mimo-v26-flash-mopd-tp2`, which served `MiMo-V2.6-Flash-MOPD-TP2`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on two Sparks of a four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
