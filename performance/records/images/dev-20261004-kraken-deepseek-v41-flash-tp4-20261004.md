# DeepSeek-V4.1-Flash on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile deepseek-v41-flash-tp4 --yes --json --image-lock LOCK`, from the published one-line command at `main`, installed source commit `9bcc9ca18b19` on one four-Spark ring, which then served `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa` as `DeepSeek-V4.1-Flash-TP4`.

## Conditions

- **Profile and image:** `deepseek-v41-flash-tp4` on installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`, selected by an explicit development image lock (`--image-lock`) with the image ID and receipts that the [installer image lock](../../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json) records; the image was loaded on every Spark before the installation; API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 530.8 s ([installer phases](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `9bcc9ca18b19`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.7.6 (the acceptance records of images up to `dev-20261001-kraken-cuda1342-nccl2323-status034` used 0.6.2) at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/functional.txt)). The screen returned 256 responses in 87.6 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/stress.json)).

**Throughput** (matrices: [run 1](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/tp4-matrix-run1.json), [run 2](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 59.9 / 179.3 / 258.3 | 25.2 / 74.7 / 118.4 | 2.39 / 2.42 / 2.28 | 4,482 / 3,798 / 3,797 |

Benchmark request errors: 0.

**Benchmark version.** One run of `llm_decode_bench.py` 0.6.2, the version of the acceptance records of images up to `dev-20261001-kraken-cuda1342-nccl2323-status034`, with the same settings ([matrix](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/tp4-matrix-bench062.json)) decoded 59.0 / 191.6 / 257.5 tok/s at 1 / 8 / 16 streams (-1.5% / +6.9% / -0.3% from the medians above) and prefilled 4,600 / 3,827 / 3,441 tok/s at 8K / 64K / 128K (+2.6% / +0.8% / -9.4%). The two 0.7.6 runs differed from each other by up to 12.0% in decode and 11.1% in prefill.

**First start and a warm comparison.** The runs above followed this profile's first start on this image, which compiled and tuned its kernels (API readiness 530.8 s). They prefilled a 64K-token prompt at 3,782 and 3,813 tok/s and ran 76.2 and 73.1 verification steps per second at 8 streams. Later on the same ring, the harness installed the profile on image `dev-20261001-kraken-cuda1342-nccl2323-status034`, the default of the installed source commit, without an image lock, then again on this image with the same lock, each followed by readiness and two benchmark runs with the settings above; both starts reused compiled kernels (API readiness 278.4 and 278.6 s). Medians of two runs each:

| Installation | Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| `dev-20261001-kraken-cuda1342-nccl2323-status034` ([run 1](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/control-default-tp4-matrix-run1.json), [run 2](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/control-default-tp4-matrix-run2.json)) | 62.1 / 198.2 / 269.7 | 25.5 / 84.8 / 115.4 | 4,620 / 4,436 / 4,180 |
| This image, again ([run 1](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/rerun-tp4-matrix-run1.json), [run 2](dev-20261004-kraken-deepseek-v41-flash-tp4-20261004/rerun-tp4-matrix-run2.json)) | 56.5 / 198.1 / 273.4 | 25.3 / 82.1 / 112.9 | 4,394 / 4,260 / 4,101 |
| This image, first start (above) | 59.9 / 179.3 / 258.3 | 25.2 / 74.7 / 118.4 | 4,482 / 3,798 / 3,797 |

On the second installation this image ran 1–3% fewer verification steps per second than `dev-20261001-kraken-cuda1342-nccl2323-status034` (-0.8% / -3.1% / -2.2% at 1 / 8 / 16 streams) and prefilled 2–5% slower (-4.9% / -4.0% / -1.9% at 8K / 64K / 128K). The first-start runs' lower 64K prefill and 8-stream step rate were not seen on the second installation; their cause was not identified.

## Conclusion

On one four-Spark ring, `install.sh` installed `deepseek-v41-flash-tp4`, which served `DeepSeek-V4.1-Flash-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The runs above followed the profile's first start on this image; a second installation on the same ring measured 1–3% fewer steps per second and 2–5% slower prefill than image `dev-20261001-kraken-cuda1342-nccl2323-status034` (warm comparison above).
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
