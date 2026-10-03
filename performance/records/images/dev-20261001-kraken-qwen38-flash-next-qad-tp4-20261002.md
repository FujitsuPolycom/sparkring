# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile qwen38-flash-next-qad-tp4 --yes --json`, from the published one-line command at `feat/qwen-mxfp8-attention-derived`, installed source commit `f06a1bfa2ee4` on one four-Spark ring, which then served `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision `60215d26cf5e` as `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`.

## Conditions

- **Profile and image:** `qwen38-flash-next-qad-tp4` on installer image `dev-20261001-kraken-cuda1342-nccl2323-status034`, selected by [`release.json`](../../../runtime/releases/dev-20261001-kraken-cuda1342-nccl2323-status034/release.json); API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 197.5 s ([installer phases](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002/install-phases.txt)).
- **Paired run:** the control for the MXFP8-attention checkpoint measured just
  before it on the same ring, image and source commit ([record](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002.md)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `b19499f10466`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002/functional.txt)). The screen returned 256 responses in 85.7 s: 0 degenerate, 0 failed and 3 wrong, to questions `a7` ([summary](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002/stress.json)).

**Throughput** (matrices: [run 1](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002/tp4-matrix-run1.json), [run 2](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 77.2 / 274.7 / 416.2 | 34.0 / 119.5 / 176.9 | 2.27 / 2.30 / 2.35 | 4,788 / 4,547 / 4,070 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 77.2 / 275 / 416 tok/s, prefill 64K 4,547 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `qwen38-flash-next-qad-tp4`, which served `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
