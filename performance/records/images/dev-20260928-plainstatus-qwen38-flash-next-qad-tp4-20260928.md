# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; single-run timing; not serving-qualified**.

`install.sh --profile qwen38-flash-next-qad-tp4 --yes --json`, from the published one-line command at `main`, installed source commit `ec407a50f020` on one four-Spark ring, which then served `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision `60215d26cf5e` as `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`.

## Conditions

- **Profile and image:** `qwen38-flash-next-qad-tp4` on installer image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`, selected by [`release.json`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json); API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 336.4 s ([installer phases](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-20260928/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `ec407a50f020`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-20260928/functional.txt)). The screen returned 256 responses in 84.8 s: 0 degenerate, 0 failed and 2 wrong, to questions `a7` ([summary](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-20260928/stress.json)).

**Wrong answers.** Both wrong responses are question `a7`, answered `5`, as on the earlier installer images.

**Throughput** ([matrix](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-20260928/tp4-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 79.6 / 276.2 / 394.0 | 33.4 / 119.0 / 172.3 | 2.39 / 2.32 / 2.29 | 4,724 / 4,515 / 4,049 |

Benchmark request errors: 0.

The step rates match the [record on `dev-20260927-b12xcache-cuda1342-nccl2323-status032`](dev-20260927-b12xcache-qwen38-flash-next-tp4-20260927.md) (33.6 / 119.1 / 172.6 steps/s) within 0.6%; the decode differences follow tokens per step.

README profile table values: decode 1 / 8 / 16 users 79.6 / 276 / 394 tok/s, prefill 64K 4,515 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `qwen38-flash-next-qad-tp4`, which served `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
