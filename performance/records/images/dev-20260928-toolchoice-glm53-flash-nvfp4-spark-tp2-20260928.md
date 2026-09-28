# GLM-5.3-Flash NVFP4-Spark on two Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one pair; single-run timing; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp2 --yes --json`, from a Git bundle on Node A, ref `sync/next`, installed source commit `f2f2c9cdcb96`, which was not published; its profile and installer files equal those of commit `a1b4ba75`, on one directly cabled Spark pair, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` as `GLM-5.3-Flash-NVFP4-Spark-TP2`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp2` on installer image `dev-20260928-toolchoice-cuda1342-nccl2323-status032`, selected by [`release.json`](../../../runtime/releases/dev-20260928-toolchoice-cuda1342-nccl2323-status032/release.json); API port 8000.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 310.6 s ([installer phases](dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp2-20260928/install-phases.txt)).
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `cc6a7875`; this record was drafted from its saved outputs at commit `d2da1b66`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp2-20260928/functional.txt)). The screen returned 256 responses in 218.9 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp2-20260928/stress.json)).

**Cache pages.** The profile uses 1,024-token split cache pages; vLLM reported a KV cache of 747,630 tokens, 2.85 requests at the full 262,144-token context ([page-size record](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928.md)). The pair profile serves at most 8 requests, so 16 streams queue behind 8.

**Throughput** ([matrix](dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp2-20260928/tp2-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 38.0 / 107.7 / 96.8 | 14.6 / 41.2 / 39.4 | 2.60 / 2.62 / 2.46 | 1,362 / 2,419 / 2,156 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 38.0 / 108 / 97 tok/s, prefill 64K 2,419 tok/s.

## Conclusion

On one directly cabled Spark pair, `install.sh` installed `glm53-flash-nvfp4-spark-tp2`, which served `GLM-5.3-Flash-NVFP4-Spark-TP2`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one directly cabled Spark pair. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
