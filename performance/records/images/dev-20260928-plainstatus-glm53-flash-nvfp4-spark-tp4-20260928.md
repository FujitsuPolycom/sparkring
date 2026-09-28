# GLM-5.3-Flash NVFP4-Spark on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp4 --yes --json`, from the published one-line command at `main`, installed source commit `dd2698da08ac` on one four-Spark ring, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` as `GLM-5.3-Flash-NVFP4-Spark-TP4`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp4` on installer image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`, selected by [`release.json`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json); API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 290.8 s ([installer phases](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-20260928/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `dd2698da08ac`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-20260928/functional.txt)). The screen returned 256 responses in 121.8 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-20260928/stress.json)).

**Throughput** (matrices: [run 1](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-20260928/tp4-matrix-run1.json), [run 2](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-20260928/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 64.0 / 205.8 / 283.8 | 25.6 / 76.5 / 102.3 | 2.50 / 2.69 / 2.77 | 3,710 / 3,644 / 3,517 |

Benchmark request errors: 0.

**Step rate at 16 streams.** The two runs measured 104.1 and 100.5 steps/s. The parent image, `dev-20260928-toolchoice-cuda1342-nccl2323-status032`, measured 100.2 in one run ([record](dev-20260928-toolchoice-glm53-flash-nvfp4-spark-tp4-20260928.md)), and `dev-20260927-b12xcache-cuda1342-nccl2323-status032`, which has no supervised peer wait, 106.8 in one run ([record](dev-20260927-b12xcache-glm53-flash-tp4-20260927.md)). Two runs of this image differ by about as much as the images do, so these runs do not resolve a cost of the supervised peer wait; any cost is a few percent at 16 streams.

README profile table values: decode 1 / 8 / 16 users 64.0 / 206 / 284 tok/s, prefill 64K 3,644 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `glm53-flash-nvfp4-spark-tp4`, which served `GLM-5.3-Flash-NVFP4-Spark-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
