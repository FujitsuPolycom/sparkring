# MiMo-V2.6-Flash-RL on two Sparks with the installer image

Status: **implemented for text; image input does not work (the model misreads image colors at every size tested); 6 of 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one pair; single-run timing; not serving-qualified**.

`install.sh --profile mimo-v26-flash-rl-tp2 --yes --json`, from a Git bundle on Node A, ref `sync/integration`, installed source commit `e336e2058f51` on one directly cabled Spark pair, which then served `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision `5711b2681699` as `MiMo-V2.6-Flash-RL-TP2`.

## Conditions

- **Profile and image:** `mimo-v26-flash-rl-tp2` on installer image `dev-20260927-b12xcache-cuda1342-nccl2323-status032`, selected by [`release.json`](../../../runtime/releases/dev-20260927-b12xcache-cuda1342-nccl2323-status032/release.json); API port 8020.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 409.9 s ([installer phases](dev-20260927-b12xcache-mimo-v26-flash-tp2-20260927/install-phases.txt)).
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `e336e2058f51`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 6 of 7 functional checks passed; failed: image ([output](dev-20260927-b12xcache-mimo-v26-flash-tp2-20260927/functional.txt)). The screen returned 256 responses in 109.3 s: 0 degenerate, 0 failed and 8 wrong, to questions `a4` ([summary](dev-20260927-b12xcache-mimo-v26-flash-tp2-20260927/stress.json)).

**Throughput** ([matrix](dev-20260927-b12xcache-mimo-v26-flash-tp2-20260927/tp2-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 44.9 / 135.7 / 215.0 | 12.9 / 42.4 / 66.6 | 3.48 / 3.20 / 3.23 | 3,746 / 2,583 / 2,107 |

Benchmark request errors: 0.

**Image input.** The image check failed: asked for the colors of a picture whose left half is red and right half blue, the model answered black and white. Follow-up requests with the same two-color picture at 64×32, 256×128 and 768×384 pixels returned black/white or white/black each time, while the prompt grew by 36, 60 and 316 tokens, so image tokens reach the model and their content is lost. The same check passed on the Qwen, Swift and GLM profiles on this image. Earlier MiMo records tested text only.

**Wrong answers.** All 8 wrong responses in the screen are question `a4`, answered `16` every time: a consistent miss of one question, not corruption.

**Comparison.** The two-Spark MiMo matrix on image `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` ([record](dev-20260927-b12xcache-mimo-v26-20260927.md)) measured 27.1 / 139.0 / 205.6 tokens/s at 12.6 / 42.9 / 64.2 steps/s. The step rates here are equal within 3%; one-stream decode differs by tokens accepted per step (3.48 against 2.14), which follows the sampled text at temperature 1.0.

README profile table values: decode 1 / 8 / 16 users 44.9 / 136 / 215 tok/s, prefill 64K 2,583 tok/s.

## Conclusion

On one directly cabled Spark pair, `install.sh` installed `mimo-v26-flash-rl-tp2`, which served `MiMo-V2.6-Flash-RL-TP2`: 6 of 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one directly cabled Spark pair. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
