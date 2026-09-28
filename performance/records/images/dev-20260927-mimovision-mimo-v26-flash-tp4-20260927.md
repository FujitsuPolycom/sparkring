# MiMo-V2.6-Flash-RL on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; single-run timing; not serving-qualified**.

`install.sh --profile mimo-v26-flash-rl-tp4 --yes --json`, from a Git bundle on Node A, ref `sync/integration`, installed source commit `6cda5ba79bc1` on one four-Spark ring, which then served `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision `5711b2681699` as `MiMo-V2.6-Flash-RL-TP4`.

## Conditions

- **Profile and image:** `mimo-v26-flash-rl-tp4` on installer image `dev-20260927-mimovision-cuda1342-nccl2323-status032`, selected by [`release.json`](../../../runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/release.json); API port 8020.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 394.3 s ([installer phases](dev-20260927-mimovision-mimo-v26-flash-tp4-20260927/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `6cda5ba79bc1`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260927-mimovision-mimo-v26-flash-tp4-20260927/functional.txt)). The screen returned 256 responses in 73.9 s: 0 degenerate, 0 failed and 8 wrong, to questions `a4` ([summary](dev-20260927-mimovision-mimo-v26-flash-tp4-20260927/stress.json)).

**Throughput** ([matrix](dev-20260927-mimovision-mimo-v26-flash-tp4-20260927/tp4-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 52.9 / 187.7 / 341.6 | 23.4 / 70.2 / 113.9 | 2.26 / 2.67 / 3.00 | 5,041 / 4,076 / 3,355 |

Benchmark request errors: 0.

**Image input.** The image check answered red for the left half and blue for the right. On image `dev-20260927-b12xcache-cuda1342-nccl2323-status032`, whose MiMo vision encoder added its attention sinks to each image's first key, the same check answered black and white; this image applies the sinks in the softmax denominator ([`derive_mimo_vision.py`](../../../runtime/images/derive_mimo_vision.py)).

**Wrong answers.** All 8 wrong responses in the screen are question `a4`, answered `16` every time, as on the parent image.

**Comparison.** The parent image measured 48.5 / 189.0 / 336.1 tokens/s at 23.4 / 71.3 / 113.1 steps/s and prefill 4,064 tokens/s at 64K (installation with extended IPv4 GIDs) ([record](dev-20260927-b12xcache-mimo-v26-20260927.md)). The step rates here are equal within 2%; the vision change does not touch text decoding.

README profile table values: decode 1 / 8 / 16 users 52.9 / 188 / 342 tok/s, prefill 64K 4,076 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `mimo-v26-flash-rl-tp4`, which served `MiMo-V2.6-Flash-RL-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
