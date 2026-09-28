# MiMo-V2.6-Flash-RL on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; single-run timing; not serving-qualified**.

`install.sh --profile mimo-v26-flash-rl-tp4 --yes --json`, from a Git bundle on Node A, ref `sync/next`, installed source commit `b9ebe632f165` on one four-Spark ring, which then served `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision `5711b2681699` as `MiMo-V2.6-Flash-RL-TP4`.

## Conditions

- **Profile and image:** `mimo-v26-flash-rl-tp4` on installer image `dev-20260928-toolchoice-cuda1342-nccl2323-status032`, selected by [`release.json`](../../../runtime/releases/dev-20260928-toolchoice-cuda1342-nccl2323-status032/release.json); API port 8020.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 394.1 s ([installer phases](dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928/install-phases.txt)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `cc6a7875`; this record was drafted from its saved outputs at commit `d2da1b66`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928/functional.txt)). The screen returned 256 responses in 74.3 s: 0 degenerate, 0 failed and 8 wrong, to questions `a4` ([summary](dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928/stress.json)).

**Wrong answers.** All 8 wrong responses are question `a4`, answered `16` every time, as on the published image.

**Tool-result contract.** `api_probe.py` sent named and required `tool_choice` requests, streaming and not, with `max_tokens` 400 and 16: all eight cases passed ([report](dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928/tool-choice-probe.json)). Complete calls returned HTTP 200; a call cut off by the token limit returned HTTP 400, or an SSE error event when streaming.

**Throughput** ([matrix](dev-20260928-toolchoice-mimo-v26-flash-rl-tp4-20260928/tp4-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 47.2 / 197.1 / 339.9 | 23.2 / 71.3 / 111.5 | 2.03 / 2.77 / 3.05 | 4,724 / 4,002 / 3,310 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 47.2 / 197 / 340 tok/s, prefill 64K 4,002 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `mimo-v26-flash-rl-tp4`, which served `MiMo-V2.6-Flash-RL-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
