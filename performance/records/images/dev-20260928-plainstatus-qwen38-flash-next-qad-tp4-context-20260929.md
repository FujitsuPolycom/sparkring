# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks: decode and prefill by context

Status: **implemented; measured on one four-Spark ring; three runs per cell; not serving-qualified**.

## Conditions

- Profile `qwen38-flash-next-qad-tp4` and installer code of `main` at `721db585`; two reinstalls used
  `59b253fd`, which differs only in `README.md`. Installer image
  [`dev-20260928-plainstatus-cuda1342-nccl2323-status033`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json).
- Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision `60215d26cf5e`, served as `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`.
- KV cache: 3,131,214 tokens, from the `GPU KV cache size` line of the rank-0 startup log. The profile serves at most 16 requests at once.
- Client: a separate machine on Node A's network.

## Measurement

- **Decode and prefill:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2, three runs of the same matrix: 1, 2, 4, 8 and 16 concurrent
  streams at 8K, 16K, 32K, 64K and 128K tokens of context, temperature 1.0, exact token
  targeting, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence
  ignored. The benchmark received the KV cache size above and skipped cells whose streams
  need more, concurrency × (context + 2,048) tokens. Decode is the aggregate output rate;
  prefill is one cold prompt of each length divided by its time to first token.
- **Coding peak:** the same benchmark's coding probe, one stream writing a Sieve of Eratosthenes
  program, five runs of up to 2,000 tokens at temperature 1.0.
- **Tool calling:** tool-eval-bench Hard Mode, 88 scenarios, pinned revision `6be685f0`, run by
  the benchmark's own settings (temperature 0, one request at a time).

## Result

Decode (tok/s): mean of the three runs, lowest and highest run in
brackets. Prefill (tok/s) per context length.

| Context | 1 user | 2 users | 4 users | 8 users | 16 users | Prefill |
|---|---|---|---|---|---|---|
| 8K | 64.8 (61.9–67.8) | 108 (105–112) | 169 (167–171) | 230 (223–235) | 340 (335–344) | 4,738 (4,724–4,766) |
| 16K | 64.9 (63.4–66.8) | 104 (101–109) | 167 (165–170) | 235 (229–239) | 341 (334–345) | 4,855 (4,832–4,878) |
| 32K | 64.6 (63.0–66.8) | 104 (100–108) | 164 (158–170) | 232 (226–235) | 340 (335–347) | 4,777 (4,766–4,789) |
| 64K | 63.3 (62.4–64.6) | 105 (102–108) | 166 (163–169) | 227 (224–229) | 332 (330–335) | 4,524 (4,505–4,539) |
| 128K | 68.1 (66.9–68.7) | 119 (115–124) | 185 (183–187) | 262 (256–270) | 373 (360–384) | 4,073 (4,068–4,078) |

— : the cell exceeds the KV cache and was skipped.

Coding peak: 102.3 tok/s mean over 5 of 5 runs (93.6–108.5).
Tool calling: 91/100 (76 passed, 8 partial, 4 failed), responsiveness 62/100 (median turn 2.2s), deployability 82/100.

Files: [matrices](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/run1-matrix.json) (runs [1](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/run1-matrix.json), [2](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/run2-matrix.json), [3](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/run3-matrix.json)), [coding peak](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/coding-peak.json), [tool calling](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/tool-eval.json).

![Decode and prefill by context](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/matrix.png)

![Tool-calling results](dev-20260928-plainstatus-qwen38-flash-next-qad-tp4-context-20260929/tool-calls.png)

## Conclusion

On one four-Spark ring, `qwen38-flash-next-qad-tp4` decoded 64.9 / 167 / 235 / 341 tok/s at 1 / 4 / 8 / 16 users with 16K tokens of context each, and prefilled a cold 64K-token prompt at 4,524 tok/s. These are the README profile table's values.

## Limitations

- One cluster. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs; the brackets give the spread of three runs.
- Prefill is one cold prompt per length and run.
- Throughput only: the correctness checks of each profile are in its acceptance records.
