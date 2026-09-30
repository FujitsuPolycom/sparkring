# DeepSeek-V4.1-Flash on four Sparks: decode and prefill by context

Status: **implemented; measured on one four-Spark ring; three runs per cell; not serving-qualified**.

## Conditions

- Profile `deepseek-v41-flash-tp4` and installer code of `main` at `721db585`; two reinstalls used
  `59b253fd`, which differs only in `README.md`. Installer image
  [`dev-20260928-plainstatus-cuda1342-nccl2323-status033`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json).
- Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa`, served as `DeepSeek-V4.1-Flash-TP4`.
- KV cache: 13,032,722 tokens, from the `GPU KV cache size` line of the rank-0 startup log. The profile serves at most 16 requests at once.
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
| 8K | 62.3 (61.3–63.7) | 97 (95–98) | 135 (134–137) | 205 (204–207) | 280 (277–282) | 4,482 (4,407–4,521) |
| 16K | 62.2 (58.9–64.5) | 95 (92–99) | 135 (130–140) | 206 (204–207) | 278 (276–280) | 4,497 (4,352–4,579) |
| 32K | 65.0 (61.2–67.4) | 93 (92–95) | 135 (133–136) | 204 (201–205) | 278 (275–282) | 4,434 (4,387–4,510) |
| 64K | 64.4 (61.5–65.9) | 94 (86–100) | 132 (131–133) | 200 (197–202) | 276 (267–282) | 4,302 (4,271–4,333) |
| 128K | 61.9 (60.2–62.9) | 96 (90–104) | 136 (133–139) | 200 (195–203) | 271 (270–272) | 4,086 (4,064–4,114) |

— : the cell exceeds the KV cache and was skipped.

Coding peak: 99.9 tok/s mean over 5 of 5 runs (98.1–101.5).
Tool calling: 93/100 (79 passed, 6 partial, 3 failed), responsiveness 67/100 (median turn 1.9s), deployability 85/100.

Files: [matrices](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/run1-matrix.json) (runs [1](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/run1-matrix.json), [2](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/run2-matrix.json), [3](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/run3-matrix.json)), [coding peak](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/coding-peak.json), [tool calling](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/tool-eval.json).

![Decode and prefill by context](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/matrix.png)

![Tool-calling results](dev-20260928-plainstatus-deepseek-v41-flash-tp4-context-20260929/tool-calls.png)

## Conclusion

On one four-Spark ring, `deepseek-v41-flash-tp4` decoded 62.2 / 135 / 206 / 278 tok/s at 1 / 4 / 8 / 16 users with 16K tokens of context each, and prefilled a cold 64K-token prompt at 4,302 tok/s. These are the README profile table's values.

## Limitations

- One cluster. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs; the brackets give the spread of three runs.
- Prefill is one cold prompt per length and run.
- Throughput only: the correctness checks of each profile are in its acceptance records.
