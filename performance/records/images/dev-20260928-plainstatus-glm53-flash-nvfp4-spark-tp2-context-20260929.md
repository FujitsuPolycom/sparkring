# GLM-5.3-Flash NVFP4-Spark on two Sparks: decode and prefill by context

Status: **implemented; measured on one pair of DGX Sparks; three runs per cell; not serving-qualified**.

## Conditions

- Profile `glm53-flash-nvfp4-spark-tp2` and installer code of `main` at `721db585`; two reinstalls used
  `59b253fd`, which differs only in `README.md`. Installer image
  [`dev-20260928-plainstatus-cuda1342-nccl2323-status033`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json).
- Checkpoint `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4`, served as `GLM-5.3-Flash-NVFP4-Spark-TP2`.
- KV cache: 747,630 tokens, from the `GPU KV cache size` line of the rank-0 startup log. The profile serves at most 8 requests at once.
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
| 8K | 37.5 (36.3–38.4) | 56 (53–60) | 78 (74–82) | 104 (99–109) | 81 (69–97) \* | 1,981 (1,971–1,986) |
| 16K | 38.0 (37.6–38.6) | 56 (54–59) | 78 (75–81) | 106 (104–109) | 85 (77–93) \* | 2,252 (2,245–2,255) |
| 32K | 37.7 (36.8–38.4) | 55 (54–56) | 78 (75–83) | 104 (103–106) | 93 (76–111) \* | 2,394 (2,394–2,394) |
| 64K | 37.8 (37.3–38.4) | 54 (52–56) | 76 (73–80) | 99 (97–101) | — | 2,454 (2,451–2,459) |
| 128K | 38.1 (37.5–39.2) | 56 (54–59) | 73 (71–75) | — | — | 2,439 (2,434–2,442) |

— : the cell exceeds the KV cache and was skipped. \* : requests queued: the profile serves at most 8 requests at once, so the cell is capacity-limited.

Coding peak: 45.9 tok/s mean over 5 of 5 runs (42.9–47.4).
Tool calling: 94/100 (80 passed, 5 partial, 3 failed), responsiveness 35/100 (median turn 4.5s), deployability 76/100.

Files: [matrices](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/run1-matrix.json) (runs [1](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/run1-matrix.json), [2](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/run2-matrix.json), [3](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/run3-matrix.json)), [coding peak](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/coding-peak.json), [tool calling](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/tool-eval.json).

![Decode and prefill by context](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/matrix.png)

![Tool-calling results](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp2-context-20260929/tool-calls.png)

## Conclusion

On one pair of DGX Sparks, `glm53-flash-nvfp4-spark-tp2` decoded 38.0 / 78 / 106 / 85\* tok/s at 1 / 4 / 8 / 16 users with 16K tokens of context each, and prefilled a cold 64K-token prompt at 2,454 tok/s. These are the README profile table's values.

## Limitations

- One cluster. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs; the brackets give the spread of three runs.
- Prefill is one cold prompt per length and run.
- Throughput only: the correctness checks of each profile are in its acceptance records.
