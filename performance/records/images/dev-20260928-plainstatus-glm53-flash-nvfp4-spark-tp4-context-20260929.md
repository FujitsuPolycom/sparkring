# GLM-5.3-Flash NVFP4-Spark on four Sparks: decode and prefill by context

Status: **implemented; measured on one four-Spark ring; three runs per cell; not serving-qualified**.

## Conditions

- Profile `glm53-flash-nvfp4-spark-tp4` and installer code of `main` at `721db585`; two reinstalls used
  `59b253fd`, which differs only in `README.md`. Installer image
  [`dev-20260928-plainstatus-cuda1342-nccl2323-status033`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json).
- Checkpoint `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4`, served as `GLM-5.3-Flash-NVFP4-Spark-TP4`.
- KV cache: 2,173,412 tokens, from the `GPU KV cache size` line of the rank-0 startup log. The profile serves at most 16 requests at once.
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
| 8K | 62.9 (60.8–64.0) | 91 (90–91) | 135 (129–138) | 195 (194–197) | 266 (260–269) | 3,809 (3,800–3,828) |
| 16K | 62.0 (57.5–65.2) | 91 (91–92) | 137 (131–143) | 199 (190–204) | 261 (256–264) | 3,745 (3,745–3,745) |
| 32K | 63.2 (63.0–63.8) | 90 (90–92) | 133 (132–135) | 195 (189–200) | 267 (258–281) | 3,683 (3,660–3,698) |
| 64K | 63.0 (60.4–64.5) | 92 (89–95) | 135 (134–136) | 195 (192–197) | 264 (262–266) | 3,610 (3,557–3,638) |
| 128K | 62.4 (59.5–64.8) | 88 (86–90) | 131 (127–134) | 190 (183–198) | 255 (252–258) | 3,512 (3,511–3,514) |

— : the cell exceeds the KV cache and was skipped.

Coding peak: 81.3 tok/s mean over 5 of 5 runs (79.4–82.6).
Tool calling: 94/100 (82 passed, 2 partial, 4 failed), responsiveness 59/100 (median turn 2.4s), deployability 84/100.

Files: [matrices](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/run1-matrix.json) (runs [1](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/run1-matrix.json), [2](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/run2-matrix.json), [3](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/run3-matrix.json)), [coding peak](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/coding-peak.json), [tool calling](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/tool-eval.json).

![Decode and prefill by context](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/matrix.png)

![Tool-calling results](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-context-20260929/tool-calls.png)

## Conclusion

On one four-Spark ring, `glm53-flash-nvfp4-spark-tp4` decoded 62.0 / 137 / 199 / 261 tok/s at 1 / 4 / 8 / 16 users with 16K tokens of context each, and prefilled a cold 64K-token prompt at 3,610 tok/s. These are the README profile table's values.

## Limitations

- One cluster. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs; the brackets give the spread of three runs.
- Prefill is one cold prompt per length and run.
- Throughput only: the correctness checks of each profile are in its acceptance records.
