# MiMo-V2.6-Flash-MOPD on four Sparks: decode and prefill by context

Status: **implemented; measured on one four-Spark ring; three runs per cell; not serving-qualified**.

## Conditions

- Profile `mimo-v26-flash-mopd-tp4` and installer code of `main` at `721db585`; two reinstalls used
  `59b253fd`, which differs only in `README.md`. Installer image
  [`dev-20260928-plainstatus-cuda1342-nccl2323-status033`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json).
- Checkpoint `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` revision `2479e2d0029e`, served as `MiMo-V2.6-Flash-MOPD-TP4`.
- KV cache: 2,185,934 tokens, from the `GPU KV cache size` line of the rank-0 startup log. The profile serves at most 16 requests at once.
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
| 8K | 50.7 (46.4–59.1) | 81 (69–89) | 127 (117–142) | 175 (165–195) | 287 (267–299) | 3,981 (3,971–4,002) |
| 16K | 47.7 (46.1–50.2) | 90 (71–121) | 119 (110–133) | 167 (157–181) | 295 (280–307) | 4,539 (4,520–4,559) |
| 32K | 47.4 (43.3–55.1) | 72 (65–78) | 111 (103–119) | 174 (169–184) | 278 (272–283) | 4,530 (4,529–4,530) |
| 64K | 40.1 (38.0–42.9) | 62 (59–65) | 106 (103–112) | 176 (163–183) | 267 (251–285) | 3,959 (3,957–3,964) |
| 128K | 31.4 (29.6–34.6) | 61 (59–62) | 90 (79–102) | 143 (137–146) | 253 (247–256) | 3,331 (3,330–3,333) |

— : the cell exceeds the KV cache and was skipped.

Coding peak: 98.3 tok/s mean over 5 of 5 runs (89.4–105.2).
Tool calling: 90/100 (74 passed, 11 partial, 3 failed), responsiveness 78/100 (median turn 1.3s), deployability 86/100.

Files: [matrices](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/run1-matrix.json) (runs [1](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/run1-matrix.json), [2](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/run2-matrix.json), [3](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/run3-matrix.json)), [coding peak](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/coding-peak.json), [tool calling](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/tool-eval.json).

![Decode and prefill by context](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/matrix.png)

![Tool-calling results](dev-20260928-plainstatus-mimo-v26-flash-mopd-tp4-context-20260929/tool-calls.png)

## Conclusion

On one four-Spark ring, `mimo-v26-flash-mopd-tp4` decoded 47.7 / 119 / 167 / 295 tok/s at 1 / 4 / 8 / 16 users with 16K tokens of context each, and prefilled a cold 64K-token prompt at 3,959 tok/s. These are the README profile table's values.

## Limitations

- One cluster. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs; the brackets give the spread of three runs.
- Prefill is one cold prompt per length and run.
- Throughput only: the correctness checks of each profile are in its acceptance records.
