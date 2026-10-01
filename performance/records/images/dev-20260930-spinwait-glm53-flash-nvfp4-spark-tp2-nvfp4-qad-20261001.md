# GLM-5.3-Flash NVFP4 QAD checkpoint on two Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one pair; single-run timing; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp2 --yes --json`, from a Git bundle on Node A, ref `feat/glm-checkpoint-options`, installed source commit `ad2a97e2ccab` on one directly cabled Spark pair, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4` revision `175ae8ce3b5a` as `GLM-5.3-Flash-NVFP4-QAD-TP2`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp2` on installer image `dev-20260930-spinwait-cuda1342-nccl2323-status033`, selected by [`release.json`](../../../runtime/releases/dev-20260930-spinwait-cuda1342-nccl2323-status033/release.json); API port 8000.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 336.6 s ([installer phases](dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001/install-phases.txt)).
- **Checkpoint:** `--checkpoint nvfp4-qad`: the main model on B12X as with NVFP4-Spark, the draft's MXFP8 experts on the Humming MoE backend, 5 GiB of KV cache per Spark instead of the profile's 10 GiB, and a 524,288-token context window instead of 1,048,576. vLLM reported a KV cache of 736,274 tokens, 1.40 requests at the full context; model loading took 93.65 GiB on rank 0.
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `ad2a97e2ccab`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001/functional.txt)). The screen returned 256 responses in 198.6 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001/stress.json)).

**Throughput** ([matrix](dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001/tp2-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 30.7 / 93.7 / 95.5 | 12.0 / 36.6 / 38.0 | 2.55 / 2.56 / 2.51 | 1,994 / 2,497 / 2,491 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 30.7 / 94 / 96 tok/s, prefill 64K 2,497 tok/s. The profile serves at most 8 requests at a time, so 16 streams measure 8.

**Node A memory.** After the acceptance run, Node A had 4.34 GiB of `MemAvailable`. One request with 8 unseen random-noise images (29,555 prompt tokens), run by the GLM memory record's [`image_load.py`](../glm53-flash/installer-memory-20260929/programs/image_load.py) under its [`guarded.py`](../glm53-flash/installer-memory-20260929/programs/guarded.py) with a 1.2 GiB floor, took Node A to a low of 2.19 GiB, and it settled at 3.0 GiB; the guard did not fire. The [GLM memory record](../glm53-flash/installer-memory-20260929.md) measured the same step on the NVFP4-Spark pair profile from 4.11 to 1.99 GiB.

## Conclusion

On one directly cabled Spark pair, `install.sh` installed `glm53-flash-nvfp4-spark-tp2`, which served `GLM-5.3-Flash-NVFP4-QAD-TP2`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one directly cabled Spark pair. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
- Concurrent multi-image requests were not run; on the NVFP4-Spark pair profile, two concurrent 8-image requests took Node A below the 1.2 GiB floor.
