# GLM-5.3-Flash NVFP4 QAD checkpoint on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; single-run timing; not serving-qualified**.

`install.sh --profile glm53-flash-nvfp4-spark-tp4 --yes --json`, from a Git bundle on Node A, ref `feat/glm-checkpoint-options`, installed source commit `986378285940` on one four-Spark ring, which then served `local-inference-lab/GLM-5.3-Flash-NVFP4` revision `175ae8ce3b5a` as `GLM-5.3-Flash-NVFP4-QAD-TP4`.

## Conditions

- **Profile and image:** `glm53-flash-nvfp4-spark-tp4` on installer image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`, selected by [`release.json`](../../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/release.json); API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 507.6 s ([installer phases](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvfp4-qad-20261001/install-phases.txt)).
- **Checkpoint:** `--checkpoint nvfp4-qad`: the main model on B12X as with NVFP4-Spark, the draft's MXFP8 experts on the Humming MoE backend, and 37 GiB of KV cache per Spark instead of the profile's 40 GiB.
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `986378285940`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. Each cell ran once.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvfp4-qad-20261001/functional.txt)). The screen returned 256 responses in 121 s: 0 degenerate, 0 failed and 0 wrong ([summary](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvfp4-qad-20261001/stress.json)).

**Throughput** ([matrix](dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvfp4-qad-20261001/tp4-matrix.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 57.7 / 183.6 / 258.5 | 21.3 / 69.3 / 95.5 | 2.71 / 2.65 / 2.71 | 3,591 / 3,603 / 3,425 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 57.7 / 184 / 258 tok/s, prefill 64K 3,603 tok/s.

## Conclusion

On one four-Spark ring, `install.sh` installed `glm53-flash-nvfp4-spark-tp4`, which served `GLM-5.3-Flash-NVFP4-QAD-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response.

## Limitations

- The benchmark ran once on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint.
- Host memory headroom with this checkpoint (free memory after startup, and concurrent image requests as in the GLM memory record) was not measured.
