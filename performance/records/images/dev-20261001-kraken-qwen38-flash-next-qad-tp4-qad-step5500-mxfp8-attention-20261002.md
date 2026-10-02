# Qwen3.8-Flash-Next step 5500 with MXFP8 attention on four Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one four-Spark ring; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile qwen38-flash-next-qad-tp4 --yes --json --checkpoint qad-step5500-mxfp8-attention`, from the published one-line command at `feat/qwen-mxfp8-attention-derived`, installed source commit `f06a1bfa2ee4` on one four-Spark ring, which then served `sparkring-derived/Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention` revision `648b194a96e5` as `Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-TP4`.

## Conditions

- **Profile and image:** `qwen38-flash-next-qad-tp4` on installer image `dev-20261001-kraken-cuda1342-nccl2323-status034`, selected by [`release.json`](../../../runtime/releases/dev-20261001-kraken-cuda1342-nccl2323-status034/release.json); API port 8015.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 570.1 s ([installer phases](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/install-phases.txt)).
- **Derived checkpoint:** this installation wrote it. Node 0 already held
  step 4000's two donor files, so nothing was downloaded; every Spark
  hard-linked 48 files from step 5500, Node 0 derived the 6 others (5.6 GiB)
  in 39.5 s, and Nodes 1 and 3 received them from Node 0 and Node 2 from
  Node 1 over the fabric in 7.7 to 9.2 s, every file matching the
  [manifest](../../../profiles/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/648b194a96e5f130ab62702113242d8e1ddd6e76.json)
  ([installer phases](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/install-phases.txt)).
- **Paired control:** right after this run, the same ring, image, source
  commit and harness settings installed and measured stock step 5500
  (`qad-step5500-ple1000`) ([record](dev-20261001-kraken-qwen38-flash-next-qad-tp4-20261002.md)).
- **Cluster:** one four-Spark ring (`direct-cycle-4`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `f06a1bfa2ee4`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/functional.txt)). The screen returned 256 responses in 82.9 s: 0 degenerate, 0 failed and 2 wrong, to questions `a7` ([summary](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/stress.json)).

**Throughput** (matrices: [run 1](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/tp4-matrix-run1.json), [run 2](dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002/tp4-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 86.5 / 283.4 / 419.1 | 36.8 / 124.5 / 178.9 | 2.35 / 2.28 / 2.34 | 4,878 / 4,650 / 4,158 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 86.5 / 283 / 419 tok/s, prefill 64K 4,650 tok/s.

**Against stock step 5500** (medians of 2 runs each, paired control above):

| | Stock step 5500 | MXFP8 attention | Change |
|---|---|---|---|
| Steps/s, 1 / 8 / 16 streams | 34.0 / 119.5 / 176.9 | 36.8 / 124.5 / 178.9 | +8% / +4% / +1% |
| Decode tok/s, 1 / 8 / 16 streams | 77.2 / 274.7 / 416.2 | 86.5 / 283.4 / 419.1 | +12% / +3% / +1% |
| Tokens/step, 1 / 8 / 16 streams | 2.27 / 2.30 / 2.35 | 2.35 / 2.28 / 2.34 | — |
| Prefill 8K / 64K / 128K (tok/s) | 4,788 / 4,547 / 4,070 | 4,878 / 4,650 / 4,158 | +1.9% / +2.3% / +2.2% |

Each MXFP8 run ran more steps per second than each stock run at every
stream count (1: 36.8 and 36.8 against 33.9 and 34.0; 8: 122.5 and 126.6
against 120.2 and 118.9; 16: 178.5 and 179.2 against 175.9 and 177.9). At one
stream, the 12% decode gain combines 8% more steps per second with 3.5% more
tokens accepted per step (2.35 against 2.27), which at temperature 1.0
follows the sampled text.
The screens missed question `a7` twice (MXFP8) and three times (stock).

## Conclusion

On one four-Spark ring, `install.sh --checkpoint qad-step5500-mxfp8-attention` installed `qwen38-flash-next-qad-tp4`, which served `Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-TP4`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response. It ran 1–8% more verification steps per second and prefilled about 2% faster than stock step 5500 on the same ring, a smaller gain than the 7–17% and 4–5% measured on two Sparks ([record](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002.md)).

## Limitations

- The benchmark ran 2 times on one four-Spark ring. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint. This run did not measure quality; the [research record](../qwen38-flash-next/mxfp8-attention-20261001.md)'s log-likelihood check found mean negative log-likelihood 0.0045 and 0.0063 nats per token higher than step 5500's, about 0.5% in perplexity.
