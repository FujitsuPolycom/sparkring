# Qwen3.8-Flash-Next step 5500 with MXFP8 attention on two Sparks with the installer image

Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response; measured on one pair; median of 2 benchmark runs; not serving-qualified**.

`install.sh --profile qwen38-flash-next-tp2 --yes --json --checkpoint qad-step5500-mxfp8-attention`, from the published one-line command at `feat/qwen-mxfp8-attention-derived`, installed source commit `f06a1bfa2ee4` on one directly cabled Spark pair, which then served `sparkring-derived/Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention` revision `648b194a96e5` as `Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-TP2`.

## Conditions

- **Profile and image:** `qwen38-flash-next-tp2` on installer image `dev-20261001-kraken-cuda1342-nccl2323-status034`, selected by [`release.json`](../../../runtime/releases/dev-20261001-kraken-cuda1342-nccl2323-status034/release.json); API port 8000.
- **Installation:** result state `complete` with exit status 0; standard output held one `sparkring-install-result/v1` document and nothing else. Node 0's API readiness step took 593 s ([installer phases](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/install-phases.txt)).
- **Derived checkpoint:** both Sparks already held it, so this installation
  verified its 54 files against the
  [manifest](../../../profiles/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/648b194a96e5f130ab62702113242d8e1ddd6e76.json)
  and wrote nothing. An installation on the same pair with the same
  `--checkpoint` wrote it, from source commit `f8bcf73de252` (whose recipe,
  derivation code and checkpoint manifests are byte-identical to
  `f06a1bfa2ee4`'s) on installer image
  `dev-20261001-statusrows-cuda1342-nccl2323-status034`: it hard-linked 48
  files from step 5500 on each Spark, downloaded the 2 donor files (2.6 GiB)
  on Node A in 31.3 s, derived 6 files (5.6 GiB) in 32.1 s and copied them to
  Node 1 over the fabric in 6.0 s, and every file matched the manifest
  ([derivation phases](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/derivation-phases.txt)).
- **Paired control:** right after this run, the same pair, image, source commit
  and harness settings installed and measured stock step 5500
  (`qad-step5500-ple1000`) ([record](dev-20261001-kraken-qwen38-flash-next-tp2-20261002.md)).
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `f06a1bfa2ee4`.

## Measurement

- **Functional checks:** counting, arithmetic and code with the profile's thinking-off request settings (`{"chat_template_kwargs": {"enable_thinking": false}}`), an automatic and a forced tool call, a description of a generated two-color image, then the arithmetic question with the chat template's default thinking, which must return reasoning text. Each check passes or fails on the reply's content.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any other response that misses the expected answer is wrong.
- **Throughput:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's length divided by its time to first token. Steps per second and tokens per step come from vLLM's speculative-decoding counters. The benchmark ran 2 times; the tables give each value's median and the sum of request errors.

## Result

**Correctness.** 7 of 7 functional checks passed ([output](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/functional.txt)). The screen returned 256 responses in 95.9 s: 0 degenerate, 0 failed and 2 wrong, to questions `a7` ([summary](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/stress.json)).

**Throughput** (matrices: [run 1](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/tp2-matrix-run1.json), [run 2](dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002/tp2-matrix-run2.json)):

| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|
| 59.7 / 210.6 / 318.3 | 27.3 / 92.2 / 130.8 | 2.19 / 2.28 / 2.43 | 3,942 / 3,829 / 3,488 |

Benchmark request errors: 0.

README profile table values: decode 1 / 8 / 16 users 59.7 / 211 / 318 tok/s, prefill 64K 3,829 tok/s.

**Against stock step 5500** (medians of 2 runs each, paired control above):

| | Stock step 5500 | MXFP8 attention | Change |
|---|---|---|---|
| Steps/s, 1 / 8 / 16 streams | 23.3 / 85.5 / 122.1 | 27.3 / 92.2 / 130.8 | +17% / +8% / +7% |
| Decode tok/s, 1 / 8 / 16 streams | 53.3 / 197.6 / 301.1 | 59.7 / 210.6 / 318.3 | +12% / +7% / +6% |
| Tokens/step, 1 / 8 / 16 streams | 2.29 / 2.31 / 2.47 | 2.19 / 2.28 / 2.43 | — |
| Prefill 8K / 64K / 128K (tok/s) | 3,744 / 3,665 / 3,348 | 3,942 / 3,829 / 3,488 | +5.3% / +4.5% / +4.2% |

Each MXFP8 run ran more steps per second than each stock run at every
stream count (1: 27.2 and 27.4 against 23.1 and 23.4; 8: 90.2 and 94.1
against 86.4 and 84.6; 16: 132.9 and 128.7 against 122.9 and 121.3). Both
screens missed the same question, `a7`, twice.

## Conclusion

On one directly cabled Spark pair, `install.sh --checkpoint qad-step5500-mxfp8-attention` installed `qwen38-flash-next-tp2`, which served `Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-TP2`: all 7 functional checks passed; a 256-request correctness screen returned no degenerate or failed response. On the installer image it ran 7–17% more verification steps per second and prefilled 4–5% faster than stock step 5500 on the same pair, close to the 14% and 6% (one and eight streams) and 3.5–5.3% that the [research record](../qwen38-flash-next/mxfp8-attention-20261001.md) measured on another image.

## Limitations

- The benchmark ran 2 times on one directly cabled Spark pair. At temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode rates vary between runs.
- Prefill is one cold prompt per length and run.
- The functional checks and the screen test correctness, not output quality against a reference checkpoint. This run did not measure quality; the [research record](../qwen38-flash-next/mxfp8-attention-20261001.md)'s log-likelihood check found mean negative log-likelihood 0.0045 and 0.0063 nats per token higher than step 5500's, about 0.5% in perplexity.
