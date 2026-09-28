# GLM-5.3-Flash on two Sparks: prefix-cache page size

Status: **implemented; 1,024-token pages chosen for `glm53-flash-nvfp4-spark-tp2`; all 7 functional checks and a 256-request correctness screen passed at 512 and 1,024 tokens; measured on one pair; single-run timing; not serving-qualified**.

GLM-5.3-Flash mixes full attention with linear-attention layers. vLLM keeps one linear-attention state per cached page, so every attention page must be at least as large as that state. With the profile's requested 256-token pages, vLLM raises the attention page to 4,864 tokens on two Sparks. Prefix caching then only reuses whole 4,864-token pages, and multi-token prediction holds back the last one, so a repeated prompt reuses (⌊(P − 1) / B⌋ − 1) × B tokens for a prompt of P tokens and page size B. Prompts under about 10K tokens get no cache hits.

The image's GLM split-page support stores each linear-attention state across several smaller pages. `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE` and `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE` set the split page size, and `--block-size` and `--mamba-block-size` must match it. Smaller pages raise cache hits and cost KV capacity, because each page keeps its own share of the linear-attention state.

## Conditions

- **Image:** `dev-20260927-mimovision-cuda1342-nccl2323-status032`, image ID `sha256:8e4de5f05f02`, selected by [`release.json`](../../../runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/release.json).
- **Profiles:** `glm53-flash-nvfp4-spark-tp2` serving `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` on API port 8000, in three variants:
  - 4,864-token pages: the profile with `--block-size 256` and no split pages.
  - 512-token pages: split pages at 512 tokens, profile revision `0d7bb587`.
  - 1,024-token pages: split pages at 1,024 tokens, profile revision `edee7557`; the settings this profile now uses.
- **Installation:** each variant was installed from a Git bundle of its revision with `sparkring install --profile glm53-flash-nvfp4-spark-tp2`, replacing the deployment before it. The API was ready 258.9 s (512) and 206.5 s (1,024) after the model started.
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`).
- **Client:** a separate machine on Node A's network sent every request.
- **Harness:** [`accept_profile.py`](../../../performance/harnesses/acceptance/accept_profile.py) at commit `a51f55ac`.

## Measurement

- **KV capacity:** the `GPU KV cache size` and `Maximum concurrency for 262,144 tokens per request` lines vLLM logs at startup.
- **Cache hits:** five prompts of about 0.5K, 2K, 8K, 20K and 32K tokens, each sent three times with identical text (`max_tokens` 4, temperature 0, thinking off). The second and third sends report `cached_tokens` in the response usage. For the 4,864-token variant only the second-send cached tokens were kept.
- **Functional checks and correctness screen:** as in the [two-Spark GLM record](dev-20260927-b12xcache-glm53-flash-tp2-20260927.md#measurement): 7 checks, then 8 rounds of 32 requests through 16 threads at temperature 0.
- **Throughput:** llm-inference-bench `llm_decode_bench.py` 0.6.2 at temperature 1.0, 1, 8 and 16 concurrent streams, 20 s per cell after a 5 s warm-up, up to 2,048 output tokens; prefill is one cold prompt's length divided by its time to first token. Each cell ran once.

## Result

**Capacity and cache hits** ([probe](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/cache-probe.json)):

| Page (tokens) | KV cache (tokens) | Concurrency at 262,144 | Cached on repeat: 2K / 8K / 20K / 32K prompt |
|---|---|---|---|
| 4,864 | 1,190,050 | 4.54× | 0 / 0 / 14,592 / 24,320 |
| 1,024 | 747,630 | 2.85× | 0 / 6,144 / 18,432 / 30,720 |
| 512 | 388,543 | 1.48× | 1,024 / 7,168 / 18,944 / 31,744 |

Every hit matches (⌊(P − 1) / B⌋ − 1) × B. With a warm cache, the 32K prompt's reply time fell from 13.7 s cold to 1.3 s at 1,024-token pages and 0.9 s at 512.

**Correctness.** Both split-page variants passed 7 of 7 functional checks ([512](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-512/functional.txt), [1,024](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-1024/functional.txt)). Each screen returned 256 responses: 0 degenerate, 0 failed and 0 wrong ([512](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-512/stress.json), [1,024](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-1024/stress.json)).

**Throughput** ([512](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-512/tp2-matrix.json), [1,024](dev-20260927-mimovision-glm53-flash-tp2-pages-20260928/pages-1024/tp2-matrix.json)):

| Page (tokens) | Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |
|---|---|---|---|---|
| 512 | 39.0 / 111.1 / 94.5 | 14.9 / 41.6 / 38.8 | 2.62 / 2.67 / 2.44 | 1,964 / 2,305 / 2,296 |
| 1,024 | 40.2 / 115.6 / 102.8 | 14.7 / 41.8 / 40.3 | 2.74 / 2.77 / 2.55 | 1,859 / 2,397 / 2,338 |

Benchmark request errors: 0. The step rates match within 4% at every stream count, so page size does not change decode speed; the decode differences follow tokens per step, which vary with the sampled text at temperature 1.0. The pair profile serves at most 8 requests, so 16 streams queue behind 8.

## Conclusion

1,024-token split pages keep 63% of the 4,864-token KV capacity (747,630 tokens, 2.85 requests at the full 262,144-token context), and a repeated prompt longer than 2K tokens reuses all but its last 1,025 to 2,048 tokens. 512-token pages reuse up to 1,024 more tokens per prompt, which matters mainly below about 4K tokens, and hold about half as many tokens as 1,024-token pages. Decode and prefill speed are the same at both sizes. `glm53-flash-nvfp4-spark-tp2` uses 1,024-token pages.

## Limitations

- One run per variant on one directly cabled Spark pair; the throughput cells ran once.
- The cache probe measures hits on exact repeats, not on shared prefixes of different prompts; shared prefixes reuse the same whole pages.
- The four-Spark profile, `glm53-flash-nvfp4-spark-tp4`, already uses 512-token split pages and was not measured here.
