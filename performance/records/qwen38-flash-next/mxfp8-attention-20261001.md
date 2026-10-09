# MXFP8 attention projections for Qwen3.8-Flash-Next step 5500 on two Sparks, 2026-10-01

Status: **research-only**. One pair, one derived checkpoint that no profile
references, one day.

## Question

The step-5500 checkpoint of local-inference-lab/Qwen3.8-Flash-Next-NVFP4
(branch `qad-step5500-ple1000`, revision `60215d26cf5e`) stores its 240 text
attention projections in BF16: in each of the 36 Gated DeltaNet layers
`linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and
`out_proj`, and in each of the 12 full-attention layers `self_attn.q_proj`,
`k_proj`, `v_proj`, `o_proj` and `indexer.index_qk_proj`. The step-4000
checkpoint (branch `qad-step-4000`, revision `629bc3218833`) stores the same
frozen projections as MXFP8: E4M3 weights with one UE8M0 scale per 32 values.
Serving step 5500 with those projections in MXFP8 halves the bytes they
occupy and that each decode step reads. Does that raise decode speed and free
memory on the `qwen38-flash-next-tp2` profile, and what does it cost in
output quality?

## Conditions

- **Derived checkpoint.**
  [transplant_mxfp8_attention.py](mxfp8-attention-20261001/transplant_mxfp8_attention.py)
  (SHA-256 `802769f24515dd89ef2b3682a54f8c7adcfd92abd2926fd80fc6e5f6e5f39055`)
  reads the step-5500 directory and the step-4000 donor and writes only a new
  output directory. For each projection it quantizes the step-5500 BF16
  weight with vLLM's `_mxfp8_e4m3_quantize_torch` and requires the result to
  equal the step-4000 `weight` and `weight_scale` bytes, then stores the
  step-4000 tensors. Shards holding a projection are rewritten; the others are
  hard-linked; `config.json` and `hf_quant_config.json` gain one
  `{"quant_algo": "MXFP8", "group_size": 32}` entry per projection under
  `quantized_layers`.
  - Donor files, from Hugging Face at revision `629bc3218833`: the index,
    `config.json`, `hf_quant_config.json`, `generation_config.json` and
    `model-00035-of-00036.safetensors`, the one shard holding all 480 donor
    tensors (SHA-256 `94994a8d5484…`, equal to the hash Hugging Face serves).
    The index and `config.json` hashes equal the `qad-step-4000` pins of the
    profile's [config.json](../../../profiles/qwen38-flash-next-tp2/config.json)
    ([donor-files.sha256](mxfp8-attention-20261001/donor-files.sha256)).
  - The program ran in installer image
    `dev-20260930-spinwait-cuda1342-nccl2323-status033` (`sha256:fcb20b0ce839`),
    CPU only and without network, as root: the pinned checkpoint's shards are
    root-owned with mode 0644 and the host sets `fs.protected_hardlinks=1`, so
    another account cannot hard-link them. The files it created were then
    given to the operator account; the hard-linked shards are the pinned
    checkpoint's own files.
- **Serving.** Image `sha256:fcb20b0ce839`, profile `qwen38-flash-next-tp2` of
  SparkRing `9f39a9f6`, on two directly cabled DGX Sparks (spark-b rank 0,
  spark-a rank 1). Three deployments:
  - stock, installer: `install.sh --profile qwen38-flash-next-tp2`;
  - stock, research Compose: each rank's installer Compose file changed by
    [research_compose.py](mxfp8-attention-20261001/research_compose.py)
    (project and container names, the deployment label, no runtime-binding
    file), with the pinned checkpoint mounted;
  - MXFP8 attention: the same research Compose files with the derived
    directory mounted at the same path on both Sparks. Rank 1's copy
    hard-links the 39 unchanged shards to its own pinned copy and holds the
    other files, whose SHA-256 equal rank 0's.

  All three use the installer's cache directory.
- **Kernels.** vLLM's ModelOpt mixed-precision method builds every MXFP8
  linear layer through `init_mxfp8_linear_kernel`, which returns the first
  supported kernel; with `--linear-backend b12x` on SM12x that is
  `B12xMxfp8LinearKernel`, which accepts every shape. Rank 0's log names no
  other MXFP8 linear kernel. B12X's activation mode is `auto` (no override in
  the profile).
- **Decode and prefill.** llm-inference-bench 0.6.2 `llm_decode_bench.py`,
  `--concurrency 1,8 --contexts 0 --duration 20 --decode-warmup-seconds 5
  --max-tokens 2048 --temperature 1.0 --token-targeting exact --no-hw-monitor
  --no-resume`, three runs per deployment between 10:55 and 11:35 local time.
  Steps per second is tokens per second divided by the server's MTP accept
  length. Each run also measures one prefill at 8K, 64K and 128K tokens.
- **Acceptance.**
  [spec_accept_probe.py](mxfp8-attention-20261001/spec_accept_probe.py): vLLM's
  `/metrics` speculative-decoding counters around 48 seeded chat requests at
  temperature 1.0 (24 prompts, two seeds, at most 384 tokens, four in flight),
  twice per deployment.
- **Output quality.**
  [logprob_probe.py](mxfp8-attention-20261001/logprob_probe.py): the
  log-probability of every token of six fixed texts (the first 6,000
  characters of six SparkRing `9f39a9f6` files, 9,708 tokens), from
  `prompt_logprobs` at temperature 0, twice per deployment; mean negative
  log-likelihood (NLL) per token and mean absolute per-token difference.
- **Correctness.** SparkRing's acceptance harness,
  `accept_profile.py --skip-install --steps readiness,functional,stress`:
  seven functional checks and a 256-request screen (16 threads, thinking off).
- **Memory.** `Model loading took` and `GPU KV cache size` from the rank logs;
  `MemAvailable` on each Spark 20 seconds after each research Compose
  deployment reported startup complete. The profile fixes the KV cache at
  24 GiB (`--kv-cache-memory-bytes`).

## Results

Derived checkpoint: the requantization check passed for all 240 projections.
Shards 17 (75 projections) and 18 (165) were rewritten and 39 hard-linked;
the checkpoint holds 100.13 GiB against 102.53 GiB
([derivation.json](mxfp8-attention-20261001/derivation.json),
[derived-own-files.sha256](mxfp8-attention-20261001/derived-own-files.sha256)).
`export-manifest.json` is copied from the source and still describes the
attention projections as BF16; nothing reads it when serving.

Decode, three runs each:

| Deployment | Streams | tok/s, mean (range) | steps/s, mean (range) | accept length, mean (range) |
|---|---|---|---|---|
| Stock, installer | 1 | 52.8 (51.4 – 53.6) | 23.6 (23.6 – 23.7) | 2.24 (2.16 – 2.28) |
| Stock, research Compose | 1 | 56.5 (55.3 – 58.4) | 23.6 (23.4 – 23.8) | 2.39 (2.32 – 2.50) |
| MXFP8 attention | 1 | 62.5 (59.7 – 64.5) | **27.0** (26.8 – 27.1) | 2.32 (2.21 – 2.40) |
| Stock, installer | 8 | 204.7 (199.7 – 212.0) | 85.4 (84.5 – 87.1) | 2.40 (2.36 – 2.43) |
| Stock, research Compose | 8 | 202.5 (197.2 – 209.1) | 85.3 (85.0 – 85.7) | 2.37 (2.32 – 2.45) |
| MXFP8 attention | 8 | 211.7 (209.6 – 214.1) | **90.6** (89.4 – 92.8) | 2.34 (2.26 – 2.39) |

Prefill, tok/s, three runs each:

| Deployment | 8,192 tokens | 65,536 tokens | 131,072 tokens |
|---|---|---|---|
| Stock, installer | 3,727 (3,717 – 3,746) | 3,673 (3,666 – 3,679) | 3,357 (3,355 – 3,359) |
| Stock, research Compose | 3,736 (3,719 – 3,746) | 3,667 (3,660 – 3,670) | 3,356 (3,351 – 3,359) |
| MXFP8 attention | **3,923** (3,912 – 3,942) | **3,806** (3,782 – 3,820) | **3,473** (3,472 – 3,474) |

Memory:

| | Stock | MXFP8 attention |
|---|---|---|
| Model loading, each rank | 55.86 GiB | 54.67 GiB |
| GPU KV cache | 2,877,721 tokens (24 GiB) | 2,877,721 tokens (24 GiB) |
| `MemAvailable`, rank 0 / rank 1 (research Compose) | 19.18 / 21.69 GiB | 20.38 / 22.92 GiB |

Acceptance over the seeded requests:

| Deployment | Pass | Acceptance rate | Accept length | Accepted at draft position 1 / 2 / 3 |
|---|---|---|---|---|
| Stock, installer | 1 | 0.565 | 2.69 | 0.772 / 0.548 / 0.373 |
| Stock, installer | 2 | 0.555 | 2.66 | 0.763 / 0.537 / 0.364 |
| MXFP8 attention | 1 | 0.569 | 2.71 | 0.775 / 0.556 / 0.374 |
| MXFP8 attention | 2 | 0.559 | 2.68 | 0.767 / 0.545 / 0.366 |

Output quality over the 9,708 tokens (stock passes on the research Compose
deployment):

| Comparison | Mean NLL per token, first / second | Difference | Mean absolute per-token difference |
|---|---|---|---|
| Stock pass 1 against stock pass 2 | 1.5651 / 1.5648 | −0.0003 | 0.179 |
| MXFP8 pass 1 against MXFP8 pass 2 | 1.5696 / 1.5711 | +0.0015 | 0.193 |
| Stock pass 1 against MXFP8 pass 1 | 1.5651 / 1.5696 | +0.0045 | 0.202 |
| Stock pass 2 against MXFP8 pass 2 | 1.5648 / 1.5711 | +0.0063 | 0.203 |

In the first paired pass, the MXFP8 NLL was higher on five of the six texts,
by 0.004 to 0.016, and lower on one, by 0.008
([logprob-comparisons.json](mxfp8-attention-20261001/measurements/logprob-comparisons.json)).

Correctness: both passed 7 of 7 functional checks. The 256-request screens had
no degenerate response and no error; stock answered one question wrong and
MXFP8 attention two, all the same question ("the remainder when 1000 is
divided by 7", answered 5), which each of the four screens run on this pair
on 2026-09-30 and 2026-10-01 missed at least once
([screens](mxfp8-attention-20261001/measurements/)).

Per-run values and the SHA-256 of each raw output are in
[results.json](mxfp8-attention-20261001/measurements/results.json); the raw
outputs, rank logs and Compose files stay on spark-b under
`~/mxattn-results/`.

## Conclusion

MXFP8 attention projections make the engine faster: 14% more verification
steps per second with one stream (23.6 to 27.0) and 6% more with eight (85.4
to 90.6), and 3.5 – 5.3% faster prefill, with every run of the derived
checkpoint above every stock run. Both stock deployments measured the same,
so the gain comes from the checkpoint, not from the research Compose files.
Draft acceptance is unchanged, so tokens per second rises by the same
proportion on average; single runs vary with the accept length.

Each rank's weights shrink by 1.19 GiB, and `MemAvailable` rose by 1.2 GiB on
each Spark. The profile fixes the KV cache at 24 GiB, so the KV cache did not
grow; at 2,877,721 tokens per 24 GiB, raising the KV allowance by the freed
1.19 GiB would hold about 143,000 more tokens.

The quality cost is small and measurable: mean NLL rose by 0.0045 and 0.0063
nats per token (about 0.5% in perplexity), against 0.0003 and 0.0015 between
two passes of one deployment. Draft acceptance, the functional checks and the
screens did not separate the two.

Recommendation: worth pursuing as a serving option for this checkpoint. Before
a profile uses it, it needs a pinned, published derived checkpoint (or a
load-time conversion with the same requantization check), a broader quality
evaluation than this probe, and a decision on whether to give the freed memory
to the KV cache.

Limits: one pair, three benchmark runs, two probe passes and two quality
passes per deployment, temperature 1.0 for throughput; quality measured on six
texts only.
