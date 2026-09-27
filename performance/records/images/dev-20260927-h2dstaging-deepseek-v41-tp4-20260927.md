# DeepSeek-V4.1-Flash on four Sparks with the installer image

Status: **research-only; measured on one four-Spark ring; not
serving-qualified**.

The `deepseek-v41-flash-tp4` profile serves `deepseek-ai/DeepSeek-V4.1-Flash`
revision `dba1be0a40aa` as four tensor-parallel ranks on image
`dev-20260927-h2dstaging-cuda1342-nccl2323-status031`. On GB10 the image's
vLLM runs the model through its native B12X path: B12X sparse MLA attention,
the DSA indexer, MXFP4 routed experts on the `b12x` MoE backend (vLLM does not
select it automatically, so the profile names it), the checkpoint's own FP8
quantization and DSpark speculative decoding.

| Setting | Value | Effect |
|---|---|---|
| Engram tables | `--engram-config '{"table_memory":"disk","disk_resident_scales":true}'` | The two Engram n-gram tables (94.6 GiB each) stay in the checkpoint's last two shards; each step reads the rows it needs with io_uring. Their scale bytes stay in host RAM, 1.4 GiB per rank |
| `VLLM_DS41_ENGRAM_OVERLAP=0` | Engram rows are read before the step | See [Engram overlap](#engram-overlap) |
| `VLLM_DS41_MARKOV_NVFP4=1`, `VLLM_DS41_DRAFT_NVFP4_HEAD=1` | NVFP4 DSpark draft layers and vocabulary head | Single-stream decode 3-8% faster; acceptance unchanged. The target model verifies every draft, so outputs do not depend on the draft's precision |
| Speculative decoding | DSpark, 5 tokens (the checkpoint's `dspark_block_size`), greedy draft, block rejection, adaptive verification off | |
| Sequences and graphs | `--max-num-seqs 16`; CUDA graphs `FULL_AND_PIECEWISE` at every multiple of 5 and 6 tokens up to 96 | |
| `NCCL_IB_EXTENDED_IPV4_GIDS=1` | NCCL publishes all four ring NIC functions | Prefill 4-9% faster from 16K to 128K than with it off |
| Other | Request limit 1,048,576 tokens, 8,192 scheduler tokens, memory utilization 0.83, 256-token pages (128 for the sliding window), prefix caching on | |

Thinking follows the model's chat template, which enables it unless a request
sets `chat_template_kwargs` `thinking` or `enable_thinking` to false.

Two Sparks cannot serve this checkpoint: with the Engram tables on disk a rank
holds 74.6 GiB of weights at TP4 and roughly twice that at TP2, more than one
Spark's 121.7 GiB.

## Method

- **Settings search:** serving containers derived from the installer
  deployment on the same four Sparks: image, network, NCCL and container
  limits from the installer's Compose files, the checkpoint copy already on
  each Spark and the arguments in
  [`variants/`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/variants/),
  one group of settings added per variant.
  [`ds-probe.sh`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/ds-probe.sh)
  ran [`prefill_probe.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/prefill_probe.py)
  (one cold prompt of about 8K, 16K, 32K, 64K and 128K tokens, one output
  token, then 16K and 64K again),
  [`decode_probe.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/decode_probe.py)
  (512-token greedy decode by prompt type, two runs) and
  [`ngram_probe.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/ngram_probe.py)
  (16K prompts of new and repeated random text).
- **Matrices:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting,
  20 s per cell after a 5 s warm-up, up to 2,048 output tokens. Prefill is one
  cold prompt's length divided by its time to first token. Decode cells give
  aggregate output tokens/s, then verification steps/s × tokens per step.
- **Correctness:** [`stress.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/stress.py)
  sends rounds of 32 requests (24 short questions with known answers, 8
  questions about a code hidden in about 6K tokens) through 16 threads and
  counts degenerate (one word repeated 8 or more times), wrong and failed
  responses.

## Settings search

Prefill is the first pass, tokens/s. Decode is end-to-end tokens/s, one
stream. Each row after the third adds only its own settings to the Engram
overlap off row, which uses the b12x loader.

| Variant | Prefill 8K | 16K | 32K | 64K | 128K | Decode prose / code / JSON | KV pool (tokens) |
|---|---:|---:|---:|---:|---:|---|---:|
| Engram overlap on, safetensors loader | 1,035 | 1,924 | 2,706 | 3,328 | 3,634 | 48.6 / 94.0 / 105.0 | 14,360,826 |
| Engram overlap on, b12x loader | 989 | 1,944 | 2,720 | 3,199 | 3,549 | 48.7 / 94.6 / 104.3 | 12,807,123 |
| **Engram overlap off** | **3,859** | **4,445** | **4,460** | **4,358** | **3,960** | **46.5 / 96.8 / 104.7** | 13,111,795 |
| + extended GIDs off | 3,403 | 4,062 | 4,126 | 4,027 | 3,815 | 46.1 / 96.7 / 105.8 | 12,923,304 |
| + pages 128 / 64 | 3,896 | 4,045 | 4,436 | 4,340 | 4,055 | 47.5 / 93.7 / 103.6 | 13,198,705 |
| **+ NVFP4 draft** | 3,225 | 4,437 | 4,444 | 4,334 | 4,092 | **50.4 / 99.9 / 108.9** | 12,388,178 |
| + 16,384 scheduler tokens | 4,088 | 3,496 | 4,435 | 4,352 | 4,091 | 48.4 / 91.1 / 102.9 | 7,431,109 |
| + 16 sequences (graphs to 48 tokens) | 3,540 | 4,424 | 4,437 | 4,353 | 4,092 | 47.1 / 93.2 / 103.6 | 12,467,771 |
| + Engram scales on disk | 3,380 | 4,296 | 4,305 | 4,161 | 3,944 | 48.4 / 93.3 / 103.8 | 14,510,301 |

The first 8K prompt after each start includes warm-up. The first two rows
differ only in the weight loader; they differ by up to 4% in prefill and 1% in
decode.
MemAvailable was 8-11 GiB per rank while serving; weights took 74.6 GiB per
rank.

### Engram overlap

With `VLLM_DS41_ENGRAM_OVERLAP=1`, the image's default, a side stream reads
each step's Engram rows while the target graph starts, and the Engram layer
waits for them with a kernel that gives up after 5 s without raising an error.
In that mode each first-pass prompt took 4-5 s longer than the repeat of the
same length (16K: 8.7 s against 3.7 s; 64K: 21.0 s against 17.0 s). With
overlap off the first pass matched the repeat, and 16K prompts of new text
prefilled at the same rate as a repeat of seen text (4,468-4,524 tokens/s),
so the rows are not served from a cache. The profile reads the rows before
each step, which also never proceeds without them.

## Sixteen sequences

The two candidates differ only in `--max-num-seqs` and the largest CUDA graph
(48 or 96 tokens). No added context:

| Streams | 8 sequences | 16 sequences |
|---:|---:|---:|
| 1 | 58.1 (24.0 × 2.42) | 51.9 (24.4 × 2.12) |
| 2 | 80.4 (34.5 × 2.33) | 78.7 (35.3 × 2.23) |
| 4 | 110.3 (48.0 × 2.30) | 105.5 (48.4 × 2.18) |
| 8 | 139.3 (63.6 × 2.19) | 145.9 (63.0 × 2.31) |
| 16 | 142.9 (60.0 × 2.38) | 216.4 (97.0 × 2.23) |

Verification steps per second match up to 8 streams; at temperature 1.0 the
tokens per step follow the generated text. Sixteen sequences add 51% at 16
streams, and the profile uses them. Sixteen sequences compile kernels for more
batch shapes: the first 16-sequence start took 539 s to initialize the engine,
against 270-310 s for 8-sequence starts whose kernels were already compiled.

## Comparison with `deepseek-v41-flash-cycle`

The [cycle profile's record](../deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md)
measured the same checkpoint on the same kind of ring with a locally built
image, packed Engram shards and SparkRing's NCCL 2.30.7, with its own prompt
sets. The conditions differ, so these values are not a matched comparison:

| | `deepseek-v41-flash-cycle` | This profile |
|---|---|---|
| Prefill | 1,873 tokens/s at 16K, 2,058 at 64K (packed Engram shards, from the [profile guide](../../../profiles/deepseek-v41-flash-cycle/README.md)) | 4,445 at 16K, 4,358 at 64K (first pass) |
| One stream | code 81-83, prose 31-33 tokens/s (decode shapes probe) | code 99.9, prose 50.4 tokens/s (greedy, 512 tokens) |
| Eight streams | 88-98 tokens/s aggregate (temperature 1.0, 256-token replies) | 145.9 tokens/s (temperature 1.0, up to 2,048 tokens) |

## Correctness screen

On the first variant (Engram overlap on), 256 requests (8 rounds of
`stress.py`) returned no degenerate, wrong or failed response, and a repeated
20K-token prompt reused 19,712 cached tokens: time to first token fell from
9.26 s to 0.26 s ([`prefix_check.py`](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/programs/prefix_check.py)).

## Files

The [measurement directory](dev-20260927-h2dstaging-deepseek-v41-tp4-20260927/)
holds the variants, the probe output of the settings search
(`tp4-probes.txt`, rank addresses replaced by `r0`-`r3`), both candidate
matrices (`tp4-matrix-8-sequences.json`, `tp4-matrix-16-sequences.json`, with
the benchmark client's host diagnostics and the server address removed) and
the programs.
