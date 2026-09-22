# MiMo-V2.6-Flash-RL runtime

Launch inputs for the two MiMo-V2.6-Flash-RL profiles,
[`mimo-v26-flash-rl-tp2`](../../profiles/mimo-v26-flash-rl-tp2/README.md)
(two Sparks) and
[`mimo-v26-flash-rl-tp4`](../../profiles/mimo-v26-flash-rl-tp4/README.md)
(four Sparks). Status: **implemented**; neither profile is qualified.

`XiaomiMiMo/MiMo-V2.6-Flash-RL` is a 309B mixture-of-experts model with 48
decoder layers (9 global-attention layers with 4 KV heads, 39 sliding-window
layers with 8 KV heads and a 128-token window), 192-wide Q/K heads, 128-wide V
heads, fp8 attention weights, MXFP4 experts stored in an fp8 quantization
config, a bundled five-layer DFlash draft (`dflash/`, block size 8) and vision
and audio encoders.

## Contents

| File | Purpose |
|---|---|
| [image.json](image.json) | Runtime contract: published image reference, vLLM build, checkpoint identity and the digest, mount target and selector of every overlay file |
| [launch-pair.sh](launch-pair.sh) | Starts one TP2 rank from a rank-local copy of [pair.env.example](pair.env.example) |
| [launch-ring.sh](launch-ring.sh) | Starts one TP4 rank from a rank-local copy of [ring.env.example](ring.env.example) and a filled [sircl-rank.env.example](sircl-rank.env.example) |
| [fix_dflash_config.py](fix_dflash_config.py) | Writes a valid-JSON copy of a draft configuration whose upstream revision carries a trailing comma |
| `overlay/` | vLLM source files bind-mounted read-only over the image's package |
| `patches/` | The upstream diffs the overlay files apply, for provenance |

## Why an overlay

The published image's vLLM build (`0.26.1rc0+sparkring.native.2160312ffde4`)
loads this checkpoint only with changes that upstream vLLM merged after the
build: fused fp8 QKV sharding by the checkpoint's pre-shard count
([vLLM PR 57508](https://github.com/vllm-project/vllm/pull/57508)), the bf16
MoE router, the draft's attention value scale and the Eagle3 mixin on the
omni class ([vLLM PR 57784](https://github.com/vllm-project/vllm/pull/57784)).
Two further files are performance changes: the split-KV dispatch for
multi-row speculative verification batches in the DiffKV kernel
([local-inference-lab/vllm PR 839](https://github.com/local-inference-lab/vllm/pull/839))
and the same rule ported to the symmetric kernel. Multimodal serving needs the
omni class to report encoder token counts and the sink slice per data-parallel
encoder shard. The image entrypoint attests site-packages hashes and rejects
any overlay, so the launchers exec the vLLM CLI directly with the argv the
entrypoint would exec, from a login shell so the image's shell initialization
activates its CUDA forward-compatibility driver (the vision and audio encoders'
FlashAttention-2 kernel is CUDA 13.3 PTX).

A rebuilt image that includes these changes removes the overlay and the
entrypoint bypass; `image.json` records each file's digest so such an image
can state what it absorbed.

## Launcher switches

Both launchers read a rank-local environment file (site addresses,
interfaces, devices, paths and image) and take serving tunables from the
process environment, defaulting to the profile recipe.

| Variable | Pair default | Ring default | Meaning |
|---|---|---|---|
| `MM` | `1` | `1` | `1`: images (3), video (1) and audio (1) per prompt; `image`: images only; `0`: text only |
| `PAD_V` | `1` | `0` | `1`: padded-V symmetric attention kernel; `0`: DiffKV kernel with the PR 839 dispatch |
| `KV_BYTES` | 12 GiB (16 GiB when `MM=0`) | 20 GiB | KV reservation per rank; the target KV cache is bf16, the draft's fp8 |
| `SPEC_TOKENS` | `5` | `5` | Drafted tokens per step |
| `CG_CAP` | `32` | `64` | Full cudagraph capture ceiling in tokens |
| `ROCE_AR` | `1` | not used | RoCEnante one-shot all-reduce up to 2 MB; NCCL above |
| `SIRCL` | not used | `1` | SIRCL graph-only and fused-prefill sessions over the managed mesh; `0` serves on PyNCCL |
| `LINEAR_BACKEND` | `b12x` | `auto` | At TP4 the 3392-wide global-attention QKV slice is not a multiple of 128 |
| `EXTRA_ENV` | empty | empty | `NAME=value,NAME2=value` added to the container environment |

The target KV cache stays bf16 on both topologies: the checkpoint carries no
K/V scales and an fp8 target cache repeats itself on long outputs at both
greedy and sampled decoding. Global and sliding layers plus the draft give
mixed KV page sizes, which only the block-outermost `BLHNC` layout expresses.
Ahead-of-time compilation of the language-model forward is disabled whenever
multimodal inputs are enabled: the compiled function is specialised on the
text-only dummy run and fails on the multimodal profiling run.
