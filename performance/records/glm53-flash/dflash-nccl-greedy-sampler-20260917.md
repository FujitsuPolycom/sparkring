# GLM DFlash greedy and sampler observations with NCCL

Status: **research-only**. The bounded test reproduced greedy output variation
and post-readiness sampler initialization on four GB10 systems. It does not
resolve [issue #161](https://github.com/FujitsuPolycom/sparkring/issues/161) or
[issue #214](https://github.com/FujitsuPolycom/sparkring/issues/214).

## Conditions

The image was
`sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`,
with vLLM composition `e02b174693e13859de61811b5e8cd13d5308e259` and B12X
`9ae41c5cb9935d740456479954b0089f80bd2ef2`. TP4/DCP4 used fixed DFlash depth
seven, 24 GiB FP8 KV per rank, 16 sequences and 8192-token batches. Patched NCCL
served collectives; SIRCL execution was explicitly disabled. SparkCache was
absent. The existing JIT namespace was reused.

The target was a local runtime conversion of
`local-inference-lab/GLM-5.3-Flash-NVFP4` revision
`520de24eabf507659eaef7c70f14fd584527facc`; the draft was
`incoai/GLM-5.3-Flash-DFlash2` revision
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
The [JSON record](dflash-nccl-greedy-sampler-20260917.json) identifies the exact
local config/index hashes, all seven installed source hashes, neutralized SIRCL
selectors and request parameters. Metadata matched across four ranks; full
weight checksums were not collected, so the converted payload is not asserted
identical to the public revision. Target sampling defaults were null.

## Greedy output

Five identical sequential band-name requests used temperature zero, 120 output
tokens and five top logprobs. All had 26 prompt tokens and ended at the token
limit. Running requests, waiting requests and KV use were zero before and after
every call. Durations were 3.047–3.359 seconds.

All five reasoning/output strings differed. At the first differing token versus
run one, the preceding token strings still matched:

| Run | Token index | Run-one token | Other token | Run-one top-two gap | Other gap |
|---|---:|---|---|---:|---:|
| 2 | 59 | ` Harvest` | ` Orchard` | 0.1875 | 0.375 |
| 3 | 28 | ` think` | ` brainstorm` | 0.625 | 0.125 |
| 4 | 37 | `Band` | `Two` | 0.625 | 0 |
| 5 | 77 | ` W` | ` Ghost` | 0.8125 | 0.125 |

Indices are zero-based; gaps are differences between the API's largest reported
logprobs, in nats. Three comparisons have positive gaps on both sides. This
does not justify a tie-breaking-only patch. Constant concurrency and maximum
draft depth do not guarantee identical GPU row shapes or accepted-draft lengths.
A matched speculation-disabled control and pre-sampling logits/shape trace are
needed. The issue author's withdrawn claim that filters are ignored is not
supported by this test.

## Sampler readiness

The frozen temperature-zero, thinking-disabled readiness sweep completed before
`05:35:42Z`. The maintained `warmup_sampling` client then sent six C1 streamed
requests with explicit unfiltered, temperature-only, top-k, top-p, combined and
seeded-combined settings. All completed 64 tokens, final usage and `[DONE]` in
9.922 seconds altogether.

Every rank reported `_gumbel_sample_kernel` initialization around `05:36:01.5Z`
and `_topk_topp_kernel` around `05:36:03.5Z`, after Docker readiness. The six-case
helper therefore exercises paths omitted by this frozen readiness sweep.
Packaging that helper before readiness is a concrete next step; this external
client test did not change the installed warmup.

The monitor can report disk-cache loading as well as compilation and suppresses
repeated warnings. This warm-cache result does not measure cold compile cost,
prove every sampler backend ran, or qualify concurrent filters and mixed
long/short prefill. The helper retains `jit_coverage_verified=false`.

All temporary GLM containers were stopped. The four exact original Qwen
containers were restored with unchanged Config/HostConfig and a healthy API.
No image, installed source, model file, network setting or public artifact changed.
