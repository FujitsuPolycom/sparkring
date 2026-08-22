# DeepSeek-V4-Flash-0731 width-4096 SIRCL/NCCL comparison

Status: **research-only**. Machine-readable observations are in
[`sircl-width4096-nccl-ab-20260822.json`](sircl-width4096-nccl-ab-20260822.json).

## Conditions

Four directly cabled NVIDIA DGX Sparks served
`deepseek-ai/DeepSeek-V4-Flash-0731` as TP4/DCP1 from the pinned serving image.
Both arms used a 1,048,576-token model limit, 32 maximum sequences, a 4,096-token
batch budget, 16 GiB of `fp8_ds_mla` key-value memory per rank, asynchronous
scheduling, full-input reservation, DSpark K5/B12X, disabled prefix caching,
and the same shared CUDA-capture stream. The only transport activation change
was SIRCL custom width-4096 mode versus disabled mode with patched NCCL fallback.

The SIRCL session used the `tiered_64k` graph kernel on the sequential wire
schedule. Dual-port striping was disabled.

## Measurement

Cold prefill used exact 8K, 64K, and 128K prompts. Coding Peak used the standard
cc1 prompt, temperature 1.0, a 2,000-token ceiling, and five sequential samples
per arm. C1 decode used three separate temperature-1.0 invocations at exact 2K
and 8K contexts.

The long decode workload used 32 fully unique 16K streams, temperature 1.0, a
32,768-token output ceiling, a 2,198,756-token key-value budget, a 600-second
readiness timeout, and a 240-second window after all 32 streams were active with
no queue. Three server-counter observations were retained per arm.

The shared harness changed its client-accounting policy during the A/B/A run.
A later copy was frozen at SHA-256
`07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851`, and one
additional `--isolated-server` C32 run was completed on each transport. Raw
whole-window rates are reported with their speculative acceptance because
acceptance materially changes generated tokens per target verification.

## Result

Prefill was effectively flat: SIRCL versus NCCL was 2,445 versus 2,395 tokens/s
at 8K, 2,364 versus 2,381 at 64K, and 2,172 versus 2,212 at 128K. This is
expected because DeepSeek prefill remains on NCCL in both arms.

Coding Peak favored NCCL. SIRCL measured 92.7 median and 93.4 mean tokens/s;
NCCL measured 95.8 median and 95.2 mean. SIRCL was 3.2% lower by median and 1.9%
lower by mean.

The three initial C32 server rates were 530.1, 567.5, and 546.0 tokens/s for
SIRCL, versus 550.6, 577.8, and 472.9 for NCCL. Their raw means were 547.9 and
533.8 tokens/s, but mean speculative acceptance was 78.95% for SIRCL and 69.70%
for NCCL, so the raw means do not isolate transport. The frozen isolated-server
pair was similarly confounded: 598.8 tokens/s at 84.06% acceptance for SIRCL
and 511.9 tokens/s at 64.69% acceptance for NCCL.

Near-matched ten-second windows isolate the transport more sharply. At mean
acceptance length 4.46 and 69.2% draft acceptance, SIRCL produced 585.1 tokens/s
and NCCL produced 599.6. At mean length approximately 4.85 and draft acceptance
approximately 77.1%, SIRCL produced 636.8 and NCCL produced 653.1. SIRCL was
2.4-2.5% lower in those matched windows. Other nearby windows placed it as much
as 3.0% lower.

Every SIRCL run retained equal published, consumed, and completed native
sequences on all four ranks, zero overflow, no fatal state, and caught-up
replay.

## Conclusion

The width-4096 SIRCL path is functional but does not improve the tested
DeepSeek decode profile. Temperature-one DSpark acceptance dominates
whole-window variance. At near-identical speculative behavior, SIRCL was
approximately 2.4-3.0% slower than patched NCCL; Coding Peak independently
showed a small regression.

Patched NCCL should remain the default for this DeepSeek profile. Further SIRCL
work should target the measured graph-transport overhead before promotion.

## Limitations

The checkpoint revision is not pinned. Temperature-one prompts and samples were
not byte-identical between arms. The harness accounting policy changed during
the sequence, although a frozen isolated-server pair was added. The frozen pair
had very different speculative acceptance and therefore cannot isolate
transport by raw rate. The acceptance-matched observations are ten-second
diagnostic windows rather than independent 240-second repetitions. Results
apply only to the sequential tiered-64K width-4096 graph path on this
four-Spark cycle.
