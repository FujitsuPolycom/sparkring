# Four-Spark DeepSeek SIRCL A/B plan

Status: **research-only; live-validated on one four-Spark appliance**. This
plan prepares a matched transport experiment for four directly cabled NVIDIA
DGX Sparks. It does not qualify SIRCL or the DeepSeek profile and does not
provide an execution command that contacts a host. The bounded live validation
is recorded in
[`sircl-width4096-live-validation-20260822.md`](../performance/records/deepseek-v4-flash/sircl-width4096-live-validation-20260822.md).

The patched-NCCL control and SIRCL candidate use the same serving contract. The
plan validates model, scheduler, memory, and speculation fields against
`recipes/deepseek-v4-flash-0731.json`, then applies the experiment's explicit
4,096-token batch budget to both arms:

- TP4 across four ranks and DCP1;
- DSpark with five draft tokens and the B12X MoE backend;
- 1,048,576 maximum model tokens;
- 32 maximum sequences;
- 4,096 maximum batched tokens;
- 17,179,869,184 key-value-cache bytes per rank;
- `fp8_ds_mla` key-value representation; and
- the immutable serving-image digest in `runtime/faststart-lock.json`.

Generate the offline plan:

```bash
python scripts/deepseek_sircl_ab_plan.py \
  --output build/deepseek-sircl-ab-plan.json
```

The command reads local files and writes one JSON document. It has no execute
mode. The generated Docker argument arrays contain placeholders and must not be
treated as an authorized deployment.

Before any live action, bind the completed patched-NCCL benchmark receipt and
the exact harness bytes into the experiment record, resolve all four rank
environments, build and hash the six SIRCL runtime mount inputs, and obtain
explicit authorization to stop the control and start the candidate.

## Candidate environment

Copy the canonical per-rank environment and the research overlay separately:

```bash
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-0.env
cp scripts/config/deepseek-v4-flash-0731-sircl-research.env.example \
  /path/to/rank-0-sircl.env
```

Repeat for ranks 1 through 3. The candidate uses both files; Docker applies the
research overlay after the canonical environment. Resolve each direct peer
address against the device named on the same line. Do not reuse the routed
`VLLM_HOST_IP` as both SIRCL peers unless that address is actually assigned on
both direct links.

Both generated commands contain identical read-only placeholders for two vLLM
CUDA-capture overlays, the native SIRCL library, and three Python runtime
modules. Mounting the same bytes in both arms keeps transport activation in the
research environment overlay as the only A/B difference. Replace every
`/path/to/sircl-runtime/` source with an immutable file built from the checkout
and record its SHA-256. These mounts are launch inputs; the pinned serving image
is not claimed to contain their bytes.

The candidate changes only transport activation. Eager prefill, width-256
drafter collectives, and unsupported signatures remain on patched NCCL. The
width-4096 target-verification collectives are admitted only during CUDA graph
capture through the existing maximum-capacity `[Q <= 512, 4096]` session.
Five draft tokens plus one target row produce six verification rows per active
sequence, so the 32-sequence quickstart contract requires at most Q192.

## Required live evidence

A later authorized deployment must retain, for every rank:

- the resolved Docker argument array and both environment files;
- image ID and manifest digest;
- model revision and file hashes;
- graph status before and after the measured request window;
- captured-node and published/consumed/completed sequence counts;
- zero overflow sequence;
- speculative draft and acceptance counters;
- bounded output comparison against the patched-NCCL control; and
- container and host health after the run.

Only a matched A/B with identical prompts, seeds, request ordering, cache
state, warmup, and serving contract can support a performance conclusion.

## Sustained-decode measurement

The generated plan reproduces the accepted sustained-decode method rather
than a finite-request latency benchmark:

- each concurrency level runs in a separate harness invocation;
- contexts are exactly 2,048 and 8,192 total chat tokens, not prompt text
  estimated from character count;
- every cell measures 90 seconds after a hidden ten-second C1 warmup at the
  largest requested context;
- every stream has a fully unique context;
- requests use temperature zero, ignore EOS, and permit up to 8,192 output
  tokens so completions do not drain the cell;
- queueing, underfilled concurrency, errors, capacity rejection, warmup
  timeout, and missing hardware samples reject a cell; and
- the headline uses client token timing only when it covers enough of the
  measured window. Otherwise it uses the exact Prometheus generation-token
  delta over the post-warmup 90-second wall window.

The control and candidate must report the same headline source for a paired
cell. A client-timestamp result and a Prometheus-fallback result are separate
measurements even when both have tokens-per-second units. The harness file
must be retained with its version, Git revision, and SHA-256 because local
uncommitted changes are part of the measurement identity.

## TP2-aligned C32 comparison

The generated plan also contains one distinct comparison invocation matching
the established two-Spark base workload: one exact 16K context, C32,
temperature 1.0, a 240-second measurement window, 32,768 maximum output tokens,
a 2,198,756-token key-value budget, and a 600-second readiness timeout. The
large output ceiling keeps admitted streams alive while all 32 unique prompts
finish prefill. Measurement is valid only after the scheduler reports all 32
requests running simultaneously and no request waiting.

This invocation is not part of the 2K/8K validation matrix. Compare it only to
another 16K/C32 receipt using the same timing, output ceiling, temperature,
key-value budget, prompt construction, and readiness gate.
