# GLM-5.2 model-down MoE round-floor probe

## Purpose

Determine whether a roughly 100 ms Q5 target round is physically and
architecturally plausible before repeatedly loading the complete 378 GiB
model.

The project has four independent hypotheses:

1. Q5 candidate positions select many of the same experts and can reuse weight
   reads.
2. B12X's existing direct-micro path already captures some of that reuse.
3. eager launches and workspace churn hide otherwise efficient grouped work.
4. the real kernel remains bandwidth-inefficient even after grouping.

No throughput claim follows until each hypothesis has its own measurement.

## Updated boundaries from Q-1 MTP4 and Q-2 MTP5

The current measured target round is **305.6 ms**, of which **291.8 ms is
already inside CUDA graphs**. Only about **13.8 ms (4.5%)** is outside the
captured region. Therefore:

- CPU launch cleanup cannot supply the proposed 100--150 ms reduction;
- eager-versus-graph is a control, not the primary experiment;
- the primary target is graph-internal MoE weight traffic, routing/grouping,
  kernel serialization, and memory-bandwidth efficiency;
- any projected round saving must be bounded by profiler-attributed
  graph-internal time.

The required graph-internal reductions are severe:

| Whole-round target | Saving from 305.6 ms | Fraction of graphed 291.8 ms |
|---:|---:|---:|
| 160 ms | 145.6 ms | 49.9% |
| 140 ms | 165.6 ms | 56.8% |
| 120 ms | 185.6 ms | 63.6% |
| 100 ms | 205.6 ms | 70.5% |

Thus 100 ms requires removing more than two thirds of current graph-internal
execution. Route reuse is worth pursuing only if the route census and kernel
differential expose an opportunity of that order, or if several separately
measured graph-internal optimizations compose to it.

The minimum useful profiler capture must put NVTX ranges around each routed
MoE layer inside graph replay and report LPDDR/DRAM bytes plus duration. A
host-side timing-only result is insufficient.

Q-2 adds an important boundary rather than replacing Q-1. MTP5's full gate
measured 333.05 ms periodic and 333.26 ms novel rounds. A repeated 128-token
timing probe covered only 146.740 ms/round inside the currently instrumented
FULL Q6 wrappers and left <=176.874 ms/round outside that timer. It recorded
exactly two FULL replays per target round with no dropped/error samples.
That remainder includes untimed draft/multistep execution and 160 known
cached-prefill stock calls; it is **not** evidence of 176.9 ms of CPU
overhead. Before projecting a MoE saving, extend instrumentation to every
graph type and MTP phase and repeat the same harness under MTP4.

## Deployed operator audit

The exact Q5/Q6 production path was identified read-only in the live
2026-07-26 image. This changes the next experiment materially.

GLM-5.2 has hidden size 6144, 256 routed experts, top-8 routing, 78 total
layers, three initial dense layers, and therefore 75 routed-MoE layers. The
deployed B12X selector uses:

```text
routed_rows = num_tokens * num_topk
direct-micro when num_tokens <= 8 and routed_rows < 64
dynamic grouped GEMM otherwise
```

Consequently Q5 has 40 routed rows and Q6 has 48: both run the implementation
named `static`, which now means the fused **direct-micro** kernel. The retired
tensor-core static kernel is not involved. Warm-up messages mentioning
`dynamic` do not prove that production Q5/Q6 used the dynamic kernel; vLLM
warms additional signatures deliberately.

The call chain is:

```text
vLLM B12xExperts.apply
  -> _plan_b12x_moe_fp4_scratch
  -> _run_b12x_moe_fp4
  -> build_tp_moe_fp4_binding
  -> b12x_moe_fp4(binding=...)
  -> _launch_compact_static
  -> torch.ops.b12x.tp_moe_compact_micro_launch
  -> MoEMicroKernelSilu.launch
```

The direct-micro FC1 scheduler computes `fc1_task_count = m * topk *
fc1_chunks`, maps tasks to routes in token-major route order, obtains the
expert ID directly from `topk_ids`, and loads that expert's W1 matrices. Its
FC2 path again iterates each token's top-k expert IDs directly. No route sort
or expert-grouped schedule is visible in this path. It does prefetch the next
expert in parts of FC2, but that is not cross-route grouping.

For Q5/Q6, `share_input_across_experts` is false because the implementation
only enables it for `m == 1`. Scalar input/output scales can enable
`share_expert_scales`, but that does not share expert weights.

Pinned source fingerprints:

| Component | SHA-256 |
|---|---|
| vLLM `b12x_moe.py` | `d534632c7aa8ee64334cfce51c946ebf2c805cd17e46993f6f6305df4cd2fda4` |
| B12X `integration/tp_moe.py` | `98f5b8b3cea77ef71253450cca412d3f6e79b95587edb184445274545dd76b27` |
| B12X `moe/fused/micro.py` | `ca1126ba045ee82084d7abaff531186a5e111ac1b8f14656198c9ecbd6f867f4` |
| B12X `moe/fused/silu.py` | `d43f95d6ea8a12e6f6c942ba92794e7c1c363bb945274123019e9ff4726691e2` |

This is encouraging but not proof of a win. Direct-micro may beat dynamic
precisely because grouping overhead and padded expert tiles dominate at 40--48
rows. The required model-down comparison is therefore actual direct-micro
against forced dynamic and a purpose-built expert-coherent micro schedule.

## Implemented: GPU-free route census

`route_reuse.py` consumes JSONL records with this compact canonical shape:

```json
{
  "schema": "glm52-target-expert-routes/v1",
  "request_key": "salted-id",
  "round": 0,
  "layers": [
    {
      "layer": 0,
      "positions": [
        {"expert_ids": [1, 4, 7, 9, 12, 18, 31, 44]}
      ]
    }
  ]
}
```

For a Q5 run, every layer needs five `positions`. Expert identity is scoped to
its layer: expert 7 in layer 10 and expert 7 in layer 11 are different weight
matrices and are never counted as reusable.

Run:

```powershell
python spark_transport/experiments/moe_round_floor/route_reuse.py `
  target-routes.jsonl --width 5
```

The conservative decision uses the p10 reuse factor across rounds:

- `>=1.8`: GO; substantial weight reuse exists;
- `1.3--1.8`: MEASURE; useful, but not independently transformative;
- `<1.3`: NO-GO for expert coherence as the primary route to 100 ms.

This is a potential-reuse measurement, not a speedup estimate. The subsequent
kernel differential determines how much reuse the current implementation has
already captured.

The analyzer also reports `logical_schedule_compaction`: token-major expert
runs divided by unique experts. It prices how much a perfectly expert-grouped
logical schedule could compact the route stream. It is deliberately labelled
as neither a cache-miss count nor a speedup estimate because direct-micro
strides work across resident CTAs and the memory system may already retain
some repeated weight lines.

## Implemented scaffold: exact B12X model-down floor

`b12x_floor_benchmark.py` is a throwaway, fail-closed benchmark scaffold
around the exact B12X binding contract inside the deployed vLLM image. It does
not substitute a generic GEMM. It reuses `TPMoEFP4Binding`, caller-owned
scratch, fixed output tensors, prepared NVFP4 weights/scales, preallocated CUDA
events, and deterministic inputs/routes.

The four pinned B12X/vLLM source hashes above are hard live-mode gates. Plan
and dry-run modes work on Windows without Torch, B12X, CUDA, or a Spark:

```powershell
python spark_transport/experiments/moe_round_floor/b12x_floor_benchmark.py `
  --mode dry-run
```

Live mode is model-down only. Run it inside the pinned container on one Spark
after vLLM is stopped:

```bash
python spark_transport/experiments/moe_round_floor/b12x_floor_benchmark.py \
  --mode live \
  --source-root /opt/venv/lib/python3.12/site-packages \
  --output /tmp/glm52-b12x-floor.json
```

Live mode requires Linux, CUDA SM12.1, exact source hashes, and the
`TPMoEScratchCaps -> plan_tp_moe_scratch -> plan.bind ->
b12x_moe_fp4(binding=...)` ABI. It launches the direct and forced-dynamic
groups in separate child interpreters so the B12X cutover environment is set
before importing B12X. Every binding is checked after construction:
direct-micro must report the deployed implementation name `static`, and the
forced group must report `dynamic`.

Synthetic GLM dimensions are hidden size 6144, 256 experts, top-8, and a
TP4-local intermediate size of 512 (global 2048 / TP4). The source format is
`modelopt_nvfp4`, the quant mode is `nvfp4`, and the W13 layout is `w31`.
Weights use a deterministic CUDA generator and occupy about 1.27 GiB before
any B12X-prepared representation. Scratch, outputs, bindings, graphs, and CUDA
events are all created before timing.

The implemented matrix is:

- GLM hidden size, 256 experts, top-8 routing, and the deployed quant format;
- random/deterministic quantized weights in a rotating arena larger than
  relevant caches;
- captured Q5 expert routes;
- fixed workspaces and no allocator calls in the timed interval;
- CUDA-event timing and profiler-visible NVTX ranges.

Required columns:

| Case | Meaning |
|---|---|
| `5xQ1-eager` | repeated-streaming/launch control |
| `Q5-direct-micro-eager` | deployed tiny-decode implementation |
| `Q5-direct-micro-graph` | production-like graph-internal baseline |
| `Q6-direct-micro-eager/graph` | MTP5 Q6 equivalents |
| `Q5-forced-dynamic-graph` | cost/benefit of existing expert grouping |
| `Q6-forced-dynamic-graph` | MTP5 grouped equivalent |
| `Q5/Q6-identical-route-graph` | optimistic physical ceilings |
| `Q5/Q6-coherent-micro-graph` | explicit fail-closed placeholders |

The JSON includes CUDA-event min/median/mean/p90/p99, output digests and
cross-case numerical deltas, allocator live-byte deltas, exact dispatch names,
source audit, platform identity, and guardrails. A live-byte delta is not an
allocation-event count. LPDDR bytes, achieved bandwidth, and kernel count
still require Nsight Compute/Systems around the emitted
`glm52-moe-floor:<case>` NVTX ranges.

First compare `Q5/Q6-direct-micro-graph` with forced dynamic. Only replace the
coherent-micro placeholder if the route census and memory counters show
repeated weight reads the current kernel does not retain. `--request-coherent`
intentionally exits with a machine-readable fail-closed result. Eager
measurements merely bound the now-small host/launch component. A route census
alone cannot distinguish an available optimization from one B12X already
obtains through cache behavior.

## Synthetic round composition

After measuring the real operator, extend `../model_loop_replay` rather than
creating another transport harness. One synthetic target round should replay
78 calibrated layer blocks and the measured production collective census.

Report:

```text
MoE kernel total
attention/other calibrated GPU work
exposed SparkRing custom-transport time (not sum of standalone calls)
host/launch gaps
whole CUDA-event round time
```

The 100 ms gate is:

```text
78 * measured optimized layer cost
+ measured non-MoE kernel cost
+ exposed communication
<= 100 ms
```

Do not use a no-op collective or zero-duration compute block: either changes
the dependency structure and produces a false floor.

## One required loaded-model artifact

The only immediate full-load dependency is a bounded route capture:

- 100--500 Q5 and/or Q6 target rounds, with the width recorded explicitly;
- all 75 routed-MoE layers;
- top-8 target expert IDs for every verification position;
- exact per-round rejected-token and accepted-prefix counts from the
  source-pinned V2 sampler seam;
- stratified code, prose, reasoning, JSON, and adversarial prompts;
- source/image/checkpoint/config fingerprints.

Capture routes immediately after `FusedMoERouter.select_experts()`. After the
single-request V2 sampler returns, associate its GPU `num_rejected` scalar
with the latest completed route slot in the preallocated sidecar. Copy both
arenas only at drain. Never call `.item()`, allocate, synchronize, or write
files inside the timed path.

Once saved, kernel and synthetic-round work can proceed model-down. The full
model returns only for correctness and final end-to-end validation.

## Status / breadcrumb

As of 2026-07-26:

- route-census implementation and unit tests exist;
- Q-1 MTP4 evidence placed 291.8/305.6 ms inside timed FULL graphs;
- Q-2 MTP5 measured 17.487 tok/s periodic, 15.532 tok/s novel, 333 ms
  rounds, useful 67.53% P4 acceptance, and 750.535 tok/s uncached prefill;
- Q-2's repeated timing probe covered 146.740/323.614 ms in the currently
  timed FULL Q6 wrappers; the unexplained boundary is now explicit;
- the deployed ABI and kernel dispatch are pinned: Q5/Q6 use fused
  direct-micro, not dynamic grouped GEMM;
- no real target-route trace has been captured;
- the model-down B12X binding/timing scaffold and Windows dry-run gate exist;
- no kernel timing or 100 ms feasibility result has been measured.

The next operator should capture Q5/Q6 target routes, replace the synthetic
routes for a second pass, and run the scaffold model-down. Then collect LPDDR
bytes and kernel counts for direct-micro versus forced dynamic. Coherent-micro
is admitted only if both route reuse and the memory-counter differential pass.
