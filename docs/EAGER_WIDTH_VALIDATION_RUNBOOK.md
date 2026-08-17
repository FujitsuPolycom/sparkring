# Eager width admission validation runbook

Status: leg 2 executed and passed; leg 3 in progress (pythia-70m window
closed, gate fail dispositioned — results below); leg 1 unexecuted. All
executions 2026-08-17. This runbook produces
functional and regression evidence only — no performance or maturity claim.
Every leg is operator-launched. Feature under test: opt-in width-generic
eager TP4 all-reduce admission (`VLLM_SPARK_TP4_EAGER_WIDTHS`), commit
`12a9fbe`, documented in
[`spark_transport/integrations/vllm/README.md`](../spark_transport/integrations/vllm/README.md).

What this runbook does NOT validate: CUDA-graph admission (unchanged,
6144-only, behind the open graph correctness gate), capacity classes /
active-bytes ABI, non-BF16 dtypes, PP6 or any topology beyond the four
directly cabled Sparks, and serving performance.

## Leg 1 — GLM-5.2 regression (production shape, no new widths)

Purpose: prove the refactor is inert for the production configuration.

Two arms, identical serve config otherwise (the currently qualified
`--enforce-eager` TP4 configuration):

- Arm 1a: `VLLM_SPARK_TP4_EAGER_WIDTHS` unset (production default).
- Arm 1b: `VLLM_SPARK_TP4_EAGER_WIDTHS=6144` (explicit; must be
  semantically identical).

Checks per arm:

1. Startup completes; `validate_active_port_namespace` passes (it runs at
   install and fails closed on any port or env problem).
2. Eager all-reduce control ports are unchanged: on each rank,
   `ss -tlnp | grep -E '110[0-9][0-9]'` shows the same
   `11000 + 2*(Q-1)` pairs as before the change.
3. Identical short traffic script against both arms; collective-audit
   snapshots show the same custom/fallback counts arm-to-arm.
4. Generation output sanity (any divergence is a hard stop).

Evidence: startup logs, audit snapshots, `ss` output, git SHA, env dump.

## Leg 2 — native transport probes at new payload sizes

Purpose: validate the width-agnostic native claim on the real links with
no vLLM in the loop. Do not run while the serving stack is up — the
probes occupy the RDMA links.

Instruments (both accept arbitrary `--bytes`; build per
[`spark_transport/README.md`](../spark_transport/README.md)):

- `spark_tp4_probe` — raw byte exchange over the TP4 cycle.
- `spark_tp4_tensor_probe` — BF16 reduce kernel with numerical
  verification (the leg's primary instrument).

Payload matrix (`bytes = Q * W * 2`, covering the leg-3 widths at Q=1 and
Q=6 plus one large point):

| W | Q=1 | Q=6 | large |
|---:|---:|---:|---:|
| 512 | 1024 | 6144 | — |
| 768 | 1536 | 9216 | — |
| 1024 | 2048 | 12288 | — |
| 2048 | 4096 | 24576 | 2097152 (Q512-equivalent) |

Template (per rank, matching `--bytes` on all four ranks; peers, HCAs, and
GID indexes from the deployed topology configuration):

```bash
./spark_tp4_tensor_probe --rank R --peer0 IP0 --peer1 IP1 \
  --device0 HCA0 --device1 HCA1 --gid0 G --gid1 G \
  --control-port0 P0 --control-port1 P1 \
  --bytes B --warmup 50 --iterations 500
```

Pass: all ranks complete every payload point with zero verification
mismatches; latency in family with the 12288-byte baseline (record, do
not gate on, the numbers).

### Executed 2026-08-17 19:37Z — ALL PASS

Operator-launched from CodesPC; stack down, links idle. `spark_tp4_tensor_probe`
from the `sparkring-native-full-20260812T0335Z` build on r0, fanned to
r1/r2/r3. Control TCP over the direct links (192.168.101/102/103/200.x)
because the 192.168.0.x network was unavailable; RDMA devices and GID
indexes per the deployed v23 per-rank env (`rocep1s0f0/f1`, odd ranks
inverted, GID 3). Every rank reported `mismatched_elements=0 correct=true`
at every point. r0 percentiles (µs):

| bytes | width x rows | p50 | p95 | p99 | max |
|---:|---|---:|---:|---:|---:|
| 1024 | 512 x 1 | 14.8 | 16.3 | 21.0 | 705.3 |
| 1536 | 768 x 1 | 17.9 | 19.7 | 25.8 | 532.3 |
| 2048 | 1024 x 1 | 18.2 | 19.5 | 24.1 | 39.5 |
| 4096 | 2048 x 1 | 20.3 | 21.4 | 23.2 | 27.9 |
| 6144 | 512 x 6 | 22.8 | 24.8 | 26.5 | 32.2 |
| 9216 | 768 x 6 | 26.7 | 28.6 | 30.4 | 34.4 |
| 12288 | 6144 x 1 (baseline) | 32.4 | 33.6 | 37.5 | 443.0 |
| 24576 | 2048 x 6 | 48.2 | 50.7 | 53.8 | 55.7 |
| 2097152 | 2048 x 512-equivalent | 2563.3 | 2810.3 | 3067.8 | 5035.4 |

Cross-rank p50 agreement at the baseline was within 0.05 µs
(32.30-32.35 µs across ranks 0-3). Numbers are recorded, not gated;
occasional first-iteration max outliers (up to ~0.8 ms) appeared only at
the smallest payloads and at rank warmup.

## Leg 3 — small-model shadow matrix

Purpose: exercise the widened adapter admission end-to-end through vLLM
dispatch with real models whose hidden widths differ from 6144. Shadow
mode executes the custom transport AND the stock path per eligible
collective and compares elementwise, so model quality is irrelevant; the
models below are chosen for width coverage, TP4 head divisibility, and
download size.

| Model | Width | Heads Q/KV | Arch | Size |
|---|---:|---|---|---:|
| `EleutherAI/pythia-70m` | 512 | 8 MHA | GPTNeoX | ~140 MB |
| `facebook/opt-125m` | 768 | 12 MHA | OPT | ~250 MB |
| `Qwen/Qwen3-0.6B` | 1024 | 16/8 | Qwen3 | ~1.2 GB |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 | 32/4 | Llama | ~2.2 GB |

Per model, identically on ALL FOUR ranks (setting the widths variable
remaps eager all-reduce control ports, so a mixed env fails closed at
session connect):

```bash
export VLLM_SPARK_TP4_MODE=shadow
export VLLM_SPARK_TP4_EAGER_WIDTHS=<model width from the table>
# Deliberately NOT set: SPARK_TP4_SHADOW_PROMOTE, SPARK_TP4_SHADOW_STRICT,
# VLLM_SPARK_TP4_GRAPH_Q1, VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40,
# VLLM_SPARK_TP4_PREFILL_Q512.
# Serve: --enforce-eager --dtype bfloat16 -tp 4 <model>
```

Traffic: batch-1 decode generation until every observed signature
accumulates its 10,000-collective shadow window (~2 eligible all-reduces
per layer per decode step):

| Model | Layers | Decode steps per window |
|---|---:|---:|
| pythia-70m | 6 | ~833 |
| opt-125m | 12 | ~417 |
| Qwen3-0.6B | 28 | ~179 |
| TinyLlama-1.1B | 22 | ~228 |

Watch:

1. The shadow verdict log line per signature at its window boundary.
2. Collective-audit snapshot: admitted `[Q, W]` shapes counted custom
   (shadow-observed); vocabulary/DCP/all-gather traffic recorded as
   stock `ineligible_signature` (expected — only the all-reduce family
   is under test).
3. Worker liveness: an `os._exit(70)` worker death means a native-path
   failure — hard stop, collect logs and the audit snapshot.

Pass per model: every accumulated signature validates
(`outside_tolerance == 0`, `nonfinite_mismatches == 0`); no fallbacks for
admitted shapes; no aborts. One failing signature: stop, save the stats,
report — do not promote, do not retry-loop.

### Executed 2026-08-17 — pythia-70m (width 512), shadow window closed

First non-GLM model ever served by this stack: four-rank TP4, shadow mode,
eager, API healthy, traffic driven per this runbook. The (1, 512) BF16
signature closed its 10,000-collective window on every rank with
bit-identical statistics:

```text
collectives=10000 exact_mismatches=1565908 outside_tolerance=3
nonfinite_mismatches=0 max_abs=0.5 max_ulp=30080
```

Configured gate verdict: FAIL (outside_tolerance must be zero). Operator
disposition (Cody, same day): accepted as explained, not promoted. The
divergence signature is BF16 reduction-order noise at near-cancellation
(SIRCL pairwise tree versus NCCL ring order): three elements of 5.12
million (6e-7) beyond the 1 percent envelope, zero nonfinite, and all four
ranks agree bit-for-bit, which corruption would not. For scale: 99.99994
percent of compared elements sat inside the 1 percent envelope, and the
worst absolute divergence was 0.5 on a bf16 activation. The operator also
notes the practical scope: pythia-70m is a ~140 MB instrument model that
nobody would serve across four Sparks; it is in this matrix purely as the
fastest width-512 traffic source. pythia-70m is a
research-grade model with documented extreme activation outliers that make
near-cancellation unusually common. The signature remains unpromoted; the
matrix proceeds to properly trained models (opt-125m, Qwen3-0.6B,
TinyLlama) to determine whether any outlier appears at all under sane
activation statistics.

Launch-shape findings recorded during bring-up, each required for non-GLM
serving from the deployed image: the entrypoint's LD_PRELOAD compat-libcuda
and baked-env unset list must be replicated when bypassing it; follower
ranks require --headless; the full v23 bind-mount set is load-bearing; and
the CuTe-DSL warmup crash documented in runtime/hotfixes/deployed-r34-20260810.

Excluded candidates, for the record: Qwen2.5-0.5B (14 heads % 4 != 0),
SmolLM2-135M/360M (9/15 heads), Gemma-3-270M (heads pass; pinned-runtime
support unverified).

## Staging (disk prep only)

Weights land in each serving rank's HF cache (~4.2 GB total for all
four):

```bash
hf download EleutherAI/pythia-70m
hf download facebook/opt-125m
hf download Qwen/Qwen3-0.6B
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Staging is preparation and may be done ahead of time; launching any
serving or probe run is the operator's action.
