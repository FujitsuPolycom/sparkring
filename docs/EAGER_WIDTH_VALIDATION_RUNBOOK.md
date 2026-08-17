# Eager width admission validation runbook

Status: offline-prepared; no leg has been executed. This runbook produces
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
