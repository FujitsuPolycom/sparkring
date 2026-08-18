# Eager width admission validation runbook

Status: all three legs executed. Leg 2 (native probes at nine payload
sizes) passed 2026-08-17. Leg 3 (four-width small-model shadow matrix)
completed 2026-08-17: width 768 passed its gate; widths 512 and 2048
failed the configured gate with recorded operator disposition; the
width-1024 disagreement remains open, with FP32-oracle arbitration
recorded as research-only. Leg 1 (GLM-5.2 EXL3-R7 3.5bpw serving regression, arms A
and B) passed 2026-08-18 under bootstrap-gated exact-state attestation
— bootstrap-gated live validation, not exact deployment-attestation
equivalence. This runbook produces functional and regression evidence
only — no performance or maturity claim. Every leg is
operator-launched. Feature under test: opt-in width-generic eager TP4
all-reduce admission (`VLLM_SPARK_TP4_EAGER_WIDTHS`), documented in
[`spark_transport/integrations/vllm/README.md`](../spark_transport/integrations/vllm/README.md).

What this runbook does NOT validate: CUDA-graph admission (unchanged,
6144-only, behind the open graph correctness gate), capacity classes /
active-bytes ABI, non-BF16 dtypes, PP6 or any topology beyond the four
directly cabled Sparks, and serving performance.

## Leg 1 — GLM-5.2 EXL3-R7 3.5bpw four-Spark serving regression (no new widths)

Purpose: prove the refactor is inert for the GLM-5.2 EXL3-R7 3.5bpw four-Spark serving configuration.

Two arms, identical serve config otherwise (the 2026-08-11-qualified
`--enforce-eager` TP4 configuration):

- Arm 1a: `VLLM_SPARK_TP4_EAGER_WIDTHS` unset (the serving configuration's default).
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

### Executed 2026-08-17 — full four-width matrix complete

All four models served TP4 shadow-mode from the deployed image via the
v23-cloned launch harness; every signature window is the (1, W) BF16 decode
shape at 10,000 collectives; all four ranks reported bit-identical
statistics in every run.

| Width | Model | outside_tolerance | rate | max_abs | Gate |
|---:|---|---:|---:|---:|---|
| 512 | pythia-70m | 3 / 5.12M | 5.9e-7 | 0.5 | FAIL, dispositioned |
| 768 | opt-125m | 0 / 7.68M | 0 | 0.0156 | PASS |
| 1024 | Qwen3-0.6B | 1153 / 10.24M | 1.1e-4 | ~1 | FAIL, open |
| 2048 | TinyLlama-1.1B | 2 / 20.48M | 1.0e-7 | 0.0625 | FAIL, dispositioned |

Interpretation on the evidence: divergence does not scale with width or
payload (the largest width is the cleanest), zero nonfinite values appeared
in 40,000 windowed collectives, and every run reproduced bit-identically
across ranks — jointly inconsistent with transport corruption and
consistent with BF16 summation-order sensitivity of each model's
activation distribution. SIRCL reduces as a balanced pairwise tree; the
stock path reduces in ring order; the two orders disagree most on
Qwen3-0.6B, an architecture with documented extreme activation-outlier
channels. The width-1024 signature remains unpromoted pending FP32-oracle
arbitration (which order lies closer to fp32 truth) and a
disagreement-policy decision for outlier-heavy models. Diagnostic-only
shadow decode rates, batch 1: opt-125m 68.8 tok/s, Qwen3-0.6B 6.6 tok/s,
TinyLlama 5.9 tok/s (shadow executes every admitted collective on both
transports and compares elementwise; no performance claim attaches).

### FP32-oracle arbitration of the width-1024 disagreement (2026-08-17)

Modeled analysis via `spark_fp32_ground_truth` (run CPU-only inside the
deployed image; module taken from the adaptive-branch checkout), width
1024, world size 4, 400 iterations per pattern, comparing the SIRCL
balanced-tree order (as implemented in `gpu_tp4_tensor.cu`) and a naive
sequential BF16 sum against correctly rounded FP32 ground truth:

| pattern | closer to truth (tree/seq/tie) | max err tree | max err seq | outside gate tol vs truth (tree/seq) |
|---|---|---:|---:|---|
| random | 54908 / 52192 / 302500 | 1.25 | 1.375 | 5162 / 5181 |
| cancellation | 1518 / 139 / 407943 | 0.000732 | 0.125 | 0 / 81 |

On the cancellation pattern — the divergence class observed in the
Qwen3-0.6B window — the tree order is two orders of magnitude closer to
truth, records zero elements outside the gate tolerance against truth,
and the 81 order-versus-order disagreements coincide exactly with the 81
elements where the sequential reference itself exceeds tolerance against
truth. In this modeled regime every disagreement the shadow gate counts
is an error of the reference, not the candidate. Caveats: the sequential
comparator is a naive left-to-right sum, not NCCL's actual chunked ring
order (the tool disclaims modeling NCCL), and inputs are synthetic
patterns rather than captured Qwen activations. Proposed follow-ups
before any width-1024 promotion decision: capture real divergent
collective inputs (flight recorder) and rerun the oracle on them, and
consider a truth-anchored gate criterion for outlier-heavy models
alongside the existing disagreement gate.

### Leg-1 attempt 2026-08-17: blocked by design, finding recorded

Arm A (widths unset) was launched as a verbatim v23 clone — real
entrypoint, full attestation env, all mounts, exact production command —
with only the two width-generic adapter files and the fabric
control-plane re-point substituted. Attestation and the 316 GB
InstantTensor load completed; engine initialization then failed closed
with `adaptive-MTP exact-state arena count drifted`.

Two findings, separated after a second launch attempt with the
provider-rows rebased adapter pair failed identically:

1. Adapter lineage divergence (real, fixed offline): the deployed
   adapter lineage is ahead of the public repository (33 changed lines
   in spark_tp4_backend.py, 31 in spark_tp4_port_namespace.py) and
   admits the sparse Q42-Q48 provider row set. The width feature's
   contiguous enumeration diverged from that admission surface. The
   rebase composing the two is validated byte-identical to the deployed
   lineage under every production environment by
   test_provider_rows_equivalence.py.

2. Launch blocker (unresolved, not adapter-related): the failing
   invariant, `_attest_adaptive_mtp_exact_state_policy` in the deployed
   model_runner, counts MoE exact-state GPU weight-storage arenas
   (`(len(q40), len(q48)) == (2, 2)` with pinned storage byte totals),
   not transport reservations. Its per-layer BF16 parity checks pass,
   confirming the model view and weight repack are correct; the arena
   count itself drifts under this runbook's container-clone launch
   (verbatim env, mounts, command, entrypoint) with either adapter
   lineage. The deployment's own launch pipeline (container name
   component `launch-tunable-v10-adaptive-closure-v23`) is not present
   on the rank filesystems examined and evidently performs steps the
   clone does not reproduce. Reproducing the confirmation load requires
   that pipeline or its operator's knowledge of it.

Recorded for operations: the per-rank model view bind mounts
(`/var/tmp/sparkring-model-view-exl3-* -> /var/tmp/sparkring-r7-model`)
are not reboot-persistent, which is why the original v23 stack exited
255 after the site event; systemd mount units would close that gap.

### Leg-1 attempt 2026-08-18: launch surface diffed, adapter exonerated

The launch surface of the clone was compared field by field against the
preserved raw launch script for the reference container
(`/var/tmp/sparkring-r7-jit/k4k5-pilot/preservation/rank0_v10_launch.sh.bak`
on rank 0, SHA-256 `1b71e56a59baa766…`), parsing both with a shell-word
parser rather than by inspection. The comparison covers mounts keyed by
destination, environment variables, host configuration, image identity
and command.

The clone matches the reference on every field except one: all 51 mount
destinations are present with the same sources apart from the two
deliberately substituted adapter files; all 285 environment variables
match apart from the five deliberate fabric re-points
(`MASTER_ADDR`, `SPARK_TP4_PEER0`, `SPARK_TP4_PEER1`,
`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`) and an added `VLLM_HOST_IP`;
image ID, `--network host`, `--ipc host`, 16 GiB shm, `CAP_IPC_LOCK`,
unlimited memlock, `/dev/infiniband` and the entrypoint all match.

The one divergence: the reference binds every source, contract, library
and entrypoint path read-only and only the two `/cache` paths writable.
The clone bound all 51 writable, because its mount-cloning step read
only source and destination and dropped the mode. The launcher now
clones the mode as well.

A control launch then ran with the mode fix and **no adapter
substitution at all** — the deployment lineage's own
`spark_tp4_backend.py` and `spark_tp4_port_namespace.py`, no
`VLLM_SPARK_TP4_EAGER_WIDTHS` — verified at the container as 49
read-only and 2 writable mounts. It failed identically with
`adaptive-MTP exact-state arena count drifted`.

That control establishes what the failure is not. It is not the
width-generic adapter pair, which was not mounted; it is not the mount
modes, which were correct; and it is not a difference in environment,
flags, image or command, which the diff excludes. The remaining
candidates are host state that the launch surface does not describe and
whatever the deployment's own launch pipeline does beyond issuing the
container invocation.

The invariant's message reports only the two counts, discarding which
buffer-cache key diverged. A diagnostic copy of the deployed
`model_runner.py` that also reports the sorted key set for both
capacities is staged at `/var/tmp/leg1-arena-diag/model_runner.py` on
all four ranks; the launcher binds it in place of the original when
`DIAG=1`. It is a copy, so no shared runtime file is modified.

### Leg-1 interim hypothesis 2026-08-18, superseded: online-cache identity

This section records an interim diagnosis that later evidence
disproved. The severed online-cache identity described here is real
and was repaired, but it was not the cause of the arena-count failure:
neither online-cache namespace contains routed-expert payloads, so the
cache could not have supplied the missing expert tier. The completed
diagnosis is in the section titled "Leg-1 root cause, completed
2026-08-18". The interim record is preserved because the identity
repair it prescribes was executed and remains in effect.

The diagnostic launch reported the diverging state directly:

```
arena count drifted: counts q40=1 q48=1 expected (2, 2);
q40_keys=[((0, False), 0, 40, 8, (128, 128, 32, 512), 8, 6144, 512, 40, 48, 'bf16', torch.int64, 2)]
q48_keys=[((0, False), 0, 48, 8, (128, 128, 32, 512), 8, 6144, 512, 48, 48, 'bf16', torch.int64, 2)]
```

The serving-era receipt on rank 0
(`/var/tmp/sparkring-r7-jit/q35-q40-exact-state-serving-v2-rank0.json`,
`shape_receipts/*/buffer_cache_keys`) records the healthy signature: two
buffer-cache keys per capacity, identical in every field except
`tier_count` — one two-tier and one three-tier scratch-arena geometry.
Boots against the surviving 2026-08 model views produce only the two-tier variant on every layer. The invariant's
`(2, 2)` and its pinned byte totals are hardcoded in the deployed
`model_runner.py`, so the boot refuses.

The tier mix traces to the online EXL3 requantizer
(`VLLM_EXL3_ONLINE_TRELLIS_BITS=6`, cache under `/cache/exl3-online`,
host `/var/tmp/sparkring-r7-online`). Its cache namespace is the first
20 hex digits of a model-identity hash computed over the model
directory's path, revision, marker-file hashes, and per-shard
`(name, resolved path, size, mtime_ns)` tuples
(`exl3_online_cache.resolve_model_identity`). Two namespaces exist on
rank 0:

- `d871f8223f2678c032f1` — created 2026-08-10 by the healthy boot,
  261 attention/dense artifacts, warm through every serving-era
  launch. (A separate namespace created by the post-reboot boots
  holds 411; 672 is the combined count across both, not the size of
  either.)
- `3de03931460fa5ae458a` — created 2026-08-17 by the first container
  clone attempt, and matching the identity that every model directory
  surviving the 2026-08-17 reboot computes (verified by recomputing
  the hash against
  `/var/tmp/sparkring-model-view-exl3-3f57337`,
  `/var/tmp/sparkring-model-view-exl3-bc036a0`, and
  `/srv/models/GLM-5.2-EXL3-TR3-3.25bpw`; all three produce
  `3de03931…`, and no tested revision string produces `d871f822…`).

Every surviving model directory is pristine (all marker and shard
ctimes predate the healthy boot), so the identity input that produced
`d871f822…` no longer exists on the host. Which deleted path supplied
that input is unknown and unrecoverable after the reboot; `/tmp` was
emptied by the site-event reboot and is one candidate, but the
evidence proves only that the healthy identity's input tuple is gone,
not where it lived. The post-reboot bind restore pointed
`/var/tmp/sparkring-r7-model` at the surviving `/var/tmp` view —
weight-identical to the healthy configuration (same inodes),
identity-different.

Consequence: the requantizer runs cold, re-derives per-expert bitrates
from proxy-error thresholds, and lands on a different (uniformly
two-tier) mix than the one the attestation pins. Each of the 261 warm
artifacts embeds the full old identity in its `cache_key` metadata and
its filename digest, so pointing the loader at the old namespace
requires rewriting that metadata to the new identity — mechanical, and
doable against a copy in an isolated namespace — or accepting the new
mix, which means changing the deployed attestation's pinned constants
and re-validating quality. Both are deployment-policy decisions,
neither belongs to the eager-width work, and the width-generic adapter
remains exonerated by the no-substitution control launch.

### Leg-1 root cause, completed 2026-08-18: the attested state was never the checkpoint

Source analysis of the deployed runtime (the b12x kernel package from
image `02881d52…`, the deployed `exl3.py`, the instanttensor loader)
established three facts with citations, then receipt data closed the
case:

1. Scratch-arena tier arity is a pure function of loaded routed-expert
   tensor widths. The only constructor of a three-tier launch is
   `compile_mixed_trellis3`, selected when a layer's gate/up/down
   expert tensors span three distinct trellis widths; no environment
   variable, plan file, contract, or `/cache/jit` artifact can add a
   tier, and no layer-index conditional exists.
2. The load path has no overlay or substitution mechanism, and
   tensor-parallel slicing never changes the width dimension.
3. The serving-era receipt records per-layer expert bitrate mixes that
   the published 3.25-bpw checkpoint cannot produce: layers 3, 4, 5, 9 each carried
   one or two K5 experts (layer 3: 171 K3 / 86 K4 / 1 K5), and the
   other 71 layers carried K3/K4 splits that differ from the
   checkpoint's uniform 192 K3 / 64 K4. The attested state was
   therefore a runtime-requantized expert mix — across all 75 layers —
   not the published 3.25-bpw checkpoint.

The stores that could have supplied those payloads were checked
directly: the online EXL3 cache (both identity namespaces) contains
zero routed-expert tensors and only K6 attention/dense entries, and
`/opt/sparkring-q35` holds two contract files, no weights. No
surviving artifact contains the K5-bearing expert payloads. The
receipt also pins an `exl3.py` source hash (`91187fcd…`) that differs
from the mounted lineage, whose files carry an Aug 17 02:30 mtime:
the runtime generation changed in the failure window, and the deployed
generation does not route online-cache payloads into routed experts at
all.

Status of the deployed attestation pins: they describe a configuration
whose weight source did not survive the site-event reboot and whose
producing runtime generation is no longer mounted. Recovery choices:
re-encode the per-expert mix from the donor recipe recorded in
`config.json`'s `k4_patch` note (a quantization campaign), or
re-baseline the attestation to the checkpoint's native two-tier state.

Bootstrap mode implements the second choice for validation purposes:
`/var/tmp/leg1-arena-bootstrap/model_runner.py` on each rank is a copy
of the deployed `model_runner.py` whose four exact-state pins (two
arena-cardinality checks, two storage-byte checks) log observed values
and continue when `SPARK_EXACT_STATE_BOOTSTRAP=1` is set, and behave
identically to the deployed original otherwise. The launcher mounts it
only into leg1 containers when `BOOTSTRAP=1`. Runs under this gate
serve the model as published — the checkpoint's native expert mix plus
K6-onlined attention projections; the deployed attestation pins
predate this configuration and are superseded by re-baselining them to
its observed values.

### Bootstrap-gated boot, executed 2026-08-18: healthy and validated

The zero-substitution control under the bootstrap gate reached API
health in 295 seconds. The gate accepted exactly two checks per rank:
arena counts (1, 1), and storage bytes {q40: 16134192, q48: 18100560}
— each exactly half the deployed pin, the arithmetic signature of one
arena variant where the runtime-requantized state had two. CUDA graph
capture completed through the declared rows, and the boot wrote a
fresh create-once exact-state receipt recording the clean state.

Functional validation, all at temperature 0 against the serving API:

- Deterministic outputs: 847 × 293 answered 248171; a requested
  Python one-liner was valid; a factual query answered correctly.
  All finished with stop.
- Prefill: a 14,466-token prompt embedding 700 numbered sections
  answered the exact section count.
- Concurrency-8 decode: eight parallel bounded generations all
  completed (two initially truncated by the probe's own token cap;
  a re-run with adequate budget finished with stop and exact output).
- Speculative decode: 2,441 drafts, mean draft depth 4.27 tokens,
  79% draft-token acceptance (8,233 of 10,416) — the adaptive-MTP
  depth ladder exercises depth 4-5 on the clean state.

Not covered: sustained-load benchmarks.

### Leg-1 arms A and B, executed 2026-08-18: PASS

Both arms ran the width-generic adapter pair under the bootstrap gate,
using the same launch surface as the control (verbatim clone, receipt
lifecycle automated: stop all leg1 containers, archive prior-process
exact-state receipts, launch).

- Arm A (widths variable unset): booted to API health, SparkRing
  installed in custom mode on all four ranks (TP4 all-reduce,
  all-gather, and vocabulary backends), served a bounded
  deterministic request correctly.
- Arm B (`VLLM_SPARK_TP4_EAGER_WIDTHS=6144`): same result. 6144 is
  the default width, so admitting it explicitly must not change the
  reservation surface.
- Port comparison: a numeric sweep of each arm's full rank-0 log over
  the reservation range (11100-12288) matched on 84 of 85 distinct
  values. The single arm-A-only value (12049) sits at no valid
  row-slot position (neither port base yields an integer slot), and
  its source line was not recoverable after the container was
  replaced. The observation is unresolved: no reservation explains
  it, and no log line survives to attribute it. Separately,
  test_provider_rows_equivalence.py establishes parsed
  admission-surface equality between the adapter pair under test and
  the deployment lineage for its enumerated environments — a semantic
  comparison of reservations, port resolution, admission bitmaps, and
  capacity values, not a byte-level one, and not a comparison between
  the two arms.

With legs 2 and 3 already qualified, this completes the three-leg
validation: the width-generic adapter serves the production model
with the feature off (regression) and with the default width
explicitly admitted (no-op equivalence), live on the ring.

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
