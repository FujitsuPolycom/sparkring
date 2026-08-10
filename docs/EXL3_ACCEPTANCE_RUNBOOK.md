# EXL3 + LMCache public acceptance runbook

This runbook advances the default four-DGX-Spark GLM-5.2 EXL3 3.25-bpw plus
LMCache CS512 profile beyond its bounded live-validation receipt. It is a
`public-functional` candidate workflow. Publishing the tools does not upgrade
the profile from `live-validated` to `accepted`; only a reviewed, passing live
evidence bundle can do that. NF3 remains the accepted deterministic
alternative.

The workflow is intentionally easy to inspect and extend. Individual offline
tests and focused read-only API gates can be run without invoking the full
acceptance orchestrator. The full orchestrator adds strict checks only where a
false pass could create a misleading public result or leave a four-node stack
running after failure.

## What the workflow proves

The profile is fixed to the public recipe:

| Setting | Required value |
|---|---|
| Hardware | four directly cabled DGX Sparks / GB10s |
| Model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| Parallelism | TP4 / DCP4 / PP1 |
| Speculation | fixed MTP2 |
| Context and KV | 524,288 tokens; 4,500,000,000 KV bytes/rank |
| Scheduling | maximum 8 sequences; 4,096 batched tokens; Q32 graphs |
| Cache | native prefix cache plus one LMCache CS512 RAM server per rank |

The generic orchestrator runs these stages in order:

1. require a passing, full-cluster read-only preflight receipt and independently
   attest the identical image and model bytes on all four ranks;
2. qualify all four physical edges and run the configured model-down collective
   probe;
3. start four LMCache servers and four EXL3 engines;
4. require exact labels, running state, zero restarts, no OOM state or OOM log
   signature, four healthy LMCache servers, and the rank-0 API/model identity;
5. compare a fixed 128-token generation against a reviewed token-ID baseline;
6. run three 128-token, `ignore_eos=true` waves at C1/C2/C4/C8 and compare the
   completed cells with the documented public-profile regression floors;
7. run the broader multi-case token-ID gate, then attribute a cold/warm cache
   hit across an engine-only restart and prove the volatile-server boundary
   across a complete LMCache-server restart; and
8. remove only containers with the exact EXL3 profile label, verify their exact
   names are absent, and confirm that the API stopped.

If any stage after the start attempt fails, stage 8 is still attempted. A
cleanup failure is retained in the same result bundle. A failure before the
start attempt performs no cleanup because the gate did not create a stack.

LMCache CS512 in this recipe has RAM-only L1 storage. The expected persistence
boundary is therefore:

- cache objects and warm reuse survive an **engine-only** restart because the
  four LMCache servers stay alive;
- cache objects are absent immediately after a **server restart**;
- the first post-server-restart request is cold, a repeated request is warm,
  and all four shards repopulate.

That is lifecycle recovery and attribution, not NVMe durability. SparkCache is
a separate implementation and is disabled in this profile.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Functional checks passed and the performance band passed or was not measured. |
| 2 | Functional failure, including a request error, short benchmark stream, output mismatch, unhealthy rank, restart/OOM, cache-boundary failure, or rollback failure. |
| 3 | Configuration or plan error; the orchestrator started nothing. |
| 4 | A deterministic baseline was recorded for review. This is deliberately not a pass. |
| 5 | Functional checks passed but the performance band failed or only a candidate band was recorded. |

The focused correctness gate uses 0/2/3/4 with the same meanings. The focused
cache-boundary gate uses 0/2/3.

## 1. Prepare the generated deployment files

Complete the default quickstart through its dry bootstrap or live deployment.
The acceptance profile expects these ignored, site-specific files:

```text
.sparkring/bootstrap-exl3/site.yaml
.sparkring/bootstrap-exl3/launch.json
```

Start with [QUICKSTART.md](QUICKSTART.md) if they do not exist. Validate the
resolved site before doing anything remote:

```bash
python scripts/sparkring_site.py .sparkring/bootstrap-exl3/site.yaml
python scripts/preflight.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --print-plan
```

The second command is offline. Read every command in its output before running
the read-only remote preflight.

## 2. Prepare a local acceptance profile

Copy both templates into the ignored `.sparkring` directory:

```bash
mkdir -p .sparkring/acceptance
cp scripts/config/gate.exl3.example.json \
  .sparkring/acceptance/gate.exl3.json
cp scripts/config/exl3-correctness.example.json \
  .sparkring/acceptance/exl3-correctness.json
```

Edit `.sparkring/acceptance/gate.exl3.json` and replace:

- `<YOUR_PUBLIC_MODEL_DOWN_PROBE_ENTRYPOINT>` with the already installed,
  hash-pinned public probe entrypoint for each rank;
- both `<RANK0_MANAGEMENT_ADDRESS>` values with rank 0's management address;
- `<UNIQUE_ACCEPTANCE_RUN_ID>` with a new identifier that has never been used
  as a cache prefix.

Do not replace immutable recipe pins, serving shape, exact container labels,
or cache geometry merely to make a failing deployment pass. Site addresses,
paths, credentials, and raw evidence stay in ignored files.

The shipped performance floors are regression guards derived below the same
profile's clean-checkout public-functional C1/C2/C4/C8 receipt. They are not
reference-lane figures and they are not advertised throughput. A contributor
may remove `performance.band` to record a candidate band; that produces exit
5, never acceptance.

## 3. Establish reviewed correctness baselines

The example correctness cases intentionally contain `null` expected hashes.
Every request probe now requires a public-safe live-arm receipt emitted only
after the attribution launcher attests the exact container name, ownership
labels, profile, arm, image, generated command, and explicitly generated
environment on all four ranks. On a fresh evidence path, activate and attest
the arm first (this stops the canonical engines and is **STOPS SERVING**):

```bash
python scripts/exl3_attribution_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --arm d-mtp2-apc1-lmcache1 \
  --execute \
  --confirmation RUN-EXL3-ATTRIBUTION-ALL-FOUR \
  --output .sparkring/acceptance/exl3-attribution-launch-private.json \
  --live-arm-receipt-output .sparkring/acceptance/exl3-live-arm-receipt.json \
  activate
```

The raw launcher report is private because it contains SSH targets and remote
commands. The live-arm receipt is closed and public-safe. Both output paths use
exclusive creation; use new paths for a later run rather than overwriting
evidence. The first correctness run then records observations and exits 4:

```bash
python scripts/exl3_correctness_gate.py \
  --config .sparkring/acceptance/exl3-correctness.json \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --attribution-arm d-mtp2-apc1-lmcache1 \
  --activation-receipt .sparkring/acceptance/exl3-live-arm-receipt.json \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --repetitions 3 \
  --execute run \
  > .sparkring/acceptance/correctness-candidate.json
```

The correctness plan and raw report are private: the plan exposes reviewed SSH
targets and the report records the exact contacted API origin. Before its first
HTTP request, the gate uses the site file for a **READ-ONLY REMOTE** four-rank
re-attestation of the receipt's runtime-unique Docker container IDs and
`StartedAt` values. A replaced or restarted engine, rank mismatch, or SSH
failure aborts with zero HTTP requests. Only the closed output of
`scripts/exl3_attribution_compare.py` is the publishable sanitized comparison.

`--attribution-arm` is also a cache-safety boundary. The gate derives a
layout-specific `cache_salt` from the locked DCP, KV, chunk, context, and MTP
contract and includes it in every completion request. Use the arm that is
actually running; the gate fails closed without one. MTP0 and MTP2 evidence
must never share the unsalted LMCache namespace. Until LMCache supplies a live
layout receipt, the attribution launcher recreates all four LMCache server
processes before every cache-attached activation, transition, or restart-arm,
including same-layout and same-arm changes, and before canonical rollback.
The correctness and cache-metric probes reject a missing, stale, wrong-arm, or
profile/image/model/KV-mismatched receipt before making any HTTP request.

The published cache-correctness configuration contains two distinct prompts.
Capture one metric probe for each case; a probe for one suffix cannot be reused
for the other. Both commands below are **READ-ONLY REMOTE** but fill transient
serving caches, and each re-attests all four runtime-unique container identities
before its first HTTP request:

```bash
python scripts/exl3_cache_metric_probe.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --activation-receipt .sparkring/acceptance/exl3-live-arm-receipt.json \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --attribution-arm d-mtp2-apc1-lmcache1 \
  --run-label cache-long-prefix-json \
  --output .sparkring/acceptance/cache-long-prefix-json-metric.json \
  --prompt-fragment 'SparkRing deterministic cache boundary evidence contains alpha beta gamma delta epsilon zeta eta theta and remains identical in every record.
' \
  --prompt-repetitions 64 \
  --prompt-suffix 'Task: Return only valid JSON with keys first and last, setting first to alpha and last to theta.' \
  --execute

python scripts/exl3_cache_metric_probe.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --activation-receipt .sparkring/acceptance/exl3-live-arm-receipt.json \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --attribution-arm d-mtp2-apc1-lmcache1 \
  --run-label cache-long-prefix-recall \
  --output .sparkring/acceptance/cache-long-prefix-recall-metric.json \
  --prompt-fragment 'SparkRing deterministic cache boundary evidence contains alpha beta gamma delta epsilon zeta eta theta and remains identical in every record.
' \
  --prompt-repetitions 64 \
  --prompt-suffix 'Task: In exactly one sentence, name the first and last Greek labels in the repeated evidence.' \
  --execute
```

Bind the probes explicitly by case ID. Missing, extra, duplicate, unknown, and
non-cache-case mappings fail before HTTP; prompt text is checked locally, then
token IDs and token count are checked against the live tokenizer before any
completion request:

```bash
python scripts/exl3_correctness_gate.py \
  --config scripts/config/exl3-correctness-cache.example.json \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --attribution-arm d-mtp2-apc1-lmcache1 \
  --activation-receipt .sparkring/acceptance/exl3-live-arm-receipt.json \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --cache-metric-probe cache-long-prefix-json=.sparkring/acceptance/cache-long-prefix-json-metric.json \
  --cache-metric-probe cache-long-prefix-recall=.sparkring/acceptance/cache-long-prefix-recall-metric.json \
  --repetitions 3 \
  --execute run \
  > .sparkring/acceptance/cache-correctness-candidate.json
```

For every case:

1. confirm all repetitions have the same `token_ids_sha256`;
2. inspect the returned `token_ids` and corresponding completion behavior;
3. copy the reviewed hash into that case's
   `expected_token_ids_sha256`; and
4. rerun the command and require exit 0.

Do the same for the generic stage-5 expected generation. On its first full
gate execution, the orchestrator writes a candidate expected-generation file
inside the evidence bundle and exits 4. Review it, then copy it to the ignored
path declared by `acceptance.expected_generation_path`:

```text
.sparkring/acceptance/exl3-expected-generation.json
```

A null hash, a newly recorded candidate, internally divergent repetitions, or
a mismatch with a reviewed expected hash can never report `PASS`.

For a small contribution, it is valid to copy the correctness template and
retain only one case while developing. The schema deliberately accepts one or
more cases. The release profile must retain the complete reviewed case set.

## 4. Review the focused cache plan

This command does not contact the cluster even though the plan describes two
controlled restarts:

```bash
python scripts/exl3_cache_acceptance.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --probe-id <unique-acceptance-run-id> \
  plan
```

The live form is `STOPS SERVING`: it restarts all four engines, then all four
engines and servers. Run it independently only on an authorized acceptance
cluster and only after reviewing the plan:

```bash
python scripts/exl3_cache_acceptance.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --probe-id <unique-acceptance-run-id> \
  --execute \
  --confirmation RUN-EXL3-CACHE-BOUNDARY-ALL-FOUR \
  run
```

The gate requires one registered GPU context, healthy L1 state, no temporary
or locked objects, and stored objects on every rank. It also requires identical
fixed-seed completion text across cold, warm, and restart samples. The broader
correctness gate remains the token-ID authority.

## 5. Produce and review the full dry-run plan

The acceptance orchestrator is dry-run by default. It creates no evidence
directory and opens no connection:

```bash
python scripts/acceptance_gate.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --gate-config .sparkring/acceptance/gate.exl3.json \
  --plan-out .sparkring/acceptance/plan.json
```

Review all eight stages. In particular, confirm:

- the four SSH targets and ring edges are the intended acceptance hosts;
- both delegated attestations reference the generated EXL3 site/profile;
- the model-down probe is real and hash-pinned;
- startup and rollback use the LMCache launcher and exact confirmation token;
- rank-specific container names expand to `...-r0` through `...-r3`;
- the correctness URL, cache URL, and unique probe identifier are resolved;
- C1/C2/C4/C8 each use three waves, 128 maximum tokens, at least 120 returned
  tokens, and `ignore_eos=true`; and
- stage 8 removes only the two exact profile-labelled containers per rank.

## 6. Capture the required read-only preflight receipt

The full gate starts from an idle cluster because stage 2's collective probe is
model-down and the generated site requires API/rendezvous ports to be free.
After receiving authorization, use the exact-label launcher rollback if this
profile is already serving, verify rollback, and only then capture preflight.
That preparatory stop is outside the acceptance gate and must be recorded in
the operator log; the gate will later start and finally stop its own stack.

```bash
python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute \
  --confirmation START-EXL3-LMCACHE-CS512-ALL-FOUR \
  rollback

python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute verify-rollback
```

After the printed probe plan has been reviewed and the intended four hosts are
idle, run the full read-only remote preflight and retain its JSON:

```bash
python scripts/preflight.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --strict-placeholders \
  --json .sparkring/acceptance/preflight.json
```

The acceptance gate rejects a missing, partial, failed, or mismatched preflight
receipt before it starts a stack.

## 7. Execute only with fresh authorization

`--execute` starts and stops the named four-host stack. Do not run it on a
production-serving cluster. Obtain fresh authorization naming all four hosts
and this exact start/restart/stop operation, then use one run ID consistently
in the gate config and command:

```bash
python scripts/acceptance_gate.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --gate-config .sparkring/acceptance/gate.exl3.json \
  --preflight .sparkring/acceptance/preflight.json \
  --bundle-dir evidence/acceptance \
  --run-id <unique-acceptance-run-id> \
  --execute \
  --confirm RUN-PUBLIC-ACCEPTANCE-GATE
```

Throughput uses OpenAI continuous `usage.completion_tokens`, not SSE event
counts. The terminal summary is not sufficient publication evidence. Inspect
`evidence/acceptance/<run-id>/result.json`, every stage artifact, the cleanup
result, four-rank logs, immutable IDs, and any candidate baseline. Evidence can
contain local identifiers and remains untracked until a maintainer sanitizes a
machine-readable receipt.

## Troubleshooting

### Configuration exits 3 before any connection

Read every listed mismatch. Common causes are an NF3 site file used with the
EXL3 profile, an unexpanded angle-bracket value, mutable model revision, wrong
MTP mode, 1M context/9 GB KV settings left from an older profile, or a missing
C2/C4 cell. Regenerate the site/profile with the EXL3 bootstrap rather than
editing around the recipe contract.

### Attestation fails

Run `verify-image` and `verify-model` independently. A tag is not an image ID;
all ranks must resolve the same local image ID. The model verifier checks all
81 shards plus the config, index, tier bitmap, manifest, repository, and
immutable revision. Re-copy the proven artifact; do not weaken the hashes.

### Startup, OOM, or restart failure

Use the launcher `status` command and inspect all eight exact containers. The
profile requires 4.5 GB/rank KV, 524,288 maximum model length, maximum eight
sequences, and lazy 1-GiB LMCache L1. A stale container with a different profile
label is intentionally not removed. Preserve the failing evidence and resolve
the named conflict manually.

### Warm cache does not beat cold cache

Confirm all four servers stored objects and have one registered GPU context.
Use a never-reused probe ID. Native prefix caching can serve a repeated prompt
within one engine lifetime, which is why the decisive attribution sample is
after the engine-only restart. A server restart must clear RAM-only L1; if it
does not, the tested deployment is not this recipe.

The default TTFT ratios are regression assertions, not universal speedup
claims. If environmental noise is suspected, preserve the failed report and
rerun with a new probe ID. Do not raise thresholds without publishing the
reason and comparative evidence.

### C8 is slow or a stream is short

Performance cells require exact effective concurrency, no request errors, and
at least 120 tokens per stream. Check `cell-C8.json`, API logs, restart/OOM
state, and KV admission. An out-of-band completed cell exits 5 and does not
rewrite the functional verdict; a request error or short stream is functional
exit 2. Do not quote suppressed or partially populated C8 measurements.

### A later stage fails

The orchestrator still attempts exact-label rollback after any start attempt.
If rollback also fails, use the captured stop and verification artifacts to
identify the exact remaining container. The launcher refuses to remove a
same-name container carrying a foreign profile/component label; that refusal
is a safety result, not a check to bypass.

## Offline contributor checks

These require no Sparks, no model weights, and no network:

```bash
python -m pytest \
  scripts/test_acceptance_gate.py \
  scripts/test_exl3_correctness_gate.py \
  scripts/test_exl3_attribution_launcher.py \
  scripts/test_exl3_attribution_compare.py \
  scripts/test_exl3_attribution_reduce.py \
  scripts/test_exl3_teacher_forced_margin_probe.py \
  scripts/test_exl3_teacher_forced_margin_reduce.py \
  scripts/test_exl3_cache_metric_probe.py \
  scripts/test_exl3_cache_acceptance.py \
  scripts/test_exl3_cache_geometry_gate.py \
  scripts/test_sparkring_exl3_lmcache_launcher.py \
  scripts/test_sparkring_startup_evidence.py \
  scripts/test_compare_benchmark_evidence.py -q

ruff check --select E,F,W --ignore E501 \
  scripts/acceptance_gate.py \
  scripts/exl3_correctness_gate.py \
  scripts/exl3_attribution_cache_contract.py \
  scripts/exl3_attribution_launcher.py \
  scripts/exl3_attribution_compare.py \
  scripts/exl3_attribution_reduce.py \
  scripts/exl3_teacher_forced_margin_probe.py \
  scripts/exl3_teacher_forced_margin_reduce.py \
  scripts/exl3_cache_metric_probe.py \
  scripts/exl3_cache_acceptance.py \
  scripts/exl3_cache_geometry_gate.py \
  scripts/sparkring_exl3_lmcache_launcher.py \
  scripts/sparkring_startup_evidence.py \
  scripts/compare_benchmark_evidence.py
```

Also verify the LMCache CS512 geometry and startup evidence classifier:

```bash
python scripts/exl3_cache_geometry_gate.py verify
python scripts/sparkring_startup_evidence.py --help
```

The full repository validation remains the command in [CONTRIBUTING.md](../CONTRIBUTING.md).

For long double 315.78-GiB verification, page-cache pressure, bounded RM
allocation retries versus fatal signals, headless Wi-Fi, rollback, and
PowerShell log tailing, see
[EXL3_TROUBLESHOOTING.md](EXL3_TROUBLESHOOTING.md).
