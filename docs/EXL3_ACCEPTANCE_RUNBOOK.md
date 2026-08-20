# EXL3 + LMCache public acceptance runbook

This runbook advances the default four-DGX-Spark GLM-5.2 EXL3 3.25-bpw plus
LMCache profile with 512-token cache chunks (CS512) beyond its bounded
live-validation receipt. It is a
`public-functional` candidate workflow. Publishing the tools does not upgrade
the profile from `live-validated` to `accepted`; only a reviewed, passing live
evidence bundle can do that. NF3 is the accepted deterministic
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
This makes a first live run record observations and exit 4:

```bash
python scripts/exl3_correctness_gate.py \
  --config .sparkring/acceptance/exl3-correctness.json \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --repetitions 3 \
  --execute run \
  > .sparkring/acceptance/correctness-candidate.json
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
correctness gate is the token-ID authority.

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
MTP mode, 1M context/9 GB KV settings left from a different profile, or a missing
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

For an offline, machine-readable classification of already captured startup
logs, run `scripts/sparkring_startup_evidence.py`. The classifier is
`offline-validated`: it does not contact a Spark, establish a safe retry bound,
or replace container inspection. It reports `clean`, `bounded_rm_retry`,
`indeterminate`, or `fatal` under the
`sparkring-startup-evidence/v1` schema. Missing engine evidence, stopped
containers, incomplete timestamp context, and unaligned per-rank inputs fail
closed.

Supply four engine logs and their matching `docker inspect` JSON documents in
rank order. When present, supply exactly one kernel or LMCache-server log for
every engine rank; server logs also require matching inspection JSON. The
following example classifies four engine and kernel logs. Replace
`REVIEWED_RM_EVENT_BOUND` only with a bound established by a separately
reviewed operating policy:

```bash
python scripts/sparkring_startup_evidence.py \
  --engine-log evidence/engine-r0.log \
  --engine-log evidence/engine-r1.log \
  --engine-log evidence/engine-r2.log \
  --engine-log evidence/engine-r3.log \
  --engine-inspect evidence/engine-r0-inspect.json \
  --engine-inspect evidence/engine-r1-inspect.json \
  --engine-inspect evidence/engine-r2-inspect.json \
  --engine-inspect evidence/engine-r3-inspect.json \
  --engine-container-name glm52-sparkring-exl3-lmcache-cs512-r0 \
  --engine-container-name glm52-sparkring-exl3-lmcache-cs512-r1 \
  --engine-container-name glm52-sparkring-exl3-lmcache-cs512-r2 \
  --engine-container-name glm52-sparkring-exl3-lmcache-cs512-r3 \
  --kernel-log evidence/kernel-r0.log \
  --kernel-log evidence/kernel-r1.log \
  --kernel-log evidence/kernel-r2.log \
  --kernel-log evidence/kernel-r3.log \
  --engine-log-year 2026 \
  --engine-log-tz=-05:00 \
  --rm-event-bound REVIEWED_RM_EVENT_BOUND \
  --cluster-ready \
  classify > evidence/startup-classification.json
```

`--cluster-ready` is an explicit operator-supplied fact, not a network probe.
The expected names bind each inspection document to its role and rank. The
inspection documents must expose `Id`, `Name`, `State.Running`,
`State.OOMKilled`, and `RestartCount`; duplicate identities and name mismatches
fail closed, while a stopped, restarted, or OOM-killed component is fatal. Do
not use a successful classification to relabel a generic CUDA OOM, Xid,
restart, fabric loss, or driver failure as recoverable.

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
  scripts/test_exl3_cache_acceptance.py \
  scripts/test_sparkring_exl3_lmcache_launcher.py \
  scripts/test_sparkring_startup_evidence.py -q

ruff check --select E,F,W --ignore E501 \
  scripts/acceptance_gate.py \
  scripts/exl3_correctness_gate.py \
  scripts/exl3_cache_acceptance.py \
  scripts/sparkring_exl3_lmcache_launcher.py \
  scripts/sparkring_startup_evidence.py
```

The full repository validation is the command in [CONTRIBUTING.md](../CONTRIBUTING.md).
