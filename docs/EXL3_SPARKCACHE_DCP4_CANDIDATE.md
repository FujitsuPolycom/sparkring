# EXL3 3.25-bpw SparkCache DCP4 candidate

## Scope

This is the smallest current-source path for testing SparkCache against the
public EXL3 3.25-bpw TP4/DCP4 fixed-MTP2 configuration. It is a
**public-functional-lane, offline-validated candidate** for four directly
cabled NVIDIA DGX Sparks. It has not been launched on the cluster and is not
an accepted configuration.

The default, main-advertised, currently running public-functional
configuration remains the live-validated EXL3 plus LMCache CS512 profile. It
is not accepted; NF3 remains the accepted public-functional alternative.
SparkCache is a different KV-Connector-V1 implementation. No LMCache result
validates SparkCache, and the two connectors must not be composed into the
same engine.

## What can remain unchanged

The connector is already DCP4-native. The current EXL3 profile uses the same
persisted-cache byte identities for which SparkCache was built:

- TP4/DCP4, so every physical TP worker has one distinct DCP shard;
- `nvfp4_ds_mla` with per-token scaling;
- the GLM-5.2 interleaved RoPE layout;
- fixed MTP2 whose draft state is colocated in the target cache pool; and
- 256-token persistent chunks, giving 64 local token rows per DCP4 rank.

The target checkpoint digest still changes to the immutable EXL3 artifact
identity. It prevents an NF3 entry, another EXL3 revision, or a rewritten
quantization from entering the EXL3 namespace.

The candidate no longer accepts a free-form 64-hex namespace. It embeds the
canonical checkpoint-manifest-v2 receipt: a complete, strictly sorted inventory
of every regular file beneath the mounted artifact root, with NFC/POSIX relative
path, byte length, and file SHA-256. The namespace is SHA-256 over
`sparkcache-checkpoint-manifest-v2`, one NUL byte, and canonical JSON of that
inventory (excluding only the display root name and recursive identity field).
The launcher never re-hashes this checkpoint while a model engine is live.
While the baseline is serving it performs only an inventory/path/size precheck
against the embedded receipt. During cutover it removes the exactly owned
baseline engines and LMCache servers on all four ranks, proves that no running
container mounts the model and that no NVIDIA compute process remains, and only
then re-walks and re-hashes the artifact. The full hash runs in a named,
GPU-less, network-isolated helper using the exact pinned image and read-only
model mount. Its derived identity must equal the connector namespace before any
candidate engine can start.

## Missing composition boundary

The base public EXL3 launcher either starts the base engine or injects LMCache
CS512. The historical SparkCache launcher is an NF3/DCP2 cutover and must not
be reused: it changes DCP, model length, container identities, and draft
policy.

[`scripts/exl3_sparkcache_config.py`](../scripts/exl3_sparkcache_config.py)
now generates the exact EXL3/DCP4 cache delta offline. It deliberately emits
`execution_supported=false`: the embedded profile is never launched directly.
[`scripts/sparkring_exl3_sparkcache_launcher.py`](../scripts/sparkring_exl3_sparkcache_launcher.py)
is its only supported consumer. That launcher adds and attests both mounts,
re-hashes the small connector bundle on every rank, checks the two vLLM patch
semantics inside an already-running exact-image engine, and carries inspected
rollback actions that restore the EXL3+LMCache baseline. Checkpoint shard
content is deliberately excluded from every live-engine attestation.

Generate the candidate from the ignored, receipt-gated public profile:

```powershell
python scripts/exl3_sparkcache_config.py `
  --profile .sparkring/bootstrap-exl3/launch.json `
  --checkpoint-receipt .sparkring/exl3-checkpoint-receipt.json `
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256> `
  --connector-staging-host /opt/sparkring/sparkcache-staging `
  --cache-root-host /var/lib/sparkring/sparkcache-exl3 `
  --output .sparkring/exl3-sparkcache-candidate.json
```

The delta preserves model/image pins, TP4/DCP4, fixed MTP2, 524,288 tokens,
4.5 GB KV/rank, Q4096/C8/Q32, graph settings, and transport. It:

- explicitly unsets the legacy `SPARK_CONTEXT_CACHE_ENABLE` label;
- adds exactly one `SparkContextCacheConnector` transfer configuration;
- selects `colocated_target` without a fictitious separate draft digest;
- adds `--disable-hybrid-kv-cache-manager` for the connector's non-HMA path;
- keeps native restore and streaming snapshots off for the first gate; and
- declares a read-only connector mount plus a writable rank-local cache mount.

The candidate also deliberately matches the canonical EXL3+LMCache engine's
allocator and Docker security envelope: `PYTORCH_CUDA_ALLOC_CONF` is
`expandable_segments:False`; host network and host IPC are retained; the
container is privileged with all GPUs, 16 GiB shared memory, unlimited
memlock, `CAP_IPC_LOCK`, and `/dev/infiniband`; and the canonical model/JIT
mounts remain unchanged. The only container-contract deltas are the distinct
SparkCache ownership labels, the read-only connector staging mount, the
writable rank-local cache mount, `PYTHONPATH`, the SparkCache context-profile
label, removal of LMCache's connector arguments/servers, and addition of the
SparkCache connector arguments. Final inspection rejects any extra label or
mount and binds the full command, merged image/runtime environment,
entrypoint, image, and model/checkpoint receipt. The HostConfig comparison is
an exact complete operational projection, including lifecycle/restart policy,
auto-remove, rootfs/security controls, capabilities, devices/GPU requests,
binds, network/IPC/PID/user/cgroup namespaces, runtime, shared memory, ulimits,
CPU/memory/blkio/PID/OOM resources, DNS/ports, sysctls, storage, and log policy.

Generate and inspect the complete cutover plan. `plan` performs no remote
operation:

```powershell
python scripts/sparkring_exl3_sparkcache_launcher.py `
  --site .sparkring/bootstrap-exl3/site.yaml `
  --baseline-profile .sparkring/bootstrap-exl3/launch.json `
  --candidate .sparkring/exl3-sparkcache-candidate.json `
  plan > .sparkring/exl3-sparkcache-cutover-plan.json
```

The launcher supports `status`, `cutover`, `restart-engines`, `restart-stack`,
and `rollback`. Every remote invocation requires `--execute` plus the exact
command-specific confirmation printed in the plan. `status` is read-only
remote: its patch checks use `docker exec` against an already-running
exact-image engine, its model check is inventory/path/size only, and it never
creates a probe container. The other commands are
**STOPS SERVING** and **MUTATES HOST**.

This candidate orchestrator is Docker-only and rejects a Podman baseline before
composing any phase. Its inspection, quiescence, exact-ownership, and cleanup
contracts intentionally share one container-engine authority.

The rollback sequence is explicit: stop a residual exactly owned hash helper,
prove or remove the exactly owned SparkCache candidate, remove any exactly
labelled baseline remnants, prove all-rank model-process quiescence, start all
four canonical LMCache servers, check them, start all four canonical engines,
then check their exact container contracts and links without re-hashing the
checkpoint live. It finishes with explicit hash-helper and candidate-absence
proofs on all four ranks. Proven absence is idempotent; Docker/SSH uncertainty
is not treated as absence.

Rollback is best effort but never bypasses the safety barrier. It attempts all
owned helper/candidate/baseline cleanup even if an earlier cleanup step fails.
If the subsequent all-rank quiescence proof still fails, it does not start the
baseline engines; it records the dependent phases as skipped and leaves serving
stopped for operator inspection. Starting a model beside an unproved hash or
model process is not an acceptable rollback.

## Inputs and read-only attestations

1. Supply a checkpoint identity receipt produced during an engine-down or
   offline preparation of the exact mounted artifact.
   `checkpoint_identity_sha256` from the receipt, not a mutable path or Hub
   tag, is the candidate input. **Never generate or verify the full receipt by
   `docker exec` inside a serving engine.** If no trustworthy receipt exists,
   schedule a separate outage: stop the exactly owned engines on every rank,
   prove no model mount/container or NVIDIA compute process remains, and run
   `checkpoint_manifest_generator.py` in the exact pinned image without
   `--gpus`, with networking disabled and the model mounted read-only. Restore
   the baseline before building this candidate. The cutover independently
   re-verifies that receipt on every rank under the same engine-down contract.

2. Stage only the allowlisted connector closure on every rank and compute its
   identity using `scripts/connector_bundle_manifest.py`. The identity must be
   recomputed remotely before any source engine is stopped; checking only the
   workstation copy is insufficient.

3. Attest the deployed vLLM image contains both published SparkCache
   compatibility semantics:

   - the exact `spark_req_id` invalid-block loop fetches
     `self.requests.get(spark_req_id)`, null-guards it, then adds every output
     placeholder to the discard count before resetting placeholders and clean
     recompute; and
   - `_validate_kv_transfer_vmm` exempts only
     `SparkContextCacheConnector` from the expandable-segments registration
     restriction.

   The source attestation rejects unreachable lookalikes (including code after
   constant-true terminal loops), duplicate/wrong request lookups, reordered
   repair statements, and a connector exemption that does not dominate the
   real rejection.

4. Verify the cache source is a writable, rank-local filesystem on every
   Spark. Do not place all ranks on one shared directory. Record free bytes,
   filesystem type, mount source, and device identity.

5. Inspect the generated cutover and rollback plans. Live execution is both
   **STOPS SERVING** and **MUTATES HOST** and needs the operator's explicit
   authorization.

## Live promotion sequence

Run each stage separately and retain rank-complete logs, output JSON, health,
restart/OOM state, link carrier, cache inventory, and wall-clock boundaries.
Do not promote a later optimization after an earlier mandatory gate fails.

### A. Store and engine-restart restore

Start with end-of-prefill snapshots, Python restore, and one 32K-class exact
prompt. Native APC is allowed during the store request, but the engine restart
is mandatory: it clears process-local APC while leaving NVMe SparkCache data.

```powershell
python scripts/context_cache_gate.py `
  --phase store `
  --base-url http://<RANK0>:8000 `
  --model glm-5.2-exl3-tr3-3.25bpw `
  --words 24000 --max-tokens 64 `
  --output-dir .sparkring/evidence/exl3-sparkcache

# Use the inspected candidate launcher's engine-restart action here.

python scripts/context_cache_gate.py `
  --phase restore `
  --base-url http://<RANK0>:8000 `
  --model glm-5.2-exl3-tr3-3.25bpw `
  --words 24000 --max-tokens 64 `
  --output-dir .sparkring/evidence/exl3-sparkcache
```

Mandatory observations:

- one durable manifest commits on each of ranks 0-3 without a nudge request;
- all four workers advertise the same context digest and distinct TP/DCP keys;
- the engine restart preserves the manifests and objects;
- restore reads and SHA-verifies every selected chunk before releasing state;
- prompt and completion SHA-256 values are recorded; and
- API health, links, restart counts, and OOM state remain clean.

The gate requests `return_tokens_as_token_ids` with streamed logprobs and
records the runtime's original generated token IDs. It requires exact,
non-empty completion-token-ID equality. Ground-truth recall is diagnostic only
and can never rescue a token mismatch. Teacher-forced per-position logits/KLD
still require an internal model-runner probe; the OpenAI API does not expose
the full logits needed for that stronger claim.

### B. Full-stack restart durability

Stop and recreate all four candidate engines, leaving only the rank-local NVMe
stores. Repeat the restore request. This is SparkCache's decisive distinction
from the volatile LMCache CS512 L1 path. Require all four ranks to rediscover
metadata, achieve quorum, verify payloads, and restore without a prefill-sized
TTFT.

### C. Corruption withdrawal and recovery

After seeding and proving a healthy restore, run the existing sabotage matrix
against the candidate paths:

```powershell
$env:SPARKRING_BASE_URL = "http://<RANK0>:8000"
$env:SPARKRING_MODEL = "glm-5.2-exl3-tr3-3.25bpw"
$env:SPARKRING_TARGETS = "<R0_SSH>,<R1_SSH>,<R2_SSH>,<R3_SSH>"
$env:SPARKRING_CONTAINER = "glm52-sparkring-exl3-sparkcache-v51-r{rank}"
$env:SPARKRING_CACHE_ROOT = "/var/lib/sparkring/sparkcache-exl3"
$env:SPARKRING_ENGINE = "/opt/sparkring/sparkcache-staging-v2/sparkcache/persistent_context_cache/cache_manifest.py"
$env:SPARKRING_VERIFY_SCRIPT = "/tmp/context_cache_verify_store.py"
python scripts/context_cache_p3_matrix.py `
  --base-url $env:SPARKRING_BASE_URL `
  --output .sparkring/evidence/exl3-sparkcache/corruption.json
```

That command is dry-run only. Inspect its plan, then authorize the destructive
cache-object drill explicitly:

```powershell
python scripts/context_cache_p3_matrix.py `
  --base-url $env:SPARKRING_BASE_URL `
  --output .sparkring/evidence/exl3-sparkcache/corruption.json `
  --execute --confirmation CORRUPT-EXL3-SPARKCACHE-RANKS-1-2-3
```

The execution artifact is deliberately **private raw evidence**: it contains
SSH authorities, host paths, verifier records, and bounded container logs.
Never commit or publish it directly. After the run reaches a terminal state,
produce a closed public receipt which binds the raw file by SHA-256 while
omitting those fields:

```powershell
python scripts/context_cache_p3_reduce.py `
  --input .sparkring/evidence/exl3-sparkcache/corruption.json `
  --output .sparkring/evidence/exl3-sparkcache/corruption.sanitized.json
```

The reducer refuses incomplete/running evidence, duplicate or non-finite JSON,
unknown error classifications, noncanonical mode/rank order, and output-name
reuse. The sanitized receipt is still only an evidence summary; it does not
replace the private raw artifact during audit.

The matrix's verifier and sabotage helpers execute on each host, not inside the
serving container. Therefore `SPARKRING_ENGINE` must name the host-side staged
path (`/opt/sparkring/sparkcache-staging-v2/...` for this candidate), while the
container continues to see the same file under `/opt/sparkcache-staging/...`.
The matrix requires explicit absolute cache/engine paths and accepts a
rank-formatted container pattern. Stage the repository's unchanged
`scripts/context_cache_verify_store.py` at the declared verifier path on every
rank before execution. Each run generates a fresh nonce and places it inside
the aligned cached prefix. For each mode, the gate binds the streamed API
request ID to exactly one common `store_committed` digest on every physical
rank, first for the target and then for a distinct sentinel. Both digests must
also verify healthy on all four ranks. This stays rerunnable with deterministic
seeds and rejects unrelated concurrent cache traffic instead of inferring
ownership from a global manifest-set delta. The sabotage tool must select that
exact 64-hex target; it may not choose the first manifest in the store. The
target must remain present but verify corrupt only on the damaged worker
immediately after sabotage.

Native vLLM Automatic Prefix Caching remains enabled during this drill, so the
matrix isolates every API request into a distinct APC namespace with the
`cache_salt` request field. For each mode, target seed, sentinel seed,
pre-sabotage probe, corruption trigger, and recovery receive deterministic but
different salts derived from the private run nonce, mode, and request role.
The target's prompt and tokenization remain identical across its four requests,
so its SparkCache digest is unchanged, while the corruption trigger cannot be
satisfied by an intact native APC entry populated before sabotage. Dry-run
plans record the derivation contract; private evidence records only SHA-256
receipts for transmitted salts, never the raw values. `--max-tokens` can bound
the streamed comparison length and defaults to 48; exact, non-empty token-ID
equality remains mandatory at every correctness boundary.

The triggering request can invalidate and republish the target before a host
snapshot observes an absent manifest. The gate therefore binds full-digest,
request-ID connector events: the damaged worker must emit
`worker_invalidated`, scheduler rank 0 must emit `scheduler_retired`, and each
must precede its same-request `store_committed` event. The connector identity
must either equal the public SSE request ID or be exactly one shared vLLM child
ID formed as the escaped public ID plus `-0-[0-9a-f]{8}` (for example,
`cmpl-a6c9ec16baa5aee1-0-b11c9acd`). Every expected rank must use the same
resolved identity. Multiple children, malformed suffixes, prefix collisions,
or an event whose claimed rank differs from its emitting container invalidate
the evidence envelope. The private evidence records both the public and
resolved connector IDs. Events from the later recovery call cannot satisfy
this proof. A subsequent identical request must return the exact
expected token IDs, and the final verifier must find the target healthy on all
four ranks. Unaffected non-scheduler workers
must retain healthy target copies; the sentinel and every unrelated entry must
remain identical and healthy throughout. The host verifier observes payload
state but does not claim to perform withdrawal. Counts, timing-dependent
absence, or an empty final store cannot substitute for this evidence.

Rank-0 scheduler and worker connectors are separate instances sharing durable
storage. If scheduler retirement removes a corrupt manifest while the worker
still has a stale in-memory `_held` offer, the worker revalidates the manifest
at the next store admission, withdraws the stale offer, and permits the clean
recompute to republish. A stale `_held` value may never suppress recovery.

### D. Rollback

Exercise rollback even after a successful candidate run:

1. remove only containers carrying the SparkCache candidate ownership label;
2. start the exact receipt-gated EXL3+LMCache CS512 baseline, including all
   four LMCache servers;
3. require four engines/four servers, HTTP health, zero OOM/restarts, and
   `carrier=1` plus `operstate=up` for both exact site-defined production ring
   interfaces on every Spark; and
4. verify the candidate containers are absent/stopped and the baseline image
   ID/profile labels match the receipt.

Any unknown SSH/container state makes rollback incomplete; it is never treated
as absence.

### E. Matched C8 interference

Only after correctness and rollback pass, compare cache-off and SparkCache
with exact tokenized prompt artifacts. Use interleaved order (ABBA or BAAB),
at least five valid repetitions per arm, 16K actual context, C8, 25 seconds,
1,024 maximum tokens, temperature zero, 100% unique contexts, no prefill
scouts, and the same DCP4/KV capacity. Novel prompts keep the benchmark from
measuring warm-cache acceleration.

Report medians and dispersion for aggregate decode, TTFT, effective
concurrency, errors, underfill, GPU/power, snapshot time, bytes written,
writer queue depth, and I/O preemption. A two-run difference is not a
performance attribution. The first candidate fails if serving freezes, links
drop, any cell is invalid, or median C8 regresses beyond a predeclared 5%
budget.

## Optimization order after the first pass

1. Enable checksum-attested native restore and repeat A-E.
2. Enable v51 streaming snapshots and prove no-nudge publication again on the
   current EXL3 image.
3. Add activity-aware I/O preemption and repeat the interleaved C8 gate.
4. Only then investigate prefix-aware partial restore, LRU/TTL, and orphan
   collection.

The phases are independent candidates. A native or streaming result cannot be
attributed to the simpler first-gate configuration.
