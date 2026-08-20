# SparkCache DCP2 dry-run launch plan — TEMPLATE

> **Historical operator snapshot:** The container names, active-service
> description, image identity, mounts, and blocker states below were recorded
> for the 2026-08-01 NF3 deployment. They are not assertions about any live
> cluster. As of the 2026-08-03 operator snapshot, the EXL3 3.25 bpw
> service used LMCache CS512 and had SparkCache disabled. This NF3 TP4/DCP2
> SparkCache path is offline-validated only; rerun discovery and every
> attestation before using this template.

**Date:** 2026-08-01
**Lane:** public-functional
**Target:** stopped NF3 TP4/DCP2 candidate containers
**Active service at snapshot time:** TP4/DCP4 fixed-MTP2 (not DCP2)
**Status:** OFFLINE preparation — no host mutation, no serving interruption

## Purpose

This document is a **plan template** defining the configuration changes
needed to enable SparkCache on the stopped NF3 DCP2 candidate
containers. At the dated snapshot, the active live service was TP4/DCP4,
not DCP2.
Four stopped DCP2 containers exist as the intended candidate for a
future cutover. Nothing here starts, stops, replaces, rebuilds, copies
into, or alters a Spark host or container. It is OFFLINE work under
the AGENTS.md safety boundary.

The executable form of this plan is
`scripts/sparkcache_config_generator.py`, a pure-function config
generator/verifier with unit tests in
`scripts/test_sparkcache_config_generator.py`.

**Offline tests do NOT establish live acceptance.** They prove code
paths only. Live acceptance requires the full validation gate in the
runbook, with checkpoint hashes and runtime tensor equivalence
independently attested.

## BLOCKERS — status after physical-worker identity fix

### B1: TP/DCP physical-worker identity gap — RESOLVED (code)

Previously, TP ranks 0 and 2 shared DCP rank 0 and produced identical
`storage_key` because `CacheIdentity` had no `tp_shard_rank` field.
This was the primary blocker.

**Resolution:** Added `tp_shard_rank` field to `CacheIdentity` and
`_physical_rank()` to the connector. Worker reports now use physical TP
rank (0..tp_degree-1), quorum requires all physical TP workers, and
persistent storage keys distinguish TP0 from TP2 and TP1 from TP3.

**Compatibility choice:** Old entries written without `tp_shard_rank`
have a different `storage_key` and will miss, causing a clean
re-prefill — the fail-closed path. Old entries are NOT silently
reinterpreted. Under DCP4 (where physical rank == DCP rank), the
new field equals the old rank, but the `storage_key` still changes
because `to_wire()` includes the new field. This is intentional.

**Proven by:**
- `test_dcp2_storage_keys_differ_for_tp_ranks_sharing_dcp_rank`
- `test_dcp2_reports_from_tp0_and_tp2_not_deduplicated`
- `test_dcp2_quorum_requires_all_four_physical_workers`
- `test_dcp2_withdrawing_one_physical_worker_removes_quorum`
- `test_dcp4_legacy_entries_without_tp_shard_rank_miss`

**Remaining:** Live attestation that deployed cache tensor inventory
matches the new identity model is still required before live acceptance.

### B2: Checkpoint identity not attested

The checkpoint identity must be a 64-character SHA-256 covering every
deployed weight shard and cache-affecting artifact. The previous
hand-computed hashes (`4ae049...`/`15202f...`) only covered
revision|config|index (and a few draft pins), omitting the target's
184 weight shards. They have been **removed** from tests and docs.

**Generator implemented:** `scripts/checkpoint_manifest_generator.py`
recursively inventories every regular file under an artifact root,
streams SHA-256 hashes in 1 MiB chunks, and emits a versioned
canonical JSON receipt. The `checkpoint_identity_sha256` field is a
domain-separated SHA-256 over the canonical serialization of the
complete inventory, independent of filesystem enumeration order, root
directory name, and platform path separators. Path normalization uses
POSIX separators plus Unicode NFC, with collision detection. The
generator rejects symlink roots, symlink files, symlink directories,
non-regular files, duplicate normalized paths, files changing during
read (size/inode/dev/mtime_ns/ctime_ns), and empty inventory. Output
uses exclusive creation (no TOCTOU race). Tests in
`scripts/test_checkpoint_manifest_generator.py` cover all of these
cases.

**Resolution requires:** Running the generator against the complete
mounted artifacts on the deployed image. This is READ-ONLY REMOTE
and may be expensive (target model has ~184 weight shards). The
generator never downloads, mutates, or creates files in the artifact
root; the only write is the local receipt file. The resulting
`checkpoint_identity_sha256` is consumed verbatim via
`--target-checkpoint` and `--draft-checkpoint` CLI arguments to the
config generator.

### B3: vLLM patch semantics — RESOLVED for exact image

Public `preimages.json` records expected preimage SHA-256 values, but
these are not attestation against the deployed image. The deployed
overlay (59 modified + 12 new files per `runtime-lock.json`) may differ.

**Resolved** for image
`sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d`
via READ-ONLY REMOTE `inspect.getsource()`:

- The deployed build does not have `Scheduler._handle_kv_load_failure`.
  The actual seam is `Scheduler._handle_invalid_blocks` in
  `vllm/v1/core/sched/scheduler.py`. It calls
  `_update_requests_with_invalid_blocks` for async requests waiting
  for remote KVs and for running sync requests. Under recompute
  policy, for every affected request with output placeholders it
  executes
  `spark_request.async_tokens_to_discard += spark_request.num_output_placeholders`
  followed by `spark_request.num_output_placeholders = 0`. It logs
  recovery, marks failed async receives for retry, and returns
  affected sync IDs to skip.
- `VllmConfig._validate_kv_transfer_vmm` in
  `vllm/config/vllm.py` returns before the expandable-segments
  incompatibility error when the connector is
  `SparkContextCacheConnector`, with the explicit rationale that
  this connector performs no GPU memory registration.

This attestation is scoped to the exact image ID above. If the image
changes, re-attestation is required.

### B4: quantization_layout and rope_layout — RESOLVED for exact image

The connector hard-codes `quantization_layout="nvfp4_ds_mla-per-token-v1"`
and `rope_layout="glm52-rope-v1"`. These are provisionally consistent
with the NF3 recipe docs but have not been verified against the
deployed image's actual KV configuration.

**Resolved** for image
`sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d`
/ model mount `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`
via READ-ONLY REMOTE observation:

- argv: `--kv-cache-dtype nvfp4_ds_mla`
- env: `VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`
- config: `head_dim=192`, `kv_lora_rank=512`,
  `qk_nope_head_dim=192`, `qk_rope_head_dim=64`,
  `rope_interleave=true`, `rope_parameters.rope_theta=8000000`,
  `rope_parameters.rope_type=default`

These facts support the connector labels
`nvfp4_ds_mla-per-token-v1` and `glm52-rope-v1` for this exact
deployed image and model mount. The label strings are connector
identifiers, not config field values read verbatim. This attestation
is scoped to the exact image ID and mount above. If either changes,
re-attestation is required.

### B5: Streaming snapshots must stay disabled for first gate

Streaming snapshots require a native ARM64/SM121 library, absolute
paths, hashes, lease seam, cache tensor names/shapes, checkpoint
manifests, and all four physical-worker participation to be attested.

**Status:** `spark_cache_streaming_snapshots=false` enforced by the
config generator/verifier for the first live gate. Do not enable until
all attestation is complete.

### B6: Connector source not staged/importable — Unresolved

The deployed image does NOT contain `spark_context_cache_connector`
(`importlib.util.find_spec('spark_context_cache_connector')` returns
`None` inside the exact deployed image on every rank). The operator
must stage the connector source on each host before prepare/cutover.

**Bundle identity verification:** The cutover script verifies a
domain-separated SHA-256 over the required connector files (17 files
covering the full import closure: 3 top-level connector modules, 1
persistent-context-cache engine, 10 streaming modules, 2 runtime-patches
modules, and 1 JSON lease contract) before proceeding. The identity is
generated offline by `scripts/connector_bundle_manifest.py` and
re-verified on every host via an inline Python verifier (`python3 -c`)
that applies fail-closed semantics: lstat (no symlink dereference),
reject non-regular/missing/extra files, hash in 1 MiB chunks, and
compare stable stat metadata before/after every read.

## Enable switch

Per `sparkcache/README.md`, the enable switch is the **presence** of
the complete `--kv-transfer-config` argument. The public connector
does **not** consume a `SPARK_CONTEXT_CACHE_ENABLE` flag. That variable
is an image environment flag with no connector authority; the generator strips
it from the env.

Omit `--kv-transfer-config` entirely to disable SparkCache.

The pinned vLLM factory also requires
`--disable-hybrid-kv-cache-manager` because this connector does not
advertise HMA support.

## Canonical --kv-transfer-config schema

```json
{
  "kv_connector": "SparkContextCacheConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "spark_context_cache_connector",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "spark_cache_root": "/var/tmp/sparkring-public-validation/context-cache",
    "spark_cache_target_checkpoint_sha256": "<64 lowercase hex>",
    "spark_cache_draft_policy": "separate",
    "spark_cache_draft_checkpoint_sha256": "<64 lowercase hex>",
    "spark_cache_store": true,
    "spark_cache_restore": true,
    "spark_cache_streaming_snapshots": false
  }
}
```

## Deployed variant summary

| Setting | Active DCP4 | Stopped DCP2 candidate |
|---|---|---|
| Container pattern | `glm52-sparkring-nf3-dcp4-fixedk2-r{rank}` | `glm52-sparkring-nf3-dcp2-r{rank}` |
| TP / DCP | 4 / 4 | 4 / 2 |
| DCP rank map | `[0,1,2,3]` (identity) | `[0, 1, 0, 1]` |
| Physical TP groups | `[0,1,2,3]` | `[0,1]` and `[2,3]` |
| Max model len | 1,048,576 | 524,288 |
| Image ID | same: `sha256:ab6bddba...` | same |
| SparkCache | disabled (no `--kv-transfer-config`) | disabled |

## Physical-worker identity model

The connector separates two concepts:

1. **Physical TP worker identity** (`_physical_rank()`): unique across
   all four TP ranks (0..3). Used for worker reports, quorum
   admission/withdrawal, and persistent namespace (`tp_shard_rank`).
2. **DCP-local shard identity** (`_worker_rank()`): 0 or 1 for
   TP4/DCP2. Used for token-position ownership and DCP slicing
   semantics. Unchanged.

Quorum requires every physical TP worker in `range(tp_degree)`, not
every DCP-local rank. A withdrawal from one physical worker
invalidates quorum even when another worker has the same DCP-local
rank.

## Draft policy: separate

The NF3 recipe defines the MTP draft as a **separate artifact** at
`aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`, distinct from the target
`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`.

**Decision: `spark_cache_draft_policy="separate"`**

## Executable config generation

The dry-run plan is executed by
`scripts/sparkcache_config_generator.py`. Note: `docker inspect` works
on stopped containers (it reads container metadata, does not require a
running process).

### Disabled (default, no checkpoints required)

```bash
docker inspect glm52-sparkring-nf3-dcp2-r0 | \
  python scripts/sparkcache_config_generator.py
```

### Enabled (requires canonical manifest identities)

First, generate checkpoint identity receipts. The script is supplied
via stdin to the active same-image container — it is not baked into
the deployed image. The receipt JSON (including
`checkpoint_identity_sha256`) is captured on stdout and written to
an explicit local file. This is READ-ONLY REMOTE (reads and hashes
every weight shard on the serving host) and may be expensive (target
model has ~184 weight shards). The operator must schedule or confirm
acceptance of the I/O and CPU load before running:

```bash
# Target model — remote read/hash, local receipt write
cat scripts/checkpoint_manifest_generator.py | \
  docker exec -i glm52-sparkring-nf3-dcp4-fixedk2-r0 \
    /opt/venv/bin/python - \
      --artifact-root /models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  > target_receipt.json

# Draft model — same pattern
cat scripts/checkpoint_manifest_generator.py | \
  docker exec -i glm52-sparkring-nf3-dcp4-fixedk2-r0 \
    /opt/venv/bin/python - \
      --artifact-root /mtp-draft \
  > draft_receipt.json
```

Then feed the `checkpoint_identity_sha256` from each local receipt
into the config generator:

```bash
docker inspect glm52-sparkring-nf3-dcp2-r0 | \
  python scripts/sparkcache_config_generator.py \
    --enable \
    --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
    --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256>
```

The `<*_CHECKPOINT_IDENTITY_SHA256>` values are the
`checkpoint_identity_sha256` field from the manifest generator's
receipt JSON. They remain unresolved placeholders until the generator
is run against the complete mounted artifacts.

## What the offline tests validate

1. DCP2 interleave math (rank 0 → even positions, rank 1 → odd).
2. Quorum requires all four physical TP workers, not just two DCP-local ranks.
3. Reports from TP0 and TP2 are not deduplicated.
4. Withdrawing one physical worker removes quorum even if its paired
   DCP-local rank remains.
5. Persistent storage keys differ for TP0 vs TP2 and TP1 vs TP3.
6. DCP-local token ownership remains based on DCP rank.
7. DCP4 compatibility: physical rank == DCP rank, old entries miss
   (fail-closed).
8. Malformed/out-of-range physical rank reports fail closed.

## What is NOT validated here

- No live request is sent.
- No container is started, stopped, or replaced.
- Checkpoint identity not attested — generator implemented (B2), not yet run against deployed artifacts. Hashes are unscheduled.
- vLLM patches attested (B3) — resolved for exact image `sha256:ab6bddba...`.
- quant/rope layouts attested (B4) — resolved for exact image / mount.
- Connector source not staged/importable (B6) — `find_spec` returns `None` in the deployed image. Operator must stage connector source on each host before prepare/cutover. The cutover script verifies a domain-separated bundle identity (`--connector-bundle-identity`) over the staged connector files; the identity is generated offline by `scripts/connector_bundle_manifest.py`.
- No live authorization (B7) — no explicit user authorization to start/stop containers.
- Runtime tensor equivalence is not attested.
- **Offline tests do not establish live acceptance.**

After B2 is resolved (checkpoint identity attested by running the
manifest generator against deployed mounts), B6 is resolved
(connector source staged on every host), B1's live attestation is
complete, and live authorization (B7) is obtained, the operator
authorizes a live validation window using the runbook at
`docs/SPARKCACHE_DCP2_LIVE_RUNBOOK.md`. B3 and B4 are already
resolved for the exact shared image ID.

The live cutover is executed by `scripts/live_dcp2_cutover.py`:

```bash
# Read-only preflight (no mutation)
python scripts/live_dcp2_cutover.py plan \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
  --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256> \
  --cache-root /var/tmp/sparkring-public-validation/context-cache \
  --connector-staging /opt/sparkcache-host-staging \
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256>

# Mutating: create target containers (confirmation-gated)
python scripts/live_dcp2_cutover.py prepare \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
  --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256> \
  --cache-root /var/tmp/sparkring-public-validation/context-cache \
  --connector-staging /opt/sparkcache-host-staging \
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256> \
  --execute --confirmation PREPARE-NF3-DCP2-SPARKCACHE-ALL-FOUR

# STOPS SERVING: cutover (confirmation-gated)
python scripts/live_dcp2_cutover.py cutover \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
  --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256> \
  --cache-root /var/tmp/sparkring-public-validation/context-cache \
  --connector-staging /opt/sparkcache-host-staging \
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256> \
  --health-timeout 120 \
  --execute --confirmation STOP-DCP4-START-DCP2-SPARKCACHE
```

Target containers use the pattern
`glm52-sparkring-nf3-dcp2-sparkcache-r{rank}` (distinct from
baseline `glm52-sparkring-nf3-dcp2-r{rank}` to preserve evidence).
Rollback is idempotent and inspect-confirmed via:

```bash
python scripts/live_dcp2_cutover.py rollback \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --execute --confirmation STOP-DCP2-SPARKCACHE-RESTORE-DCP4
```
