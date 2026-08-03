# SparkCache DCP2 live validation runbook

> **Historical operator snapshot:** The container names, active-service
> description, image identity, mounts, and blocker states below were recorded
> for the 2026-08-01 NF3 deployment. They are not assertions about the current
> cluster. As of the 2026-08-03 operator snapshot, the current EXL3 3.25 bpw
> service uses LMCache CS512 and has SparkCache disabled. This NF3 TP4/DCP2
> SparkCache path remains offline-validated only. Rediscover the live topology,
> regenerate the plan, and re-attest every identity before executing it.

**Date:** 2026-08-01
**Lane:** public-functional
**Prerequisite:** `docs/SPARKCACHE_DCP2_DRY_RUN_PLAN.md` reviewed,
B2 resolved (checkpoint identity attested), B1 live attestation
complete, B3/B4 resolved for exact shared image ID, connector source
staged on every host with bundle identity verified, and operator
authorization obtained
**Safety class:** STOPS SERVING — requires explicit user authorization

## Purpose

Step-by-step procedure to validate SparkCache on the stopped NF3
TP4/DCP2 candidate containers. At the dated snapshot, the active live service
was TP4/DCP4 fixed-MTP2
(`glm52-sparkring-nf3-dcp4-fixedk2-r{rank}`).
Four stopped DCP2 containers exist as the intended candidate. This
runbook assumes the DCP2 variant will be started (via
`scripts/live_dcp2_cutover.py`) and SparkCache is currently disabled
(no `--kv-transfer-config` present).

**Offline tests do NOT establish live acceptance.** They prove code
paths only. Live acceptance requires the full validation gate below,
with checkpoint hashes and runtime tensor equivalence independently
attested.

## BLOCKERS — status

| ID | Blocker | Status |
|---|---|---|
| B1 | TP/DCP physical-worker identity gap | **RESOLVED (code):** `tp_shard_rank` added to `CacheIdentity`, `_physical_rank()` added to connector. Quorum uses physical TP ranks. **Remaining:** live attestation of cache tensor inventory. |
| B2 | Checkpoint identity not attested | **Unresolved.** `scripts/checkpoint_manifest_generator.py` implemented (offline tests pass). Requires running generator against mounted model artifacts. Hashes are unscheduled — no live authorization to run the I/O-heavy scan. |
| B3 | vLLM patch semantics not attested | **RESOLVED** for image `sha256:ab6bddba...`: `Scheduler._handle_invalid_blocks` performs async rollback via `async_tokens_to_discard`; `VllmConfig._validate_kv_transfer_vmm` exempts `SparkContextCacheConnector`. |
| B4 | `quantization_layout`/`rope_layout` not attested | **RESOLVED** for image `sha256:ab6bddba...` / model mount `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`: argv `--kv-cache-dtype nvfp4_ds_mla`, env `VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`, config confirms DS-MLA + interleaved RoPE. Labels `nvfp4_ds_mla-per-token-v1` and `glm52-rope-v1` are consistent with these facts. |
| B6 | Connector source not staged/importable | **Unresolved.** `importlib.util.find_spec('spark_context_cache_connector')` returns `None` inside the deployed image on every rank. The operator must stage the connector source on each host before prepare/cutover. The staging step is MUTATES HOST and is not yet executed. The cutover script verifies a domain-separated bundle identity (`--connector-bundle-identity`) over the staged connector files before proceeding; the identity is generated offline by `scripts/connector_bundle_manifest.py`. |
| B7 | No live authorization | **Unresolved.** No explicit user authorization to start/stop containers. |

Do not proceed to pre-flight until B2 and B6 are resolved, B1's live
attestation is complete, and live authorization (B7) is obtained.
B3 and B4 are resolved for the exact shared image ID
`sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d`
(see steps 5–6 for the attestation evidence).

## Pre-flight (READ-ONLY REMOTE)

1. **Verify the stopped DCP2 candidate containers match the recorded snapshot.**
   ```bash
   docker inspect glm52-sparkring-nf3-dcp2-r{rank} --format '{{.Image}}'
   ```
   Confirm all four report
   `sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d`.

2. **Verify image equality between active DCP4 and stopped DCP2.**
   The stopped DCP2 containers cannot be `docker exec`'d (they are
   stopped). For read-only source/config attestation, use the active
   DCP4 rank-0 container — but **only after** verifying it has the
   exact same full image ID as the stopped DCP2 candidate:
   ```bash
   docker inspect glm52-sparkring-nf3-dcp4-fixedk2-r0 --format '{{.Image}}'
   docker inspect glm52-sparkring-nf3-dcp2-r0 --format '{{.Image}}'
   ```
   If the image IDs are not identical, do NOT proceed — leave
   attestation blocked. The active DCP4 container is the attestation
   proxy only when image equality is established.

3. **Verify DCP2 configuration.** When the DCP2 candidate was last
   running, logs showed `decode_context_parallel_size=2` and DCP rank
   map `[0,1,0,1]` with worker mappings `TP0_DCP0`, `TP1_DCP1`,
   `TP2_DCP0`, `TP3_DCP1`. Verify these mappings are still present
   in the container definitions.

4. **Verify the deployed entrypoint shape.** The active DCP4
   containers use `Entrypoint=["/usr/bin/env"]` and
   `Path="/usr/bin/env"`, with a `Cmd` of
   `["-u", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "/opt/venv/bin/vllm", "--model", ...]`.
   The cutover script normalizes this wrapper into a clean vLLM argv
   before applying DCP2 rewrites and cache config. Verify the
   entrypoint and `Cmd` shape match:
   ```bash
   docker inspect glm52-sparkring-nf3-dcp4-fixedk2-r0 --format '{{.Config.Entrypoint}}'
   docker inspect glm52-sparkring-nf3-dcp4-fixedk2-r0 --format '{{.Config.Cmd}}'
   ```

5. **Attest vLLM patch semantics in the deployed image** (B3 RESOLVED).
   Read-only `docker exec` into the **active DCP4 rank-0 container**
   (only if image equality from step 2 is confirmed). The deployed
   build does not have `Scheduler._handle_kv_load_failure`; the
   actual seam is `Scheduler._handle_invalid_blocks`:
   ```bash
   docker exec glm52-sparkring-nf3-dcp4-fixedk2-r0 \
     /opt/venv/bin/python -c "
   import inspect
   from vllm.v1.core.sched.scheduler import Scheduler
   src = inspect.getsource(Scheduler._handle_invalid_blocks)
   print('async_rollback_present:', 'async_tokens_to_discard' in src and 'num_output_placeholders' in src)
   from vllm.config.vllm import VllmConfig
   src2 = inspect.getsource(VllmConfig._validate_kv_transfer_vmm)
   print('vmm_exemption_present:', 'SparkContextCacheConnector' in src2)
   "
   ```
   **Observed evidence** (image `sha256:ab6bddba...`):
   - `_handle_invalid_blocks` calls `_update_requests_with_invalid_blocks`
     for async requests waiting for remote KVs and for running sync
     requests. Under recompute policy, for every affected request with
     output placeholders it executes
     `spark_request.async_tokens_to_discard += spark_request.num_output_placeholders`
     followed by `spark_request.num_output_placeholders = 0`. It
     logs recovery, marks failed async receives for retry, and returns
     affected sync IDs to skip.
   - `_validate_kv_transfer_vmm` returns before the
     expandable-segments incompatibility error when the connector is
     `SparkContextCacheConnector`, with the explicit rationale that
     this connector performs no GPU memory registration.

6. **Attest quantization_layout and rope_layout** (B4 RESOLVED).
   The model mount destination is `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`
   and the draft mount is `/mtp-draft` (discovered from `docker inspect`).
   Read the config and environment to confirm the connector labels:
   ```bash
   docker exec glm52-sparkring-nf3-dcp4-fixedk2-r0 \
     /opt/venv/bin/python -c "
   import json, os
   cfg = json.load(open('/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid/config.json'))
   print('kv_cache_dtype:', cfg.get('kv_cache_dtype'))
   print('nvfp4_per_token_scale:', os.environ.get('VLLM_NVFP4_MLA_PER_TOKEN_SCALE'))
   print('head_dim:', cfg.get('head_dim'))
   print('kv_lora_rank:', cfg.get('kv_lora_rank'))
   print('qk_nope_head_dim:', cfg.get('qk_nope_head_dim'))
   print('qk_rope_head_dim:', cfg.get('qk_rope_head_dim'))
   print('rope_interleave:', cfg.get('rope_interleave'))
   print('rope_theta:', cfg.get('rope_parameters', {}).get('rope_theta'))
   print('rope_type:', cfg.get('rope_parameters', {}).get('rope_type'))
   "
   ```
   **Observed evidence** (image `sha256:ab6bddba...`, mount
   `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`):
   - argv: `--kv-cache-dtype nvfp4_ds_mla`
   - env: `VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`
   - config: `head_dim=192`, `kv_lora_rank=512`,
     `qk_nope_head_dim=192`, `qk_rope_head_dim=64`,
     `rope_interleave=true`, `rope_parameters.rope_theta=8000000`,
     `rope_parameters.rope_type=default`
   These facts support the connector labels
   `nvfp4_ds_mla-per-token-v1` and `glm52-rope-v1` for this exact
   deployed image and model mount. The label strings are connector
   identifiers, not config field values read verbatim.

7. **Attest checkpoint identity** (resolves B2). Run the canonical
   manifest generator against the mounted model artifacts. The target
   artifact root is `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` and
   the draft artifact root is `/mtp-draft` (both discovered from
   `docker inspect` in step 6).

   The script is **not** baked into the deployed image. Instead,
   pipe the checked-out script from the local repository into the
   container via stdin. The container runs `/opt/venv/bin/python -`
   (read script from stdin) with `--artifact-root` pointing at the
   discovered mount destination. The receipt JSON (including
   `checkpoint_identity_sha256`) is captured on stdout and written
   to an **explicit local file** — nothing is persisted inside the
   container:

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

   The `checkpoint_identity_sha256` field in each local receipt JSON
   is the value passed to `--target-checkpoint` and
   `--draft-checkpoint` in steps 9–11.

   **READ-ONLY REMOTE — load impact.** This step reads and hashes
   every weight shard on the serving host's storage. On the active
   DCP4 container, this competes for disk I/O and CPU with live
   inference traffic. The operator must schedule or confirm
   acceptance of this load before running. Do not run during peak
   traffic. The generator never writes to or mutates the artifact
   root; the only write is the local receipt file.

8. **Verify the cache bind mount.** The writable context-cache bind
   mount on every active rank has source and destination
   `/var/tmp/sparkring-public-validation/context-cache`. The cutover
   script requires `--cache-root` to match this destination exactly
   and validates it is a writable bind mount on every rank before
   proceeding. Verify:
   ```bash
   docker inspect glm52-sparkring-nf3-dcp4-fixedk2-r0 \
     --format '{{range .Mounts}}{{if eq .Destination "/var/tmp/sparkring-public-validation/context-cache"}}{{.Source}} RW={{.RW}}{{end}}{{end}}'
   ```

9. **Verify streaming snapshots are disabled.** The config generator
   enforces `spark_cache_streaming_snapshots=false`.

## Connector staging (MUTATES HOST — not yet executed)

The deployed image does NOT contain `spark_context_cache_connector`
(`importlib.util.find_spec('spark_context_cache_connector')` returns
`None` inside the exact deployed image on every rank). A KV config
alone cannot launch. The operator must stage the connector source on
each host at an explicit directory. The complete canonical file list
is defined in ``scripts/connector_bundle_manifest.py`` as
``REQUIRED_FILES`` — 17 files (16 Python modules + 1 JSON contract)
beneath ``sparkcache/``:

- `sparkcache/spark_context_cache_connector.py`
- `sparkcache/spark_context_cache_codec.py`
- `sparkcache/spark_context_cache_store.py`
- `sparkcache/persistent_context_cache/cache_manifest.py`
- `sparkcache/streaming/` — 10 `.py` files (eager + runtime)
- `sparkcache/runtime_patches/` — 2 `.py` files + 1 JSON contract

The `prepare` command bind-mounts this directory read-only into the
target at `/opt/sparkcache-staging` and sets
`PYTHONPATH=/opt/sparkcache-staging:/opt/sparkcache-staging/sparkcache:/opt/spark-vllm`
so that both top-level `spark_context_cache_{connector,codec,store}`
and the `sparkcache.streaming` package are importable.

**This staging step is MUTATES HOST** — it copies files to each
Spark host. It must be performed by the operator before running
`prepare`. The cutover script's `plan` command checks for the
staging directory's existence on every host (read-only) and reports
a blocker if it is missing. The script never stages or copies
anything itself.

Before running `prepare`, generate the connector bundle identity from
the staged directory using the offline manifest script:

```bash
python scripts/connector_bundle_manifest.py --staging-root /opt/sparkcache-host-staging
```

This computes a domain-separated SHA-256 over the required connector
files (17 files covering the full import closure: 3 top-level connector
modules, 1 persistent-context-cache engine, 10 streaming modules,
2 runtime-patches modules, and 1 JSON lease contract). The resulting
64-hex digest is passed to `--connector-bundle-identity` in
plan/prepare/cutover. The cutover script re-verifies this identity on
every host before stopping any source container by sending an inline
Python verifier via `python3 -c` that applies fail-closed semantics:
lstat (no symlink dereference), reject non-regular/missing/extra
files, hash in 1 MiB chunks, and compare stable stat metadata
before/after every read.

## Enable SparkCache via cutover script (STOPS SERVING — requires authorization)

The cutover is executed by `scripts/live_dcp2_cutover.py`, which
automates the full stop-source / start-target / health-poll / rollback
sequence. It uses **distinct target container names**
(`glm52-sparkring-nf3-dcp2-sparkcache-r{rank}`) that do NOT
collide with the baseline DCP2 candidates
(`glm52-sparkring-nf3-dcp2-r{rank}`), preserving the stopped
baseline containers as evidence.

The CLI requires four `--node RANK=SSH_TARGET` arguments (one per
rank 0–3). All mutating operations require `--execute` and an
require `--target-checkpoint`, `--draft-checkpoint`,
`--cache-root`, `--connector-staging`, and
`--connector-bundle-identity`. The `--cache-root` and
`--connector-staging` arguments are validated as absolute POSIX paths
(no `.`/`..`, no NUL/newline, not root `/`).

### Command reference

```bash
python scripts/live_dcp2_cutover.py --help
```

Subcommands: `plan` (read-only preflight), `status` (read-only),
`prepare` (mutating — creates targets), `cutover` (STOPS SERVING),
`rollback` (STOPS SERVING).

### Step 10: Plan — read-only preflight

```bash
python scripts/live_dcp2_cutover.py plan \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
  --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256> \
  --cache-root /var/tmp/sparkring-public-validation/context-cache \
  --connector-staging /opt/sparkcache-host-staging \
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256>
```

Inspects all four DCP4 source containers, normalizes the
`/usr/bin/env` wrapper, generates/verifies the exact target config,
validates source state/image consistency, cache mount, connector
staging layout, and target collision/state. Prints explicit
readiness/blockers and returns nonzero on blockers. Does NOT mutate
anything. Checkpoint IDs are optional for `plan` — if omitted, B2
is reported as a blocker.

### Step 11: Prepare — mutating, confirmation-gated

```bash
python scripts/live_dcp2_cutover.py prepare \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --target-checkpoint <TARGET_CHECKPOINT_IDENTITY_SHA256> \
  --draft-checkpoint <DRAFT_CHECKPOINT_IDENTITY_SHA256> \
  --cache-root /var/tmp/sparkring-public-validation/context-cache \
  --connector-staging /opt/sparkcache-host-staging \
  --connector-bundle-identity <CONNECTOR_BUNDLE_IDENTITY_SHA256> \
  --execute --confirmation PREPARE-NF3-DCP2-SPARKCACHE-ALL-FOUR
```

Creates the four target containers
(`glm52-sparkring-nf3-dcp2-sparkcache-r{rank}`) as clones of the
DCP4 sources, then applies DCP2 rewrites (decode-context-parallel-size
4→2, max-model-len 1048576→524288, kv-cache-memory-bytes adjusted)
and the cache-enabled config from `sparkcache_config_generator.py`.
Checkpoint identities are embedded inside the `--kv-transfer-config`
JSON's `kv_connector_extra_config`, not as env vars. The
connector staging directory is bind-mounted read-only into the
target at `/opt/sparkcache-staging` with `PYTHONPATH` set. Requires
`--execute` and the exact confirmation phrase above.

### Step 12: Cutover — STOPS SERVING, confirmation-gated

```bash
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

Before stopping any source, the cutover command re-verifies every
prepared target against the current source's exact argv/env (recomputed
via `_apply_cache_config`) and the requested checkpoint identities.
Any drift (unrelated flags, env changes, or checkpoint mismatch) aborts
before any service interruption.

Stops all four DCP4 sources, starts all four target containers, then
polls readiness: all four running, no OOM, no exited target (including
clean exit 0), rank-0 `http://127.0.0.1:8000/health` returns HTTP
success. Once any source-stop command may have been issued, every
exception (including unexpected non-`CutoverError` exceptions)
triggers a single rollback path: stop targets, restart sources, then
inspect-verify completeness. The original error is combined with
inspect-verified `ROLLBACK COMPLETE` or `ROLLBACK INCOMPLETE` — the
original failure is never erased. All SSH calls use `ConnectTimeout=10`
and `BatchMode=yes`. Requires `--execute` and the exact confirmation
phrase above.

### Step 13: Verify readiness

Watch logs for:
- `spark-context-cache: registered N layers ... policy=separate`
- All four physical workers report distinct ranks
- No `RuntimeError` from the connector

## Validation gate (live, four-rank)

14. **Store test.** Send a known prompt. Watch logs for all four
    physical TP workers committing.

15. **Evict.** Send a different prompt to fill the KV pool.

16. **Restore test.** Re-send the exact same prompt. The output hash
    must match the store output exactly.

17. **Corruption drill.** Corrupt one chunk file on one rank. The
    connector must detect corruption, invalidate the entry, and the
    request re-prefills cleanly.

18. **All-four-physical-worker participation.** Verify all four
    physical TP ranks participated in store and restore.

19. **Attest physical-worker identity** (completes B1 live
    attestation). In the running DCP2 candidate logs, verify:
    - All four physical TP workers (TP0..TP3) report distinct
      physical ranks via `get_kv_connector_stats()`.
    - The quorum set contains all four physical ranks
      `{0, 1, 2, 3}`, not just DCP-local ranks `{0, 1}`.
    - TP0 and TP2 (both DCP rank 0) produce distinct storage keys
      in the persistent cache directory.
    - TP1 and TP3 (both DCP rank 1) produce distinct storage keys.
    These observations can only be made after the cache-enabled
    candidate is running — they are live validation, not pre-flight.

## Rollback — STOPS SERVING, confirmation-gated

If any mandatory check fails, use the cutover script's rollback command:

```bash
python scripts/live_dcp2_cutover.py rollback \
  --node 0=spark0 --node 1=spark1 --node 2=spark2 --node 3=spark3 \
  --execute --confirmation STOP-DCP2-SPARKCACHE-RESTORE-DCP4
```

This stops all target containers using actual return codes (not stdout),
then performs a read-only `docker inspect` confirmation that every
target is stopped (or proven absent) and every DCP4 source is running.
Existence is checked with fail-closed tri-state semantics: SSH failures
(rc 255, timeout) are `UNKNOWN`, not absence — `UNKNOWN` is treated
as failure, never as "safely stopped." If any target cannot be
confirmed stopped or any source cannot be confirmed running, rollback
raises `CutoverError` with `ROLLBACK INCOMPLETE` and the affected
ranks — manual intervention required.

Automatic rollback during a failed cutover uses the same return-code
and inspect-confirmation path; the raised error states whether
rollback was complete or incomplete. Rollback failures do not erase
the original failure — both errors are combined in the exception
message.

Manual rollback (if script is unavailable):
1. Stop the cache-enabled DCP2 target containers.
2. Restart the original DCP4 containers
   (`glm52-sparkring-nf3-dcp4-fixedk2-r{rank}`).
3. Verify via `docker inspect` that all targets are stopped and all
   sources are running.

## Pass criteria

| Check | Requirement |
|---|---|
| B1 live attestation complete | mandatory |
| Physical-worker identity attested (B1 live: distinct ranks, full quorum, distinct keys) | mandatory |
| All four ranks start with cache enabled, no OOM | mandatory |
| `--kv-transfer-config` uses canonical schema | mandatory |
| `--disable-hybrid-kv-cache-manager` present | mandatory |
| No `SPARK_CONTEXT_CACHE_ENABLE` in env | mandatory |
| Store committed on all four physical ranks | mandatory |
| Restore output byte-identical to store output | mandatory |
| Corruption drill: clean re-prefill | mandatory |
| Streaming snapshots remain disabled | mandatory (B5) |
| Checkpoint identity attested (B2) | mandatory |
| vLLM patches attested (B3) | resolved for image `sha256:ab6bddba...` |
| quant/rope layouts attested (B4) | resolved for image `sha256:ab6bddba...` / mount `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` |
| Connector source staged on every host (B6) | mandatory |
| API health HTTP 200 after all tests | mandatory |
| Connector bundle identity verified (B6) | mandatory |

## Evidence collection

Use `scripts/collect_evidence.py` to bundle:
- All four rank startup logs
- Store/restore log lines
- Output hashes
- Corruption drill logs
- API health checks
- Physical-worker identity observations (step 19)
- B1–B7 resolution evidence

Redact site identities before committing the bundle.
