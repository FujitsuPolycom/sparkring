# Generic Runtime Launcher

> **Lane:** public-functional tooling; a native generic profile is outside the
> accepted/current matrix until it is named and gated | **Maturity:**
> offline-validated (focused tests
> and golden-equivalence tests) | **Validation hardware:** CPU-only Windows
> workspace; no Spark contacted | **Target hardware:** four-Spark GPU/RDMA
> cluster | **Evidence scope:** offline plan output, structural validation,
> lifecycle safety, and byte-identical bridge actions. No live validation or
> acceptance.

The generic runtime launcher (`scripts/sparkring_generic_launcher.py`) lets
you operate a four-node SparkRing with a vLLM-style serving stack through a
single, profile-driven interface. It puts the native generic plan/lifecycle
implementation in `scripts/sparkring_runtime.py`; the canonical NF3 launcher
also consumes that module's remote-action and SSH-execution primitives.

## Compatibility boundary

The generic launcher targets **four-Spark GPU/RDMA clusters running
vLLM-style serving with TP4/DCP4 parallelism**. It is not a universal
model runner — it assumes the same transport topology, container
lifecycle, and serving shape as the existing EXL3 and NF3 launchers.

## Existing architecture audit

Before this slice, the public tree had three related but intentionally
model-specific paths:

- `sparkring_launcher.py` validates the pinned NF3 contract, including its
  target/draft layout, KV profiles, graph settings, and GLM-specific vLLM
  arguments.
- `sparkring_exl3_launcher.py` validates the pinned EXL3 artifact, shard and
  manifest receipts, Trellis settings, model verifier, and bounded batch-token
  experiment override.
- `sparkring_exl3_lmcache_launcher.py` composes the EXL3 engines with the
  separate per-rank LMCache server lifecycle and health phases.

Those launchers duplicated the `RemoteAction` shape and remote SSH execution;
their topology/environment builders were similar but not interchangeable with
all family-specific rules. This slice extracts `RemoteAction`, `run_remote`,
`execute`, and `action_succeeded` for use by the native generic path and the
canonical NF3 launcher. EXL3/NF3 contract validation and action construction
remain canonical extension seams, while LMCache remains an explicit service
composition boundary. The EXL3 executor is deliberately not migrated because
its timeout/result behavior differs.

## Dispatch by source schema

Dispatch is by the profile's `schema` field, not `model_family`. A
native generic profile named `exl3` or `nf3` is still a generic
profile — it uses generic action builders and generic execution
semantics. Only a profile whose `schema` is
`sparkring-public-exl3-launch/v1` or `sparkring-public-launch/v1` takes
the EXL3 or NF3 bridge path respectively.

## What is generic vs. model-specific

|Layer|Lane / maturity|Hardware and evidence scope|What it does|
|---|---|---|---|
|**Generic orchestration** (`sparkring_runtime.py`)|public-functional tooling / offline-validated|CPU-only Windows workspace; focused tests; no cluster|Shared remote execution plus native profile validation, action builders, and plans|
|**Generic launcher** (`sparkring_generic_launcher.py`)|public-functional tooling / offline-validated|CPU-only Windows workspace; focused tests; no cluster|Loads profiles, dispatches by schema, and plans or executes actions|
|**EXL3 bridge** (static dispatch)|public-functional tooling / offline-validated|CPU-only Windows workspace; golden-equivalence tests|Delegates operations present in `sparkring_exl3_launcher`|
|**NF3 bridge** (static dispatch)|public-functional tooling / offline-validated|CPU-only Windows workspace; golden-equivalence tests|Delegates operations present in `sparkring_launcher`|
|**EXL3 model/runtime support**|public-functional / live-validated|Four directly cabled Sparks; clean-checkout bootstrap and bounded live gates in [EXL3_RECIPE.md](EXL3_RECIPE.md) and [EXL3_LMCACHE_CAMPAIGN_20260803.md](EXL3_LMCACHE_CAMPAIGN_20260803.md)|Pinned EXL3 3.25-bpw runtime/model support|
|**NF3 model/runtime support**|public-functional / accepted|Four directly cabled Sparks; deterministic public validation in [NF3_NVFP4_PUBLIC_VALIDATION.md](NF3_NVFP4_PUBLIC_VALIDATION.md)|Pinned NF3 hybrid runtime/model support|

### Generic-only operations

Some operations exist only in the generic launcher and have no canonical
launcher counterpart:

|Operation|Builder|Notes|
|---|---|---|
|`verify-rollback` (EXL3 bridge)|`runtime.verify_rollback_actions`|EXL3 canonical launcher has no verify-rollback; generic builder checks container absence|
|`verify-image` (NF3 bridge)|`runtime.verify_image_actions`|NF3 canonical launcher has no verify-image; generic builder checks exact image digest from `site.runtime`|

These are **not** byte-identical to a canonical launcher because no
canonical counterpart exists. They are tested for structure only.

### Execution semantics by source schema

|Source schema|Execution mode|Start rollback|Success check|
|---|---|---|---|
|EXL3 (`sparkring-public-exl3-launch/v1`)|`exl3`|No rollback|Exit status only|
|NF3 (`sparkring-public-launch/v1`)|`nf3`|Partial-start rollback|`action_succeeded` (container ID check)|
|Generic (`sparkring-runtime-profile/v1`)|`generic`|Partial-start rollback|`action_succeeded` (container ID check)|

EXL3 canonical execution uses exit status only and does not perform
partial-start rollback. NF3 and generic execution use `action_succeeded`
(with container ID verification) and perform partial-start rollback.
These semantics are preserved by the generic launcher's schema-aware
dispatch.

## Composition boundary for LMCache

The EXL3+LMCache two-component lifecycle (separate LMCache server and
engine phases, health checks, restart sequences) is **not** part of the
generic launcher. It remains in `scripts/sparkring_exl3_lmcache_launcher.py`,
which composes the EXL3 engine lifecycle with the LMCache server
lifecycle. The generic launcher handles single-container start/stop per
rank. If you need LMCache, use the canonical EXL3+LMCache launcher
directly.

## Quick start

```bash
# OFFLINE: validate a generic profile and print four exact remote commands.
python scripts/sparkring_generic_launcher.py \
  --site scripts/config/site.yaml \
  --profile scripts/config/generic.example.json plan

# BACKWARD-COMPATIBLE: use the same EXL3 launch profile the bootstrap generated.
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json plan

# BACKWARD-COMPATIBLE: use the same NF3 launch config.
python scripts/sparkring_generic_launcher.py \
  --site scripts/config/site.yaml \
  --profile scripts/config/launch.json plan

# EXL3 bridge with bounded --max-num-batched-tokens override:
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json plan \
  --max-num-batched-tokens 3072
```

`plan` is always offline. `start` and `stop` require `--execute`. If the
profile declares a `confirmation` token, mutating commands require
`--confirmation <token>`. The `--max-num-batched-tokens` flag applies
only to EXL3 bridge `plan`/`start` and accepts 2048, 3072, or 4096.

## Profile format

A generic profile is a JSON file with schema
`sparkring-runtime-profile/v1`:

```json
{
  "schema": "sparkring-runtime-profile/v1",
  "profile_id": "my-model-profile",
  "model_family": "my-family",
  "engine": "docker",
  "container_name": "sparkring-my-model",
  "image": "registry.example/org/runtime:tag",
  "image_id": "sha256:<64-hex>",
  "model_host_path": "/srv/models/my-model",
  "model_container_path": "/models/my-model",
  "shm_size": "16g",
  "startup_timeout_seconds": 3600,
  "environment": {"VLLM_SPARK_MAX_NUM_SEQS": "8"},
  "extra_vllm_args": ["--quantization", "my-quant", "..."],
  "extra_volumes": [],
  "extra_labels": {},
  "privileged": false,
  "entrypoint": null,
  "confirmation": null,
  "identity": {
    "model_repository": "org/my-model",
    "model_revision": "<40-hex>",
    "model_config_sha256": "<64-hex>"
  },
  "attestation_hook": null,
  "health_check": null
}
```

See `scripts/config/generic.example.json` for a complete sanitized template.

### Boundaries

|Section|Owns|Does not own|
|---|---|---|
|**Site (`site.yaml`)**|Topology, ranks, RDMA devices, management addresses, serving shape|Model identity, environment, vLLM args|
|**Profile identity**|Model repository, revision, SHA256 pins (attestation metadata)|Transport, RDMA, serving shape|
|**Profile serving**|Engine, container name, image, environment, vLLM args, volumes|Site topology, rank count, site-owned vLLM options|
|**Profile lifecycle**|Confirmation token, privileged flag, entrypoint, attestation hook, per-rank health-check argv|Multi-service orchestration and acceptance policy|

Environment keys derived from the site (`RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, `NCCL_IB_*`, `SPARK_TP4_*`) are rejected if they appear
in a profile — the site owns transport topology. Keys with the
`SPARKRING_` prefix are reserved for the runtime; identity keys are
automatically prefixed with `SPARKRING_ATTEST_` to avoid collision.

Site-owned vLLM options (`--tensor-parallel-size`, `--port`,
`--nnodes`, `--headless`, etc.) are rejected in `extra_vllm_args` —
both `--option value` and `--option=value` forms. Profile-controlled
model/runtime flags (e.g. `--quantization`, `--kv-cache-dtype`) remain
easy to add.

Reserved labels (`org.sparkring.managed`, `org.sparkring.profile`) are
rejected in `extra_labels` — the runtime sets them automatically.

### Attestation and identity

The `identity` field carries declared model metadata (repository, revision,
SHA256 pins). These are **declared identity metadata**, not verified
attestation. They are set as `SPARKRING_ATTEST_*` environment variables
inside the container.

An optional `attestation_hook` is a validated argv array. Its first element is
used as the verification container's explicit entrypoint and the remaining
elements are its arguments. It runs after exact image verification and before
the main `docker run`. The composition is fail-closed: if the hook exits
non-zero, the start action aborts. Because the generic runtime cannot judge
what an arbitrary hook proves, the plan reports
`attestation-hook-configured`, not an independent correctness claim. Without a
hook a native profile reports `image-verified-before-start`; the NF3 bridge
reports only `declared-site-image`; and the canonical EXL3 bridge reports its
own `canonical-model-verification` scope.

An optional `health_check` is a validated argv array that runs inside every
rank container via exact-profile-guarded `docker exec`. Placeholders like
`{api_port}` are expanded from site context. `health` is dry-run by default and
prints all four actions. The supplied argv is arbitrary and may mutate state or
stop serving, so `health --execute` is classified as both `MUTATES HOST` and
`STOPS SERVING`; a configured profile confirmation token applies. Family
bridges continue to use their canonical health/acceptance paths and reject this
generic hook.

```bash
python scripts/sparkring_generic_launcher.py \
  --site scripts/config/site.yaml \
  --profile scripts/config/myfamily.json health
```

## Adding a new model/runtime profile

This is the minimal example for a contributor adding a new model family.

### 1. Write a profile

Copy `scripts/config/generic.example.json` and fill in your model's
identity, environment, and vLLM args. Set `model_family` to a short
identifier for your family (e.g. `"my-family"`).

### 2. Write a focused test

```python
# scripts/test_sparkring_myfamily.py
import json
from pathlib import Path
import sparkring_runtime as runtime
from sparkring_site import load_site

def test_myfamily_profile_validates(tmp_path):
    doc = json.loads(
        Path("scripts/config/generic.example.json").read_text()
    )
    doc.update({"profile_id": "my-model", "model_family": "my-family", ...})
    path = tmp_path / "myfamily.json"
    path.write_text(json.dumps(doc) + "\n")
    profile = runtime.load_runtime_profile(path)
    site = load_site(Path("scripts/config/site.example.yaml"))
    actions = runtime.start_actions(site, profile)
    assert len(actions) == 4
    # Verify image identity is checked before run
    assert "image inspect" in actions[0].shell_command
    # Verify profile label is set
    assert "org.sparkring.profile=my-model" in actions[0].shell_command
```

A focused profile/test does not require running the full acceptance gate or
owning four Sparks. The offline plan is deterministic and asserts
structure only.

### 3. Run the plan

```bash
python scripts/sparkring_generic_launcher.py \
  --site scripts/config/site.yaml \
  --profile scripts/config/myfamily.json plan
```

If your model family needs exact pins beyond structural validation
(e.g. a required KV cache dtype, a specific speculative config, or exact
shard counts), enforce them in your test. The generic launcher does not
use a plugin registry or family-adapter registration — native generic
profiles work without any registration step. EXL3 and NF3 are handled
by static schema dispatch in the generic launcher, which delegates to
their canonical launchers.

## Backward compatibility

The generic launcher accepts the existing EXL3 and NF3 profile formats
directly; no migration is required. The EXL3 and NF3 launchers remain the
canonical executable paths. Operations listed as golden-equivalent below are
delegated and byte-identical; the two generic-only operations documented above
have no canonical counterpart.

|Existing format|Schema|Generic launcher support|
|---|---|---|
|EXL3 launch profile|`sparkring-public-exl3-launch/v1`|Delegates to `sparkring_exl3_launcher` — golden-equivalence tested for start (default + --max-num-batched-tokens), stop, status, verify-image|
|NF3 launch config|`sparkring-public-launch/v1`|Delegates to `sparkring_launcher` — golden-equivalence tested for start, stop, status, verify-rollback|
|Generic profile|`sparkring-runtime-profile/v1`|Native|

### Shared primitives extraction

`sparkring_runtime.py` is the single source for `RemoteAction`,
`run_remote`, `execute`, and `action_succeeded`. The NF3 canonical
launcher (`sparkring_launcher.py`) imports these from
`sparkring_runtime` instead of maintaining independent copies. Existing
NF3 tests pass unchanged, proving byte-identical behavior. The EXL3
canonical launcher retains its own copies because its `execute` has
slightly different error-handling structure; a future extraction could
unify them if the EXL3 launcher is refactored.

## Safety

The generic launcher inherits the same proportional safety as the
existing launchers:

- **OFFLINE**: `plan` and dry-run `start`/`stop` make no SSH connection.
- **READ-ONLY REMOTE**: executed `status`, `verify-image`, and
  `verify-rollback` connect to ranks but do not mutate them. Inspect their
  dry-run plan first.
- **MUTATES HOST**: `start --execute` creates containers.
- **MUTATES HOST and STOPS SERVING**: `stop --execute` removes running
  containers. `health --execute` runs profile-supplied argv inside all four
  serving containers, so it must also be treated as potentially mutating or
  service-stopping despite its name. Both operations are exact-profile-label
  guarded. If the profile declares a confirmation token, it applies to
  executed `start`, `stop`, and `health`.
- **Image verification**: each generic `start` action verifies the exact
  image digest before `docker run` — fail-closed on identity drift.
- **Attestation hook**: if set, runs after image verification and before
  `docker run` (fail-closed).
- **Profile-label guard**: stop checks both `org.sparkring.managed=true`
  and `org.sparkring.profile=<profile_id>` so a foreign same-named
  container with a different profile is never removed (exit 73).
- **Schema-aware rollback**: EXL3 bridge uses exit-status-only (no
  rollback); NF3 bridge and generic profiles use `action_succeeded` with
  partial-start rollback (only containers whose `docker run` succeeded).
