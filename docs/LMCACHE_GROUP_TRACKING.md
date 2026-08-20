# Per-group key-value layout in the external cache tier

Status: **unsupported** for every checkpoint whose key-value cache is laid out
in more than one layer group, on the cache package pinned by
[`runtime/exl3/pins.json`](../runtime/exl3/pins.json).

This document specifies the package property such a checkpoint requires, the
package selection that property implies for this hardware, the local delta that
must survive a package change, the verification that precedes deployment, and
the gate that decides whether external reuse is functional. It does not
authorize remote action.

Lane: public-functional. Hardware: four directly cabled NVIDIA DGX Sparks,
GB10 / SM121, `linux/arm64`. The external cache tier here is LMCache; it is not
the repository's [`sparkcache/`](../sparkcache) implementation, and the two
systems carry separate maturity and evidence claims.

## The layout property that decides serving

A checkpoint's key-value cache is either **uniform** — one packed format across
every layer — or **grouped**, where layer groups differ in block size, element
size, or head size. A grouped checkpoint may additionally be internally
heterogeneous, carrying several geometries inside one engine group.

Conditions: cache package `0.5.2+glm52dcp4.1`, one multiprocess cache server
per rank started inside the engine container, four-rank tensor-parallel
serving.

Measurement, reported by the operator and not reproduced in this checkout:
serving `deepseek-ai/DeepSeek-V4-Flash-0731` registers 170 layers across five
engine groups, after which every retrieve aborts with `Size mismatch:
memory_obj nbytes=985664, gpu_buffer nbytes=15644672`, 21,525 blocks are marked
invalid, and the request stalls in the waiting queue instead of completing by
recomputation. The same package contains no reference to that checkpoint's
model identifier.

Result: the device-side read path sizes one staging buffer per transfer and has
no notion of kernel groups.

Conclusion: one staging-buffer size cannot satisfy a registration whose groups
differ in block size or element size. Serving a grouped checkpoint through this
tier requires a package that tracks key-value layout per layer group.

## The required package property

Upstream added multiprocess per-group tracking in LMCache pull request 3171,
`[MP][Feat] Support DeepSeek V4`, merged 2026-05-14 into the `dev` branch as
commit `384d79df5c3a023ccfebedc2b69b094b0d7b7084`. A package carrying that
change is the required input for any grouped checkpoint.

The change is identified by four properties:

| Property | Artifact | Why it decides |
|---|---|---|
| Layers are partitioned into transfer-kernel dispatch units | `lmcache/v1/kv_layer_groups.py` defines `KVLayerGroupsManager` | The module has no counterpart in packages predating the change |
| Block size separates groups | `block_size` is a field of `KernelGroupIdentity` | Without it a compressed and an uncompressed group collapse into one entry and the transfer is sized from the wrong geometry |
| Padded pool views are described explicitly | `block_stride_elems` on `PageBufferShapeDesc` | Dim-0-padded views from the engine's unified pool are not contiguous and cannot be assumed so |
| Registration carries the engine block size | `vllm_block_size` in the `REGISTER_KV_CACHE` payload | The server sizes buffers per group rather than from one global block size |

The kernel-group identity is the tuple of key-value size, head count, head
size, block size, engine group index, dtype, and engine key-value format. That
identity is what separates several geometries inside one engine group, which is
the case a grouped, internally heterogeneous checkpoint presents.

### The property is already present in the released package line

Conditions: the source distributions of `lmcache` 0.5.2 and 0.5.3, downloaded
from the package index and checked with
[`runtime/exl3/verify_lmcache_group_tracking.py`](../runtime/exl3/verify_lmcache_group_tracking.py).

| Package | `sha256` of source distribution | Group tracking present |
|---|---|---|
| 0.5.2, published 2026-07-22 | `90e747898ef304026c9e8a8475dd970f31cfdd622213f9034f98b48f417fb0ec` | yes |
| 0.5.3, published 2026-08-05 | `aa2e1313d6dcfa719638b3cd62cb42875469841dec6e62e32111856bc8d04d96` | yes |

Measurement: both trees contain `lmcache/v1/kv_layer_groups.py`, define
`KernelGroupIdentity` with `block_size`, carry `block_stride_elems`, and send
`vllm_block_size` in the registration payload. Both release dates postdate the
2026-05-14 merge.

Result: stock 0.5.2 — the release this deployment's version string is derived
from — already carries the required property.

Conclusion: the failure observed on a grouped checkpoint is **not** explained by
an upstream package that predates per-group tracking, and obtaining a newer
release is therefore not by itself the remedy. The deployed package is not
stock 0.5.2: it is composed from fork base `9cebd405` dated 2026-07-28, eleven
integration heads, and a topology patch. Since that base also postdates the
capability, the cause lies in the composition rather than in the upstream
baseline, and the composed tree is where diagnosis continues. This has not been
determined; see [Open items](#open-items).

### Two checks that do not decide it

Two package checks in circulation do not establish this property and must not
gate a deployment:

- **Matching a file against a model name.** The module that implements the
  capability, `lmcache/v1/kv_layer_groups.py`, does not mention any model name.
  The capability is layout-driven, not checkpoint-driven.
- **Requiring that `lmcache/v1/gpu_connector/gpu_ops.py` sizes buffers per
  kernel group.** Measurement: in both the 0.5.2 and the 0.5.3 source
  distributions that file is 146 lines defining three memory-copy and staging
  helpers, with zero occurrences of `layer_group`, `block_size`,
  `block_stride_elems`, or `KernelGroup`; pull request 3171 does not modify it.
  Result: the check fails against packages that do carry per-group tracking.
  Conclusion: applying it rejects a correct package, which is the opposite of
  its purpose.

[`runtime/exl3/verify_lmcache_group_tracking.py`](../runtime/exl3/verify_lmcache_group_tracking.py)
implements the deciding properties instead. It parses source text and never
imports the package, so it runs on a workstation with no CUDA toolchain.

```bash
python runtime/exl3/verify_lmcache_group_tracking.py --package-dir PATH
```

Safety class: OFFLINE. Exit `0` pass, `2` fail, `3` configuration error.

## Package selection for this hardware

Condition: the serving hardware is `linux/arm64`.

Measurement: `lmcache` 0.5.3 is the latest release on the Python Package Index,
published 2026-08-05, and its release files are four `manylinux` `x86_64`
wheels plus a source distribution. The project's `v0.5.4rc2`, `v0.5.4rc3`, and
`v0.5.4rc4` tags are prereleases.

Result: no release wheel targets this hardware.

Conclusion: preferring a release over a source build does not apply here. An
ARM64 wheel must be built from a source tree, exactly as
[`runtime/exl3/pins.json`](../runtime/exl3/pins.json) already records through
its `validated_arm64_wheel_sha256` field. The release to select is therefore a
source **ref**, and 0.5.3 is the earliest released ref that both postdates the
2026-05-14 merge and has been confirmed to contain `kv_layer_groups.py`.

## The local delta a package change must carry

The installed suffix `+glm52dcp4.1` denotes local changes. Those changes are
enumerated in [`runtime/exl3/pins.json`](../runtime/exl3/pins.json) and do not
need to be rediscovered by comparison against a stock tree:

| Input | Value |
|---|---|
| Source repository | `https://github.com/local-inference-lab/LMCache.git` |
| Base commit | `9cebd405d0caf4bebe01d694b5a8bf4e3e354314` |
| Integration heads | 11 commits |
| Topology patch | [`runtime/exl3/patches/lmcache-tp4-dcp4-four-local-servers.patch`](../runtime/exl3/patches/lmcache-tp4-dcp4-four-local-servers.patch) |
| Composed tree | `7dddbfde874d123e5b5785e6e56b4b7baf4baa82` |

Comparing the installed tree against a stock `lmcache==0.5.2` source
distribution reports the union of three unrelated deltas — fork-versus-release,
the eleven integration heads, and the topology patch — as one undifferentiated
difference set, and so attributes fork changes to this deployment. The pinned
inputs above separate them and are the source to work from.

The topology patch encodes a requirement of this deployment. It relaxes the
cache adapter's rule that MLA plus decode context parallelism permits exactly
one cache server, admitting one server per decode-context shard, and it adds
`local_server_url_for_worker` to select a worker's node-local server from its
contiguous rank block. The repository's serving topology,
`tp4-dcp4-pp1-four-local-servers`, cannot start without that relaxation.

That patch is **superseded** by the released package line.

Conditions: the patch applied against the stock 0.5.2 source distribution with
`git apply --check`.

Measurement: the two connector hunks apply with offsets, and the adapter hunk
fails, because the single-server rejection the patch rewrites does not exist in
stock 0.5.2. In that tree `n_servers` is derived from the length of the
configured server-URL list, and the worker's node-local server is selected as:

```python
ranks_per_node = parallel_strategy.vllm_world_size // n_servers
local_server_url = server_urls[parallel_strategy.vllm_worker_id // ranks_per_node]
```

Result: that is the contiguous-block routing the patch's `local_server_url_for_worker`
helper extracts, with the same semantics, and the constraint the patch relaxes
is already absent.

Conclusion: both halves of the patch are carried upstream. Re-applying it to a
0.5.2-or-later base will fail on the adapter hunk, and the local helper symbol
is not required for the deployed topology. Rebuild without it, and verify the
resulting tree with the checker above rather than by re-applying the patch.

A package rebuilt on a different base must not reuse the version string
`0.5.2+glm52dcp4.1`, which denotes a package without per-group tracking.

## Coupling to the published serving configuration

A package change is not confined to the runtime image. The following are
pinned in executable configuration and fail closed when they drift.

| Pinned value | Owner | Effect of a package change |
|---|---|---|
| `runtime.lmcache.version` `0.5.2+glm52dcp4.1` | [`scripts/sparkring_recipe.py`](../scripts/sparkring_recipe.py) | The recipe validator rejects any other value; a new version string must be introduced there and in the recipe together |
| `serving.lmcache.connector` `LMCacheMPConnector` | [`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json) | Pull request 3171's description states the connector was renamed; measurement against `v0.5.3` finds `LMCacheMPConnector` still defined in `lmcache/integration/vllm/lmcache_mp_connector.py` and no `LMCacheMPConnectorDynamic` there, so the pinned binding holds at that ref and must be re-verified at any other |
| `serving.lmcache.mq_timeout_seconds` `10` | [`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json) | Rendered as `lmcache.mp.mq_timeout`; a checkpoint whose registration takes longer than the deployed profile's value cannot register, so the deadline is a per-checkpoint value and not one constant |
| The `REGISTER_KV_CACHE` payload | the package on each rank | The payload gained `vllm_block_size`, so a client and a server at different versions are wire-incompatible |

The wire-compatibility item sharpens the identical-package requirement: a mixed
deployment is not merely inconsistent, it is a protocol mismatch, and it
presents as a stalled request rather than an error.

## Deployment constraints

These hold for every checkpoint and are independent of the package version.

- **Each cache server runs inside its rank's engine container.** A server in a
  separate container cannot import the engine's device memory; registration
  fails with `cudaErrorInvalidResourceHandle`.
- **The registration deadline exceeds the observed registration time.** The
  operator reports about 75 seconds to register 170 layers against a 32 GiB
  per-rank reservation. Set `lmcache.mp.mq_timeout` above the observed time for
  the largest reservation in use, per checkpoint.
- **The heartbeat guard must test contents, not existence, and no released
  package satisfies this.** Conditions: `lmcache/integration/vllm/vllm_multi_process_adapter.py`
  in the 0.5.2 and 0.5.3 source distributions. Measurement: `_heartbeats` is
  initialized to an empty dictionary, and `_ensure_heartbeat_started` opens with
  `if self._heartbeats is not None: return`, both before and inside the lock.
  Result: the guard is taken on every call, so the heartbeat thread never
  starts. Conclusion: the server reaps a live engine while stores continue and
  lookups stop silently, and **upgrading does not fix this** — a local
  correction is required on whichever base is chosen. The verifier locates the
  pattern and fails on it. Confirm the correction additionally by holding a
  deployment idle past the reaping interval, which the operator reports at
  roughly 150 seconds.
- **Native prefix caching is disabled during validation.**
  `--no-enable-prefix-caching` routes all reuse through the external tier.
  Without it the engine's own cache serves replays and no measurement
  attributes a result to the external tier.
- **The package is byte-identical on every rank.**

## External reuse is optional per checkpoint

Each checkpoint declares whether the external tier is load-bearing for it.

| Policy | Meaning | Gate failure |
|---|---|---|
| `required` | Serving depends on the external tier | Blocks serving; the evaluator exits `2` |
| `optional` | Serving proceeds without the external tier | Withholds reuse only; the evaluator exits `4` |

`deepseek-ai/DeepSeek-V4-Flash-0731` is **`optional`**. External key-value reuse
is not functional for it on this deployment, and serving it without the
external tier is the supported configuration. Nothing in this document may
become a serving dependency for that checkpoint: the tier is disabled for it,
the engine serves normally, and the enablement work above is a separate,
non-blocking track.

Two properties keep that boundary honest:

- The policy is declared in the evidence document before collection, not chosen
  after reading a result. An unqualified capability cannot be reclassified as
  optional in order to make a failure disappear.
- An optional failure gets its own exit code rather than reusing the passing
  one. A caller never has to read a serving break out of an absent accelerator,
  and never has to suppress a real serving break in order to tolerate one.

## Validation gate

External key-value reuse is functional for a checkpoint only when a request
served by an engine that has been destroyed and recreated satisfies all of:

1. The request completes.
2. Its prompt is identical to the store-phase prompt.
3. Its output hash matches the store-phase hash for that prompt.
4. Native prefix caching was disabled and its hit counter reads zero.
5. The external hit counter is non-zero.
6. The cache server log records no `Size mismatch`.
7. The engine log records no `LMCache retrieve failed`.

Counters alone do not carry the gate, because two instrument properties make
partial evidence misleading. Both are operator-reported and both are rejected
explicitly by the evaluator:

- The external hit counter counts lookup matches, not completed transfers. A
  configuration that failed every transfer reported 104,960 hits against
  105,246 queries while recomputing every token and never returning a response.
  A stalled request with a non-zero hit counter is a failure, not a slow
  success.
- A replay inside a live engine process is served by that engine's own prefix
  cache. Such a replay measured 36 to 40 times faster than cold with
  byte-identical output while the external counter read zero against 47,738
  queries and the native counter read 47,104.

[`scripts/lmcache_external_reuse_gate.py`](../scripts/lmcache_external_reuse_gate.py)
prints the collection procedure and evaluates a collected evidence document.

```bash
python scripts/lmcache_external_reuse_gate.py plan
```

```bash
python scripts/lmcache_external_reuse_gate.py evaluate --evidence PATH
```

Safety class of the evaluator: OFFLINE. It is fail-closed — a missing or
malformed required field is a configuration error, never a pass and never a
refutation. Exit codes: `0` pass, `2` a required capability failed, `3`
configuration error, `4` an optional capability did not qualify. The collection
steps the plan describes stop serving and require explicit authorization for
the named hosts.

## Order of work across checkpoints

Ascending layout difficulty, so that a failure isolates to the capability
introduced at that step. A step is complete only when the gate above passes for
it; a partial result is not carried forward.

| Order | Checkpoint | Layout | Package requirement | Purpose of the step |
|---:|---|---|---|---|
| 1 | `Qwen/Qwen3.8-27B` | Uniform across layers | Original path; no group tracking needed | Validates deployment shape — server placement, registration deadline, filesystem tier, and the gate — without depending on group tracking |
| 2 | `zai-org/GLM-5.2` at 3.5 bits per weight | Multiple groups, from dynamic sparse attention | Per-group tracking | First checkpoint that depends on the property above |
| 3 | `deepseek-ai/DeepSeek-V4-Flash-0731` | Multiple groups, one internally heterogeneous | Per-group tracking | Adds several geometries inside one engine group, and speculative draft caches |

Record each checkpoint's registered kernel group inventory before configuring
it; the server logs that inventory at registration. Chunk size interacts with
block size, so a chunk size is chosen against the recorded inventory rather
than carried across checkpoints. Upstream states a chunk size of 1024 for
`zai-org/GLM-5.2`; record that checkpoint's registered block sizes before
departing from it.

The registrations for `Qwen/Qwen3.8-27B` and `zai-org/GLM-5.2` on this ring are
unmeasured. The registration reported by the operator for
`deepseek-ai/DeepSeek-V4-Flash-0731` is:

| Engine group | Layers | Tokens per block | Head size | Element size |
|---|---:|---:|---:|---:|
| 0 | 21 | 64 | 584 | 1 |
| 0 | 21 | 64 | 132 | 1 |
| 0 | 20 | 2 | 584 | 1 |
| 1, 2 | 23 each | 64 | 584 | 1 |
| 3 | 21 | 4 | 512 and 2048 | 4 |
| 4 | 20 | 8 | 1024 | 4 |

Engine group 0 carries three distinct geometries, which is the case that a
kernel-group identity including block size is required to separate.

### Configuration specific to `deepseek-ai/DeepSeek-V4-Flash-0731`

Values below differ from a default, each with its reason. They are engine-side
configuration and are independent of the cache package version.

- `--kv-cache-dtype fp8_ds_mla`. The engine gates its layout on an exact string
  comparison against this value. `fp8` selects a generic path that declares a
  geometry differing from the one it allocates. Both allocate identically, so
  the difference is invisible to the engine's own kernels and incorrect for any
  external consumer of the layout.
- `--tokenizer-mode deepseek_v4`.
- Speculative decoding requires depth five or greater together with the `b12x`
  mixture-of-experts backend. The operator reports measured draft acceptance of
  42 percent without that backend and 86 percent with it.
- The engine's load-failure path must handle one block-identifier list per
  key-value cache group. `KVCacheManager.get_block_ids` returns one list per
  group, and a single-list unpack terminates the engine core on the first
  invalid block.

## Open items

The single open question is **why the deployed composition fails when its
upstream baseline does not**. The required property is present in stock 0.5.2,
and the fork base postdates it, so the cause lies in the eleven integration
heads or in how the composed tree was assembled. That has not been determined,
and it is not determinable from this checkout, which does not carry the composed
tree.

The next measurement is to run the checker against the package directory
installed on a rank:

```bash
python runtime/exl3/verify_lmcache_group_tracking.py --package-dir "$(python -c 'import lmcache, os; print(os.path.dirname(lmcache.__file__))')"
```

Safety class: READ-ONLY REMOTE. It reads one directory, starts nothing, and
changes nothing, so it can run against a serving rank. Its output distinguishes
the two remaining possibilities: a composition that dropped the capability
fails `group_module_present`, while a composition that retains it but
misbehaves passes every check and moves diagnosis to configuration.

## What this document does not establish

- No checkpoint has passed the gate on this hardware. Every grouped checkpoint
  whose policy is `required` remains `unsupported` until its gate passes, its
  configuration is recorded with its registered kernel group inventory, and its
  package version is pinned in the runtime inputs.
- No package has been built, deployed, or measured on this ring for this work.
  The upstream properties above are verified by source inspection of downloaded
  source distributions, not by execution on the target hardware.
- The `Size mismatch` symptom is operator-reported and has not been reproduced
  here, and no cause has been established for it.
- Serving any checkpoint without external key-value reuse is unaffected by
  everything here, and is the supported configuration wherever the policy is
  `optional`.
- Comparisons across a key-value cache dtype change are invalid. Measurements
  taken under one cache dtype do not compare to measurements under another even
  when both produce identical output for identical input, because they are
  separate configurations.

Machine-readable companion:
[`docs/configurations/lmcache-group-tracking.json`](configurations/lmcache-group-tracking.json).
