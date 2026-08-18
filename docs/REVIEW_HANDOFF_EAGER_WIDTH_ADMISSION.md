# Review handoff: width-generic eager TP4 all-reduce admission

## Scope

This document hands a reviewer everything needed to assess one body of
work: opt-in width-generic admission for the eager TP4 all-reduce
adapter, its composition with the sparse Q42/Q48 provider row contract,
the equivalence evidence that pins the composed adapter pair to the
deployment adapter lineage, a pip-installable `sparkring` plugin that
packages the adapter as a `vllm.general_plugins` entry point, an
operator preflight console script, a bind-mount hotfix for the deployed
container image, and the validation record for all of it. The reviewer
has no access to the conversation that produced the work; every claim
below is derived from the code and documents named here.

## Where the work lives

Three local branches, stacked. Each contains everything below it.

| Branch | Commits (oldest first) | Contents |
|---|---|---|
| `claude/sparkring-memory-capacity-classes-0f80e0` | `12a9fbe`, `2e531f2`, `9b88af9` | The `VLLM_SPARK_TP4_EAGER_WIDTHS` admission feature and its dispatch/namespace tests; the three-leg validation runbook; the leg-2 native probe results. |
| `claude/sparkring-vllm-plugin` | `61ad142`, `1b8e4da`, `8729280`, `ca6c01d`, `7cab210`, `2e5bf13`, `8e60acf`, `358b3b4`, `69fa1fd`, `2c05d89`, `243ac98` | The `sparkring_plugin/` package (entry point, vendored import closure, vendor-parity test), build-artifact ignores, the feature-detected vLLM compatibility gate, the `sparkring-preflight` diagnostic, the deployed-image `kernel_warmup` hotfix, the SIRCL ring-size design note, and the leg-3 and leg-1 validation records. |
| `claude/sparkring-provider-rows-rebase` (current tip, `7b961f9`) | `79d7170`, `384ee39`, `7b961f9` | The deployment-lineage equivalence oracle (harness plus test), the composition of width-generic admission with the sparse Q42/Q48 provider contract, and the separated leg-1 findings. |

Stacking: `claude/sparkring-provider-rows-rebase` contains
`claude/sparkring-vllm-plugin`, which contains
`claude/sparkring-memory-capacity-classes-0f80e0`, which is three
commits ahead of `main` at `b7e8fb0`. Reviewing the tip of
`claude/sparkring-provider-rows-rebase` covers all seventeen commits.

**Nothing is pushed.** All three branches exist only in the local
repository; no remote carries them, so a review must run against a
local clone or worktree.

### Not reviewable from git alone

Two paths are load-bearing for the work but are deliberately absent
from version control.

- [`spark_transport/integrations/vllm/_private_fixtures/`](../spark_transport/integrations/vllm/_private_fixtures/)
  — untracked by its own `.gitignore` (`*` with `!.gitignore`). It holds
  three deployment-lineage files fetched from the cluster runtime:
  `deployed_backend.py`, `deployed_namespace.py`, and
  `spark_tp4_sparse_q42_q48_contract.py`. The first two are the golden
  oracles the equivalence test compares against. The third is a runtime
  dependency: both
  [`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):22
  and
  [`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):17
  import `spark_tp4_sparse_q42_q48_contract`, and no copy of that module
  is tracked anywhere in the repository.
  [`conftest.py`](../spark_transport/integrations/vllm/conftest.py):17-23
  appends the fixtures directory to `sys.path` when the module is not
  otherwise importable, so the offline suite runs; without the
  directory, the import fails and the equivalence tests skip with an
  explicit message.
- `sparkring_plugin/src/sparkring/_native/libspark_transport_capi.so` —
  a built aarch64 artifact, ignored via `sparkring_plugin/.gitignore`.
  [`plugin.py`](../sparkring_plugin/src/sparkring/plugin.py):51-54
  defaults `SPARK_TP4_LIBRARY` to that path when it exists, and
  `[tool.setuptools.package-data]` in
  [`pyproject.toml`](../sparkring_plugin/pyproject.toml) ships
  `_native/*.so` into the wheel. A reviewer working from git sees the
  resolution logic but not the binary.

A reviewer without cluster access can still run the offline suite (the
equivalence tests skip loudly) and can read every code path; the
byte-for-byte equivalence claim itself is only re-executable with the
fixtures present.

## What the change does

### Control surface

`VLLM_SPARK_TP4_EAGER_WIDTHS` sets the list of hidden widths (elements
per row) the eager TP4 all-reduce adapter admits. Unset or empty, the
admitted width tuple is exactly `(6144)`
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):157-196).
Set, it is a comma-separated integer list, returned sorted ascending
and deduplicated by rejection: an empty token, a non-integer token, a
width outside `[1, 1048576]`, or a repeated width each raise a
`ValueError` naming the variable. The upper bound is chosen so that
`1048576 elements x 512 rows x 2 bytes = 1 GiB` stays under the native
single-RDMA-write bound of `UINT32_MAX` bytes
(comment at
[`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):150-153).

The variable is default-off and fail-closed. `install()` evaluates the
width parse, the Q512 flag, the sparse-contract flag, and the full port
namespace before it patches vLLM
([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):1215-1218),
so a malformed value aborts installation rather than degrading
admission. All four ranks must carry the same value, because the value
determines control-port assignment.

Scope is eager all-reduce only. CUDA-graph capture and replay, the
dual-port striped Q40 path, vocabulary all-gather, DCP, and the
all-gather family remain bound to width 6144. A non-6144 tensor seen
during an active graph capture routes to the stock path and is recorded
as `graph_width_ineligible`
([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):1250-1257)
rather than reaching the capture path, whose failure handler terminates
the worker with `os._exit(70)`.

### Two-regime port mapping

Eager all-reduce control ports come from a slot index over a base pair
(`SPARK_TP4_CONTROL_PORT0` / `SPARK_TP4_CONTROL_PORT1`, defaulting to
`11000` / `11001`) with stride 2. There are two disjoint slot regimes,
split by
[`_eager_allreduce_size_regimes`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):214-244
and resolved by
[`eager_allreduce_ports_for_payload`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):247-293.

**Legacy regime.** Payload sizes of the form `rows * 6144 * 2` for every
row in the supported row set. The supported row set is
`1..512` when `VLLM_SPARK_TP4_PREFILL_Q512=1`, the sparse provider row
set (`1..40` plus `42` and `48`) when `VLLM_SPARK_TP4_SPARSE_Q42_Q48=1`,
and `1..VLLM_SPARK_MAX_QUERY_ROWS` otherwise
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):139-146).
The slot is `row - 1`
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):266),
so rows the row set omits leave permanent holes in the port sequence:
under the sparse contract, rows 41 and 43-47 are never reserved and
their slots stay vacant, while rows 42 and 48 occupy slots 41 and 47.
Preserving those holes is the point — the deployment's exact-state
invariant counts the reservation set at startup and refuses to serve on
any drift.

**Extension regime.** Payload sizes generated by every admitted
non-default width over the contiguous row range `1..maximum`, where
`maximum` is 512 under Q512 and `VLLM_SPARK_MAX_QUERY_ROWS` otherwise
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):131-136,
234-244). Extension slots start at 512, past the largest legacy span
(Q512 ends at slot 511), and increase with the payload size's position
in the sorted extension tuple
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):271-273).
The base 512 is a constant, not a function of the active row set, so no
sparse or Q512 toggle moves an extension port.

**Regime precedence.** A non-default-width payload whose byte count
equals a supported legacy payload is excluded from the extension set
and resolves to the legacy slot
([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):242-243,
262-266). An identical byte count is the same native operation, so it
shares one session and one port pair. A payload in neither regime raises
a `ValueError` naming the byte count.

Reservations are emitted by
[`active_port_reservations`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):436-453:
one `eager_allreduce:q=N` owner per supported row, then one
`eager_allreduce:payload=B` owner per extension size, in that order.
[`validate_active_port_namespace`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):554-581
range-checks every pair and rejects any port claimed by two owners.

### Composition with the sparse Q42/Q48 provider contract

Admission mirrors reservation branch for branch.
[`_admitted_default_width_rows`](../spark_transport/integrations/vllm/spark_tp4_backend.py):249-263
returns `1..512` under Q512, the provider row set under the sparse
contract, and `1..MAX_QUERY_ROWS` otherwise — the same three branches in
the same order as `_supported_allreduce_query_rows`.
[`_target_shape_eligible`](../spark_transport/integrations/vllm/spark_tp4_backend.py):266-271
(the graph gate) tests membership in that set at width 6144.
[`_eager_shape_eligible`](../spark_transport/integrations/vllm/spark_tp4_backend.py):297-315
uses that same set for the default width and the contiguous
`1..MAX_QUERY_ROWS` range (or `1..512` under Q512) for every other
admitted width, matching what the namespace enumerates for extension
payloads. The sparse contract is a default-width serving constraint and
does not govern extension widths.

`sparse_q42_q48_enabled` itself is fail-closed and mutually exclusive
with Q512, and requires `VLLM_SPARK_MAX_QUERY_ROWS=40`.

### Documented deviation from the deployment adapter lineage

The docstring of
[`_admitted_default_width_rows`](../spark_transport/integrations/vllm/spark_tp4_backend.py):249-258
records one intentional difference from the deployment adapter lineage
held in `_private_fixtures/deployed_backend.py`.

The deployment lineage's `_target_shape_eligible` calls
`provider_query_rows()` unconditionally whenever Q512 is off
(`deployed_backend.py`:248-257). `provider_query_rows` returns
`1..40` plus `{42, 48}` under the sparse contract and `1..40` otherwise
— it never consults `VLLM_SPARK_MAX_QUERY_ROWS`. The port namespace, in
both lineages, reserves only `1..MAX_QUERY_ROWS` when the sparse
contract is off. So whenever the row cap is below 40, the deployment
lineage admits rows for which no control-port pair is reserved.

The adapter pair in this work instead returns `range(1, MAX_QUERY_ROWS + 1)`
on that branch, which makes admission and reservation agree by
construction. The two behaviors coincide exactly when
`VLLM_SPARK_MAX_QUERY_ROWS` is 40, which is the value the sparse
contract enforces and the value the production configuration sets. The
default value of that variable is 6
([`spark_tp4_query_contract.py`](../spark_transport/integrations/vllm/spark_tp4_query_contract.py):19),
so the divergence is reachable outside production.

## Equivalence evidence

[`test_provider_rows_equivalence.py`](../spark_transport/integrations/vllm/test_provider_rows_equivalence.py)
compares the adapter pair in this repository against the deployment
adapter lineage using a golden-oracle, subprocess-per-permutation
design.

[`_provider_equivalence_harness.py`](../spark_transport/integrations/vllm/_provider_equivalence_harness.py)
takes a backend file and a namespace file, loads them under the
invoking environment, and prints one canonical JSON object describing
the complete eager all-reduce admission surface: the ordered reservation
tuple and its length (the arena set the deployment's exact-state
invariant counts), per-row control-port resolution for rows 1-520 with
rejections recorded as such, `_target_shape_eligible` and
`_eager_shape_eligible` bitmaps over rows 1-520 at widths 2880, 4096,
and 6144, `_maximum_allreduce_query_rows`, `_graph_capacity_bytes`, and
backend-side control ports at eight probe payloads including the sparse
rows 42 and 48 and the rejected gap row 41. The namespace is loaded
under its real module name first so the backend's `from
spark_tp4_port_namespace import ...` binds to the paired namespace
([`_provider_equivalence_harness.py`](../spark_transport/integrations/vllm/_provider_equivalence_harness.py):42-43).
A fresh subprocess per permutation is required because
`spark_tp4_query_contract` bakes `VLLM_SPARK_MAX_QUERY_ROWS` at import.
The test builds each subprocess environment by stripping every
`VLLM_SPARK`, `SPARK_TP4`, and `SPARK_Q42` variable from the parent and
applying a fixed base plus the permutation's overrides.

Permutations with `VLLM_SPARK_TP4_EAGER_WIDTHS` unset, where the two
surfaces must be equal object-for-object:

- `production_replay_sparse` — sparse contract on, row cap 40, mode `custom`.
- `contiguous_q40` — sparse off, row cap 40, mode `custom`.
- `contiguous_q512` — row cap 40, `VLLM_SPARK_TP4_PREFILL_Q512=1`, mode `custom`.
- `shadow_sparse` — sparse on, row cap 40, mode `shadow`.

`eager_shape_admission` has no counterpart in the deployment lineage and
is removed from both sides before comparison; it is then checked
separately to equal the target-width admission bitmap at 6144 and to be
all-zero at 2880 and 4096.

Permutations with widths set, where the deployment surface must be a
strict prefix of this one:

- `sparse_with_width_2880` — sparse on, row cap 40, widths `2880`.
- `contiguous_with_widths` — sparse off, row cap 40, widths `2880,4096`.

Each checks that the deployment reservation list is a prefix of this
one, that the remainder is non-empty, that every added owner is named
`eager_allreduce:payload=`, that every added port starts at or past
`base + 2 * 512`, and that per-row port resolution is unchanged.

**What this proves.** For the six environment permutations listed, with
the width variable unset the adapter pair's complete admission surface
— reservation tuple, ordering, owner strings, port numbers, shape
admission, and capacity values — is identical to the deployment adapter
lineage's; and with the width variable set, the additions are strictly
appended past the legacy slot span and disturb neither the reservation
prefix nor per-row port resolution. That is the offline form of the
deployment's exact-state arena-count invariant.

**What this does not prove.** It exercises no GPU, no RDMA link, no
native library, and no live vLLM. It does not run the deployment's
attestation. It covers only the environment permutations enumerated in
the test, all of which pin `VLLM_SPARK_MAX_QUERY_ROWS=40`, so the
documented deviation above is never exercised at a value where it
diverges. Widths beyond 2880 and 4096 are not compared, nor is any
combination of widths with Q512. It says nothing about numerical
correctness, performance, or whether the deployment will in fact admit
the pair at launch.

## Validation status

Detail, evidence, and per-run numbers are in
[`docs/EAGER_WIDTH_VALIDATION_RUNBOOK.md`](EAGER_WIDTH_VALIDATION_RUNBOOK.md).
The runbook's leg sections carry chronological execution records; the
status summary is:

| Item | Status |
|---|---|
| Width-generic eager admission code path | Implemented, default-off. No serving qualification. |
| Widths-unset admission surface versus deployment lineage | Implemented and verified offline, byte-for-byte, over six environment permutations. |
| Leg 1 — GLM-5.2 production-shape regression on the cluster | Unexecuted. Blocked by a launch failure that is independent of this work: the deployment's `_attest_adaptive_mtp_exact_state_policy` raises "adaptive-MTP exact-state arena count drifted", counting MoE exact-state GPU weight-storage arenas rather than transport reservations. A control launch that substitutes no adapter files at all — the deployment lineage's own modules, with every bind mount restored to the reference container's read-only mode — reproduces the same failure, so neither the width-generic adapter pair nor the mount modes cause it. See the runbook's leg-1 section for the launch-surface diff and the remaining candidates. |
| Leg 2 — native transport probes at extension payload sizes | Qualified. Nine payload points from 1 KiB to 2 MiB, four ranks, zero mismatched elements everywhere; latencies recorded, not gated. |
| Leg 3 — width 768 (`opt-125m`) shadow window | Qualified against the configured gate: 0 of 7.68M elements outside tolerance. |
| Leg 3 — widths 512 (`pythia-70m`) and 2048 (`TinyLlama-1.1B`) shadow windows | Configured-gate FAIL with recorded operator disposition (accepted as explained, not promoted): 3 of 5.12M and 2 of 20.48M elements outside tolerance, zero nonfinite, bit-identical across ranks. |
| Leg 3 — width 1024 (`Qwen3-0.6B`) shadow window | Configured-gate FAIL, open: 1153 of 10.24M elements (1.1e-4). Modeled FP32-oracle arbitration favors the SIRCL balanced-tree reduction order over a naive sequential order on the cancellation pattern; the arbitration uses synthetic inputs and a comparator that does not model NCCL's chunked ring, so it is research-only and no promotion follows from it. |
| CUDA-graph capture and replay at non-default widths | Unsupported by construction; non-6144 tensors under capture route to the stock path. |
| Prefill capacity pool (`VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL`) | Research-only; `install()` raises when it is requested. |
| `sparkring` plugin package | Implemented; GPU-free tests and a wheel build only. Not deployed, not qualified. |
| `kernel_warmup` CuTe-DSL hotfix for image `r34-sm121a-flat2-20260810` | Implemented and live-validated on four ranks by bind mount. Held deliberately outside `runtime/patches/` so the fail-closed applier never consumes it; its preimage matches the deployed image, not the current pinned base. |
| SIRCL ring sizes beyond N=4 | Research-only design note in [`docs/SIRCL.md`](SIRCL.md); no implementation or evidence exists beyond N in {2, 4}. |

## Review questions

1. **Regime disjointness under every toggle.** Legacy slots are `row - 1`
   over the supported row set; extension slots start at the literal 512
   ([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):266,
   271-273). Is 512 provably past every reachable legacy slot? The
   largest legacy row is 512 under Q512 (slot 511) and 48 under the
   sparse contract (slot 47), and `VLLM_SPARK_MAX_QUERY_ROWS` is capped
   at 40. Does any reachable configuration — present or plausibly next —
   produce a legacy row above 512, and if the Q512 cap ever moves, what
   fails first?
2. **Extension slot ceiling.** Extension slot count is unbounded by
   anything except the port range check inside
   `validate_control_port_pair`. With Q512 and several widths, the
   extension tuple can exceed 26000 entries and push port numbers past
   65535. Is the resulting `ValueError` — which names a payload size,
   not the width list — an acceptable failure mode, and should a width
   count or slot-space bound be checked at parse time instead?
3. **Extension slot stability.** `extensions.index(payload_bytes)`
   ([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):272)
   makes every extension slot a function of the entire admitted width
   list. Adding one width silently renumbers every extension port. Is
   the "all ranks share the same value" requirement enforced anywhere
   other than a native session-connect failure, and is a content-derived
   slot (for example a deterministic hash or a sorted payload-size
   ordering that is stable under insertion) worth the complexity?
4. **Sparse-row hole handling.** Under the sparse contract the supported
   set is `1..40, 42, 48`; slots 40 and 42-46 are never reserved
   ([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):139-146,
   266). Confirm that an extension payload can never land in one of
   those holes, and that a payload whose byte count coincides with a
   legacy payload resolves to the legacy slot rather than allocating a
   second reservation for the same bytes. (The deployment's exact-state
   arena invariant counts MoE weight-storage arenas, not transport
   reservations, so it is not the check that would catch a collision
   here — nothing outside these functions and their tests is.)
5. **Stale docstring on the port resolver.** The docstring of
   `eager_allreduce_ports_for_payload`
   ([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):252-254)
   still states "Index = position of payload_bytes in
   `eager_allreduce_payload_sizes`; the pair is
   `(base0 + 2*index, base1 + 2*index)`". The body no longer does that;
   it uses the two-regime slot. Is the docstring simply out of date, or
   does it record an intended contract that some caller still relies on?
6. **Untested branch of the documented deviation.** All six oracle
   permutations pin `VLLM_SPARK_MAX_QUERY_ROWS=40`
   ([`test_provider_rows_equivalence.py`](../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):45-74),
   which is exactly the value at which the deviation described in
   `_admitted_default_width_rows`
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):249-263)
   vanishes. The default value of the variable is 6. Should a
   permutation at a lower row cap be added — asserting divergence
   deliberately rather than leaving it unobserved — and is the
   admission/reservation-agreement behavior the one the deployment
   should eventually adopt?
7. **Capture guard ordering.** In `spark_all_reduce`, `_eligible` runs
   first and admits any listed width; only then does the capturing
   branch re-gate on `_target_shape_eligible`
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):1231,
   1245-1257). Is there any path where a non-6144 tensor reaches
   `graph_session.capture` — for instance when `_graph_q1_enabled()` is
   false but a capture is active, or when
   `torch.cuda.is_current_stream_capturing` is absent so
   `_is_stream_capturing` returns false
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):331-333)
   during a real capture? The failure mode on that path is
   `os._exit(70)`.
8. **Fail-closed behavior on malformed environment input.** Width
   parsing rejects empty tokens, non-integers, out-of-range values, and
   duplicates
   ([`spark_tp4_port_namespace.py`](../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):157-196),
   and `install()` evaluates it before patching vLLM
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):1215-1218).
   But module-level `_TARGET_SHAPES`
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):44-46)
   calls `provider_query_rows()` at import, which can raise on a
   malformed `VLLM_SPARK_TP4_SPARSE_Q42_Q48` before any install gate
   runs. Is an import-time raise the intended fail-closed point, given
   that vLLM loads plugins inside worker processes? Relatedly,
   `_TARGET_SHAPES` has no remaining runtime reader; the only reference
   outside its definition is a `patch.object` in
   [`test_spark_tp4_backend_dispatch.py`](../spark_transport/integrations/vllm/test_spark_tp4_backend_dispatch.py):614-623,
   where the patch appears inert because the co-patched `MAX_QUERY_ROWS`
   is what actually drives admission. Should the constant be deleted and
   that test tightened?
9. **Dead-but-compared backend helper.**
   `_maximum_allreduce_query_rows`
   ([`spark_tp4_backend.py`](../spark_transport/integrations/vllm/spark_tp4_backend.py):233-238)
   has no caller in the runtime; its only reader is the equivalence
   harness
   ([`_provider_equivalence_harness.py`](../spark_transport/integrations/vllm/_provider_equivalence_harness.py):86-88).
   It returns 48 under the sparse contract while the namespace function
   of the same name returns `MAX_QUERY_ROWS` (40). Is comparing a value
   that drives nothing a meaningful part of the equivalence surface, or
   does keeping it hide the fact that the two same-named functions
   disagree by design?
10. **Vendored-module hash parity in the plugin.**
    [`test_vendor_parity.py`](../sparkring_plugin/tests/test_vendor_parity.py):28-47
    asserts every vendored module is byte-identical to its source and
    that the vendor directory contains no unlisted module. At the tip of
    `claude/sparkring-provider-rows-rebase` the vendored
    `spark_tp4_backend.py` and `spark_tp4_port_namespace.py` do not
    match their sources — the vendor copies predate `384ee39`, so the
    parity test fails until `scripts/sync_vendor.py` is re-run. More
    seriously, the `MODULES` tuple in
    [`sync_vendor.py`](../sparkring_plugin/scripts/sync_vendor.py):18-35
    is the fifteen-module import closure computed before
    `spark_tp4_sparse_q42_q48_contract` became an import of
    `spark_tp4_backend.py` and `spark_tp4_port_namespace.py`. That
    module has no tracked copy in the repository, so re-syncing does not
    fix the closure: a wheel built from this tree cannot import
    `spark_tp4_backend` at all. How should a deployment-private
    dependency be carried by a distributable wheel — vendor a copy,
    declare it an external requirement resolved at install time, or
    invert the dependency so the adapter degrades to the contiguous row
    contract when the module is absent?
