# Review handoff: width-generic eager TP4 all-reduce admission

## Scope

This document is the reviewer entry point for one body of work,
proposed as GitHub pull request #26: opt-in width-generic admission
for the eager TP4 all-reduce adapter, a generic query-row provider
seam that removes all deployment-specific row policy from the tracked
tree, a pip-installable `sparkring` plugin that packages the adapter
as a `vllm.general_plugins` entry point with an operator preflight,
and the evidence that pins the result to the adapter pair running in
the four-Spark cluster's serving container (the deployment adapter
lineage). A reader with the pull-request diff and this document has
everything needed to review; every claim below is derived from the
files named here.

## Component map

### Generic runtime — `spark_transport/integrations/vllm/`

| File | What it is |
|---|---|
| [`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py) | The query-row policy resolver. Single owner of the default-width row set for both admission and reservation. |
| [`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py) | The eager TP4 all-reduce adapter: shape admission, dispatch, native session management, `install()`. Consumes the resolver. |
| [`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py) | Deterministic control-port reservation namespace for every TP4 collective family. Consumes the resolver. |
| [`spark_tp4_query_contract.py`](../../spark_transport/integrations/vllm/spark_tp4_query_contract.py) | The `VLLM_SPARK_MAX_QUERY_ROWS` contract (default 6, ceiling 40), baked at module import. |
| [`test_spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/test_spark_tp4_query_row_provider.py) | Seam unit tests: default/Q512/provider resolution, fail-closed validation, composition with widths, and the guarantee that the generic tree never imports the deployment contract. |
| [`test_spark_tp4_backend_dispatch.py`](../../spark_transport/integrations/vllm/test_spark_tp4_backend_dispatch.py) | Admission and dispatch suite for the backend. |
| [`test_spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/test_spark_tp4_port_namespace.py) | Reservation-namespace suite. |
| [`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py) | Equivalence oracle against the deployment adapter lineage. Self-skips without the private fixtures. |
| [`_provider_equivalence_harness.py`](../../spark_transport/integrations/vllm/_provider_equivalence_harness.py) | Subprocess harness that loads one backend/namespace file pair and prints its admission surface as canonical JSON. |
| [`_private_fixtures/`](../../spark_transport/integrations/vllm/_private_fixtures/) | Deployment-lineage copies fetched from the cluster runtime (`deployed_backend.py`, `deployed_namespace.py`, `spark_tp4_sparse_q42_q48_contract.py`). Maintainer-held and untracked: only the directory's `.gitignore` is in version control. |

### Plugin — `sparkring_plugin/`

| File | What it is |
|---|---|
| [`src/sparkring/plugin.py`](../../sparkring_plugin/src/sparkring/plugin.py) | The `vllm.general_plugins` entry point. No-op with no `VLLM_SPARK_*` mode set; exits with code 78 on any installation failure while a mode is enabled. |
| [`src/sparkring/_compat.py`](../../sparkring_plugin/src/sparkring/_compat.py) | Feature-detected vLLM compatibility gate, checked before any family patches vLLM. |
| [`src/sparkring/preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py) | The `sparkring-preflight` console script: read-only configuration and host checks, including the query-row-provider check. |
| [`src/sparkring/_vendor/`](../../sparkring_plugin/src/sparkring/_vendor) | Vendored byte-identical copies of the backend's import closure — sixteen modules including the resolver. External row-policy providers are deliberately not vendored. |
| [`scripts/sync_vendor.py`](../../sparkring_plugin/scripts/sync_vendor.py) | The vendor sync script; its `MODULES` tuple is the authoritative closure manifest. |
| `src/sparkring/_native/` | Where a built `libspark_transport_capi.so` may sit locally. The `.so` is untracked (repository-root `.gitignore` ignores `*.so`); see "Native library" below. |
| [`tests/`](../../sparkring_plugin/tests) | Plugin suites: registration contract, vendored-module hash parity, compatibility gate, preflight. |
| [`pyproject.toml`](../../sparkring_plugin/pyproject.toml) | Packaging: entry point, `sparkring-preflight` script, `_vendor/*.py` and `_native/*.so` package data. |

### Evidence and gates

| File | What it is |
|---|---|
| [`docs/EAGER_WIDTH_VALIDATION_RUNBOOK.md`](../EAGER_WIDTH_VALIDATION_RUNBOOK.md) | The three-leg validation runbook with per-run records: offline equivalence, native transport probes, shadow windows, and the bootstrap-gated cluster runs. |
| [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | Public CI. The blocking `offline-tests` job runs the `spark_transport` suites and the `sparkring_plugin` suites ([`ci.yml`](../../.github/workflows/ci.yml):135-139); it is a hygiene and contract gate on a GPU-free runner, not hardware qualification. |

## Architecture

### The query-row provider seam

The eager TP4 all-reduce admits default-width payloads by query-row
count. Which row counts exist is policy, and
[`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py)
is the single owner of that policy. `resolve_query_rows(environ)`
returns a sorted ascending tuple from one of three mutually exclusive
sources
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):123-151):

- `VLLM_SPARK_TP4_PREFILL_Q512=1`: the broad prefill geometry, rows
  `1..512`.
- `VLLM_SPARK_TP4_QUERY_ROW_PROVIDER=<module>`: an external provider
  module owns the row set. The module is imported lazily, only when
  the variable is set, and must expose
  `provider_query_rows(environ) -> Iterable[int]`
  ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):104-120).
  The provider owns any policy of its own, including reading
  provider-specific environment variables.
- Neither: the contiguous range `1..MAX_QUERY_ROWS` from
  [`spark_tp4_query_contract.py`](../../spark_transport/integrations/vllm/spark_tp4_query_contract.py).

Validation is fail-closed and lives in the generic core, not in
providers: a configured provider that cannot be imported, lacks the
interface, or returns an empty, non-integer, out-of-range,
or duplicated row set raises `ValueError` naming
`VLLM_SPARK_TP4_QUERY_ROW_PROVIDER` and the module
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):78-120).
Provider rows are bounded to `[1, 512]` because rows occupy
row-denominated reservation slots below the extension span
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):45-47).
Configuring both Q512 and a provider raises rather than letting two
geometry sources compete
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):135-139).
Resolution is cached per `(provider, q512, row-cap)` triple
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):50-54,
140-151), keeping the admission hot path allocation-free.

The provider module name is rank-consistent launch identity: every
rank must configure the same value or none. A mismatch produces
differing port reservations and fails native session establishment;
it is not detectable from a single rank
([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):28-31).

The generic tree contains no Q42/Q48 row semantics and never imports
`spark_tp4_sparse_q42_q48_contract`. The deployment's sparse row
contract (rows `1..40` plus 42 and 48) is one provider implementation
of this interface; its module exists only in the deployment runtime
and in the untracked `_private_fixtures/` copy.
[`test_spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/test_spark_tp4_query_row_provider.py):35-38
asserts the contract module is absent from a clean-checkout import of
the adapters.

### One resolution for admission and reservation

Both consumers call the same resolver, so admission and reservation
cannot disagree:

- The reservation namespace derives the supported row set from
  `resolve_query_rows`
  ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):17,
  142-145) and emits one `eager_allreduce:q=N` reservation per
  resolved row
  ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):441-447).
- The backend's default-width admission set is the same call
  ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):22,
  234-243), used by both the graph gate `_target_shape_eligible`
  ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):246-251)
  and the eager gate `_eager_shape_eligible`
  ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):277-295).

`install()` resolves the row policy, parses the width list, and
validates the full port namespace before it patches vLLM, so a broken
provider or malformed value aborts installation rather than degrading
admission
([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):1195-1201).

### Width admission control surface

`VLLM_SPARK_TP4_EAGER_WIDTHS` sets the list of hidden widths
(elements per row) the eager TP4 all-reduce admits. Unset or empty,
the admitted width tuple is exactly `(6144,)`
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):156-195).
Set, it is a comma-separated integer list, returned sorted ascending;
an empty token, a non-integer token, a width outside `[1, 1048576]`,
or a repeated width each raise a `ValueError` naming the variable.
The upper bound keeps `1048576 elements x 512 rows x 2 bytes = 1 GiB`
under the native single-RDMA-write bound of `UINT32_MAX` bytes
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):149-152).
All four ranks must carry the same value, because the value
determines control-port assignment.

Scope is eager all-reduce only. CUDA-graph capture and replay, the
dual-port striped Q40 path, vocabulary all-gather, DCP, and the
all-gather family remain bound to width 6144. A non-6144 tensor seen
during an active graph capture routes to the stock path and is
recorded as `graph_width_ineligible`
([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):1233-1240)
rather than reaching the capture path, whose failure handler
terminates the worker with `os._exit(70)`
([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):87-89).

### Two-regime port mapping

Eager all-reduce control ports come from a slot index over a base
pair (`SPARK_TP4_CONTROL_PORT0` / `SPARK_TP4_CONTROL_PORT1`,
defaulting to `11000` / `11001`) with stride 2. There are two
disjoint slot regimes, split by
[`_eager_allreduce_size_regimes`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):213-243
and resolved by
[`eager_allreduce_ports_for_payload`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):246-297.

**Legacy regime.** Payload sizes of the form `rows * 6144 * 2` for
every row in the resolved query-row set. The slot is `row - 1`
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):265-269),
so rows a provider omits leave permanent holes in the port sequence.
Under the deployment's sparse row contract, rows 41 and 43-47 are
never reserved and their slots stay vacant, while rows 42 and 48
occupy slots 41 and 47. Preserving those holes is the point: the
resolved set determines the exact reservation tuple the deployment's
startup accounting observes.

**Extension regime.** Payload sizes generated by every admitted
non-default width over the contiguous row range `1..maximum`, where
`maximum` is 512 under Q512 and `MAX_QUERY_ROWS` otherwise
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):128-139,
233-243). A row-policy provider is a default-width serving constraint
and does not restrict extension widths. Extension slots start at 512,
past the largest legacy span (Q512 ends at slot 511), and increase
with the payload size's position in the sorted extension tuple
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):270-277).
The base 512 is a constant, not a function of the active row set, so
no provider or Q512 toggle moves an extension port.

**Regime precedence.** A non-default-width payload whose byte count
equals a supported legacy payload is excluded from the extension set
and resolves to the legacy slot
([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):239-242,
265-269). An identical byte count is the same native operation, so it
shares one session and one port pair. A payload in neither regime
raises a `ValueError` naming the byte count.

Reservations are emitted by
[`active_port_reservations`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):427-457:
one `eager_allreduce:q=N` owner per resolved row, then one
`eager_allreduce:payload=B` owner per extension size, in that order.
[`validate_active_port_namespace`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):558-585
range-checks every pair and rejects any port claimed by two owners.

### Plugin packaging

[`plugin.py`](../../sparkring_plugin/src/sparkring/plugin.py) registers
the transport as a `vllm.general_plugins` entry point
([`pyproject.toml`](../../sparkring_plugin/pyproject.toml):25-26).
`register()` is a no-op unless a `VLLM_SPARK_*` mode variable is set;
with a mode enabled, any installation failure terminates the process
with exit code 78 before vLLM can serve traffic
([`plugin.py`](../../sparkring_plugin/src/sparkring/plugin.py):69-91).

The wheel carries the backend's import closure as byte-identical
vendored copies — the sixteen modules listed in
[`sync_vendor.py`](../../sparkring_plugin/scripts/sync_vendor.py):25-42,
including the query-row resolver.
[`test_vendor_parity.py`](../../sparkring_plugin/tests/test_vendor_parity.py):28-47
fails on any hash drift between vendor and source and on any unlisted
vendored module. External row-policy providers are deliberately not
vendored: the wheel is complete without any provider, a provider is
imported lazily only when `VLLM_SPARK_TP4_QUERY_ROW_PROVIDER` names
it, and the preflight reports a configured provider's importability
as its own check
([`sync_vendor.py`](../../sparkring_plugin/scripts/sync_vendor.py):19-24).

**Native library.** The compiled transport
(`libspark_transport_capi.so`) is an external input, not a repository
artifact: the repository-root `.gitignore` ignores `*.so`, so a wheel
built from a clean checkout contains no native library. Plainly: a
source-built wheel does not carry the transport binary. At runtime
the library comes from `SPARK_TP4_LIBRARY`; when that variable is
unset and a locally built `.so` sits at
`src/sparkring/_native/libspark_transport_capi.so`, the plugin
defaults the variable to that path
([`plugin.py`](../../sparkring_plugin/src/sparkring/plugin.py):51-54),
and `[tool.setuptools.package-data]` would ship it if present at
build time
([`pyproject.toml`](../../sparkring_plugin/pyproject.toml):37-38). The
preflight's `native-library` check reports which resolution applies
([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py):162-186).

The `sparkring-preflight` console script
([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py))
runs read-only checks: the query-row-provider resolution, width
parsing, mode, the full port namespace, HCA devices, the native
library, vLLM compatibility, and opt-in control-peer reachability
([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py):252-265).
The query-row-provider check passes with the variable unset (generic
contiguous rows) and otherwise requires the named module to import
and return a valid row set, reporting any failure as a required
diagnostic
([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py):47-80).

The plugin suites — registration contract, vendor parity,
compatibility gate, preflight — run in the blocking `offline-tests`
CI job ([`ci.yml`](../../.github/workflows/ci.yml):138-139) on a
GPU-free runner, with no native library and no external provider
present by design.

## Equivalence evidence

[`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py)
pins this repository's adapter pair to the deployment adapter lineage
using a golden-oracle, subprocess-per-permutation design. It is
maintainer evidence: it runs only where `_private_fixtures/` holds
the deployment-lineage files, and it self-skips with an explicit
message in a clean checkout
([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):100-105).

[`_provider_equivalence_harness.py`](../../spark_transport/integrations/vllm/_provider_equivalence_harness.py)
takes a backend file and a namespace file, loads them under the
invoking environment, and prints one canonical JSON object describing
the complete eager all-reduce admission surface: the ordered
reservation tuple and its length, per-row control-port resolution for
rows 1-520 with rejections recorded as such, admission bitmaps over
rows 1-520 at widths 2880, 4096, and 6144, the maximum query-row and
graph-capacity values, and backend-side control ports at eight probe
payloads including the sparse rows 42 and 48 and the gap row 41
([`_provider_equivalence_harness.py`](../../spark_transport/integrations/vllm/_provider_equivalence_harness.py):48-100).
The namespace is loaded under its real module name first so the
backend's import binds to the paired namespace
([`_provider_equivalence_harness.py`](../../spark_transport/integrations/vllm/_provider_equivalence_harness.py):42-43).
A fresh subprocess per permutation is required because
`spark_tp4_query_contract` bakes `VLLM_SPARK_MAX_QUERY_ROWS` at
import. The test strips every `VLLM_SPARK`, `SPARK_TP4`, and
`SPARK_Q42` variable from the parent environment before applying a
fixed base plus the permutation's overrides
([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):85-88).

Permutations with `VLLM_SPARK_TP4_EAGER_WIDTHS` unset, where the two
parsed surfaces must be equal object-for-object
([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):48-67):

- `production_replay_sparse` — provider set to the sparse contract,
  row cap 40, mode `custom`.
- `contiguous_q40` — no provider, row cap 40, mode `custom`.
- `contiguous_q512` — row cap 40, `VLLM_SPARK_TP4_PREFILL_Q512=1`,
  mode `custom`.
- `shadow_sparse` — provider set to the sparse contract, row cap 40,
  mode `shadow`.

`eager_shape_admission` has no counterpart in the deployment lineage;
it is removed from both sides before comparison, then separately
checked to equal the default-width admission bitmap at 6144 and to be
all-zero at 2880 and 4096
([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):115-126).

Permutations with widths set, where the deployment surface must be a
strict prefix
([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):69-80,
128-160): `sparse_with_width_2880` and `contiguous_with_widths`
(widths `2880,4096`). Each checks that the deployment reservation
list is a prefix of this one, that the remainder is non-empty, that
every added owner is named `eager_allreduce:payload=`, that every
added port starts at or past `base + 2 * 512`, and that per-row port
resolution is unchanged.

**What this proves.** For the enumerated environment permutations,
semantic admission-surface equivalence: the parsed reservation tuple
(owners, ordering, port numbers), per-row port resolution, admission
bitmaps, and capacity values produced by this repository's adapter
pair are identical to the deployment adapter lineage's with widths
unset, and with widths set the additions are strictly appended past
the legacy slot span without disturbing the reservation prefix or
per-row resolution.

**What this does not prove.** It is not byte-for-byte file
equivalence — the compared objects are parsed surfaces per
environment, not adapter file contents, and the files differ by
construction (the seam replaces direct contract imports). It
exercises no GPU, no RDMA link, no native library, and no live vLLM.
It does not run the deployment's attestation. It covers only the
enumerated permutations, all of which pin
`VLLM_SPARK_MAX_QUERY_ROWS=40`; widths beyond 2880 and 4096 are not
compared, nor is any combination of widths with Q512. It says nothing
about numerical correctness or performance.

## Status

Detailed evidence and per-run numbers are in
[`docs/EAGER_WIDTH_VALIDATION_RUNBOOK.md`](../EAGER_WIDTH_VALIDATION_RUNBOOK.md).
Offline suite totals are emitted by the runs themselves (`pytest -rs`
in CI); this table deliberately does not restate counts.

| Item | Status |
|---|---|
| Width-generic eager admission code path | Implemented; offline-validated by the `spark_transport` suites in blocking CI. Default-off. |
| Generic query-row provider seam | Implemented; offline-validated by the seam, dispatch, and namespace suites in blocking CI. |
| Default-width live run on the four-Spark cluster | Bootstrap-gated live validation: the adapter pair booted to serving health and answered deterministic, prefill, concurrency, and speculative probes, with the widths variable unset and with `6144` explicitly admitted. The launch ran against a copy of the deployed `model_runner.py` whose four stale exact-state pins (two arena-cardinality checks, two storage-byte checks) log observed values and continue under `SPARK_EXACT_STATE_BOOTSTRAP=1`. This is NOT exact deployment-attestation equivalence; that is not established, because the attested state's weight source no longer exists (see the runbook's leg-1 record). |
| `sparkring` plugin package | Implemented; offline-validated (GPU-free suites in blocking CI, wheel build). Not deployment-qualified. |
| Private provider-equivalence oracle | Optional maintainer evidence: requires the untracked `_private_fixtures/` deployment-lineage files and is not reproducible from a public checkout. Self-skips cleanly. |
| CUDA-graph capture at non-default widths | Unsupported by construction; non-6144 tensors under capture route to the stock path. |

## Review questions

1. **Coupled span constants.** The provider row bound
   `MAX_PROVIDER_QUERY_ROW = 512`
   ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):45-47)
   and the extension-slot base
   `_EAGER_ALLREDUCE_PREFILL_MAX_QUERY_ROWS = 512`
   ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):24,
   274-276) are two literals in two modules whose equality is what
   keeps legacy slots (at most 511) disjoint from extension slots
   (from 512). The seam tests reject row 513 and assert the extension
   base
   ([`test_spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/test_spark_tp4_query_row_provider.py):80,
   164-167), but nothing ties the two constants to each other. Should
   they be one shared constant, or at least one cross-module
   assertion?
2. **Extension slot ceiling.** Extension slot count is unbounded by
   anything except the port range check inside
   `validate_control_port_pair`
   ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):86-108).
   With Q512 and several widths, the extension tuple can push port
   numbers past 65535, and the resulting `ValueError` names a payload
   size, not the width list that caused it. Is that an acceptable
   failure mode, or should a width-count or slot-space bound be
   checked at parse time
   ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):156-195)?
3. **Extension-slot renumbering when the width list changes.**
   `extensions.index(payload_bytes)`
   ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):274-277)
   makes every extension slot a function of the entire admitted width
   list: adding or removing one width silently renumbers every
   extension port. The all-ranks-identical requirement for
   `VLLM_SPARK_TP4_EAGER_WIDTHS` is enforced nowhere except by native
   session-connect failure. Is a content-derived slot (stable under
   insertion) worth the complexity, or should the renumbering hazard
   at least be surfaced by the preflight?
4. **Provider mismatch across ranks.** The provider module name is
   launch identity: ranks configuring different providers (or one
   rank omitting the variable) derive different reservation sets and
   fail at native session establishment, which is not detectable from
   a single rank
   ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):28-31).
   The preflight runs on one rank
   ([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py):1-7).
   Should `install()` or the preflight emit a row-set digest that a
   launcher can compare across ranks before the first collective?
5. **Resolver cache keying.** The cache key is
   `(provider, q512, MAX_QUERY_ROWS)`
   ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):140),
   but a provider may read provider-specific environment variables
   ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):16-18)
   that are outside the key. For the immutable `os.environ` of a
   launch this is a per-process constant, but callers passing
   explicit `environ` mappings can receive rows resolved from a
   different mapping — the seam's own tests must clear the cache in
   `setUp`
   ([`test_spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/test_spark_tp4_query_row_provider.py):23-24).
   Should the key incorporate a provider-declared signature, or
   should the cache be documented as valid for `os.environ` only?
6. **Provider row bound versus the extension row range.** A provider
   may return rows up to 512
   ([`spark_tp4_query_row_provider.py`](../../spark_transport/integrations/vllm/spark_tp4_query_row_provider.py):94-98),
   while extension widths enumerate only `1..MAX_QUERY_ROWS` (at most
   40) when Q512 is off
   ([`spark_tp4_port_namespace.py`](../../spark_transport/integrations/vllm/spark_tp4_port_namespace.py):128-139;
   [`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):290-295).
   A provider admitting rows 41..512 at the default width therefore
   composes with extension widths capped at 40 rows. Is that
   asymmetry intended, and should the resolver document (or bound)
   provider rows relative to the extension range?
7. **Production configuration must now select the provider.** The
   deployment adapter lineage consults its row-contract module
   unconditionally whenever Q512 is off; the generic pair consults a
   provider only when `VLLM_SPARK_TP4_QUERY_ROW_PROVIDER` names one.
   Every sparse equivalence permutation sets that variable
   ([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):48-67).
   Launched with the deployment's existing environment — which does
   not set it — this pair resolves the contiguous `1..40` set: rows
   42 and 48 silently stop being admitted and fall back to the stock
   path (recorded only as `ineligible_signature`), and if all ranks
   agree, nothing fails. Where is the required environment addition
   recorded for operators, and should the sparse deployment refuse to
   start without it?
8. **Capture guard ordering.** In `spark_all_reduce`, `_eligible`
   runs first and admits any listed width; only then does the
   capturing branch re-gate on `_target_shape_eligible`
   ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):1214,
   1227-1240). Is there any path where a non-6144 tensor reaches
   `graph_session.capture` — for instance when `_graph_q1_enabled()`
   is false but a capture is active, or when
   `torch.cuda.is_current_stream_capturing` is absent so
   `_is_stream_capturing` returns false during a real capture
   ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):311-313)?
   The failure mode on that path is `os._exit(70)`
   ([`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):87-89,
   1247-1259).
9. **Oracle permutation coverage.** Every permutation pins
   `VLLM_SPARK_MAX_QUERY_ROWS=40`
   ([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):48-80);
   no permutation exercises a lower row cap, widths beyond 2880 and
   4096, or widths combined with Q512, and `eager_shape_admission` is
   popped from the comparison and side-checked
   ([`test_provider_rows_equivalence.py`](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py):115-126).
   Is that coverage sufficient for the equivalence claim as stated,
   and should a divergent-by-design permutation (lower row cap) be
   added so the boundary is observed rather than avoided?
10. **Preflight query-row-provider check semantics.**
    `_check_query_row_provider` resolves the row policy inside the
    preflight process, with the vendored modules prepended to
    `sys.path`, and passes when the variable is unset
    ([`preflight.py`](../../sparkring_plugin/src/sparkring/preflight.py):47-80).
    A provider importable in the operator's preflight environment but
    absent from the serving container's `sys.path` passes preflight
    and then kills serving at install time (plugin exit 78:
    [`plugin.py`](../../sparkring_plugin/src/sparkring/plugin.py):74-91;
    install-time resolution:
    [`spark_tp4_backend.py`](../../spark_transport/integrations/vllm/spark_tp4_backend.py):1197-1200).
    Is single-process importability the intended check semantics, and
    since providers are deliberately not vendored
    ([`sync_vendor.py`](../../sparkring_plugin/scripts/sync_vendor.py):19-24),
    where is the supported delivery mechanism for a
    deployment-supplied provider module documented?
