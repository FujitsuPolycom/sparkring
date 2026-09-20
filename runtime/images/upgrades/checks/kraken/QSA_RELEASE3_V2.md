# QSA source preservation for SparkRing 2026.09.3, version 2

`qsa_release3_contracts_v2.py` is a separate CPU oracle for the Kraken B12X
candidate compared with the verified SparkRing 2026.09.3 source baseline. It
supersedes `qsa_release3_contracts_v1.py` in the B12X Kraken suite; version 1
remains on disk as the record of the first admitted change only. The retained
prepared-contract oracle is unchanged; the Kraken suite selects version 2.

The oracle pins the baseline byte hashes of `_contract.py`, `_kernels.py`, and
`_stable_select_cute.py`. It requires the exact 19-name support-kernel inventory
and reconstructs both admitted changes from the immutable baseline AST; the
candidate must equal that reconstruction exactly, so the candidate's own hash
is never accepted as proof of correctness.

## Admitted change 1: raw-ring boundary guard

The raw-ring commit kernel gains a runtime `rows` argument with
`do_not_specialize`, requires nonnegative starts and ends within live rows,
requires a nonempty interval, and checks error status before loading request
IDs. The host launch supplies the live request-ID row count at the matching ABI
position. Bounds-checked CPU pointers execute the production guard statements
for empty, negative, reversed, oversized, errored, valid, and ring-capacity
boundary cases; the released unchecked loads must fail the adversarial cases.

## Admitted change 2: prepared programs for shared compressed/raw storage

A binding whose compressed cache and raw ring share one pool launches the
shared-pool ownership validator and the `SHARED_COMPRESSED_RAW_POOL` variant of
the page-table validator before any state update. Prepared programs are
resolved from the support map that `compile_qsa` records, and that transaction
compiles a separate-storage layout: `_overlaps` reports no overlap while
compile-only launches are enabled. The release therefore left two defects for
shared-pool bindings: the ownership programs were absent from the map, and the
shared page-table variant resolved to the separate-storage program under the
same key, so its alias validation was silently skipped.

The admitted change has two parts, both reconstructed from the baseline:

- `_support_kernel_key` appends `/shared` when the launch's
  `SHARED_COMPRESSED_RAW_POOL` constexpr is set, so both page-table variants
  coexist in the support map.
- `compile_qsa` imports `launch_validate_page_tables` and
  `launch_validate_shared_pool_ownership` beside `_support_context` and calls
  both inside its support compile context after the decode transaction, with
  `shared_compressed_raw_pool=True`, the transaction's compile-time operands,
  and the occupancy view at the work-metadata offset that the runtime uses.

Effective mutations must fail preservation: removed, changed, or extra kernels;
selector removal or change; raw-ring reversion; removal of the `/shared` key
suffix; contract reversion to the release; compiling the page-table validator
for separate storage; dropping the ownership compile; any other contract edit;
and baseline tampering. Executed support-key checks show that the candidate
distinguishes the shared variant while the release aliases it. Selector compile
ownership and prepared-program reuse checks are unchanged from version 1.

Set `SPARKRING_B12X_SOURCE_ROOT` and `SPARKRING_ORACLE_BASELINE` to B12X checkout
roots, then run:

```text
python -m pytest runtime/images/upgrades/checks/kraken/qsa_release3_contracts_v2.py -q
```

## Stable-selection ordering

The exact sorted comparisons in `tests/attention/test_qsa_stable_selection.py`
are consistent with the public `launch_stabilize_topk` implementation and the
reference consumer contract. The CuTe selector emits intermediate winners in
input order above the threshold, followed by chosen threshold ties.
`launch_stabilize_topk` then calls `_copy_stable_topk_kernel`, which sorts packed
score and inverse global-ID keys in descending order. For QSA's nonnegative
scores this produces descending score order with lower global IDs first for
ties. The reference in `attention/qsa/reference.py` likewise uses stable
descending selection; `_qsa_decode_impl` carries the resulting values and IDs
into subsequent score chunks. The GPU test targets the complete launch, not
the unsorted CuTe intermediate. Its sorted assertions must remain intact.

## Remaining GPU qualification

CPU checks do not establish CUDA safety or numerical correctness. Run the
unchanged stable-selection GPU test across its budgets, score patterns,
offsets, live-row counts, frozen resolution, and graph replay. Run the QSA
contract GPU tests, including the malformed raw-ring boundary regression,
transactional error poisoning and state immutability, score-chunk carry and
ties, large physical-page IDs, graph replay under fixed workspace, and the
shared-pool replay test that prepares a binding over one shared pool. The
prewarm replay test prepares one plan per request-ID dtype, because prepared
programs bind the declared operand dtype exactly. Use memory-sanitizer
evidence for the malformed-boundary launch where available: accessible
allocator padding must not hide out-of-bounds reads. Offline program-key tests
still need the pinned compiler environment to establish reuse across live/pool
geometries. Report these separately from CPU preservation; no serving,
performance, or GPU acceptance is implied here.
