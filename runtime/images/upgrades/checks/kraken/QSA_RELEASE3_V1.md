# QSA source preservation for SparkRing 2026.09.3

`qsa_release3_contracts_v1.py` is a separate CPU oracle for the Kraken B12X
candidate compared with the verified SparkRing 2026.09.3 source baseline. It does not
change the retained prepared-contract oracle or any suite manifest.

The oracle pins the baseline byte hashes of `_contract.py`, `_kernels.py`, and
`_stable_select_cute.py`. It requires the exact 19-name support-kernel inventory,
the entire contract AST, and the entire support-module AST except for an
explicit raw-ring boundary fix. The admitted fix adds a runtime `rows` argument
with `do_not_specialize`, requires nonnegative starts and ends within live rows,
requires a nonempty interval, and checks error status before loading request
IDs. The host launch must supply the live request-ID row count at the matching
ABI position. The exception is constructed from the immutable baseline; it
does not accept the candidate's hash as proof of correctness.

Bounds-checked CPU pointers execute the production guard statements for empty,
negative, reversed, oversized, errored, valid, and ring-capacity boundary cases.
The released unchecked loads must fail adversarial cases. Effective source
mutations must fail preservation, including removed and changed kernels,
selector removal/change, raw-ring reversion, and baseline tampering. Separate
execution checks require selector compile ownership and prepared-program reuse;
mutations that remove either requirement must fail.

Set `SPARKRING_B12X_SOURCE_ROOT` and `SPARKRING_ORACLE_BASELINE` to B12X checkout
roots, then run:

```text
python -m pytest runtime/images/upgrades/checks/kraken/qsa_release3_contracts_v1.py -q
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
ties, large physical-page IDs, and graph replay under fixed workspace.
Use memory-sanitizer evidence for the malformed-boundary launch where
available: accessible allocator padding must not hide out-of-bounds reads.
Offline program-key tests still need the pinned compiler environment to
establish reuse across live/pool geometries. Report these separately from CPU
preservation; no serving, performance, or GPU acceptance is implied here.
