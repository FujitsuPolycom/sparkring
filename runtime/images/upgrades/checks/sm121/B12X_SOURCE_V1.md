# B12X SM121 source oracle migration

This versioned suite preserves the published Kraken oracle files and uses the
immutable SparkRing 2026.09.3 source baseline. It admits the composed SM121 source
without interpreting changed candidate hashes as evidence of correctness.

The common KDA checkpoint tests are unchanged and run against both baseline and
candidate. Candidate tests retain the existing checkpoint, preparation,
compiler-worker, lifetime, PLE and GDN checks.

Two fixture adaptations preserve every assertion and test parametrization:

- The score compiler fixture declares DCP size 1, rank 0 and token interleave 1.
  These are new mandatory static inputs; its cache reuse and prepared error-state
  ABI assertions are unchanged.
- The distributed cache fixture uses schema 6's SM121/48-SM silicon identity and
  carries `rejected_count` through the source coordinator's six-field tuning
  wire. Actual source cache validation, agreement, reracing and cancellation run.
  The 27 original tests and their assertions remain intact.

`package_b12x_suite.py` verifies the frozen parent inventory and AST equality of
assertions and test decorators before producing `b12x-sm121-suite-v1.json`.

## QSA safety with context parallelism

The old whole-module QSA equality rule permits only the raw-ring load guard and
shared-pool preparation fixes. It cannot admit the reviewed DCP implementation.
`qsa_dcp_contracts_v3.py` replaces that rule with explicit baseline-derived
transformations and executed semantic checks:

- Baseline identities remain fixed. Every one of the 19 existing support kernel
  definitions must equal the immutable baseline plus the recorded ring fix and
  explicit DCP edits. Those edits change local token/group counts, ownership,
  table-width guards and tail positions. No other kernel-body changes pass.
- The stable selector remains byte-identical after LF/CRLF normalization only.
  The score kernel's additional stripe-count calculation and constructor fields
  are reconstructed from the baseline AST; all other scoring arithmetic and
  error gates remain exact. The paired scorer and draft-record kernel remain
  unchanged. Draft preparation changes only rank-local tail mapping.
- Shared/separate validator program keys and compilation inside the owned
  support context remain required. Runtime ABI/alias validation precedes state
  mutation, the five transaction validators retain their order, launches carry
  prepared programs, and failed outputs are poisoned.
- Frozen adversarial raw-ring tests, baseline-negative controls, selector
  ownership tests and their effective mutations are reused unchanged.
- CPU execution of actual source kernel bodies checks local counts against
  independent stripe enumeration, active table read/write bounds, malformed
  physical pages, undersized tables, shared ownership, prior errors, idle live
  owners, global group/tail expansion, and draft source bounds/error propagation.
  Both draft reuse call sites must pass the plan handle and error offset.
- Effective mutants remove/change kernels, revert ring/table guards and local
  counts, alias shared keys, replace the selector, remove validation/prepared
  ownership and alter the immutable baseline. Every mutant must fail admission.

The complete `_contract.py` module is no longer compared as an opaque AST. Its
specific safety and preparation obligations above are checked instead; new DCP
collective integration and other unrelated contract behavior are outside this
CPU oracle. The original files remain in the bound inventory as both evidence
and imported frozen checks.

## Limits and execution

The NumPy interpreter bounds-checks active pointer accesses and executes source
control flow; it removes DSL numeric casts. It does not emulate GPU integer
overflow, floating-point arithmetic, parallel atomic races, CUDA allocation or
stream ordering. Native compile/replay ABIs are checked separately by the
retained preparation tests. CUDA numerical, graph replay, sanitizer, distributed
collective and full serving qualification remain necessary.

Run the normal `kraken_gate.py` with this suite, the exact source roots, immutable
baseline, and a hash-bound paired vLLM coordinator. Run the baseline common scope
as well as the candidate common and candidate-only scopes. Use CPU-only bounded
containers; test success is source admission, not image or serving qualification.
