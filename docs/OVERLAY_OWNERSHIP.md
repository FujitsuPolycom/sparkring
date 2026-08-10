# vLLM overlay ownership inventory

The machine-readable inventory is
[`configurations/vllm-overlay-ownership-v1.json`](configurations/vllm-overlay-ownership-v1.json).
It is a draft, offline-validated inventory of exactly 73 ordered build
operations:

- 59 modified upstream files recovered from the reference runtime;
- 12 fork-only files recovered from the reference runtime; and
- 2 independently authored SparkCache compatibility patches applied after the
  recovered changes.

All 61 modifications (59 recovered plus 2 SparkCache) are preimage-pinned. The
12 additions are content-addressed and refuse to overwrite existing targets.
All 73 ordered operations are applied fail closed.

This is an ownership-planning artifact, not an authorship claim or a maturity
promotion. The recovered overlay's `spark`, `b12x`, and `unknown` markers are
best-effort provenance signals only. A proposed natural destination still
requires review by that prospective owner.

Exactly two generic overlay builders, `runtime/Containerfile` and
`runtime/Containerfile.faststart`, declare the patch tree as an input. This
proves only source-builder consumption. Each row separately records the
reference runtime, the live-validated default EXL3+LMCache CS512
configuration, and the accepted NF3 alternative. EXL3 and NF3 consumption
remain `unknown` because this checkout does not contain an exact image and
build-receipt chain from the current lock to either named live image.

The inventory intentionally has 73 unique operation IDs but 71 unique target
paths. The SparkCache operations follow recovered operations on
`vllm/config/vllm.py` and `vllm/v1/core/sched/scheduler.py`; these are ordered
operations, not duplicate inventory records.

Regenerate and validate the inventory offline with:

```bash
python scripts/generate_overlay_ownership_inventory.py
python scripts/generate_overlay_ownership_inventory.py --check
```

The generator uses explicit failures if row fields, counts, operation IDs,
recovered provenance coverage, lock hashes, manifest linkage, or basic
sanitization checks drift. `--check` also fails if the committed JSON is stale.
It does not contact a Spark or any other remote host.

Mixed operations use `mixed/requires-decomposition` and list the behavior
proposed for each destination. In particular, the DSpark model files and the
B12X-backed DCP all-to-all operation are not assigned to one owner. No
operation is assigned to NCCL, the SIRCL adapter/plugin, or SparkRing
distribution merely because those modules exist; their independently
published sources and patches are outside this inventory.
