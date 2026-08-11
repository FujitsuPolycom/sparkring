# OMP handoff: EXL3 R7 stand-up public chain

## Completed

The public input chain for the EXL3 R7 3.5-bpw fixed-MTP4, DCP4, KV9.25
candidate is published and test-covered:

- `recipes/glm52-exl3-r7-3.5bpw.json` — tracked recipe with model pins and
  serving contract.
- `scripts/config/exl3-r7-pins.json` — public pins derived from the recipe
  (replaces the builder-branch `runtime/exl3-r7/pins.json` reference).
- `scripts/config/exl3-r7-site.example.yaml` — R7 site template with
  placeholder addresses and R7 serving values.
- `scripts/generate_exl3_r7_candidate.py` — baseline profile generator
  (updated to use the public pins path).
- `scripts/generate_exl3_r7_stock_dcp4.py` — **new**: derives the stock-DCP4
  baseline from tracked inputs, eliminating the maintainer-held profile
  dependency. Passes `prepare_exl3_r7_mtp2.validate_stock_control`.
- `scripts/prepare_exl3_r7_mtp{2,3,3_kv925,4}.py` — seeded derivative
  generators (unchanged).
- `scripts/exl3_r7_standup.py` — **new**: dry-run-by-default stand-up
  entrypoint chaining the full profile derivation. Separates OFFLINE,
  READ-ONLY REMOTE, MUTATES HOST, and STOPS SERVING steps. No live
  execution is performed or allowed.
- `scripts/download_exl3_r7.py` — checkpoint downloader/verifier (seeded).
- `docs/EXL3_R7_QUICKSTART.md` — **new**: full stand-up quickstart with
  prerequisites, model download, image selection, site template, offline
  validation, dry-run plan, start confirmation, health checks, MTP3
  rollback, and candidate maturity statements.
- `scripts/sparkring_recipe.py` — updated to validate the R7 recipe.
- Tests: `test_generate_exl3_r7_stock_dcp4.py` (new),
  `test_exl3_r7_standup.py` (new), `test_generate_exl3_r7_candidate.py`
  (entrypoint test skip-fixed), plus all seeded prepare tests pass.

Full suite: 2497 passed, 15 skipped. Ruff E/F/W clean. Release-safety
patterns clean.

## Blocker: runtime/exl3-r7/ builder branch

The `runtime/exl3-r7/` directory (Containerfile, entrypoint.sh,
build-image.sh, builder-specific pins) is on a separate branch and is
not present in this worktree. The task instruction says "Do not modify
runtime/exl3-r7; the builder is a separate branch."

Consequences:

1. The generator test `test_r7_entrypoint_applies_explicit_environment_unsets`
   skips when `runtime/exl3-r7/entrypoint.sh` is absent. The entrypoint
   SHA-256 is pinned in the generator and verified when the builder branch
   is present.

2. A user cannot build the ARM64 image from this worktree alone. The
   quickstart documents both options: pull-by-digest (if a local image
   exists from the builder branch) or local ARM64 build (requires the
   builder branch). The quickstart explicitly states "A public registry
   image does not exist."

3. The recipe's `runtime.pins` field points to `runtime/exl3-r7/pins.json`
  (builder branch). The public pins at `scripts/config/exl3-r7-pins.json`
  are a derived subset sufficient for the profile generator. The builder
  pins contain additional overlay/builder metadata not needed for
  offline profile generation.

This blocker does not prevent the public profile chain from being
generated, validated, or dry-run. It only prevents building the serving
image from this checkout.
