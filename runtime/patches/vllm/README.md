# vLLM patch overlay — intentionally empty (provenance gate)

This directory ships **empty on purpose**. The production reference-lane
runtime carries a vLLM patch overlay (61 modified + 12 new files, ~12.9k
lines) that is **withheld pending provenance review**. Until that review
clears, no `*.patch` files or `preimages.json` are published here, and the
build's `apply-patches.py` step is a verified no-op for this component.

When the overlay clears review, its patch files land here with per-file
sha256 pins recorded in `runtime/runtime-lock.json` (`overlays`) and a
`preimages.json` covering every patch.

See `docs/RUNTIME_GAPS.md` and `runtime/README.md` for the full gap
accounting.
