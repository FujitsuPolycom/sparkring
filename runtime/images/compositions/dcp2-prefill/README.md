# TP2/DCP2 prefill source-derived image

Status: **Experimental**; CPU-tested composition, no image build or DGX
serving qualification. This layer derives from installer image
`sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`.

**Source-read:** [sources.json](sources.json) pins the exact preimage and
result of four Python files: vLLM's KDA checkpoint admission, both mHC
admission gates, and SIRCL's pins and hook inventory. B12X kernels and native
libraries do not change. [The builder](../../derive_dcp2_prefill.py) refuses
another parent receipt or changed source. It uses the shared derived-layer
machinery to hash each installed replacement and re-record the external-base,
toolchain and SIRCL receipts. SIRCL's receipt records a source overlay over
its original wheel; it does not claim that the wheel was rebuilt.

The derived lock names vLLM build
`sparkring-kraken-bc9ea774-tp2-dcp2-prefill`, whose identity requires both
modified vLLM files. The original build's hashes and mHC admissions stay
unchanged. Native SIRCL libraries, ABI, tuning key and libsircl keep their
parent identities. Published release locks and frozen inputs are unchanged.

## Prepare, build and record locally

Run from the repository root on a build host with the ARM64 parent image
available locally, or pass `--parent-root` with an exported parent root to
prepare without Docker. The export must include receipts as well as sources.

```bash
python runtime/images/derive_dcp2_prefill.py prepare \
  --parent-lock runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json \
  --output /tmp/dcp2-prefill-context
python runtime/images/derive_dcp2_prefill.py build \
  --context /tmp/dcp2-prefill-context --tag sparkring:dcp2-prefill-local \
  --name dev-dcp2-prefill --output /tmp/dcp2-prefill-image.json \
  --profiles glm53-flash-nvfp4-spark-tp2-dcp2
```

These are build instructions, not evidence that a build ran. No registry
push is part of the builder. `record` verifies the built image, its carried
SIRCL/libsircl receipts and installer profile admission. Its image ID and
receipt hashes must come from that build; there is no fabricated image lock
in this composition. Use the resulting lock with the research profile's
`--image-lock` option only in a separately authorized hardware window.

The installer refuses the enabled profile on a lock without the derived
vLLM pin, even if someone adds the profile ID to the parent lock. Both
existing environment switches support independent off/on controls on the
derived image. See [component exactness](../../../../integrations/vllm/dcp2_prefill/README.md)
and the [hardware plan](HARDWARE-PLAN.md).
