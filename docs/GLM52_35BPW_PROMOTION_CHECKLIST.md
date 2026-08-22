# Promote a rebuilt GLM-5.2 EXL3 3.5-bpw image

Use this checklist for an image built from the tracked GLM recipe
`recipes/glm52-exl3-r7-3.5bpw.json`. The qualified status of the operator image
does not transfer to another image ID.

## Offline qualification

- [ ] Build from `runtime/exl3-r7/build-image.sh` with an immutable parent image
  ID and an audited parent license expression.
- [ ] Record the resulting Docker image ID.
- [ ] Download and verify
  `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`
  using `scripts/download_exl3_r7.py`.
- [ ] Generate the candidate, dynamic-NVFP4, CKV-gather, SIRCL, and exact-Q40
  profile layers from the tracked scripts.
- [ ] Confirm every generated profile and site references the recorded image ID.
- [ ] Inspect the final preflight and launcher plans.

## Four-rank live qualification

- [ ] Place the identical rebuilt image ID on all four directly cabled DGX
  Sparks.
- [ ] Pass the reviewed fabric and remote preflight for the final site.
- [ ] Start four ranks with the generated operator profile.
- [ ] Confirm `/health` returns HTTP 200 and `/v1/models` reports
  `glm-5.2-exl3-r7-3.5bpw` with a 262,144-token maximum length.
- [ ] Run the fixed-seed equivalence and bounded C1, C2, and C8 workload in
  [`GLM52_35BPW_ACCEPTANCE_RUNBOOK.md`](GLM52_35BPW_ACCEPTANCE_RUNBOOK.md)
  against the image ID and preserve its receipts.
- [ ] Confirm post-run rank and transport health.

## Promotion record

- [ ] Preserve sanitized receipts identifying the image ID, recipe, generated
  profile hashes, and all four ranks without recording site addresses,
  credentials, or local paths.
- [ ] State the hardware topology, evidence scope, and remaining limitations in
  any result report.

The profile becomes qualified only when every item above passes for the same
image ID.
