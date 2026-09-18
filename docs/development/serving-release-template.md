# SparkRing serving runtime — VERSION

Status: **unpublished template**. Replace every `TBD` field before requesting
publication. This document creates no GitHub Release and selects no image.

## Overview

Describe the serving capabilities of this exact version in two or three
sentences. State the supported platform and direct readers to the profile table.
Do not imply that a shared image supports every model or node topology.

## Install

- GHCR image reference: `TBD` — use `ghcr.io/fujitsupolycom/sparkring@sha256:DIGEST`
  with the actual verified registry digest, not the local Docker image ID.
- Human-readable version tag: `TBD` — same published image bytes.
- Platform: `TBD`.
- Anonymous digest-pull verification: `TBD` — receipt or dated result.
- Installation command: `TBD` — insert the exact digest-qualified Docker pull
  command only after verification. Pulling does not configure or launch a model.

The image remains stored in GitHub Packages/GHCR. This Release is its versioned
entry point for installation guidance and compatibility, not a separate image
download. Follow a supported quickstart to obtain model weights and configure
the hosts, transport and serving process.

## Supported configurations

Use one row per distinct qualified configuration. Replace the placeholder row
with evidence-backed selections; label research-only or unsupported selections
explicitly rather than implying they inherit another row's result.

| Model and checkpoint revision | Nodes / TP / DCP | Runtime / speculation | Context and KV basis | SparkCache / media | Status, evidence and pinned quickstart |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

Quickstarts must use repository links pinned to the corresponding source commit
or release tag. A mutable `main` link alone cannot reproduce a released version.

## Resulting changes

- `TBD`: capability or corrected behavior, technical reason and compatibility
  impact. Link the relevant contribution and preserve contributor credit.
- Keep changes specific to the announced image. Do not list unbuilt source
  patches or unrelated merged PRs as installed functionality.

## Validation and limitations

- Source/build/installed-payload verification: `TBD`.
- Serving correctness and failure/recovery checks: `TBD`.
- Cache capture/restore scope, if applicable: `TBD`.
- Performance: `TBD` — report workload, metric, hardware, sample count,
  baseline and cache state; disclose regressions and uncertainty.
- Known defects and untested configurations: `TBD`. Record unresolved restart
  failures even if a subsequent fresh-container launch succeeded.

Do not equate an image build, API health response, or bounded smoke test with
long-duration reliability or universal profile compatibility.

## Host tools and other assets

- Required host executable or native build input: `TBD`, or explicitly “none.”
- For each required asset, provide its purpose, unchanged versioned URL,
  checksum and consuming profile. Distinguish runtime host requirements from
  inputs needed only when rebuilding the image.
- License, third-party notices and provenance locations: `TBD`.

## Provenance and rollback

- Repository source commit and Release tag: `TBD`.
- Composition descriptor and installed receipt: `TBD`.
- vLLM/SGLang, B12X, SparkCache and transport revisions: `TBD` where applicable.
- Compatible rollback image digest, configuration and procedure: `TBD`.
- Cache compatibility/invalidation and model-directory requirements: `TBD`.

Keep dependency releases and original artifact URLs available. Removing a
recommendation does not authorize deleting its pinned image or build inputs.
