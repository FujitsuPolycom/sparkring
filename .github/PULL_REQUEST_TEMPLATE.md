<!--
  SparkRing supports two serving configurations:
  - GLM-5.2 EXL3 3.5-bpw with the R7 runtime and site/candidate contracts.
  - DeepSeek-V4-Flash-0731 with the published serving image and per-rank
    environment contract.

  Every checked box is a real gate. If a section does not apply, write "n/a"
  and explain why. Do not claim validation that was not run.
-->

## What this changes

<!-- State the resulting behavior and why it is needed. Link the issue. -->

Closes #

## Supported profile

<!-- Select every profile intentionally changed; state "neither" for metadata-only changes. -->

- [ ] GLM-5.2 EXL3 3.5-bpw — R7 runtime and site/candidate contracts
- [ ] DeepSeek-V4-Flash-0731 — published image and per-rank environment contract
- [ ] Neither profile; repository metadata or tooling only

**Status:** <!-- implemented, qualified, research-only, or unsupported -->

## Provenance and release safety

- [ ] This change contains no code copied or adapted from another project.
- [ ] Any third-party source is identified below and reflected in
      `THIRD_PARTY_NOTICES.md` where required.
- [ ] I introduced no private identifiers: no site IPs, hostnames, SSH
      accounts, absolute host paths, private image names, or credentials.
- [ ] Documentation addresses use RFC 5737 ranges and site values use
      `<PLACEHOLDER>` tokens.
- [ ] I did not modify release-safety checks or weaken fail-closed behavior.
- [ ] Any release-affecting change has a rollback or refusal condition stated
      in this PR.

Third-party source, if any:

| Source project | Upstream URL / commit | License | Files or ideas | Notice updated? |
|---|---|---|---|---|
| | | | | |

## Tests and serving validation

Paste exact commands and relevant output. A green offline check does not
qualify CUDA, RDMA, four-rank serving, or a performance result.

### Offline checks (OFFLINE; CPU-only)

Consolidated Python contract suites:

```console
$ python -m pytest spark_transport runtime/exl3-r7 runtime/test_public_overlay.py performance/harnesses scripts -q -rs

```

Static checks for maintained Python trees:

```console
$ ruff check --select E,F,W --ignore E501 spark_transport runtime scripts performance

```

- [ ] I ran the consolidated Python suites and pasted their result.
- [ ] I ran the static checks and pasted their result.
- [ ] If tests were not added or changed, I explained why the observable
      contract is already covered.

### Four-rank live validation (hardware gate; not CI)

Required for serving, transport, topology, image, environment, or performance
claims. State the exact profile, image/model identity, rank layout, and evidence
artifact. Do not run host-mutating commands without explicit authorization for
the named hosts and actions.

- Profile: <!-- GLM-5.2 EXL3 3.5-bpw / DeepSeek-V4-Flash-0731 / n/a -->
- Hosts and authorization: <!-- names or sanitized identifiers; authorization for MUTATES HOST or STOPS SERVING -->
- Exact live command and result:
- Evidence artifact:
- [ ] Four-rank live validation passed.
- [ ] Four-rank live validation was not run; maintainer hardware gating is
      required before a serving or performance claim is accepted.
- [ ] This change is not serving-related; live validation is n/a.

## Performance evidence

If this PR states throughput, latency, speedup, capacity, or another number,
provide immutable evidence from `performance/records/` and include conditions,
measurement, result, conclusion, and limitations.

- [ ] This PR makes no performance claim.

| Claim and units | Profile / rank layout | Context and concurrency | Window and cells | Baseline | Evidence JSON |
|---|---|---|---|---|---|
| | | | | | |

- [ ] I did not present offline results as four-rank qualification.
- [ ] The claim is labeled `implemented`, `qualified`, `research-only`, or
      `unsupported` as applicable.

## Housekeeping

- [ ] I used surviving repository paths and present-state semantic names.
- [ ] I updated documentation made inaccurate by this change.
- [ ] The contribution is offered under Apache-2.0.
