# SparkRing agent guide

This file is the entry point for coding agents and other zero-context
contributors. Read it before changing or running anything in this repository.

## Write without hidden context

All repository prose — documentation, comments, docstrings, commit and pull
request text, plans, reports, names, errors, and technical summaries — must
make sense to a technically capable reader who has the repository but none of
the conversation or development history.

- Describe the system as it exists: purpose, behavior, invariants, interfaces,
  evidence, and limitations. Do not narrate the development journey.
- Do not use lifecycle labels as identities. `Phase 2`, `pilot`, `next`,
  `current`, `new`, `old`, and `latest` are not technical names.
- Do not use false definite references. A phrase such as `the 1M-token
  capture`, `the experiment`, or `this approach` is invalid unless that exact
  object was introduced locally and unambiguously. Counts, dates, and versions
  are attributes, not identities.
- On first reference, give the object's semantic role and, where relevant, its
  durable identifier: artifact name, path, schema, revision, manifest, or hash.
- Explain concepts before identifiers. Internal codenames, experiment labels,
  profile numbers, and implementation shorthand are not the vocabulary of the
  design. Mention a literal identifier only after describing what it means, and
  only where the reader must use it.
- Treat canonical documentation as a present-state specification, not a
  changelog. Replace stale claims rather than layering history on top. Put
  chronology, rejected attempts, and retrospectives only in explicitly
  historical documents.
- Label status as `implemented`, `qualified`, `research-only`, or
  `unsupported`. State evidence as conditions, measurement, result, and
  conclusion.
- Comments explain invariants, intent, and non-obvious constraints, never
  change history. A TODO names the missing condition and the criterion for
  removing it.
- Commits and pull requests state resulting behavior, technical reason,
  compatibility impact, and validation. They do not recount attempts or pivots.

If understanding a sentence requires conversation history, rewrite it.

## Present state

- The **reference lane** produced the historical measurements retained in
  `README.md` and `docs/RESULTS.md`; public-functional results are
  labelled separately. The recovered 71-file reference-runtime delta is published
  under `runtime/patches/00-reference-vllm/`; the exact historical launch
  artifacts and evidence harness are maintainer-held, and public-lane
  reproduction of those measurements has not been performed.
- The **default and main advertised public-functional configuration** is EXL3
  3.25-bpw plus LMCache CS512. Its receipt-gated public bootstrap built
  one ARM64 image from a clean checkout, distributed the identical image ID to
  four directly cabled Sparks, and passed startup, graph, API, repeated
  fixed-seed 128-token, bounded C1/C2/C8, and post-run health gates. This is
  clean-checkout live validation, not blanket correctness, persistence,
  release promotion, or full public-functional acceptance.
- The operator-accepted **3.5-bpw profile** is EXL3 fixed-MTP4, DCP4, dynamic
  NVFP4 plus FP8 RoPE, and 9.25 GB KV/rank. The profile is documented in
  `docs/EXL3_R7_FIXED_MTP4_PROFILE.md`; `R7` is its durable recipe identifier.
  Acceptance applies to one four-Spark appliance. The public builder is
  source-complete and offline-validated, but a clean-checkout image has not
  passed the live promotion gate. This does not change the public default or
  the accepted alternative built on the GLM-5.2 hybrid checkpoint
  `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`, whose routed experts use the
  NF3 weight format.
- That NF3 lane is an accepted deterministic public-functional alternative.
  Its recipe, bootstrap, and quickstart are published and explicitly
  selectable.
- Never describe a reference-lane number as a result from this checkout.

Machine-readable status is in `docs/STATUS.json`.

## Where truth lives

When prose and executable configuration disagree, stop and report the drift.
Do not silently choose the more convenient value.

| Subject | Canonical source |
|---|---|
| Public-lane definition and open blockers | `docs/PUBLIC_FUNCTIONAL_TARGET.md` |
| Public headless-startup ABI audit | `docs/PUBLIC_STARTUP_SHIM_AUDIT.md` |
| Runtime input pins | `runtime/runtime-lock.json` |
| Site-config schema | `scripts/sparkring_site.py` |
| Acceptance behavior and exit codes | `scripts/acceptance_gate.py` |
| Measured claims and their evidence labels | `docs/RESULTS.md` |
| Reference deployment reconstruction | `docs/SETUP.md` |
| Component maturity | `docs/STATUS.json`, then the component README |

Tables in README files summarize these sources; they are not independent pin
sets.

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport, probes, vLLM adapters, and experiments |
| `sparkcache/` | Persistent DCP-sharded context cache |
| `sparkring_plugin/` | Pip-installable vLLM plugin packaging of the TP4 adapters |
| `runtime/` | Candidate public runtime builder, lock, and published patches |
| `scripts/` | Site schema, read-only preflight, dry-run-first launcher, acceptance gate, and evidence tooling |
| `scripts/config/` | Sanitized templates; real local configs must remain untracked |
| [`docs/README.md`](docs/README.md) | Task-routing index for specifications, runbooks, evidence, and history; it does not replace the canonical owners above |
| `.github/` | CPU-only CI, contribution forms, and public-release safety checks |

## Safety classes

- **OFFLINE**: reads or writes only the checkout/build directory.
- **READ-ONLY REMOTE**: contacts configured hosts but is guarded against remote
  mutation.
- **MUTATES HOST**: changes packages, networking, files, containers, or power
  state on a Spark.
- **STOPS SERVING**: can interrupt or replace a running model stack.

Agents may run OFFLINE checks without asking. Inspect the generated plan before
a READ-ONLY REMOTE command. Do not perform MUTATES HOST or STOPS SERVING work
without explicit user authorization for the named hosts and action.

Safe starting commands:

```bash
# OFFLINE
python scripts/sparkring_site.py scripts/config/site.example.yaml
python scripts/preflight.py --site scripts/config/site.example.yaml --print-plan
python -m pytest spark_transport sparkcache runtime scripts -q
python -m pytest sparkring_plugin -q

# READ-ONLY REMOTE, after reviewing --print-plan and filling site.yaml
python scripts/preflight.py --site scripts/config/site.yaml
```

`scripts/acceptance_gate.py` is dry-run by default, but its `--execute` mode is
**STOPS SERVING**: it starts and stops the configured stack. Never point execute
mode at a production-serving cluster.

## Common task recipes

### Documentation or Python change

1. Install `requirements-dev.txt`.
2. Install the CPU torch wheel as shown in `CONTRIBUTING.md`.
3. Run `ruff check --select E,F,W --ignore E501 .`.
4. Run `python -m pytest spark_transport sparkcache runtime scripts -q` and `python -m pytest sparkring_plugin -q`.
5. If Markdown changed, verify repo-relative links using the same checker as
   the `docs-links` CI job.

### Site configuration

1. Copy `scripts/config/site.example.yaml` to `scripts/config/site.yaml`.
2. Replace every placeholder.
3. Run the site validator.
4. Run preflight with `--print-plan`.
5. Only then run the read-only remote preflight.

The canonical local files `scripts/config/site.yaml`,
`scripts/config/launch.json`, and `scripts/config/gate.json` are ignored by
Git. Do not commit site addresses,
SSH targets, local paths, registry identities, or credentials.

### Public acceptance work

Start with `docs/QUICKSTART.md` for the default EXL3+LMCache deployment. Its
bootstrap generates ignored resolved site/profile files and its launcher is
dry-run by default; inspect the `plan` output before execution. The bounded
EXL3 gate is `scripts/exl3_live_gate.py`; the candidate full workflow is
`docs/EXL3_ACCEPTANCE_RUNBOOK.md`. For the accepted NF3 alternative,
start with `docs/NF3_QUICKSTART.md` and
`scripts/config/{launch,gate}.example.json`. Any reported blocker is not a
check to bypass, and a successful plan is not acceptance.

For the operator-accepted 3.5-bpw profile, start with
`docs/EXL3_R7_QUICKSTART.md`. A rebuilt image remains a candidate until every
item in `docs/EXL3_R7_PROMOTION_CHECKLIST.md` passes against its immutable ID.

### Native or cluster work

Follow `docs/SETUP.md` only for stages explicitly marked runnable from the
public tree. Treat its launch and reference-serving stages as a historical
reconstruction unless the user supplies an independently built launcher and
runtime.

## Terminology

- `lane`: `reference` or `public-functional`
- `maturity`: `planned`, `candidate`, `offline-validated`,
  `live-validated`, or `accepted`
- `candidate`: not accepted; always state which gate remains
- `published`: source is present here; it does not imply live validation

Always state lane, maturity, hardware, and evidence scope with a result.
