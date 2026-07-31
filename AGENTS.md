# SparkRing agent guide

This file is the entry point for coding agents and other zero-context
contributors. Read it before changing or running anything in this repository.

## Current truth

- The **reference lane** produced the measurements in `README.md` and
  `docs/RESULTS.md`. The recovered 71-file reference-runtime delta is published
  under `runtime/patches/00-reference-vllm/`; the exact historical launch
  artifacts and evidence harness remain maintainer-held, and public-lane
  reproduction of those measurements is pending.
- The **public-functional lane** now has a source bootstrap for the validated
  NF3 target. It builds one receipt-gated ARM64 image on rank 0, distributes
  that exact image ID, and launches the C8/Q40 graph profile. A clean-checkout
  external reproduction is still requested; do not equate bootstrap readiness
  with independent reproduction.
- The non-default **EXL3 lane** has an offline-validated, receipt-gated public
  source bootstrap under `runtime/exl3/` and `scripts/bootstrap_exl3.py`. Its
  serving configuration is live-validated, but the new public bootstrap still
  needs its clean-checkout four-Spark acceptance run.
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
| `runtime/` | Candidate public runtime builder, lock, and published patches |
| `scripts/` | Site schema, read-only preflight, dry-run-first launcher, acceptance gate, and evidence tooling |
| `scripts/config/` | Sanitized templates; real local configs must remain untracked |
| `docs/` | Architecture, measured results, lane contract, setup reconstruction, and gaps |
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
4. Run `python -m pytest spark_transport sparkcache runtime scripts -q`.
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

Start with `docs/QUICKSTART.md`, then
`scripts/config/{launch,gate}.example.json`. The public launcher is dry-run by
default; inspect its `plan` output first. Dry-run validates the current lock,
filled site identity, and local launcher contract without executing them. Any reported
blocker is not a check to bypass, and a successful plan is not acceptance.

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
