# SparkRing agent guide

Read this file before changing or running repository content.

## Write without hidden context

Repository prose, comments, docstrings, reports, errors, plans, and pull
request text must describe the implemented system without relying on
conversation or development history.

- State purpose, behavior, interfaces, invariants, evidence, and limitations.
- Explain a concept before using its internal identifier.
- Do not use lifecycle labels or indefinite references as technical names.
- Canonical documentation describes present behavior; replace stale claims
  instead of layering chronology over them.
- Label status as `implemented`, `qualified`, `research-only`, or
  `unsupported`.
- State measurement evidence as conditions, measurement, result, conclusion,
  and limitations.
- Comments explain non-obvious intent or invariants. TODOs name the missing
  condition and removal criterion.

### Prefer plain language

- Lead with the outcome. State a formal status once, then immediately explain
  what it means in ordinary words.
- Keep operator instructions focused on what to do and what result to expect.
- Put lane, maturity, hardware, and evidence metadata in one table or callout
  instead of repeating it throughout the prose.
- Prefer `the new settings are still being tested` over phrases such as
  `candidate target changes`, `silent promotion`, `historical qualification
  scope`, or `revalidate the composition` when the plain statement is accurate.
- Use exact formal vocabulary only where a machine-readable contract or release
  decision requires it.
- Use short sentences and short paragraphs.

## Supported repository surface

SparkRing supports three model families across eight deployment profiles:

- GLM-5.2 EXL3 3.5-bpw at four-Spark TP4/DCP4, as a base profile and a
  SparkCache composition, using the R7 runtime and site/candidate contracts.
- DeepSeek-V4-Flash-0731 at two-Spark TP2/DCP1 and four-Spark TP4/DCP1, as
  base profiles and SparkCache compositions, using the published serving image
  and per-rank environment contracts.
- Qwen3.8-27B EXL3 K5/K6 at two-Spark TP2/DCP1 as a research-only base profile
  and at four-Spark TP4/DCP1 as a base candidate, using the clean-checkout
  local image builder and checkpoint/source pins from the companion recipe.

Qwen3.8-27B with SparkCache is Pending. No composition recipe or live cache
evidence is published for that combination.

Six-Spark GLM and KIMI profiles are in dev and are not part of the supported
repository surface.

Maintained Python trees are `spark_transport/`, `runtime/`, `scripts/`, and
`performance/`. The GLM and Qwen runtime builders are `runtime/exl3-r7/` and
`runtime/qwen38/`. Do not add
references, CI jobs, or contributor commands for removed native cache,
plugin, legacy runtime-builder, or deleted configuration-example surfaces.

## Canonical inputs

When prose and executable inputs disagree, report the drift rather than
choosing one.

| Subject | Canonical source |
|---|---|
| GLM-5.2 EXL3 runtime build | `runtime/exl3-r7/README.md` |
| Qwen3.8-27B runtime build | `runtime/qwen38/README.md` |
| Runtime base image and model identity pins | `runtime/faststart-lock.json` |
| Public Python overlay allowlist | `runtime/public-overlay-files.json` |
| R7 site and candidate templates | `scripts/config/exl3-r7-site.example.yaml`, `scripts/config/exl3-r7-candidate.example.json` |
| DeepSeek per-rank environment | `scripts/config/deepseek-v4-flash-0731.env.example` |
| Qwen3.8-27B pair environment | `scripts/config/qwen38-27b-exl3-k5k6-pair.env.example` |
| Qwen3.8-27B four-rank environment | `scripts/config/qwen38-27b-exl3-k5k6.env.example` |
| Performance claim requirements | `performance/README.md` |

## Safety classes

- **OFFLINE**: reads or writes only the checkout or local build directory.
- **READ-ONLY REMOTE**: contacts configured hosts without remote mutation.
- **MUTATES HOST**: changes host files, packages, networking, containers, or
  power state.
- **STOPS SERVING**: interrupts or replaces a running model stack.

Agents may run OFFLINE checks. Inspect a generated plan before a READ-ONLY
REMOTE command. Do not perform MUTATES HOST or STOPS SERVING work without
explicit authorization for the named hosts and action.

## Offline checks

Install the development dependencies, then install the CPU torch wheel when a
test imports torch:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
ruff check --select E,F,W --ignore E501 spark_transport runtime scripts performance
python -m pytest spark_transport runtime/exl3-r7 runtime/qwen38 runtime/test_public_overlay.py performance/harnesses scripts -q -rs
```

The test suite is CPU-only contract coverage. It does not validate CUDA,
RDMA, live pair/cycle serving, or a performance result.

## Runtime and configuration work

`runtime/exl3-r7/` builds the GLM-5.2 EXL3 R7 image. `runtime/qwen38/` builds
the Qwen3.8-27B ARM64 image. `runtime/faststart-lock.json` pins the GLM/DeepSeek
foundation image and GLM model identity. `runtime/build-public-overlay.py`
produces a content-manifested bundle from the explicit allowlist in
`runtime/public-overlay-files.json`.

Use `scripts/config/exl3-r7-site.example.yaml` and
`scripts/config/exl3-r7-candidate.example.json` as sanitized R7 inputs. Keep
resolved site addresses, image identities, host paths, and credentials out of
version control. Use `scripts/config/deepseek-v4-flash-0731.env.example` only
as a per-rank environment template for DeepSeek-V4-Flash-0731.

Use the topology-specific Qwen environment in `scripts/config/` with the image
built by `runtime/qwen38/build-image.sh`, as described in
`docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md` and
`docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md`. The profiles do not use the GLM R7
builder or the DeepSeek serving image.

## Performance work

Put reusable measurement programs in `performance/harnesses/` and immutable
evidence in `performance/records/`. Follow `performance/README.md` before
claiming a measurement. A green offline test does not qualify a hardware or
serving result.
