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

SparkRing supports four model families across ten deployment profiles:

- GLM-5.2 EXL3 3.5-bpw at four-Spark TP4/DCP4, as a base profile and a
  SparkCache composition, using the R7 runtime and site/candidate contracts.
- GLM-5.3 Flash with the public BF16 DFlash2 drafter at four-Spark TP4/DCP1,
  as a cache-disabled base profile and a SparkCache composition, using the
  immutable source contract in `runtime/glm53-flash/` and the sanitized site
  and runtime templates in `scripts/config/`.
- DeepSeek-V4-Flash-0731 at two-Spark TP2/DCP1 and four-Spark TP4/DCP1, as
  base profiles and SparkCache compositions, using the published serving image
  and per-rank environment contracts.
- Qwen3.8-27B EXL3 K5/K6 at two-Spark TP2/DCP1 and four-Spark TP4/DCP1 as
  implemented base profiles, using topology-specific launchers and the
  checkpoint/source pins from the companion recipe.

Qwen3.8-27B with SparkCache is unsupported. No composition recipe or live cache
evidence is published for that combination.

Six-Spark GLM and KIMI work is research-only and is not part of the supported
repository surface.

Maintained Python trees are `spark_transport/`, `runtime/`, `scripts/`, and
`performance/`. The GLM-5.2, GLM-5.3, and Qwen runtime builders are
`runtime/exl3-r7/`, `runtime/glm53-flash/`, and `runtime/qwen38/`. Do not add
references, CI jobs, or contributor commands for removed native cache,
plugin, legacy runtime-builder, or deleted configuration-example surfaces.

## Canonical inputs

When prose and executable inputs disagree, report the drift rather than
choosing one.

| Subject | Canonical source |
|---|---|
| GLM-5.2 EXL3 runtime build | `runtime/exl3-r7/README.md` |
| GLM-5.3 Flash runtime, model, DFlash, NCCL, and SparkCache pins | `runtime/glm53-flash/pins.json` |
| GLM-5.3 Flash four-rank site and runtime profiles | `scripts/config/glm53-flash-tp4-site.example.yaml`, then the selected `scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1*.example.json` |
| Qwen3.8-27B runtime build | `runtime/qwen38/README.md` |
| GLM/DeepSeek runtime base and model pins | `runtime/faststart-lock.json` |
| Qwen runtime/source pins and model identities | `runtime/qwen38/pins.json`, then `recipes/qwen38-27b-exl3-k5k6{,-pair}.json` |
| Public Python overlay allowlist | `runtime/public-overlay-files.json` |
| R7 site and candidate templates | `scripts/config/exl3-r7-site.example.yaml`, `scripts/config/exl3-r7-candidate.example.json` |
| DeepSeek two-rank and four-rank environments | `scripts/config/deepseek-v4-flash-0731-pair.env.example`, `scripts/config/deepseek-v4-flash-0731.env.example` |
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
python -m pytest spark_transport runtime/exl3-r7 runtime/glm53-flash runtime/deepseek0731-gb10 runtime/qwen38 runtime/test_public_overlay.py performance/harnesses scripts -q -rs
```

The test suite is CPU-only contract coverage. It does not validate CUDA,
RDMA, live pair/cycle serving, or a performance result.

## Runtime and configuration work

`runtime/exl3-r7/` builds the GLM-5.2 EXL3 R7 image.
`runtime/glm53-flash/` records the GLM-5.3 Flash runtime, model, public BF16
DFlash2, patched NCCL, SparkCache, and vLLM lease-contract identities.
`runtime/deepseek0731-gb10/` builds the hardened DeepSeek-V4-Flash-0731 image,
and `runtime/qwen38/` builds the Qwen3.8-27B ARM64 image.
`runtime/faststart-lock.json` pins the generic GLM/rollback image, the hardened
DeepSeek image, and the GLM model identity. `runtime/build-public-overlay.py`
produces a content-manifested bundle from the explicit allowlist in
`runtime/public-overlay-files.json`.

Use `scripts/config/exl3-r7-site.example.yaml` and
`scripts/config/exl3-r7-candidate.example.json` as sanitized R7 inputs. Keep
resolved site addresses, image identities, host paths, and credentials out of
version control. Use `scripts/config/deepseek-v4-flash-0731-pair.env.example`
for a two-rank pair and `scripts/config/deepseek-v4-flash-0731.env.example` for
a four-rank cycle.

Use `scripts/config/glm53-flash-tp4-site.example.yaml` with exactly one of the
two GLM-5.3 runtime-profile templates. Keep resolved site addresses, host
paths, image IDs, and credentials outside version control. Both GLM-5.3
profiles use the same SparkCache-capable image and preserve asynchronous
scheduling, native prefix caching, and chunked prefill; only the
SparkCache-named profile enables the external connector.

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
