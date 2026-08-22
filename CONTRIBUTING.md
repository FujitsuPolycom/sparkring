# Contributing to SparkRing

SparkRing supports GLM-5.2 EXL3 3.5-bpw and DeepSeek-V4-Flash-0731 serving
surfaces. Contributions must preserve the explicit pins, candidate contracts,
and evidence boundaries that make those surfaces reviewable.

## Pull requests

- Match the surrounding code and follow [the repository writing
  standard](AGENTS.md#write-without-hidden-context).
- Do not commit site addresses, SSH targets, credentials, local paths, model
  files, or mutable local image names. Sanitized evidence records may include
  immutable registry digests and Docker image IDs when they are required to
  bind a result to exact bytes and reveal no site identity.
- Do not add compatibility code, documentation, or CI coverage for removed
  cache, plugin, runtime-builder, or configuration-example surfaces.
- Include the relevant focused validation command and its result.
- Contributions are licensed under Apache-2.0; copied or adapted code requires
  provenance and a corresponding `THIRD_PARTY_NOTICES.md` entry.

## Maintained Python surfaces

The offline Python suite covers only these maintained trees:

| Tree | Purpose |
|---|---|
| `spark_transport/` | Native-transport adapters, probes, and vLLM integration contracts |
| `runtime/` | Runtime overlay, lock, and R7 builder contract checks |
| `scripts/` | R7 configuration, candidate generation, and operational-plan tooling |
| `performance/` | Measurement harness parsing and accounting logic |

Set up a CPU-only development environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
ruff check --select E,F,W --ignore E501 spark_transport runtime scripts performance
python -m pytest spark_transport runtime/exl3-r7 runtime/test_public_overlay.py performance/harnesses scripts -q -rs
```

These checks are offline and CPU-only. They do not prove CUDA compilation,
RDMA transport, four-rank startup, model correctness, or performance.

## Runtime contributions

The GLM-5.2 EXL3 3.5-bpw build lives in
[`runtime/exl3-r7/`](runtime/exl3-r7/README.md). The R7 builder must retain
immutable source and artifact verification. The adjacent
[`runtime/faststart-lock.json`](runtime/faststart-lock.json) owns the pinned
ARM64 image and model identity. The public Python overlay is built by
[`runtime/build-public-overlay.py`](runtime/build-public-overlay.py) from the
explicit allowlist in
[`runtime/public-overlay-files.json`](runtime/public-overlay-files.json).

Change a pin, build input, or overlay member only with a focused test that
would fail if the intended contract drifted. Do not accept a hash mismatch by
rewriting the expected hash from an arbitrary local tree.

## Configuration contributions

The R7 site and candidate templates are:

- [`scripts/config/exl3-r7-site.example.yaml`](scripts/config/exl3-r7-site.example.yaml)
- [`scripts/config/exl3-r7-candidate.example.json`](scripts/config/exl3-r7-candidate.example.json)
- [`scripts/config/exl3-r7-pins.json`](scripts/config/exl3-r7-pins.json)

They are sanitized templates. Resolved inputs remain local and untracked.
Validate configuration changes through the affected tests in `scripts/`; do
not use an example configuration as evidence of a deployable appliance.

DeepSeek-V4-Flash-0731 uses separate per-rank environment templates for the
[four-Spark cycle](scripts/config/deepseek-v4-flash-0731.env.example) and
[two-Spark pair](scripts/config/deepseek-v4-flash-0731-pair.env.example). Keep
their placeholders explicit and document environment-variable invariants in
the applicable template.

## Performance contributions

Reusable measurement programs belong in `performance/harnesses/`; evidence
records belong in `performance/records/`. Before publishing or changing a
measurement claim, follow [`performance/README.md`](performance/README.md).
Every record must state conditions, measurement, result, conclusion, and
limitations. Hardware-specific evidence must identify the serving
configuration and must not be generalized beyond the recorded conditions.

## Documentation checks

CI resolves repo-relative Markdown links and heading anchors without fetching
external URLs. Run the same link check through CI or inspect changed
repo-relative targets and anchors before submitting documentation changes.

## Operational safety

Commands that only read the checkout are **OFFLINE**. Commands that contact
configured hosts without changing them are **READ-ONLY REMOTE**. Starting,
stopping, rebuilding, pulling, or changing configured hosts is **MUTATES
HOST** or **STOPS SERVING** and requires explicit authorization for the named
hosts and action.
