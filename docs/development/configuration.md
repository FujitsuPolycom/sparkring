# Deployment configuration

Four objects remain separate:

| Object | Owner | Meaning |
|---|---|---|
| Deployment profile | `profiles/*/profile.json` and its referenced recipe | Model variant, topology, serving defaults and evidence scope |
| Release selection | `runtime/releases/*/release.json` | Published image or pinned build inputs |
| Private site input | Untracked `*.site.json` | Rank addresses and host paths |
| Evidence | Referenced performance record or receipt | What was actually tested under stated conditions |

The catalog contains only stable IDs and definition paths. A definition uses
`sparkring-deployment/v1`; unknown fields, versions, duplicate JSON keys, missing
references and invalid topology values are errors. Future incompatible schemas
need a separate loader and migration; do not reinterpret v1 fields.

The optional `quickstart_status` field describes a primary guide that selects a
different release. Tables use that status; `status`, release selection and
configuration evidence retain their original meaning. The resolver exposes both
statuses when they differ, and the detailed catalog links to the retained record.

Table-only capability annotations live in `profiles/capabilities.json`. They
describe integration work and do not enable features or change serving defaults.

## Inspect and resolve

Operators can obtain a compact installation selection without contacting hosts:

```bash
python3 scripts/sparkring.py setup show glm53-flash-spark-tp2-dcp1-sparkcache
python3 scripts/sparkring.py setup show qwen38-flash-next-tp2 --format json
```

The helper resolves the existing profile, validates the release's pinned inputs,
and reports its publication and checkpoint. `--format shell` emits quoted
assignments for reviewed local use; it does not create an installed-image receipt.
GLM's `--variant nvfp4-qad` reads its declared target variant rather than changing
the profile's image. Unsupported selections fail with a pointer to their own guide.

`setup storage` applies [planning allowances](../../profiles/storage-planning.json)
to local model, Docker and cache destination filesystems. It sums allocations
sharing a filesystem, probes existing ancestors without creating directories,
and returns failure when space is insufficient. Reuse flags are explicit planning
assumptions and never replace asset verification. These helpers do not configure
hosts or replace launch adapters. Follow [setup](../operations/setup.md) for the
operator sequence.

```bash
python scripts/profiles.py list
python scripts/profiles.py validate
python scripts/profiles.py resolve deepseek-v41-flash-cycle
python scripts/profiles.py resolve deepseek-v41-flash-cycle --set max_num_seqs=4
python scripts/profiles.py resolve --legacy recipes/deepseek-v41-flash-cycle.json
```

Precedence, highest first: supported explicit overrides, profile defaults
including a declared preferred DCP selection, and common defaults (DCP1 and
pipeline parallelism 1 when omitted). The result names
each value's origin. Model identity, quantization variant, rank count, topology
and release cannot be overridden through serving knobs. A changed serving value
is research-only; it does not inherit the default's evidence. An explicit value
equal to a default produces the same behavior and retains the default's evidence
scope.

The resolver does not read process environment variables, run a shell, contact
hosts or inspect model weights. A legacy recipe must match a catalog recipe;
edit its authoritative `profiles/` source and regenerate the compatibility
export. This prevents independent configuration drift. Canonical composition base references
are repository-relative; the exporter restores their historical relative paths.
A composition that records a checkpoint digest without a model revision retains
that evidence limit instead of inheriting an unverified revision from its base.

## Private site configuration

Copy [site.example.json](../../profiles/site.example.json) to a `*.site.json`
file outside tracked inputs. Replace every placeholder. `--site PATH` validates
one ordered entry per rank plus distinct absolute model and cache paths. The
resolved output contains private inputs; keep it local and redact before sharing.
Credentials are not accepted in the common site schema. Model and cache files
are neither read nor modified during resolution.

Rank-level network details remain in the selected adapter's explicit site/env
contract. Do not copy real addresses into profile defaults. Existing YAML and
ENV deployment inputs remain supported by their original launchers.

## Launch and release boundaries

`python scripts/launch.py PROFILE -- ADAPTER_ARGUMENTS` prints an argv plan.
Add `--execute` before the profile only to run the adapter locally. GLM Python
rank adapters expose `plan`, `create` and `start`; Bash adapters expose `--check`
and `--run`. The SGLang Python adapter exposes `--check`, `--prepare`, `--pack`
and `--run`. Existing guards and receipt validation remain active. Profiles using
the [published SparkRing image release](../../runtime/releases/sparkring-r33-dcp4/release.json)
supply the catalog's exact runtime receipt. Compositions requiring managed
multi-host steps direct the operator to their guide instead of inventing a
single unsafe start command.

Adapters whose private ENV file owns serving settings declare
`launcher.configuration_input: "env"`. This includes the SGLang Python adapter;
Bash adapters retain that input contract by default. Their argv plan does not
read or validate the supplied ENV file. Its `catalog_defaults` field identifies the profile's baseline settings
and status; `configuration_status` is `unresolved`, and `serving` and
`modified_defaults` are `null`. This applies even when the ENV uses the defaults.
Run the adapter's `--check` action to inspect its input. The JSON printed before
`--execute` remains a plan, not a receipt for the adapter's effective settings.

For the DeepSeek and Qwen Bash adapters, `render-env` renders an actual adapter
input from the same profile defaults and supported overrides:

```bash
python scripts/profiles.py render-env deepseek-v41-flash-cycle --site-values rank0.site.json --set max_num_seqs=4 > rank0.local.env
python scripts/launch.py deepseek-v41-flash-cycle -- --check rank0.local.env
```

`--site-values` accepts a separate per-rank placeholder-values JSON object, not
the common `*.site.json` schema consumed by `--site`. The per-rank object supplies
only the environment keys whose template values
contain placeholders, such as `NODE_RANK`, `MASTER_ADDR` and `MODEL_HOST_PATH`.
Use the selected `runtime.env.template` to identify them. It cannot replace
serving or identity settings. The renderer rejects shell syntax and unresolved
placeholders; it never sources a file. Generated legacy ENV examples derive the
shared context, sequence and batch-token defaults from the same recipe.

A resolved JSON document is not an activation receipt or a replacement for the
adapter's host checks. GLM managed and frozen image procedures retain their
specialized site contracts and explicit receipt gates.

The retained source-image pair configuration,
[`glm53-flash-spark-tp2-mtp3`](../../runtime/sparkring/source_image/README.md),
and the [published-image pair deployment](../../profiles/glm53-flash-spark-tp2-dcp1/README.md)
have different context/cache settings. They remain distinct contracts. Never transfer image
qualification across those selections or interchange NVIDIA NVFP4 and
NVFP4-Spark checkpoints. Context limits are not measured KV capacity.
