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

## Inspect and resolve

```bash
python scripts/profile.py list
python scripts/profile.py validate
python scripts/profile.py resolve deepseek-v41-flash-cycle
python scripts/profile.py resolve deepseek-v41-flash-cycle --set max_num_seqs=4
python scripts/profile.py resolve --legacy recipes/deepseek-v41-flash-cycle.json
```

Resolution applies supported explicit overrides, profile defaults, then the small
common defaults (DCP1 and pipeline parallelism 1 when omitted). The result names
each value's origin. Model identity, quantization variant, rank count, topology
and release cannot be overridden through serving knobs. A changed serving value
is research-only; it does not inherit the default's evidence. An explicit value
equal to a default produces the same behavior and retains its original scope.

The resolver does not read process environment variables, run a shell, contact
hosts or inspect model weights. A legacy recipe must match a catalog recipe;
edit its authoritative `profiles/` source and regenerate the compatibility
export. This prevents independent configuration drift.

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
Add `--execute` before the profile only to run the adapter locally. Python rank
adapters expose `plan`, `create` and `start`; Bash adapters expose `--check` and
`--run`. Existing guards and receipt validation remain active. R33 selections
supply the catalog's exact runtime receipt. Compositions requiring managed
multi-host steps direct the operator to their guide instead of inventing a
single unsafe start command.

For the DeepSeek and Qwen Bash adapters, `render-env` renders an actual adapter
input from the same profile defaults and supported overrides:

```bash
python scripts/profile.py render-env deepseek-v41-flash-cycle --site-values rank0.site.json --set max_num_seqs=4 > rank0.local.env
python scripts/launch.py deepseek-v41-flash-cycle -- --check rank0.local.env
```

The per-rank JSON object supplies only the environment keys whose template values
contain placeholders, such as `NODE_RANK`, `MASTER_ADDR` and `MODEL_HOST_PATH`.
Use the selected `runtime.env.template` to identify them. It cannot replace
serving or identity settings. The renderer rejects shell syntax and unresolved
placeholders; it never sources a file. Generated legacy ENV examples derive the
shared context, sequence and batch-token defaults from the same recipe.

A resolved JSON document is not an activation receipt or a replacement for the
adapter's host checks. GLM managed and frozen image procedures retain their
specialized site contracts and explicit receipt gates.

The frozen source-image TP2 profile and published R33 TP2 profile have different
context/cache settings. They remain distinct contracts. Never transfer image
qualification across those selections or interchange NVIDIA NVFP4 and
NVFP4-Spark checkpoints. Context limits are not measured KV capacity.
