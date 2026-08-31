# Serving configuration templates

`scripts/config/` contains sanitized topology and serving inputs. Templates
describe operator contracts; they are not deployment receipts or evidence of a
healthy cluster.

## Use

1. Choose a deployment in the [profile registry](../../docs/profiles/README.md).
2. Copy its template to a private path outside version control.
3. Replace every documented placeholder for each rank.
4. Run the validator or `--check` command named by the profile.
5. Preserve the resolved, sanitized inputs with any resulting evidence.

Do not commit addresses, credentials, SSH targets, private paths, checkpoint
files, or locally produced image identities.

## Shared topology input

`site.yaml` is the ignored canonical path used by cluster preflight and Ring
Doctor. Copy a topology-compatible `*.example.yaml` file to that path and
validate it before contacting hosts:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

The first command validates structure and invariants. The plan command is
offline. Running preflight without `--print-plan` contacts the configured
hosts read-only.

## Profile template index

Model-specific settings remain in profile-owned templates and documentation.

| Template family | Deployment route |
|---|---|
| `exl3-r7-*` | [GLM-5.2 EXL3 profile](../../docs/GLM52_35BPW_QUICKSTART.md) |
| `glm53-flash-*` | [GLM-5.3 profile router](../../docs/GLM53_FLASH_QUICKSTARTS.md) |
| `deepseek-v4-flash-0731*` | [DeepSeek profile](../../docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| `qwen38-27b-exl3-k5k6*` | [Qwen profile registry](../../docs/profiles/README.md) |

The public GLM-5.3 image pair uses
[`runtime/glm53-flash-jj-r7-gb10/runtime.env.example`](../../runtime/glm53-flash-jj-r7-gb10/runtime.env.example)
as its operator input. That environment is kept with its runtime artifact
rather than duplicated in this directory.

Research overlays are secondary inputs, not complete profiles. Apply one only
when its named procedure defines the base template, source identity, expected
behavior, and evidence boundary.

## Template invariants

A resolved configuration must bind:

- rank and topology identity;
- management and direct-fabric interfaces;
- immutable runtime and checkpoint identities;
- mounted model, cache, and log paths;
- rendezvous and API ports;
- serving limits and memory budgets; and
- the transport selected by the profile.

A mismatch between a template, recipe, runtime manifest, and image label is a
configuration error. Do not normalize or guess conflicting values.

## Direct-fabric image distribution

A validated four-rank site file can drive
[`scripts/fanout_image_archive.py`](../fanout_image_archive.py). See the
[archive fanout guide](../../docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md) for
planning, checksum verification, create-only placement, and image import.

## Safety

Copying or validating a template is **offline**. Remote preflight is
**read-only remote**. Pulling an image, creating a container, changing host
files, or stopping a serving stack mutates hosts and requires authorization
for the named hosts and action.
