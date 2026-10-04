# Compose deployments from SparkRing profiles

`sparkring compose` renders a profile and your private site file into one Docker
Compose project per Spark, checks the files, and starts or stops all ranks from
a plan you approve by its hash.

The normal path is `sudo sparkring install` ([Install SparkRing](install.md));
it also writes each rank's runtime-binding file and verifies the image before
any model downtime.

[All commands and flags](commands.md) · [Published Compose files](compose-files.md) ·
[Install Builder](compose-builder.md), a web page that writes these deployments for your Sparks

## Supported profiles

| Profile | Image | `start` and `stop` |
|---|---|---|
| `qwen38-flash-next-tp2-sparkcache`, `qwen38-flash-next-qad-tp4-sparkcache` | Shared 2026.09.3 image named in the profile's release | Yes |
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `deepseek-v41-flash-tp4`, `swift15-qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp4` | Installer image, from the installer image lock | No: `render` and `check` only |

Every other profile is rejected. For the nine installer-image profiles:

- Each rank's container matches the one `sparkring install` runs;
  `deployment.json` records the image lock.
- Compose reads the loader seccomp policy from
  `runtime/common/loader-seccomp.json` in each host's checkout.
- Without the per-rank runtime-binding file, the status plugin reports worker
  identity as `binding_not_configured`. Serving is unaffected.
- `start` stops at image admission:
  `Shared toolchain images are admitted by the installer image lock`.
  Instead, pull the image by its registry digest and run each rank's
  `compose.yaml` with `docker compose` on its host, workers before rank 0.

Other Compose routes:

- `sparkring export --share`: an example site and per-rank Compose templates
  from a deployment saved by `sparkring init`, without private inputs
  ([lower-level commands and Compose sharing](install-reference.md#lower-level-commands-and-compose-sharing)).
- [Standalone TP2 Compose file](../../profiles/qwen38-flash-next-tp2/compose/README.md):
  one file per Spark for `qwen38-flash-next-tp2`
  ([measurements](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md#installation-from-the-published-branch-and-the-standalone-compose-recipe)).
- [GLM TP4 Compose creation](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/compose/README.md)
  for `glm53-flash-spark-tp4-dcp1-sparkcache`, through `sparkring deploy`.

## What each input controls

A site cannot change serving settings or select another image.

| Input | Controls |
|---|---|
| Profile configuration | Model, context, KV allocation, batching, MTP, media limits, engine arguments and integrations |
| `render --checkpoint NAME` | Another checkpoint the profile lists, with the settings it needs, as `sparkring install --checkpoint` selects it |
| `render --image NAME` | Another installer image that runs the profile: a name or release tag that [`sparkring images`](commands.md#images) lists, as `sparkring install --image` selects it |
| `render` serving-setting flags | The [serving settings](install-reference.md#serving-settings) that `sparkring install` accepts, such as `--max-concurrency 8`, replacing the profile's values in every rank; `--reasoning-effort` and `--thinking off` set the model's [thinking](install-reference.md#thinking) default in the API rank only; `--api-bind` replaces rank 0's listen address only |
| Image release or installer image lock | Registry digest, image ID, platform and image verification |
| Site file | SSH targets, rank addresses, bootstrap interface, HCA order, GID index, and model, cache, repository and deployment paths; TP4 adds the prepared fabric reference |
| Generated container specification | Effective arguments, environment, entrypoint, mounts, GPU/RDMA access, limits and health checks |

The site schema is `sparkring-compose-site/v1`, with only `schema`, `name`,
`master` and `ranks`. `name` is lowercase, at most 40 characters; containers
are named `sr-<name>-r<rank>`. `master` must be rank 0's `host_ip`.

## Prepare the hosts

Compose does not download models, pull images, configure networking or stop
other workloads. Before you render:

- The profile's two or four Linux ARM64 Sparks, one GPU each, with working
  fabric and matching HCA/GID selection. TP4 also needs the prepared mesh
  fabric and its site reference.
- Each fabric address's RoCE v2 GID at the rank's `gid` index when the
  container starts; NCCL uses that one index for every HCA.
  [`spark_roce_gid.py`](../../integrations/vllm/spark_roce_gid.py) prints the
  index on a host. `sparkring install` restores index 3 before it starts a
  model; a host started only from these files does not.
- The model downloaded with every shard checked, the image pulled and the host
  prerequisites met, as the profile guide describes:
  [Qwen TP2](../../profiles/qwen38-flash-next-tp2/README.md) or
  [Qwen QAD TP4](../../profiles/qwen38-flash-next-qad-tp4/README.md).
  Generated files use `pull_policy: never`.
- The same SparkRing checkout on the controller and every host, with Python
  3.12, PyYAML and the Docker Compose plugin. The controller, Linux or Windows,
  needs SSH to each host; hosts need NVIDIA Container Toolkit, `nvidia-smi` and
  `ip`.
- For TP4, `sudo -n` on each host: the read-only host checks read root-owned
  fabric markers and logs with the RDMA and traffic-control tools from
  [mesh host setup](../GLM53_SPARK_MESH_HOST_SETUP.md#3-complete-shared-bootstrap-and-check-additional-tools).
- Cache and deployment-root directories you create. Model, cache, repository
  and deployment-root directories must not overlap, and deployment roots must
  not contain symlinks. Generated files are mode 0600 in 0700 directories; on
  Windows, keep them in the account's private directory.

## Render and inspect

This example uses `qwen38-flash-next-tp2-sparkcache`, a profile that `start`
supports; for TP4, copy `profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml`
instead. Site examples hold documentation addresses and example SSH aliases.
From the repository root:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-tp2/compose/site.example.yaml .sparkring/qwen.site.yaml
# Edit .sparkring/qwen.site.yaml for the two hosts and their existing directories.
python3 scripts/sparkring.py compose render qwen38-flash-next-tp2-sparkcache \
  --site .sparkring/qwen.site.yaml --output .sparkring/deployments/qwen
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen
```

- `render` works offline and refuses an existing output directory; inside the
  repository, output must be under `.sparkring/`. Use one output directory per
  variant.
- `--image NAME`, `--checkpoint NAME` and the serving-setting flags of
  `sparkring install` (`--max-concurrency`, `--context-length`,
  `--kv-cache-gib`, `--max-images`, `--max-videos`, `--save-cpu`) select
  another installer image, checkpoint and settings.
  `deployment.json` records them, so `check`, `start` and `stop` derive the
  same containers; a deployment rendered without them keeps its ID. The
  profile's default checkpoint, by name or alias, selects the deployment
  without `--checkpoint`. `render` prints a warning for a KV cache above the
  profile's value, as `sparkring install` does.
- `check` runs `docker compose config` locally, with `.env` loading disabled
  and literal `$` signs escaped, and compares every resolved setting with the
  specification: arguments, image, environment, GPU reservation, RDMA devices,
  read-only model mounts and health check.
- `check` and `start` refuse an edited export as a custom configuration. To
  change a deployment, edit the site file and render again.

```text
deployment.json          # Profile, release, site, source and generated-file identities
rank0/container.json     # Effective container specification
rank0/compose.yaml       # One host's Compose project
rank1/container.json
rank1/compose.yaml
```

The deployment ID in `deployment.json` is each container's
`io.sparkring.deployment` label: a digest of the profile ID, site, image
selection and source inventory. `deployment.json` also records every input
file's SHA-256, which each host verifies before it runs SparkRing's host code,
so a rendered deployment is refused once the checkout changes. Changes to
[descriptive profile fields](../development/installer-profiles.md#files-in-profilesid)
only (`DESCRIPTIVE_FIELDS` in `runtime/common/profiles.py`) keep the ID and
Compose files, so a deployment rendered again checks and stops the containers
already running.

## Check and coordinate hosts

```bash
python3 scripts/sparkring.py compose check \
  --deployment .sparkring/deployments/qwen --hosts
python3 scripts/sparkring.py compose start \
  --deployment .sparkring/deployments/qwen
```

`check --hosts` runs read-only checks over SSH: source agreement, directories,
model metadata hashes, local image ID and platform, active HCA ports and GIDs,
bootstrap address, idle GPU, free API and bootstrap ports, and Compose
equivalence on the host. It does not repeat the full shard checksum or test
RDMA between peers.

`start` writes `start-plan.json` and prints its hosts, phases and SHA-256; it
runs nothing. Review the YAML, hosts and plan, then run `start` again with
`--approve SHA256` to execute exactly that plan:

1. Check every host before changing any host.
2. Stage the deployment's private control files.
3. Verify each image in isolated containers without GPU, network or host mounts.
4. Create stopped containers once every image passes.
5. Start the worker ranks, then rank 0, and wait up to 15 minutes for rank 0's health check.
6. Confirm every rank is running.

Each host runs its own Compose project; SparkRing, not Compose `depends_on`,
orders the hosts. Restart policy is `no`. Rank 0 has the API health check;
other ranks are checked for a running process. `sparkring compose` uses each
host's default Docker context, never recreates an existing container or removes
orphans, and keeps SSH credentials and registry login out of the YAML.

## Stop and recover

```bash
python3 scripts/sparkring.py compose stop --deployment .sparkring/deployments/qwen
# Review the stop plan and repeat with its printed --approve SHA-256.
```

- `stop` checks each container's deployment label, image and settings, then
  stops it by container ID. It does not run `compose down`; containers, caches,
  models and networks stay in place.
- `start-receipt.json` and `stop-receipt.json` record each action. A failed
  phase blocks later phases but does not stop ranks already started. Read the
  receipts and rank logs, then repeat the approved command with `--resume`: it
  rechecks completed actions and refuses any action whose outcome is uncertain.
- `start` only creates containers. To restart retained ones, check each one's
  image and deployment label and run `docker start` on workers before rank 0,
  or render a separate deployment.
- Keep the matching checkout to check or stop an exported deployment.

## Local source-image trials

For developers testing a locally built image from the
[Qwen prefill source-image composition](../../runtime/images/compositions/lil-r37-qwen-prefill/README.md),
which describes the build and the fields a registered release must pin. Only
the two SparkCache Qwen profiles accept these options; the installer image lock
selects the image for the others.

```bash
python3 scripts/sparkring.py compose render qwen38-flash-next-tp2-sparkcache \
  --site /path/to/private/site.yaml --output /path/to/tp2-source-trial \
  --local-source-extension lil-r37-qwen-prefill --local-image-id "$IMAGE_ID" \
  --local-kv-cache-gib 33
```

- `IMAGE_ID` is the full local `sha256:` image ID; every rank needs that image
  under the local tag recorded in `deployment.json`.
- KV stays at 24 GiB per rank unless `--local-kv-cache-gib` selects the single
  alternative: 33 on TP2, 40 on TP4. `--local-master-port` sets a separate
  bootstrap port.
- TP2 keeps HC sharding off and enables recurrent-checkpoint coalescing; TP4
  keeps its HC sharding and coalescing.
- SparkCache uses a separate persistent namespace
  (`qwen38-flash-next-qad-tp2-lil-r37-qwen-prefill` on TP2). Use a dedicated
  host cache directory.

## Regenerate the public examples

```bash
python3 scripts/generate_compose_examples.py
python3 scripts/generate_compose_examples.py --check
python3 -m pytest runtime/common/test_compose.py scripts/test_sparkring_compose.py -q
```

The equivalence tests need a real Docker Compose CLI; they start no containers.
A profile's model adapter returns a structured container specification; do not
copy serving defaults into YAML or translate a launcher's shell output.
