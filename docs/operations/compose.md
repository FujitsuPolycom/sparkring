# Compose deployments from SparkRing profiles

Status: **implemented; Compose serving is unqualified**. Offline tests cover
configuration equivalence and simulated coordinator failures. They do not prove
GPU, RDMA, cache restore or inference behavior under Compose.

Supported profiles:

- `qwen38-flash-next-tp2`
- `qwen38-flash-next-tp2-sparkcache`

Other profiles are rejected. GLM TP4 requires a structured adapter that preserves
its managed fabric, source-verification and recovery contracts before it can use
this coordinator. Its existing quickstarts remain the supported launch paths.

## Configuration ownership

The Qwen adapter returns a shared container specification. Docker arguments and
Compose YAML are rendered from that specification, without parsing shell commands.

| Input | Owns |
| --- | --- |
| Canonical profile | Model identity, context, KV allocation, batching, MTP, media limits, engine arguments and integrations |
| Registered image release | Registry digest, image configuration ID, platform and source-verification contract |
| Private site | SSH targets, rank addresses, interfaces, HCA order, GID index, model/cache paths and deployment locations |
| Container specification | Effective arguments, environment, entrypoint, mounts, GPU/RDMA access, limits and rank health policy |

The [Qwen configuration](../../profiles/qwen38-flash-next-tp2/config.json) owns
serving defaults. A site cannot change them or select an arbitrary image. The
[SparkCache configuration](../../profiles/qwen38-flash-next-tp2/sparkcache.json)
selects its registered extension image. Image capabilities do not automatically
enable GLM-specific features in Qwen.

## Prepare the hosts

Use two Linux ARM64 Sparks with a working direct cable and matching HCA/GID
selection. Complete the selected profile's model download, full shard checksum,
image pull and host prerequisites before generating a deployment:

- [Qwen TP2 quickstart](../../profiles/qwen38-flash-next-tp2/README.md)
- [Qwen TP2 with SparkCache](../../profiles/qwen38-flash-next-tp2/SPARKCACHE.md)

Install the same SparkRing source files on the controller and each host, with
Python 3.12, PyYAML and the Docker Compose plugin. The controller needs SSH access;
the hosts need NVIDIA Container Toolkit, `nvidia-smi` and `ip`. The coordinator
compares source fingerprints before importing the host helper. A matching branch
name alone is insufficient. The controller can run on Windows or Linux.

Create the cache and deployment-root directories yourself. Model, cache,
repository and deployment-root directories must be disjoint. Deployment roots
must not contain symlink components. Keep private files accessible only to the
operator; on Windows, use the account's private directory and appropriate NTFS
permissions. POSIX output directories and files use modes 0700 and 0600.

No command here downloads models, pulls images, installs networking, changes NIC
steering or stops another workload. The image wrapper still verifies the installed
payload before serving.

## Render and inspect

Run from the repository root. Copy and edit the public site example before
rendering. It contains documentation addresses and example SSH aliases.

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-tp2/compose/site.example.yaml .sparkring/qwen.site.yaml
# Edit .sparkring/qwen.site.yaml for the two hosts and their existing directories.
python3 scripts/sparkring.py compose render qwen38-flash-next-tp2 \
  --site .sparkring/qwen.site.yaml --output .sparkring/deployments/qwen
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen
```

To enable SparkCache, render `qwen38-flash-next-tp2-sparkcache` instead. Choose a
separate deployment name and output directory when preparing multiple variants.
Rendering refuses an existing output directory. Exports inside the repository
must live under the ignored `.sparkring/` directory.

Each export contains:

```text
deployment.json          # Profile, release, site, source and generated-file identities
rank0/container.json     # Effective container specification
rank0/compose.yaml       # One host's Compose project
rank1/container.json
rank1/compose.yaml
```

`render` is offline and does not require Docker. `check` runs
`docker compose config --format json` without contacting a daemon. It compares
every resolved setting with the specification, including argument arrays, image,
environment, GPU reservations, RDMA devices, read-only model mounts and health.
It accounts for Compose's output escaping and normalization without accepting
additional settings. `.env` is disabled, and literal dollar signs are escaped.

Edited exports remain ordinary Compose files, but become custom configurations.
SparkRing refuses their canonical check/start association even if someone updates
the recorded file hash. Regenerate from the profile/site inputs for a coordinated
deployment. No generated-file hash grants hardware qualification.

## Check and coordinate hosts

```bash
python3 scripts/sparkring.py compose check \
  --deployment .sparkring/deployments/qwen --hosts
python3 scripts/sparkring.py compose start \
  --deployment .sparkring/deployments/qwen
```

`check --hosts` performs read-only checks: source agreement, required directories,
model metadata hashes, local image identity/platform, active HCA ports and GIDs,
bootstrap address, available GPU and API/bootstrap ports, and host-side Compose
equivalence. It does not repeat the full weight-shard checksum or prove RDMA peer
connectivity. Complete those prerequisites before start.

`start` saves a private `start-plan.json` and prints its hosts, phases and identity
without executing it. Review the exported YAML, target hosts
and plan. Repeat with `--approve` followed by the printed plan SHA-256 to execute
that exact plan. This uses the shared deployment engine's barriers and receipts:

1. Check every host before writing to any host.
2. Stage only this deployment's private control files.
3. Verify each image payload using the existing adapter's isolated verification
   containers, without GPU access, networking or host mounts.
4. Create stopped containers after every image is admitted.
5. Start rank 1, then rank 0; wait up to 15 minutes for rank 0's API health check.
6. Confirm both ranks are running.

Each host has a separate Compose project. Cross-host barriers are implemented by
SparkRing, not by Compose `depends_on`. Restart policy is `no`. Rank 0 has an API
health check; headless rank 1 has health checks disabled and is checked for process
liveness by the coordinator. An API-ready result is not an inference benchmark or
a stability qualification.

The command uses each host's default Docker context. It never recreates an
existing named container, removes an orphan or silently replaces a workload.
SSH credentials and registry login remain outside generated YAML. Arbitrary
environment overrides and secret injection are not supported by this adapter.

## Stop and recover

```bash
python3 scripts/sparkring.py compose stop --deployment .sparkring/deployments/qwen
# Review the stop plan and repeat with its printed --approve SHA-256.
```

Stop verifies both containers' deployment labels, image identities and effective
settings, then stops their inspected immutable container IDs. It leaves containers,
cache directories, models and networks in place. It does not call `compose down`.

Local `start-receipt.json` and `stop-receipt.json` record outcomes. A phase failure
blocks later phases; it does not automatically stop a partially launched group.
Inspect receipts and rank logs before recovery. `--resume` rechecks completed
actions and refuses any mutation with an uncertain outcome. It does not silently
retry a possibly running model. Keep the matching source checkout available for
checking or stopping an exported deployment.

After an intentional stop, either inspect and manage those retained containers
directly or create a distinct deployment. The start command is a guarded creation
workflow, not a general cluster restart service.

## Maintaining the backend

Generate public examples with:

```bash
python3 scripts/generate_compose_examples.py
python3 scripts/generate_compose_examples.py --check
python3 -m pytest runtime/common/test_compose.py scripts/test_sparkring_compose.py -q
```

The Linux CI job requires a real Compose CLI for equivalence tests; it does not
start containers. Serving adoption still requires an isolated two-host launch of
each supported profile, a text response, SparkCache restart/restore checks for the
cache profile, and coordinated stop/recovery checks.

To add another adapter, return a structured specification from that adapter,
preserve its image admission and site contracts, and test both renderers. Do not
copy serving defaults into YAML or bypass a managed launcher by translating its
shell output.

Docker references: [GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/),
[configuration resolution](https://docs.docker.com/reference/cli/docker/compose/config/),
[interpolation](https://docs.docker.com/reference/compose-file/interpolation/).
