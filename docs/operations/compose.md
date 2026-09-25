# Compose deployments from SparkRing profiles

For one saved deployment operated through `init`, `up`, `status` and `down`, use
the [profile installer](install.md). It also generates GLM TP2 Compose files
from the TP2 adapter and exports portable templates with `export --share`.
That installer path is offline-tested; its hardware rehearsal remains pending.
The lower-level `sparkring compose` coordinator described below retains its
existing supported-profile list and recorded serving scope.

Status: **Development**. The [Qwen QAD TP4 serving record](../../performance/records/qwen38-flash-next/r37-shared-tp4.json)
covers four-rank startup, bounded inference/performance checks and coordinated
stop/restart. The [TP2 smoke record](../../performance/records/qwen38-flash-next/compose-tp2.json)
covers native/cache-enabled startup, shutdown and persistent-cache restore in a
fresh deployment. Offline tests cover configuration equivalence and coordinator failures.
The [QAD TP4 cache record](../../performance/records/qwen38-flash-next/sparkcache-tp4.json)
also covers four-rank disk restore and corrupted-object rejection/recomputation.

The `sparkring compose` coordinator supports:

- `qwen38-flash-next-tp2`
- `qwen38-flash-next-tp2-sparkcache`
- `qwen38-flash-next-qad-tp4` ([Development quickstart](../../profiles/qwen38-flash-next-qad-tp4/README.md))
- `qwen38-flash-next-qad-tp4-sparkcache` ([cache quickstart](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md))

Other profiles are rejected by this coordinator. [GLM TP4 Compose creation](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/compose/README.md)
uses the shared container specification through `sparkring deploy`. Its managed
coordinator retains fabric, source-verification, readiness and recovery gates.
GLM Docker/Compose creation has been checked without starting a model; its
quickstart carries the separate model-serving evidence.

## Configuration ownership

The model adapters return a shared container specification. Docker arguments and
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
selects the same registered shared runtime with persistence enabled. Image capabilities do not automatically
enable GLM-specific features in Qwen.

All four Qwen profiles select the shared native image by immutable registry
digest. The host verifier checks its native receipt, source identities and
pinned payload without requiring a separate R37 parent pull. Published
selections reject an isolated `--local-image-id` override. Explicit R37 trials
remain a separate developer selection below.

### Local source-image trials

The Qwen TP2 and TP4 adapters also support an explicit
[source-image test selection](../../profiles/qwen38-flash-next-qad-tp4/README.md#local-source-image-testing):
`--local-source-extension lil-r37-qwen-prefill --local-image-id IMAGE_ID`.
It uses the registered source descriptor, the image's complete installed inventory
and retained feature/cache receipts. Its verified entrypoint checks that inventory
before serving. The public profile remains selected unless these arguments are
provided. R37 trials select their own hooks, transport and source contracts.
Local KV and bootstrap-port alternatives are recorded in the
deployment manifest.

For an experimental TP2 trial, select either `qwen38-flash-next-tp2` or
`qwen38-flash-next-tp2-sparkcache`. The source route keeps HC sharding off,
enables recurrent-checkpoint coalescing and preserves the pair's existing
feature and transport selection. It does not activate TP4 HC fusion or the
`qwen-prefill` feature. KV allocation stays at 24 GiB per rank unless the explicit
33 GiB local alternative is selected:

```bash
python3 scripts/sparkring.py compose render qwen38-flash-next-tp2-sparkcache \
  --site /path/to/private/site.yaml --output /path/to/tp2-source-trial \
  --local-source-extension lil-r37-qwen-prefill --local-image-id "$IMAGE_ID" \
  --local-kv-cache-gib 33
```

Set `IMAGE_ID` to the complete local `sha256:` image ID before rendering. Every
rank must have that image under the local tag recorded in `deployment.json`.
The [combined vLLM/SGLang image](../../runtime/deepseek-v41-sglang/combined-image/README.md)
can use this route only when its inherited vLLM inventory passes source admission.
TP2 SparkCache uses the packaged source lease contract and a separate
`qwen38-flash-next-qad-tp2-lil-r37-qwen-prefill` persistent namespace. Use a
dedicated host cache directory for a controlled trial. The route admits a test
configuration; it does not qualify TP2 serving, performance or cache restoration.
TP4 retains its existing HC sharding/coalescing selection and optional 40 GiB KV
alternative. `--local-master-port` remains an explicit local option for either width.

For a registered TP4 source-image release, the adapter resolves the same source
entrypoint and admission from the canonical profile's `image_extension`.
The release must pin the publication, source descriptor and registry digest;
the publication binds the exact image ID. Ordinary render/check/start commands
then need no local flags. See the [promotion fields](../../runtime/images/compositions/lil-r37-qwen-prefill/README.md#promote-a-qualified-image).
Feature settings remain explicit profile configuration; image selection does
not silently increase KV allocation or enable optional features. TP2 source-image
selection remains local and cannot be promoted by changing its public image field.

## Prepare the hosts

Use the profile's two or four Linux ARM64 Sparks with working fabric and matching
HCA/GID selection. TP4 also requires the pinned mesh site, hardware rules and
persistent source markers described in its quickstart. Complete the model download, full shard checksum,
image pull and host prerequisites before generating a deployment:

- [Qwen TP2 quickstart](../../profiles/qwen38-flash-next-tp2/README.md)
- [Qwen TP2 with SparkCache](../../profiles/qwen38-flash-next-tp2/SPARKCACHE.md)
- [Qwen QAD TP4](../../profiles/qwen38-flash-next-qad-tp4/README.md)
- [Qwen QAD TP4 with SparkCache](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md)

Install the same SparkRing source files on the controller and each host, with
Python 3.12, PyYAML and the Docker Compose plugin. The controller needs SSH access;
the hosts need NVIDIA Container Toolkit, `nvidia-smi` and `ip`. The coordinator
compares source fingerprints before importing the host helper. A matching branch
name alone is insufficient. The controller can run on Windows or Linux.

TP4 preflight uses `sudo -n` to inspect root-owned marker executables and live
attachment logs. It performs read-only network checks and requires the mesh
guide's RDMA/traffic-control inspection tools. It does not install or repair fabric.

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
5. Start all worker ranks, then rank 0; wait up to 15 minutes for rank 0's API health check.
6. Confirm every rank is running.

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
GPU occupancy is checked whenever the owned container is stopped, including
resume, and immediately before each worker/API start. An already running owned
container can pass read-only checks without being mistaken for another workload.

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
start containers. TP2's hardware smoke check restored a 5,696-token prefix on
both ranks after creating fresh containers, with matching requests and correct
answers. QAD TP4's four-host evidence is scoped to its recorded image and bounded
workloads. GLM creation checks and model-serving evidence are scoped separately
in its quickstart. These records do not qualify additional model/media settings
or long-duration workloads.

To add another adapter, return a structured specification from that adapter,
preserve its image admission and site contracts, and test both renderers. Do not
copy serving defaults into YAML or bypass a managed launcher by translating its
shell output.

Docker references: [GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/),
[configuration resolution](https://docs.docker.com/reference/cli/docker/compose/config/),
[interpolation](https://docs.docker.com/reference/compose-file/interpolation/).
