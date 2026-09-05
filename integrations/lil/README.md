# Launch SparkRing images with lil

Status: implemented on a local lil branch, with a bounded DGX4 MTP3 startup and
persistent-recall trial. See [hardware evidence](HARDWARE_VALIDATION.md).
Fresh-host mesh installation and unattended lifecycle remain untested.
These commands require the companion lil `codex/image-runtime-adapter` branch;
they are not available in upstream lil.

SparkRing supplies model settings and the Docker arguments. lil starts ranks,
checks their ownership, shows status and logs, and stops them. The image contains
vLLM, B12X, SIRCL, and SparkCache; hosts do not need those source checkouts.

## Prepare locally

Use Linux or WSL with Python 3.11+, Bash, and a lil binary built from the companion
branch (`go build -o lil ./cmd/lil`, Go 1.26). Copy `site-mtp3.example.json` and
`fabric.example.json` to private files and enter your hosts, directories, peer
addresses, and RDMA devices. Example addresses are documentation placeholders;
they do not describe a wired cluster. Fabric routing must be verified on hardware.
Preserve each rank's device and peer ordering from a working launch. The two
connection slots need not use the same physical port numbers on every rank.

`site.settings` overrides context, sequences, batching, KV bytes, and port.
The default profile is the published NVFP4-Spark native-MTP3 mesh at TP4/DCP4,
pinned by `runtime/glm53-spark-mtp3-mesh/public-image.json`. It requires the
managed mesh fabric to be installed and healthy. This adapter does not install
or transfer ownership of that host service. DFlash7 remains an explicit alternative
in `glm53.json` with `site.example.json`; it is not the default trial profile.
Set `cache` to
`{"enabled": false}` for vLLM caching alone, or select `read-write`, `restore-only`,
or `store-only` with a namespace. Target, draft, JIT, and cache directories must
not overlap. Directory paths are interpreted independently on each rank.

From the SparkRing checkout:

```bash
python integrations/lil/export.py --site site.json --fabric fabric.json --id glm53-local > bundle.json
lil image validate bundle.json
lil image render bundle.json
```

These commands do not contact hosts. Export calls the canonical operator launcher
in its execution-disabled rendering mode, preserving its image entrypoint and
model arguments. It adds a separate persistent cache mount when enabled. The
bundle contains image, checkpoint, directory, and native-library preflight checks.
Declared hashes are not proof that remote files exist or match.

## Prepare files and run when hardware is available

Make model files and the pinned image available on every node, and create the
mount source directories shown in the bundle. `distribute.py` can stage model
files or an image archive once on the controller and copy them to explicit SSH
destinations. Its manifest format is:

```json
{"schema":"sparkring-artifacts/v1","artifacts":[{"path":"target/config.json","source":"/absolute/local/config.json","sha256":"REPLACE_WITH_FILE_SHA256"}],"destinations":[{"host":"spark0.example.invalid","root":"/srv/models"}]}
```

List every file and destination. HTTPS sources are also accepted. Preview with
`python integrations/lil/distribute.py artifacts.json --cache /local/downloads`.
Add `--execute` to transfer. Verified downloads and matching destinations are
reused; different destination contents are refused. Transfers run from the
controller through SCP. Select reachable fabric endpoints to use that network;
`fanout.py:copy_edge` also supports explicit rank-to-rank push or pull routes with
source and destination checksum verification. Pull is useful when SSH access is
authorized in only one direction. The caller supplies the authenticated route;
the helper does not provision keys or discover connectivity. Image archives need
`docker load --input ARCHIVE` on each rank before lil's image check succeeds.

The following commands contact the configured hosts:

```bash
lil image check bundle.json
lil image start bundle.json
lil image status bundle.json
lil image logs bundle.json
lil image stop bundle.json
```

`start` performs all preflight checks, starts workers before the head, and returns
after Docker accepts the launches. This is not an API-readiness guarantee; use
`status` to inspect the image health state. `logs` prints the last 100 lines per
rank. `stop` retains containers, logs, models, and caches. Existing container names
are refused on start; inspect and remove stopped containers explicitly before
reusing names. A partial launch failure reports the failed rank and leaves started
containers available for inspection and coordinated stop.

Bundles are trusted operator programs containing commands and host mounts. Review
them before execution. lil validates their shape and checks ownership labels; it
does not sandbox them or interpret SparkRing's model-specific checks.

## Validation and maintenance

```bash
python -m pytest integrations/lil -q
```

Tests exercise canonical launch rendering, cache toggles, graph sizes, identity
drift, invalid mounts/ranks, artifact checksums, resume, and fake SSH failures.
The companion Go tests exercise lifecycle order, ownership, and failure reporting.
The hardware record covers a 12,288-token persistent restore on the named
image/topology. It does not establish every profile or context size.
The separate `plan.py` command remains a non-executable summary.

Runtime and model pins are read from canonical files named in `glm53.json`; a
normalized UTF-8 SHA-256 detects changes requiring descriptor review. Source
baseline: SparkRing `f78d3b1b06b1bd57c2d660bbd3838f91db244a09`, lil
`11df08a793596b0a5b09e72e90d9a1ece51c9306`. See [ownership](OWNERSHIP.md).
Local Inference Lab supplies the vLLM/B12X performance work and target quantization;
the pin files identify the component and draft-checkpoint sources.
