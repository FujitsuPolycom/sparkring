# Launch SparkRing images with lil

Start and stop a prepared SparkRing model across four hosts with the `lil image`
commands. SparkRing turns model and network settings into one JSON launch file;
lil checks that file and runs its per-rank container commands. vLLM and B12X run
from the image, so hosts do not need those source checkouts.

Status: **implemented** in the [FujitsuPolycom/lil fork](https://github.com/FujitsuPolycom/lil/pull/1).
The MTP3 profile passed startup and cached-context restore after a restart on
four NVIDIA DGX Sparks. See the [bounded test results](HARDWARE_VALIDATION.md).
Upstream lil does not provide these commands.

## Before starting

The default is GLM-5.3 Flash NVFP4-Spark with built-in MTP3 speculation, TP4/DCP4,
and optional SparkCache. **Install and verify the managed mesh separately** using
the [MTP3 mesh quickstart](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md).
This adapter does not install the mesh or take over its systemd model units.
Use it for supervised trials on a prepared fabric.
Its preflight compares the proposed rank, image, peer addresses, devices and
GID settings to the installed mesh and requires the managed model to be stopped.
Keep that model unit stopped during the trial. The mesh supervisor does not
automatically monitor the separately named LIL trial containers.

For SparkRing-owned preparation and managed operation, see the
[standalone deployment suite](../../docs/DEPLOYMENT_SUITE.md). It is locally
implemented and offline-tested; it does not establish unattended fresh-host support.

Use Linux or WSL with Python 3.11+, Bash, and Go 1.26. Build the tested lil fork:

```bash
git clone --branch codex/image-runtime-adapter --single-branch https://github.com/FujitsuPolycom/lil.git
cd lil
git checkout 329cde801b847294005cb16765692032a6cdf206
go build -o lil ./cmd/lil
install -D lil "$HOME/.local/bin/lil"
```

Put `$HOME/.local/bin` on `PATH`, then return to the SparkRing checkout.
Copy `integrations/lil/site-mtp3.example.json` to `site.json` and
`integrations/lil/fabric.example.json` to `fabric.json`. Enter your hosts,
directories, peer addresses, and RDMA devices. The examples contain placeholders,
not a usable cable map. Preserve the device/peer ordering from your working mesh;
it can differ by rank. The MTP3 managed mesh fixes all NCCL and SIRCL GID indices
at `3`; export rejects other values. The DFlash descriptor permits explicit
indices, with optional NCCL and secondary-rail values defaulting to `3`.

In `site.json`, `settings` overrides context, sequences, batching, KV bytes, and
port. `"cache": {"enabled": false}` omits the SparkCache connector and its
separate storage mount; vLLM's in-process prefix cache remains enabled. To use SparkCache,
set `enabled: true`, a namespace, and `access_mode` to `read-write`, `restore-only`,
or `store-only`. Mount source directories must exist on each host and not overlap.
Model files must come from the pinned snapshot. See [file distribution](DISTRIBUTION.md).

## Create and inspect the launch file

```bash
python integrations/lil/export.py --site site.json --fabric fabric.json --id glm53-local > bundle.json
lil image validate bundle.json
lil image render bundle.json
```

These commands only run locally. The exporter uses SparkRing's canonical launcher
to preserve its image entrypoint, model arguments, and mounts. It adds a separate
persistent cache mount when enabled. Review `bundle.json` before running it:
the file contains trusted commands and host mounts, not sandboxed configuration.

## Run on the configured hosts

```bash
lil image check bundle.json
lil image start bundle.json
lil image status bundle.json
lil image logs bundle.json
lil image stop bundle.json
```

Preflight checks the pinned image, target metadata, native libraries, directories,
and managed-mesh readiness. It does **not** hash every target weight shard.
`start` checks all ranks, launches workers, then launches rank zero. It returns
after Docker accepts the containers; use `status` and a model request to confirm
serving readiness. `logs` prints the last 100 lines per rank.

`stop` retains containers, logs, models, and caches. To start again with the same
names, inspect and remove or rename the stopped containers first. Partial launch
failures leave containers available for inspection and stop. Automatic recovery
after a failed launch or host restart is not implemented.

Keep the original `bundle.json` for the deployment. Ownership labels bind its
contents, including checks; regenerating it after changing settings or exporter
code can prevent inspection or stop. For containers created without content-digest
labels, explicitly allow legacy ownership:

```bash
lil image status --allow-legacy-owner original-bundle.json
lil image logs --allow-legacy-owner original-bundle.json
lil image stop --allow-legacy-owner original-bundle.json
```

This accepts a missing digest, never a conflicting one. Status, logs, and stop
continue on reachable, verified ranks when another rank fails. An error therefore
does not mean nothing happened; inspect the reported results before retrying.

The optional external-draft profile uses the BF16 DFlash2 checkpoint with seven
speculative tokens. Its stable ID is `glm53-flash-tp4-dflash7`; `dflash7` denotes
the configured proposal depth. Select it with
`--descriptor integrations/lil/glm53.json` and `site.example.json`.
The default MTP3 descriptor is `glm53-mtp3.json`.

## Planned additions

- Finish hardware testing and integration of the SparkRing-owned setup command.
- Connect LIL to one managed model-lifecycle owner; do not independently control
  the same containers through both LIL and systemd.
- An operator command for direct rank-to-rank distribution, plus large-file and
  interrupted-transfer tests.
- Broader serving and failure-recovery tests before unattended use.

## Evidence and ownership

Run offline tests with `python -m pytest integrations/lil -q`.
[Offline checks](VALIDATION.md) and [hardware results](HARDWARE_VALIDATION.md)
describe their coverage. The hardware result covers a 12,288-token restore,
not every context size or concurrency. `plan.py` validates the descriptor and
site inputs and prints a non-executable summary; `export.py` creates the launch bundle.
The summary's `research-only` status applies to unresolved deployment intent,
not to the qualification of its image. It has no resolved fabric or host checks;
qualification remains scoped to the separate hardware record.

SparkRing owns profiles, networking, caching, and support; the lil fork extension
owns generic container operations. See [interface ownership and pins](OWNERSHIP.md).
Local Inference Lab supplies the vLLM/B12X performance work and target quantization;
the profile pin files identify all component and draft-checkpoint sources.
