# Install a profile

Status: **offline-tested implementation; hardware rehearsal pending**. Use two
or four prepared Linux ARM64 Sparks with GPU Docker support, Compose, Python 3.12,
PyYAML and verified SSH access. The data fabric must already work. This installer
does not flash firmware, install drivers, change networking or replace workloads.

Run from a clean, committed SparkRing checkout. Commands below use
`python3 scripts/sparkring.py`; an installed `sparkring` command is equivalent.

```bash
python3 scripts/sparkring.py init --model glm53 --host spark0 --host spark1
python3 scripts/sparkring.py up
# Review the host list and stages, then apply:
python3 scripts/sparkring.py up --execute
python3 scripts/sparkring.py status --refresh
python3 scripts/sparkring.py down --execute
```

`init` discovers addresses read-only and saves `.sparkring/deployment`. Use
`--model qwen38` for Qwen. Host count selects the default profile; `--profile`
selects an explicit cache alternative. GLM accepts `--variant nvfp4-qad`.
For offline initialization, copy [the site example](../../profiles/install-site.example.json),
fill its hosts/paths, and replace the `--host` arguments with `--site YOUR_FILE`.

Each host's workspace parent must already be writable by the SSH user; the
default is `/srv/sparkring`. Choose a new name/workspace for a different deployment.
An existing checkpoint can be selected with a host's `model` path and
`reuse_verified_model: true`, explicitly declaring that it was independently
verified. Otherwise the installer downloads the pinned revision to a fresh path.
All indexed shards are required; later use checks recorded content hashes.

`up` stages the exact source, verifies image and weights, creates stopped containers,
starts workers before rank 0, waits for readiness and sends one test request.
Rerun it to resume a known failed operation; uncertain SSH outcomes stop for
inspection. `down` stops owned containers and retains weights/cache. Saved
`status` is progress; `--refresh` reads hosts. No live command runs without
`--execute` except explicit discovery and status refresh.

Four-node Qwen requires each host's prepared `fabric` reference from its existing
Compose site. Four-node GLM uses the existing managed preparation, native tests
and lifecycle; it requires a Linux controller reachable at `controller_address`
and noninteractive sudo. Existing managed installations that cannot be verified
are refused, not adopted. Test the pair first and obtain separate approval for TP4.

## Share or inspect Compose

For a pasteable TP2 file, download one of these and save it as `compose.yaml`:

- [GLM TP2](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/compose/standalone.yaml)
- [Qwen TP2](../../profiles/qwen38-flash-next-tp2/compose/standalone.yaml)
- [Qwen TP2 with SparkCache](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/standalone.yaml)

Validate a downloaded recipe locally before trying hardware:

```bash
python3 scripts/sparkring.py validate-compose compose.yaml
# Or check all maintained recipes and save a new report:
python3 scripts/sparkring.py validate-compose --all --output .sparkring/compose-validation.json
```

Requires the Docker Compose CLI, but no daemon, images or GPUs. It checks profile
identity/settings, resolves both ranks with example inputs, tests missing-variable
guards and simulates rank registration. It rejects external includes/env files
before invoking Compose. A pass does not establish GPU/RDMA or inference behavior.

The same file goes on both hosts. Set the five local values listed at its top,
then run `docker compose --profile rank1 up -d` on the worker and
`docker compose --profile rank0 up -d` on the head. The pinned image pulls
automatically. No SparkRing checkout or installer is needed on this standalone
path; prepared networking and downloaded weights are still required. It keeps
the image's own startup verifier but does not run the installer's host/admission
workflow. Do not use these standalone files to operate installer-owned containers.

Generate another copy directly from a profile, without initializing a site:

```bash
python3 scripts/sparkring.py export --format compose \
  --profile glm53-flash-spark-tp2-dcp1-sparkcache --output compose.yaml
```

ZIP exports remain available for complete configuration bundles:

```bash
python3 scripts/sparkring.py export --share --output profile-template.zip
python3 scripts/sparkring.py export --output private-deployment.zip
```

The shareable ZIP contains the pinned profile, a **new example site** and per-rank
Compose templates for Qwen and GLM TP2. It excludes private addresses, paths,
receipts and source bundles. Recipients fill the site and run `init` to regenerate.
The private ZIP contains your actual configuration. Managed GLM TP4 produces
its Compose files during staging and retains managed start/stop coordination.

Compose is the container format, not a multi-host scheduler. Each host runs its
own rank. Swapping a model requires a matching profile, weights and image; changing
only `image:` can invalidate the runtime contract. Edit the site/profile and
reinitialize to change installer-managed deployments. Standalone files are ordinary
Compose recipes; changes to their serving settings need their own testing.
