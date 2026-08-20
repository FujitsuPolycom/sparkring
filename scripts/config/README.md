# Site configuration

Everything the public SparkRing tooling knows about *your* cluster lives in one
file: `scripts/config/site.yaml`. It describes a single four-node ring — the
ranks, the four 200GbE cables that join them, the control-channel rendezvous
addresses, the pinned runtime and artifacts, and the paths each rank must have.

`site.example.yaml` in this directory is a complete, schema-valid template with
every field commented. Its topology and serving shape match the supported
public-functional matrix, but it contains **no runnable site or runtime
identity**: every address comes from an IANA-reserved documentation or
benchmarking range, while local artifact hashes and image identity are obvious
placeholders. The public model repository and revision are real matrix pins.

## Quick start

**Safety: OFFLINE** for the copy, edit, and validator. The preflight command is
**READ-ONLY REMOTE**: it opens one guarded SSH session per configured rank and
does not mutate them.

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml

# 1. Does the file describe a coherent cluster at all?  (offline, instant)
python scripts/sparkring_site.py scripts/config/site.yaml

# 2. Can the controller reach every rank and rank 0 reach its own management
#    identity plus every follower over the bootstrap network? (read-only)
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml --scope bootstrap

# Optional: repair missing bootstrap public-key/host-key relationships
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml --scope bootstrap --fix

# 3. Are the exact direct-ring bulk-image hops ready? (read-only)
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml --scope image-fanout

# Optional: repair only missing direct-hop SSH trust/authorization
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml --scope image-fanout --fix

# 4. Does the live ring/RDMA fabric match it before model work? (read-only)
python scripts/preflight.py \
  --site scripts/config/site.yaml --scope fabric
```

Step 1 needs nothing but Python and PyYAML. The remaining steps need key-based
management SSH. `bootstrap` checks rank-0 orchestration paths.
`image-fanout` checks the exact three direct-ring payload hops used for the
image archive: rank0 to both neighbors, then the lower-ID neighbor to the
opposite rank. Archive bytes never use management addresses. The verifier is
read-only unless `--fix` is explicitly supplied; preflight never mutates
anything on the ranks. Use `--scope all-adjacent` only for the optional audit
of every direct-ring direction.

The early preflight `fabric` scope intentionally excludes final image,
artifact, disk-path, and service-port gates. The bootstrap runs it before
downloading the model or building an image, then runs the default full scope
against the generated resolved site.

Keep your own `site.yaml` out of version control — it describes your real
addressing. The repository already ignores the canonical local path
`scripts/config/site.yaml`; only `site.example.yaml` belongs in the repo.

## Default EXL3 launcher

The public default consumes this shared site schema, then generates its exact
ignored launch contract during bootstrap:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

Continue with [`docs/QUICKSTART.md`](../../docs/QUICKSTART.md). Do not hand-edit
an NF3 example launch file into an EXL3 profile.

## NF3 alternative launcher

Copy the launch profile beside the site configuration:

```bash
cp scripts/config/launch.example.json scripts/config/launch.json
$EDITOR scripts/config/launch.json

# OFFLINE: prints four exact remote docker commands; no SSH connection.
python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json plan
```

The profile owns the two host-local model paths and the NF3 vLLM contract.
This generic launcher remains the accepted NF3 executable path. See
[`docs/NF3_QUICKSTART.md`](../../docs/NF3_QUICKSTART.md).

`launch.example.json` is the conservative `fp8` source profile. Do not edit
individual KV flags to obtain the larger-capacity layout. Use:

```bash
python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8 \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

The bootstrap writes an ignored generated launch file containing the complete
NVFP4-latent/FP8-RoPE contract and selects its matching derived image.
Topology, ranks, image digest, model identity, TP/DCP sizing, ports and cache
budgets come from `site.yaml`; the launcher refuses model identity drift from
`recipes/glm52-nf3-hybrid.json`. It derives per-rank neighbors, RDMA devices,
GID indices, management interfaces, `--node-rank`, and `--headless`
automatically. `start`, `stop`, and `verify-rollback` remain dry-run unless
`--execute` is supplied. A partial start triggers best-effort removal only for
containers whose `docker run` succeeded; removal additionally requires the
launcher-managed label, so a same-named foreign container is never deleted.

Environment values are strings except for an explicit JSON `null`, which the
launcher renders as Docker `--env NAME` without `=value`. The checked-in GLM
profile uses this to remove the base image's incompatible
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`; changing it to an empty string does not
have the same semantics and is rejected.

The current public image contains an executable capability gate. With the
published recovered overlay, the gate has passed natively on DGX Spark for
`B12X_MLA_SPARSE` and `nvfp4_ds_mla`. It still fails closed if either capability
is absent or the installed sources drift. Bypassing that gate is not a
supported workaround.

## Generic runtime contributor profiles

The generic runtime's hardware-free authoring files are:

- `native-profile.template.json`: minimal native-profile template. Its obvious
  placeholders intentionally make `validate` exit `1` with
  `template/unresolved` until the required image and identity pins are filled.
- `contributor-example.json`: filled, sanitized structural example used by the
  conformance tests. Its documentation-only identities are not live evidence.
- `generic.example.json`: older feature-rich placeholder example retained for
  backward-compatible plan examples; it is also unresolved, not deployable.
- `fixtures/snapshot-*.json`: deterministic semantic projections for the
  native generic, EXL3 bridge, and NF3 bridge plans.

Run the focused offline workflow with:

```bash
python scripts/sparkring_generic_launcher.py \
  --profile scripts/config/contributor-example.json validate
python -m pytest scripts/test_runtime_conformance.py -q
```

See [`docs/GENERIC_RUNTIME.md`](../../docs/GENERIC_RUNTIME.md) for resolved
plan validation, explanation, semantic diff, and exit-code details.

## Requirements

* Python 3.10+
* PyYAML (`python -m pip install -r requirements-dev.txt`) — the only
  third-party dependency for these two tools.
  If it is missing, both tools say so plainly instead of raising an ImportError.
* For preflight only: an `ssh` client on the machine you run it from, and
  key-based (`BatchMode`) auth to each rank. Preflight will never prompt for a
  password.
* `verify_ssh_mesh.py --fix` also needs ordinary unprivileged write access to
  `~/.ssh` on each rank. It never requests sudo, handles a password, or copies
  a private key.

## What the validator enforces

`sparkring_site.py` is fail-closed. It stops at the first problem and names the
exact field (`ranks[2].ring_ports[0].address: ...`). Beyond type and range
checking it enforces the structure of the ring:

* exactly four ranks with ids 0–3, unique ssh targets
* exactly four edges, each with its own distinct, non-overlapping `/24`
* every edge is claimed by exactly two ring ports, one per declared endpoint
* the edge set forms **one closed cycle through all four ranks** — two disjoint
  pairs, a star, or a triangle with a spare cable are all rejected
* both addresses on an edge live inside that edge's subnet, are usable hosts
  (not the network or broadcast address), and differ from each other
* no address is used twice anywhere, and no management address sits inside a
  ring subnet
* each rank's two ring ports use distinct interfaces, distinct RDMA
  device/port pairs, and neither is the management interface
* each rank's two control-channel peers are exactly its two ring neighbours,
  and a peer address that belongs to a *different* rank's management interface
  is rejected — that is the shape a stale peer table takes
* every sha256 is exactly 64 lowercase hex; the image digest is
  `sha256:<64 hex>`; the model revision is an immutable 40-hex commit
* MTU, link speed, TP/DCP, context length, KV bytes and port numbers are in
  sane ranges, and TP must be divisible by DCP

Interface names, device names, remote paths and image references are also
restricted to a safe character set. Preflight interpolates those values into
remote shell commands, so this is what keeps an edited config from becoming
command injection.

## Deriving the non-obvious values

Run these on each node. The template repeats them inline next to the fields.

| Value | How to find it |
| --- | --- |
| interface names | `ip -br link`, or `ls /sys/class/net` |
| netdev ↔ RDMA device | `ls -l /sys/class/infiniband/*/device/net/`, `rdma link show`, or `ibdev2netdev` |
| `rdma_port` | `ls /sys/class/infiniband/<device>/ports/` — usually just `1` |
| `roce_gid_index` | dump `gids/<i>` and `gid_attrs/types/<i>` for i in 0..15; pick the index whose type is `RoCE v2` and whose GID is the IPv4-mapped form of that port's address |
| MTU | `cat /sys/class/net/<if>/mtu` — must agree on all eight ring ports and end to end |
| link speed | `cat /sys/class/net/<if>/speed` (200000 for 200GbE) |
| management interface | `ip -o -4 addr \| grep <management-ip>` — the interface that *really* holds it |
| per-edge subnets | any four unused, mutually distinct `/24`s, one per physical cable; they never need to route |
| image digest | `docker image inspect <ref> --format '{{.Id}}'`, or a registry digest from `--format '{{json .RepoDigests}}'` |
| artifact hashes | `sha256sum <path>` on a node you trust |

### IPv4-mapped GIDs

The RoCEv2 GID for an IPv4 address is the address in the low 32 bits of an
IPv4-mapped IPv6 address:

```
192.0.2.10  ->  0000:0000:0000:0000:0000:ffff:c000:020a
                                            ^^^^ ^^^^  c0.00.02.0a
```

Preflight recomputes this from your configured ring address and compares it
byte-for-byte against `gids/<roce_gid_index>`, so a GID table that shifted
after a driver upgrade — or an index that is actually RoCEv1 or link-local —
fails immediately instead of degrading traffic silently.

## Preflight check ids

Every check has a stable id, safe to alert on. `python scripts/preflight.py
--list-checks` prints the same table.

| Check id | Meaning |
| --- | --- |
| `SSH.REACHABLE` | rank answers over ssh with BatchMode (key auth, no prompt) |
| `MGMT.ADDRESS_PRESENT` | the configured management address is held by some interface |
| `MGMT.INTERFACE_MATCH` | it is held by exactly the configured management interface |
| `RING.LINK_UP` | ring interface exists and its operstate is up |
| `RING.MTU` | ring interface MTU equals `topology.mtu` |
| `RING.LINK_SPEED` | ring interface negotiated `topology.link_speed_mbps` |
| `RING.ADDRESS` | ring interface holds exactly the configured address, nothing else |
| `RING.RDMA_PORT_ACTIVE` | backing RDMA device exists and its port state is ACTIVE |
| `RING.RDMA_LINK_LAYER` | that RDMA port's link layer is Ethernet (RoCE, not InfiniBand) |
| `RING.ROCE_GID` | GID at the configured index is the RoCEv2 IPv4-mapped ring address |
| `RING.JUMBO_PING` | don't-fragment ping fills the MTU across the edge to the far end |
| `PEER.CONTROL_CHANNEL` | control-channel rendezvous peer answers ICMP from this rank |
| `ARTIFACT.PRESENT` | pinned artifact exists on this rank |
| `ARTIFACT.SHA256` | pinned artifact hashes to the configured sha256 |
| `ARTIFACT.EXECUTABLE` | pinned artifact marked executable carries the +x bit |
| `PORT.FREE` | a port required to be free has no listener |
| `DISK.PATH_PRESENT` | cache/JIT directory exists on this rank |
| `DISK.FREE` | cache/JIT directory has at least the configured free space |
| `IMAGE.PRESENT` | the configured container image exists in the local image store |
| `IMAGE.DIGEST` | that image matches the configured digest (image ID or RepoDigest) |

Preflight exits non-zero if any check fails, prints a per-rank table, and
writes a machine-readable evidence document (schema
`sparkring-preflight/v1`) to `--json` or to `paths.evidence_dir`.

### Read-only, by construction

Preflight opens one ssh session per rank and runs a single probe script built
from `sysfs` reads, `ip`/`ss`/`df` queries, `sha256sum`, `ping`, and
`docker image inspect`. It never starts, stops, pulls or removes a container,
never touches interfaces or routes, and writes nothing on the nodes. Every
command is passed through an in-process guard before it can reach ssh, so a
future edit that introduces a mutating command fails locally rather than
quietly on a serving cluster.

Inspect exactly what would run, without contacting anything:

```bash
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

### A note on `required_free_ports`

`preflight.required_free_ports` is for pre-launch validation. If you are
checking a cluster that is *already serving*, set it to `[]` — the API and
rendezvous ports are legitimately bound in that case and `PORT.FREE` would
fail by design.

## Placeholder warnings

Both tools warn (without failing) when the config still looks like the shipped
example: addresses from reserved documentation ranges, or hashes made of a
single repeated hex digit. Pass `--strict-placeholders` to turn those warnings
into a non-zero exit — useful in CI so a half-filled `site.yaml` cannot ship.

## Secrets

There are none in this file, and none belong in it. It carries addresses,
device names, paths and hashes. Authentication is your ssh key material and
your container registry credentials, both of which stay outside the repo.

## Scope

These are scaffolding tools: a schema, a fail-closed validator, and a
read-only checker. They tell you whether your cluster matches what you
declared. They do not launch, tune, or benchmark anything, and passing
preflight is a precondition for a healthy run, not a guarantee of one.

## Acceptance-gate configuration

`gate.example.json` is the companion template for
`scripts/acceptance_gate.py` and retains the accepted NF3 matrix defaults.
`gate.exl3.example.json` selects the EXL3 candidate with LMCache using
512-token cache chunks (CS512) and
adds its correctness and cache-boundary extensions. Follow
[`docs/EXL3_ACCEPTANCE_RUNBOOK.md`](../../docs/EXL3_ACCEPTANCE_RUNBOOK.md) for
that profile. Copy the relevant template to a Git-ignored local path and
replace every angle-bracket command:

```bash
cp scripts/config/gate.example.json scripts/config/gate.json
$EDITOR scripts/config/gate.json

# OFFLINE: dry-run is the default and executes no command or connection.
python scripts/acceptance_gate.py \
  --site scripts/config/site.yaml \
  --gate-config scripts/config/gate.json
```

The site example now carries the supported matrix values and pinned public
model repository/revision, so the gate should not report serving-shape drift
after you fill it. Dry-run also validates the current runtime lock and your
launcher contract. Treat every reported blocker as real; do not weaken the
checks to make the example pass. A successful dry-run plan is still not an
acceptance result.

The gate's `runtime.model_identity` paths are not labels copied from the site
file. During execution it independently reads the repository and immutable
revision sidecars and hashes the deployed `config.json` on **every** rank. The
accepted NF3 entrypoint expects:

```text
/models/your-model/config.json
/run/sparkring/model-identity/repository
/run/sparkring/model-identity/revision
```

The entrypoint creates the two sidecars only after hashing `config.json`,
passing the public capability gate, and verifying the runtime manifest.
Missing files, a hash mismatch, or one rank carrying a different revision is a
functional failure.

`--execute` is **STOPS SERVING**: it runs your configured start and stop
commands. It requires an explicit confirmation token and must never target a
production-serving cluster.

The EXL3 correctness case template is
`exl3-correctness.example.json`. Null expected hashes are intentional: a first
live run records a candidate and exits 4. Acceptance requires review and an
exact rerun against populated token-ID hashes.

## Tests

```bash
python -m pytest scripts/test_sparkring_site.py scripts/test_preflight.py \
  scripts/test_sparkring_launcher.py -q
```

GPU-free and offline: the malformed-config table, the ring-topology validator,
the GID vectors and the whole preflight pipeline run against synthetic probe
transcripts and a fake runner. No cluster or network is involved.
