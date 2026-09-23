# Bootstrap a blank SparkRing cluster

This host and network bootstrap supports four- and six-Spark direct rings.
It does not select a serving profile or qualify six-rank inference.
Two-node deployments use [pair networking](pair-network.md). For a first
installation, start at [setup](setup.md) and complete [host preparation](host-preparation.md)
on every rank, including noninteractive sudo for ring repairs, before continuing.

This procedure starts with one blank DGX Spark whose management IPv4 address,
username, and password are known. That first Spark becomes rank 0 and the
normal Ring Doctor controller. Passwords are used only by the interactive
OpenSSH `ssh-copy-id` command; SparkRing never reads or stores them.

## 1. Connect to the head Spark

From a laptop on the management network, replace both uppercase placeholders:

```bash
ssh RANK0_USERNAME@RANK0_MANAGEMENT_IP
```

## 2. Download and inspect the installer

Complete [install and record one checkout](host-preparation.md#3-install-and-record-one-checkout)
on every rank. The installer accepts a branch, tag or full commit; use the same
recorded commit on every host. It installs only the local checkout and CLI.
For an already installed checkout, do not install a second copy. On rank 0:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/.local/share/sparkring"
```

For a supplied test checkout, use its actual directory and define
`sparkring() { python3 scripts/sparkring.py "$@"; }` in that Bash session.

## 3. Check the blank Spark

Before adding other nodes, run the read-only single-host check:

```bash
sparkring host check
```

It verifies DGX release metadata, GPU driver visibility, Docker, NVIDIA
Container Toolkit, ConnectX-7 PCI/RDMA inventory, failed systemd units, root
free space. To make
enabled telemetry a failing policy check:

```bash
sparkring host check --require-telemetry-disabled
```

The default check does not inspect telemetry or change the user's first-boot
consent choice. The optional flag adds that policy check.

## 4. Cable and initialize the ring

Use the standard direct cycle:

- Four Sparks: `0-1-2-3-0`
- Six Sparks: `0-1-2-3-4-5-0`

Rank `N` port `enp1s0f0np0` connects to rank `N+1` port
`enp1s0f1np1`. The last rank connects back to rank 0.

On rank 0:

```bash
sparkring cluster init --size 4
```

The command prompts for `username@IPv4` for rank 0 and every worker. Before
enrollment it prints the complete plan. It idempotently authorizes rank 0's
generated key for self-SSH, then verifies rank 0's management host key. For
each worker it:

1. scans the Ed25519 SSH host key;
2. displays the fingerprint and requires the operator to type `yes`;
3. invokes the system `ssh-copy-id`, which prompts for the password once;
4. proves that non-interactive key access works;
5. inventories hostname, management interface, ConnectX-7 interfaces, and
   RDMA mappings; and
6. writes `~/.config/sparkring/cluster.yaml`.

The default fabric allocation is `198.18.0.0/21`, divided into one `/24`
per direct cable. Override it when that range overlaps the management network:

```bash
sparkring cluster init \
  --size 4 \
  --fabric-supernet 10.77.0.0/21
```

Initialization fails rather than generating an inventory when management
addresses overlap the selected fabric range, expected ConnectX-7 interfaces
are absent, RDMA mappings differ, ranks are duplicated, or SSH enrollment does
not produce key-only access.

## 5. Review and install fabric addresses

First print the exact netplan for every rank:

```bash
sparkring cluster configure
```

Generated netplans contain only the two fabric interfaces. They never name the
management interface or a default route. After review:

```bash
sparkring cluster configure --apply
```

The command backs up any prior SparkRing fabric netplan, runs
`netplan generate`, applies it, and immediately verifies that the configured
management address remains on the same interface. A failed management check
restores the prior netplan and stops before changing another rank.

## 6. Run the read-only diagnosis

```bash
sparkring doctor --verify
```

Before applying a repair, require:

- controller `rank0`;
- one valid four- or six-node cycle;
- canonical fabric preflight `PASS`;
- management repair guard `READY`; and
- no unknown observations or failed host/link/GID prerequisites.

Missing fabric routes, forwarding rules and resulting nonadjacent reachability
failures may be the findings the printed repair plan addresses. Review those
findings; they are not a reason to bypass a failed management or fabric guard.
Require the full reachability matrix and diagnostics to pass after repair.

The command prints a repair plan but changes nothing without `--apply`.

## 7. Apply fabric routing only after review

```bash
sparkring doctor --verify --apply
sparkring doctor --verify
```

Ring Doctor can change only observed fabric routes, IPv4 forwarding, and
fabric-to-fabric `DOCKER-USER` accepts. It checks the active management address
and return route before and after every individual operation and stops at the
first mismatch. Before any repair operation, it checks `sudo -n true` on every
rank whose plan has commands. If any check fails, Ring Doctor identifies the
rank and applies no repair command anywhere in the ring.

## 8. Persist routing and firewall state across boots

Routes, IPv4 forwarding, and `DOCKER-USER` rules applied by Ring Doctor are
runtime state. Do not restore the firewall rules with a cron `@reboot` job.
Cron can run before Docker creates its firewall chains, and Docker can replace
rules installed that early.

Use `sparkring doctor --emit-unit DIR` after reviewing the complete repair plan.
The command writes one fail-closed repair program and systemd service per rank.
Each program contains the complete idempotent route, forwarding, and
fabric-to-fabric firewall plan. The service declares
`After=network-online.target docker.service` and retries after ten seconds when
the recorded management address or Docker firewall chain is not ready.

The command writes files but does not install or enable them. The absolute
program path recorded in `ExecStart` must exist on the corresponding rank
before its service is installed. The installation requirements and management
safety checks are described in [SparkRing prerequisites](fabric-repair.md#management-safety-during-repair).

## Worker-controller recovery

Prepare and test worker recovery while rank 0 is healthy. Install the CLI
on the recovery worker using step 2, and place a private copy of the verified
cluster inventory at `~/.config/sparkring/cluster.yaml` there. Head-node
enrollment alone does not install the CLI or authorize worker-to-peer SSH.
Then, if rank 0 cannot run Doctor, execute from that enrolled worker:

```bash
sparkring doctor --allow-worker-controller --verify
```

The flag never permits an arbitrary laptop or unknown host. The local machine
must match one configured worker rank. Worker recovery also requires that
worker to have verified key access to every rank; automated recovery-key
preparation is tracked separately from basic head-node bootstrap.
