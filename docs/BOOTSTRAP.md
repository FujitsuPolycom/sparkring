# Bootstrap a blank SparkRing cluster

Status: implemented for four- and six-Spark direct rings.

This procedure starts with one blank DGX Spark whose management IPv4 address,
username, and password are known. That first Spark becomes rank 0 and the
normal Ring Doctor controller. Passwords are used only by the interactive
OpenSSH `ssh-copy-id` command; SparkRing never reads or stores them.

## 1. Connect to the head Spark

From a laptop on the management network:

```bash
ssh <username>@<rank0-management-ip>
```

## 2. Download and inspect the installer

```bash
curl -fLO \
  https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/bootstrap.sh
less bootstrap.sh
bash bootstrap.sh
```

The installer checks for Git, Python, OpenSSH, `ssh-copy-id`, and PyYAML. If
PyYAML is absent, it asks before installing Ubuntu's `python3-yaml` package.
It installs a managed checkout under `~/.local/share/sparkring` and the command
`~/.local/bin/sparkring`. It does not contact another Spark or configure a
network.

If `~/.local/bin` is not already in `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## 3. Check the blank Spark

Before adding other nodes, run the read-only single-host check:

```bash
sparkring host check
```

It verifies DGX release metadata, GPU driver visibility, Docker, NVIDIA
Container Toolkit, ConnectX-7 PCI/RDMA inventory, failed systemd units, root
free space, and reports the `nvidia-dgx-telemetry.service` state. To make
enabled telemetry a failing policy check:

```bash
sparkring host check --require-telemetry-disabled
```

The default check reports telemetry without changing or condemning the user's
first-boot consent choice.

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

Require:

- controller `rank0`;
- one valid four- or six-node cycle;
- canonical fabric preflight `PASS`;
- full reachability matrix `PASS`;
- management repair guard `READY`; and
- no failed or unknown diagnostic checks.

The command prints a repair plan but changes nothing without `--apply`.

## 7. Apply fabric routing only after review

```bash
sparkring doctor --verify --apply
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
safety checks are described in [SparkRing prerequisites](PREREQUISITES.md#management-safety-during-repair).

## Worker-controller recovery

Prepare and test worker recovery while rank 0 is healthy. Then, if rank 0
cannot run Doctor, execute from an enrolled worker:

```bash
sparkring doctor --allow-worker-controller --verify
```

The flag never permits an arbitrary laptop or unknown host. The local machine
must match one configured worker rank. Worker recovery also requires that
worker to have verified key access to every rank; automated recovery-key
preparation is tracked separately from basic head-node bootstrap.
