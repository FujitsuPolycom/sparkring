# Bootstrap a blank SparkRing cluster

Ring Doctor bootstrap enrolls SSH, assigns fabric addresses and repairs routing
on a four- or six-Spark direct ring, starting from one blank DGX Spark. Use it
for manual ring setup, six-Spark rings or ring diagnosis. The normal path for
pairs and four-Spark rings is `sudo sparkring install`, which does its own
discovery and fabric setup; see [Install SparkRing](install.md). Two-Spark
manual setups use [pair networking](pair-network.md).

It prepares hosts and network only; it does not select or start a model. For a
manual installation, start at [setup](setup.md) and complete
[host preparation](host-preparation.md) on every rank first, including
noninteractive sudo for ring repairs.

You need the first Spark's management IPv4 address, username and password.
That Spark becomes rank 0 and the Ring Doctor controller. Passwords are typed
only into OpenSSH's `ssh-copy-id`; SparkRing never reads or stores them.

## 1. Connect to the head Spark

From a laptop on the management network:

```bash
ssh RANK0_USERNAME@RANK0_MANAGEMENT_IP
```

## 2. Download and inspect the installer

Complete [install and record one checkout](host-preparation.md#3-install-and-record-one-checkout)
on every rank, with the same recorded commit on every host. If the checkout is
already installed, do not install a second copy. On rank 0:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/.local/share/sparkring"
```

For a checkout in another directory, `cd` there and define
`sparkring() { python3 scripts/sparkring.py "$@"; }` in that Bash session.

## 3. Check the blank Spark

```bash
sparkring host check
```

This read-only check covers DGX release metadata, the GPU driver, Docker,
NVIDIA Container Toolkit, ConnectX-7 PCI and RDMA inventory, failed systemd
units and at least 20 GiB free on the root filesystem. To also fail when NVIDIA
telemetry is enabled:

```bash
sparkring host check --require-telemetry-disabled
```

## 4. Cable and initialize the ring

Cable a direct cycle:

- Four Sparks: `0-1-2-3-0`
- Six Sparks: `0-1-2-3-4-5-0`

Rank `N` port `enp1s0f0np0` connects to rank `N+1` port `enp1s0f1np1`; the
last rank connects back to rank 0. Then, on rank 0:

```bash
sparkring cluster init --size 4
```

It prompts for `username@IPv4` for rank 0 and each worker (or pass `--head`
and `--node`), prints the plan and asks to continue. It authorizes rank 0's
own key for self-SSH, then for each worker:

1. scans the Ed25519 host key and asks you to accept its fingerprint;
2. runs `ssh-copy-id`, which asks for the password once, if key login does not already work;
3. confirms key-only SSH works;
4. records hostname, management interface, ConnectX-7 interfaces and RDMA mappings.

It writes `~/.config/sparkring/cluster.yaml`. Fabric addresses come from
`198.18.0.0/21`, one `/24` per cable. If that overlaps the management network,
choose another range:

```bash
sparkring cluster init \
  --size 4 \
  --fabric-supernet 10.77.0.0/21
```

It stops without writing an inventory if management addresses overlap the
fabric range, ConnectX-7 interfaces are missing, RDMA mappings differ, ranks
repeat or key-only SSH does not work.

## 5. Review and install fabric addresses

Print the netplan for every rank:

```bash
sparkring cluster configure
```

The netplans contain only the two fabric interfaces, never the management
interface or a default route. After review:

```bash
sparkring cluster configure --apply
```

On each rank it backs up any earlier SparkRing fabric netplan, runs
`netplan generate`, applies it and checks that the management address is still
on the same interface. If that check fails, it restores the earlier netplan and
stops before the next rank.

## 6. Run the read-only diagnosis

```bash
sparkring doctor --verify
```

Before applying a repair, require:

- controller `rank0`;
- one valid four- or six-node cycle;
- canonical fabric preflight `PASS`;
- management repair guard `READY`; and
- no unknown observations or failed host, link or GID prerequisites.

Missing fabric routes, forwarding rules and the resulting nonadjacent
reachability failures are what the printed repair plan fixes. A failed
management or fabric guard is not. Without `--apply`, nothing changes.

## 7. Apply fabric routing only after review

```bash
sparkring doctor --verify --apply
sparkring doctor --verify
```

Ring Doctor changes only fabric routes, IPv4 forwarding and fabric-to-fabric
`DOCKER-USER` accepts. It first checks `sudo -n true` on every rank with
commands in the plan; if any rank fails, it names it and changes nothing
anywhere. It checks the management address and return route before and after
each operation and stops at the first mismatch. The second run must show the
full reachability matrix passing.

## 8. Persist routing and firewall state across boots

Routes, forwarding and `DOCKER-USER` rules applied by Ring Doctor last until
reboot. Do not restore them from a cron `@reboot` job: it can run before Docker
creates its firewall chains, and Docker can replace rules added that early.

After reviewing the repair plan, run `sparkring doctor --emit-unit DIR`. It
writes, per rank, one fail-closed repair program and a systemd service with
`After=network-online.target docker.service` that retries every ten seconds
until the recorded management address and Docker's firewall chain are ready.
It does not install or enable them; the program path in `ExecStart` must exist
on that rank before you install its service. See
[management safety during repair](fabric-repair.md#management-safety-during-repair).

## Worker-controller recovery

Set this up while rank 0 is healthy. On the recovery worker, install the CLI
(step 2) and place a private copy of the verified inventory at
`~/.config/sparkring/cluster.yaml`. That worker also needs key access to every
rank; head-node enrollment does not provide it. If rank 0 cannot run Doctor,
run from that worker:

```bash
sparkring doctor --allow-worker-controller --verify
```

The flag works only on a machine that matches a configured worker rank.
