# Least privilege: what runs as root, what does not, and the one privileged path

SparkRing installs and serves on directly cabled DGX Sparks. Today almost every
`sparkring` command runs with `sudo`, the workers grant the controller
`NOPASSWD:ALL` sudo, and serving runs through the Docker socket, whose group is
root-equivalent. This document is the least-privilege model for the repository:
which operation needs which privilege, what an ordinary user may do, and how the
operations that must change the host or start serving reach root through one
narrow, audited path.

Sections: the [privilege inventory](#1-privilege-inventory), the
[run-time requirements of RDMA without root](#2-rdma-without-root), the
[implemented read-only path](#3-read-only-commands-without-sudo), the
[privileged mutation path](#4-one-privileged-path-for-mutations), the
[rootless serving assessment](#5-serving-without-the-docker-group), the
[fabric tool](#6-the-fabric-tool), the [staged plan](#7-staged-plan) and the
[45-minute hardware test plan](#8-hardware-test-plan-one-pair-of-sparks-45-minutes).

Three classifications are used throughout:

- **SETUP/BOOT ONLY** — fabric addresses, routes and neighbour entries, the
  ConnectX relay rules and hairpin setting, the systemd units that restore them,
  package installation, `/srv` state. Root's work; it never touches a running
  model.
- **MUTATES SERVING** — starts, stops or replaces a served model, or slows it.
- **READ-ONLY** — status, logs, the dashboard, observations, traffic
  measurements that carry no model operation.

## 1. Privilege inventory

The capabilities that matter on a Spark:

| Short name | Grants |
|---|---|
| NET_ADMIN | `ip addr/route/neigh/link`, `tc` qdiscs and ingress offload filters, `devlink dev param set` and `reload`, `wg set`, `sysctl net.*`, `iptables` |
| NET_RAW | `iptables` (raw socket), ICMP echo without `ping_group_range`, RDMA raw-packet QPs and mlx5 flow steering |
| SYS_ADMIN | `devlink dev reload` driver reinit, namespace and cgroup operations |
| IPC_LOCK | `ibv_reg_mr` beyond `RLIMIT_MEMLOCK` (root euid, or a raised limit instead) |
| DAC | root-owned writes: `/etc/sparkring`, `/var/lib/sparkring`, `/var/log/sparkring`, `/etc/systemd/system`, `/etc/wireguard`, `/srv/sparkring` |
| SYSTEMD | `systemctl enable/start/stop/daemon-reload` (root, or polkit) |
| NMCLI | NetworkManager connection changes |
| DOCKER | `/var/run/docker.sock` — root-equivalent |
| APT | `apt-get`/`dpkg` — root |
| ROOT-CODE | `ssh user@host 'sudo -n python3 -I -B -c <source>'` — arbitrary Python as root on a worker; not narrowable as issued |

### 1.1 The transport that forces blanket root

Every change to a Spark is delivered as
`ssh USER@mgmt-ip 'sudo -n python3 -I -B -c <source>'`
(`scripts/installer_runner.py:384-402` for the elevation check, `:417` for the
runner, `:635` for the step, `:649` for status). The source is shipped on the
command line, so the sudoers grant must be `ALL`, and `check_access` prints the
`USER ALL=(ALL) NOPASSWD:ALL` fix it requires
(`runtime/host/install_workflow.py:99-117`). There is no `sparkops` account in
the repository: SSH reaches `USER@management-IPv4` with the account that ran
setup, and the administration network reaches `root@control-address` with
root's own keys. Narrowing this transport is a prerequisite for narrowing any
sudoers rule; the `sparkring node <verb>` dispatcher
(`scripts/sparkring_node.py`) is the shape to move those steps behind.

### 1.2 Operator commands (`sparkring <cmd>`)

| Command | Privileged operation | Capability | Where | Class |
|---|---|---|---|---|
| `setup` | root gate (`single_uplink.py:443`); controller key, `/var/lib/sparkring/controller` (0700); package install on workers (`packages.py:53,73`); administration network on workers (`control_node.py:294-350`); NetworkManager fabric plan; hairpin step; `node verify/configure/workspace` on every Spark; writes `cluster.json` (0600, `controller.py:405`); starts the dashboard | root; ROOT-CODE, APT, NET_ADMIN, NMCLI, SYSTEMD, DAC | `runtime/host/single_uplink.py:443,51,65,144,159,168`; `runtime/host/controller.py:129-217` | SETUP/BOOT ONLY |
| `setup --plan` / `--worker-bundle` | state directory and controller key; `apt-get download`/`dpkg-repack` for the bundle; no remote change | root euid, DAC | `single_uplink.py:443,450-452`; `packages.py:104-112` | SETUP/BOOT ONLY |
| `setup --node ...` | SSH enrollment; `sudo -n sparkring node inspect/verify/configure/adopt/native-mesh/workspace` per Spark; NetworkManager; `cluster.json` | root; ROOT-CODE + NET_ADMIN + NMCLI + SYSTEMD on workers | `controller.py:267-412` | SETUP/BOOT ONLY |
| `setup show` / `setup storage` | offline profile resolution and filesystem stats | none | `runtime/common/setup.py:230-270` | READ-ONLY |
| `install` | root gate (`install_workflow.py:85-92`); `check_access` mandates NOPASSWD:ALL (`:99-117`); image and checkpoint transfer (`install_assets.py:169,262,354,560,1104`); runs the `up` path per rank; mesh park; retention | root; ROOT-CODE, DOCKER, FOWNER, SYSTEMD, NET_ADMIN, DAC | `runtime/host/install_workflow.py:78,263,918,977,1252`; `runtime/host/memory_settle.py:24` | SETUP/BOOT ONLY + MUTATES SERVING |
| `install --plan` | same gates and the root checkpoint survey; writes a plan | root; ROOT-CODE survey | `install_workflow.py:91,109,78,309` | READ-ONLY (root-gated) |
| `up`/`down` without `--execute` | reads controller records and reviews the retained source; no Spark contacted | none beyond DAC (0700/0600 records) | `controller.py:616,949`; `retained_source.py:80-100` | READ-ONLY (root by ownership) |
| `up PROFILE --execute` | takes `install.lock`; per rank `sudo -n python3 -c` runs image/create/start/gid-serve/ring/ready steps (`installer_host.py:1795,1820,2134,2257,2268`); `ip addr del/add` (`roce_gid.py:40-41`); mesh and relay units (`native_mesh.py`, `relays.py:292-308`); sysctl `vm.*` (`installer_host.py:1795`); deployment records; dashboard site file | ROOT-CODE, DOCKER, NET_ADMIN, NET_RAW, SYSTEMD, DAC, FOWNER | `scripts/installer_runner.py:384-402,417,635,649`; `controller.py:949,980,1132` | MUTATES SERVING |
| `down --execute` | `docker stop --time 15`, ring stop/park, records state | ROOT-CODE, DOCKER, SYSTEMD | `installer_host.py:2178`; `native_mesh.py:116-136` | MUTATES SERVING |
| `status` | reads `/run/sparkring/status.json`, controller records, saved deployments; SSHes `node status` without sudo | none (read-only); root only by record ownership | `controller.py:616-767`; `node.py:1156-1167` | READ-ONLY |
| `status --refresh` | adds `sudo -n node status --refresh` per Spark and live per-deployment sources | DOCKER, NET_ADMIN, NET_RAW, SYSTEMD (read) | `controller.py:693`; `node.py:1044-1153` | READ-ONLY (root-gated) |
| `logs` | reads `/var/log/sparkring/install*.log` | none; root only by ownership | `runtime/host/progress.py:74,140-144` | READ-ONLY |
| `models`, `images` | read the packaged catalog | none | `runtime/host/models.py:78-106`; `runtime/host/images.py:8-61` | READ-ONLY |
| `checkpoints` (list) | root gate; `sudo -n node checkpoints` per Spark; walks `/srv/sparkring` | DAC | `runtime/host/checkpoints.py:1042,799` | READ-ONLY (root by ownership) |
| `checkpoints --release` | removes one checkpoint directory on every Spark | DAC, FOWNER | `checkpoints.py:1042,919,402,663` | MAINTENANCE |
| `storage` (list) | root gate; `sudo -n node storage` per Spark; `docker inspect --size`, `df` | DAC, DOCKER | `runtime/host/storage.py:1561,1268,439,505` | READ-ONLY (root by ownership) |
| `storage --release PATH` | removes an unreferenced cache or workspace on every Spark | DAC, FOWNER | `storage.py:1561,1380,791` | MAINTENANCE |
| `storage --retain-deployments` | writes the retention setting under `install.lock` | DAC | `storage.py:1561-1572`; `retention.py:156` | MAINTENANCE |
| `hairpin --plan` | root gate; `sudo -n node hairpin status` per Spark | none beyond DAC | `runtime/host/hairpin_ring.py:1274-1276,586,160` | READ-ONLY (root-gated) |
| `hairpin` (apply) | per Spark `devlink dev param set` + `devlink dev reload ... driver_reinit` (`hairpin.py:1139,1146`), `nmcli connection up`, `systemctl enable`; refuses while serving | NET_ADMIN, SYS_ADMIN, NMCLI, SYSTEMD, DAC | `hairpin_ring.py:1339-1376`; `hairpin.py:1139-1354` | SETUP/BOOT ONLY |
| `hairpin --revoke` | `systemctl disable`, unlink approval | SYSTEMD, DAC | `hairpin_ring.py:1255-1266`; `hairpin.py:1545-1559` | SETUP/BOOT ONLY |
| `recover` / `recover status` | reads `recovery.json` (0644) and the active deployment records; `systemctl is-enabled` | none; root by record ownership | `runtime/host/recovery.py:999-1016,948-955,122` | READ-ONLY |
| `recover on\|off` | writes `recovery.json` under locks; `systemctl enable --now sparkring-recover.timer` | DAC, SYSTEMD | `recovery.py:658-678,257` | MUTATES SERVING (policy) |
| `recover --auto` | probes every rank as root over SSH (`recovery.py:355`); restarts the model through the `down`/`up` path | ROOT-CODE, DOCKER, DAC | `recovery.py:709-900,864-867`; `sparkring-recover.service` | MUTATES SERVING |
| `cabling` | root gate (`cabling.py:769`); `lldpctl` (root or `_lldpd` group), `ip -j`, per-Spark survey | LLDPD, DAC | `cabling.py:753,769,777-784`; `survey.py:27,138` | READ-ONLY |
| `cabling --bandwidth` | `ib_write_bw` per cable; refuses used cables without `--while-serving` | RDMA verbs + memlock; DAC | `runtime/host/fabric_bandwidth.py:313,399-416,691` | READ-ONLY (traffic); MUTATES SERVING with `--while-serving` |
| `fabric show` | reads `controller/fabric.json` and `cluster.json` | none; root by ownership | `runtime/host/fabric.py:646-690` | READ-ONLY (root by ownership) |
| `fabric verify` | root gate; `node fabric-check` per Spark (`ip -j`, `sysctl -n`, `tc -j`, ICMP echo) | NET_RAW (ICMP), DAC | `fabric.py:754-755,228-320,506` | READ-ONLY |
| `fabric verify --traffic light` | adds one short RDMA write per relayed lane | as above + RDMA verbs | `fabric.py:690-720` | READ-ONLY (traffic); MUTATES SERVING with `--while-serving` |
| `fabric tune` (plan) | root gate; prints the plan | none beyond DAC | `runtime/host/fabric_tune.py:1006-1012,1035` | READ-ONLY (root-gated) |
| `fabric tune --execute` | stops serving with `--stop-serving`; runs the SIRCL harness in Docker per Spark; installs measured tables | DOCKER, DAC, ROOT-CODE, NET_ADMIN | `fabric_tune.py:1035,221,465-495,761-850` | MUTATES SERVING |
| `fabric spread-check` | writes test files on Node A and streams them over bulk root SSH | DAC, root SSH | `runtime/host/spread_check.py:31,168` | MAINTENANCE |
| `check` | root gate (`check.py:337`); functional requests per model; `installer transport` (ROOT-CODE) reads receipts and `docker logs`; `nvidia-smi` over plain ssh | DAC, ROOT-CODE, DOCKER | `runtime/host/check.py:95-96,139-165,337`; `transport_receipts.py:151-165` | READ-ONLY |
| `host-load show` | reads `/etc/sparkring/host-load.json` (0644) | none | `runtime/host/host_load.py:355,428` | READ-ONLY |
| `host-load set` | writes `/etc/sparkring/host-load.json` and `/etc/systemd/system/*.d` drop-ins; `systemctl daemon-reload` and restarts | DAC, SYSTEMD | `host_load.py:429,145-146,176-190,196-197` | SETUP/BOOT ONLY |
| `host-load ... --all` | same over the administration network; `show --all` adds a `sudo -n` prefix it does not need (`host_load.py:436`) | NOPASSWD sudo | `host_load.py:433-440` | SETUP/BOOT ONLY |
| `dashboard serve` (service) | unprivileged: `DynamicUser=yes`, empty capability set, reads 0644 files only | none (already least-privilege) | `packaging/debian/sparkring-dashboard.service`; `runtime/host/dashboard.py:104-108` | READ-ONLY |
| `dashboard tui` | HTTP to the local dashboard | none | `runtime/host/dashboard_tui.py:590-620` | READ-ONLY |
| `dashboard quiet ...` | writes `/run/sparkring/dashboard.quiet` | DAC (`/run/sparkring`) | `dashboard.py:108,1306-1324` | MAINTENANCE |
| `dashboard export` | root gate; reads 0600 records, writes `site.json`, reloads the service | DAC, SYSTEMD | `runtime/host/dashboard_admin.py:167,77,114` | MAINTENANCE |
| `dashboard enable\|disable` | writes or unlinks `/etc/sparkring/dashboard.disabled`; `systemctl enable/disable --now` | DAC, SYSTEMD | `dashboard_admin.py:178-183` | MAINTENANCE |

MAINTENANCE labels commands that change disk or configuration without touching
the serving path (checkpoint and cache release, retention, spread test files,
dashboard configuration).

### 1.3 `sparkring node <verb>`

`scripts/sparkring_node.py:94-99` refuses every verb without euid 0 except
`status` (without `--refresh`), `host-load show` and `hairpin status` (without
`--busy`). Several refused verbs are read-only by effect and are root-gated by
policy, not by capability: `verify`, `inspect`, `native-mesh`, `assets`,
`unit-overrides`, `fabric-check`, `tuning-facts`, `hairpin apply --dry-run`.
The mutating and boot verbs genuinely need root:

| Group | Verbs | Capability | Where |
|---|---|---|---|
| Boot fabric | `restore` (sparkring-fabric.service), `relay-markers` | NET_ADMIN + NET_RAW (`node.py:241-250,346`; `relays.py:292-308`) | `node.py:224-257`; `relays.py:315-550` |
| Hairpin | `hairpin apply` (boot unit), `approve`, `start/resume/revoke` | NET_ADMIN, SYS_ADMIN (devlink reload), NMCLI, SYSTEMD, DAC | `runtime/host/hairpin.py:1139,1146,1264,1354,1524,1782,1810` |
| Enrollment | `initialize`, `configure`, `adopt`, `workspace`, `seed`, `inspect` | DAC, CHOWN (`node.py:714`), NET_ADMIN, NMCLI, DOCKER, SYSTEMD | `node.py:89-106,567-594,629-715`; `seed.py:44-170` |
| Administration network | `control-key`, `control-configure`, `control-up` | NET_ADMIN, NET_RAW, DAC (0600 key material), SYSTEMD | `runtime/host/control_node.py:30-40,111,294-350,372-450` |
| Observations | `agent` (30 s daemon), `status --refresh` | NET_ADMIN + NET_RAW for the repair leg; DOCKER, DAC for the reads | `node.py:331-445,1044-1153,1170-1195` |
| Read-only by effect | `verify`, `inspect`, `native-mesh`, `assets`, `unit-overrides`, `fabric-check`, `tuning-facts`, `hairpin status --busy`, `checkpoints`/`storage` list legs | DOCKER, LLDPD, NET_RAW (ICMP), DAC — each root-gated by policy | `scripts/deploy_inventory.py:215-667`; `fabric.py:228-320`; `hairpin.py:1597-1684` |

### 1.4 Units and package scripts

| Unit | Runs as | Does | Capability | Class |
|---|---|---|---|---|
| `sparkring-hairpin.service` | root | `devlink` param set + driver reinit, `ethtool -K hw-tc-offload`, `nmcli`, restarts fabric units | NET_ADMIN, SYS_ADMIN | SETUP/BOOT ONLY (boot, enabled after `hairpin approve`) |
| `sparkring-fabric.service` | root | routes, per-interface sysctls, `iptables -I FORWARD`, relay table `ip route/neigh replace` + `tc` ingress offload | NET_ADMIN, NET_RAW | SETUP/BOOT ONLY (boot) |
| `sparkring-relay-marker.service` | root | supervises RDMA tagging markers | RDMA verbs, likely NET_RAW + NET_ADMIN | SETUP/BOOT ONLY (continuous; carries relayed serving traffic) |
| `sparkring-control.service` / `-refresh` (+timer, udev rule, NM dispatcher) | root | WireGuard `sr-control`, firewall, sysctl, starts dns/access/fabric | NET_ADMIN, NET_RAW | SETUP/BOOT ONLY (administration network) |
| `sparkring-access.service` | root | sshd on `sr-control` admitting root with Node A's keys | root (sshd) | MUTATES SERVING in effect: it is the root remote-execution transport |
| `sparkring-seed.service` | root | one-time preparation sshd on port 2222 admitting root | root (sshd) | SETUP/BOOT ONLY (a root login surface while enabled) |
| `sparkring-dns.service` | root | dnsmasq on port 53 | NET_BIND_SERVICE, NET_RAW | SETUP/BOOT ONLY |
| `sparkring-agent.service` | root (hardened, no capability bound) | 30 s observation; restores missing approved routes, sysctls and relay objects; writes `/run/sparkring/status.json` | NET_ADMIN, NET_RAW, DOCKER (read) | SETUP/BOOT ONLY (adds only approved objects) |
| `sparkring-recover.service` (+timer) | root, `HOME=/root` | probes as root over SSH; restarts the model through `down`/`up` | ROOT-CODE, DOCKER, DAC | MUTATES SERVING |
| `sparkring-ssh-mux.service` | root | holds root SSH master connections | root's keys | MUTATES SERVING (carries recovery's root commands) |
| `sparkring-dashboard.service` | **DynamicUser, empty capability set** | the read-only web page | none | READ-ONLY |
| `sparkring-memory-guard.service` (not in the .deb) | root | `docker kill` of opted-in model containers below 4 GiB free | DOCKER, SYS_NICE | MUTATES SERVING |
| `sparkring-hairpin-mesh-check` (generator) | root | adds the hairpin start check drop-in to mesh units | DAC (`/run/systemd/generator`) | SETUP/BOOT ONLY |
| postinst / prerm / postrm | root | initialize, enable and restart units, record enabled units | root | SETUP/BOOT ONLY |

The package creates no user and no group; `build_deb.py:277` forces root:root
ownership and the code creates `/var/lib/sparkring`, `/var/log/sparkring` and
`/etc/sparkring` at run time. The dashboard unit is the existing template for
non-root services.

## 2. RDMA without root

What SIRCL (`spark_transport/libsircl`, the `libnccl.so.2`-compatible transport
loaded with `LD_PRELOAD`), the `sparkring_sircl` Python path and an
NCCL-plugin run time need from an ordinary user. libsircl reaches RDMA through
`dlopen("libibverbs.so.1")` and RC queue pairs; it uses no rdma_cm, no raw
sockets and no capability checks, and it never opens a device node itself —
libibverbs does, on its behalf.

| Requirement | Evidence | Privilege | Failure without it | Fix |
|---|---|---|---|---|
| Enumerate devices | `libsircl/src/transport/verbs_dl.c:45`; `sircl_roce_proxy.c:676-690` | none; sysfs and libibverbs config are world-readable | "RDMA device not found" | `rdma-core` + provider installed (already in package Depends) |
| Open `/dev/infiniband/uverbs*` | `sircl_roce_proxy.c:695-699` | read+write on the node; the kernel creates them mode 0666 by default | EACCES | nothing on a stock host; udev rule only if a site tightens modes |
| Query port and GID | `sircl_roce_proxy.c:703-720`; GID auto-resolution `libsircl/src/engine.c:1454-1499` | none (sysfs 0444) | "port 1 is not active" / "no readable GID table" | link up; correct `SIRCL_GID_INDEX` (or let libsircl resolve) |
| Register the pinned arena | `sircl_roce_proxy.c:728-732`; arena size `libsircl/README.md:123-125` | `RLIMIT_MEMLOCK` ≥ registered bytes, or `CAP_IPC_LOCK` | `ibv_reg_mr` ENOMEM → communicator creation fails | the limits.d file below; containers already set `--ulimit memlock=-1:-1` |
| Create CQ and RC QPs, RTR/RTS | `sircl_roce_proxy.c:734-768,972-1001` | none (raw-packet QPs and flow steering need NET_RAW; SIRCL uses neither on the serving path) | — | — |
| Out-of-band TCP setup | `libsircl/src/bootstrap.c:585-665` | none (ephemeral ports) | firewall only | allow TCP between ranks |
| Fork behaviour | `bootstrap.c:557,650,723-747` | none | registered MRs are not fork-safe | exec, never fork, after library use |

The arena is registered once **per RDMA device** (up to four), so the memlock
demand is a multiple of the arena: about 4 × 32 MiB = 128 MiB at TP4 defaults,
4 × 64 MiB = 256 MiB at eight ranks, plus the p2p arena and headroom. The
64 MiB `ulimit -l` this development host shows, and systemd's 8 MiB default,
are both too low.

User setup, as text for the operators' guide:

`/etc/security/limits.d/90-sparkring-rdma.conf` (applies to PAM sessions —
SSH, login, `su`; not to systemd services or containers):

```
# SparkRing / SIRCL: ibv_reg_mr pins the transport arena once per RDMA device.
# Without CAP_IPC_LOCK the kernel enforces RLIMIT_MEMLOCK and fails with ENOMEM.
# Apply to members of the "rdma" group (sudo groupadd -r rdma; sudo usermod -aG rdma USER).
@rdma    soft    memlock    unlimited
@rdma    hard    memlock    unlimited
```

For systemd-launched processes, `/etc/systemd/system.conf.d/` and
`/etc/systemd/user.conf.d/` files with:

```
[Manager]
DefaultLimitMEMLOCK=infinity
```

(or `LimitMEMLOCK=infinity` per unit, as `runtime/glm53-spark-mtp3-mesh/managed_units.py:59`
already does), then `sudo systemctl daemon-reexec` and a new login.

Only if a site tightens the device nodes (a stock kernel creates them 0666,
`drivers/infiniband/core/uverbs_main.c` and `ucma.c`), add
`/etc/udev/rules.d/70-sparkring-rdma.rules`:

```
SUBSYSTEM=="infiniband_verbs", KERNEL=="uverbs*", GROUP="rdma", MODE="0660"
SUBSYSTEM=="misc", KERNEL=="rdma_cm", GROUP="rdma", MODE="0660"
```

libsircl and stock NCCL IB never open `rdma_cm`; keep that line only for tools
that do (perftest `-R`, NCCL with rdmacm). For containers the device and limit
contract is already met by the shared container specification
(`runtime/common/container_spec.py:33-34`: `--device /dev/infiniband`,
`--ulimit memlock=-1:-1`, `--ipc host`, `--network host`); the switched launch
paths already run the container as a non-root `--user` with no added
capability, which is the pattern to keep.

CPU results in this environment (WSL, uid 1000, no RDMA hardware):
`/dev/infiniband` is absent, `ibv_get_device_list` returns zero devices, and
the shared-memory verbs stand-in suite passes
(`spark_transport/libsircl/tests/test_shm_verbs.py`, 5 tests) — it proves the
session protocol only, because the stand-in never checks memlock or
permissions. The full `libsircl.so` cannot be built here (no `nvcc`).

Needs hardware to confirm: the device modes on DGX OS
(`ls -l /dev/infiniband/`); that a non-root, no-capability process completes
communicator creation (rlimit alone suffices, no hidden CAP_IPC_LOCK or
CAP_NET_RAW demand in the GB10 mlx5 driver); the memlock actually consumed
(`VmPin` in `/proc/<pid>/status`) to size a finite limit; whether
`cuMemHostAlloc` and torch `pin_memory=True` count against `RLIMIT_MEMLOCK` on
GB10's unified memory; sysfs readability under RDMA netns `exclusive` mode;
and the relay markers' minimum capability (raw QP / flow steering).

## 3. Read-only commands without sudo

Implemented on this branch. A system group `sparkring` is created by the
package (`packaging/debian/postinst`). Members of the group read the
artifacts root publishes for them; everything secret stays root-only:

| Artifact | Mode | Written by | Read by |
|---|---|---|---|
| `/var/log/sparkring/install.log`, `install-details.log` | 0640, group `sparkring`, directory 2750 root:sparkring | `progress.run` (root) | `sparkring logs` |
| `/var/lib/sparkring/operator-status.json` (`sparkring-operator-status/v1`) | 0640, group `sparkring` | `sparkring status`, and `dashboard_admin.refresh_site` after every `install`/`up`/`down`/setup/recovery | `sparkring status`, `sparkring recover status` |
| `/run/sparkring/status.json` | 0644 (unchanged; the unprivileged dashboard service reads it) | the agent | `sparkring node status`, any local user |
| `/var/lib/sparkring/recovery.json` | 0644 (unchanged) | recovery | the dashboard |

`/var/lib/sparkring/controller` (0700), its 0600 records, `/etc/sparkring/control.json`
(0600, administration-tunnel key material), the SSH identities under the
controller directory and the deployment records stay root-only. The operator
view is redacted: `runtime/host/privilege.redact` replaces every IPv4 site
address with `<site-address>` before the document is written, so membership of
the group grants states, ranks, model names and ports — not addresses.

- `sparkring status` for a process the controller records refuse (root's
  0700 directory) falls back to the last published view
  (`runtime/host/controller.py`, `_operator_status`); the live path is
  unchanged for root and for an operator whose account owns the records
  (`node workspace` chowns the controller directory). `--refresh`, which
  signs in to every Spark, is refused for the recorded view. A user outside
  the group gets the instruction to join or use sudo. The view's age is
  printed in every output, so a snapshot is never mistaken for a live
  observation.
- `sparkring recover status` falls back the same way to the recovery lines of
  the same view (`runtime/host/recovery.py`, `_operator_recovery`).
- `sparkring logs` works for a group member; the log files are 0640 group
  `sparkring` and the directory 2750 (`runtime/host/progress.py`); a user who
  can neither read nor enter the log directory is told how to gain access.
- `sparkring models`, `images`, `host-load show`, `node status`,
  `dashboard tui` and the dashboard service already ran without root and are
  unchanged.

Tests: `runtime/host/test_privilege.py` (group membership, redaction, share
modes, publication, the non-root status and recovery paths, and that a group
member's read records nothing), `runtime/host/test_helper.py` (below), plus the
existing `test_progress.py` and `test_dashboard_admin.py` suites, which cover
the changed writers. What remains root-only by ownership or policy after this
stage: `fabric show`, `checkpoints`/`storage` listings, `up`/`down` plan
review, `fabric tune` plan, `hairpin --plan`, `check`, `cabling` and every
`--refresh`. They are stage 3 of the plan.

## 4. One privileged path for mutations

The mutation set the brief names — `install`, `down`, `up`, `recover`, `setup` —
today runs either through the operator's `sudo` (blanket `NOPASSWD:ALL`) or,
for recovery, as a root timer. Three mechanisms were compared.

**A root systemd helper with a socket and a fixed command set** (recommended,
implemented behind a flag). A root `sparkring-helper.service`, socket-activated
on `/run/sparkring/helper.sock` (mode 0660, group `sparkring`), accepts one
JSON request `{"argv": [...]}` per connection, validates the argv against a
fixed allowlist (`runtime/host/helper.py`, `COMMANDS`), runs
`/usr/bin/sparkring <argv>` as its child and streams the output back; every
request writes one audit line (time, peer uid, argv, exit) to the journal. No
shell, no free-form command, no environment forwarding.

**polkit** authorizes well but executes whatever the caller asked for with
full root argv freedom, needs an authentication agent for non-interactive use,
and adds a policy authoring surface (`.policy` XML) whose rules still cannot
inspect argument values as precisely as code can.

**A narrow sudoers rule** for `sparkring <subcommand>` cannot actually be
narrow today, because the commands deliver their remote steps as
`sudo -n python3 -c <source>` (section 1.1); a sudoers rule for
`/usr/bin/sparkring` would also grant every subcommand and every argument
value, and sudoers argument matching is order-sensitive and easy to bypass
with flag reordering.

The helper is the narrowest mechanism that can validate argument values in
code, refuses anything the allowlist does not name, and audits each request in
the journal with the caller's kernel-verified uid.

Threat model:

- **Who can trigger what.** Only members of the `sparkring` group can connect
  (socket mode 0660, group-checked again from `SO_PEERCRED`); only the argv
  shapes in `COMMANDS` run (`up|down [PROFILE] --execute`, `recover --auto`).
  A group member cannot run arbitrary commands, cannot pass extra flags
  (`--fresh-mesh`, `--allow-loopback-bind` are refused), and cannot reach
  `setup` or `install` (not in the table; widening it is a reviewed change).
- **Input validation.** The whole argv is matched before any process starts;
  a profile or placement value must match its grammar; a request line over
  1 MiB is refused; the connection times out.
- **Audit.** One journal line per request: time, peer uid, exact argv, exit
  status; the child's output is attributed to the same journal entry. The
  operator runs the command themselves, so their shell history plus the
  journal reconstruct every mutation.
- **How agents run unprivileged.** An agent account holds membership of
  `sparkring` (read-only surface) and nothing else; its mutations go through
  the helper and appear in the journal with its uid. It never needs the
  Docker group and never holds a private key.
- **Residual risks.** The allowlist grants `up`/`down` with any profile and
  placement — a group member can start or stop serving, which is the point,
  but also the blast radius; `sparkring-helper@.service` runs as root with a
  light sandbox because the child legitimately needs Docker, systemd and SSH;
  the SSH transport to workers is still `sudo -n python3 -c`, so a compromised
  Node A is still root on every Spark — narrowing that transport is stage 5.

Implemented on this branch, behind a flag: `runtime/host/helper.py` (grammar,
protocol, server handler, client), `sparkring node helper` (the per-connection
entry point), `packaging/debian/sparkring-helper.socket` and
`sparkring-helper@.service` (shipped, **not enabled** — an operator opts in
with `systemctl enable --now sparkring-helper.socket`), and the client hook in
`scripts/sparkring.py`, which routes a non-root `up`/`down`/`recover` through
the socket only when `SPARKRING_HELPER_SOCKET` is set. Without the variable,
or as root, behaviour is exactly as before. Tests:
`runtime/host/test_helper.py` (allowlist acceptance and refusal, protocol and
audit line, an end-to-end request over a real unix socket, and both CLI
interception cases). The unit wiring and a live `up --execute` through the
socket need a Spark.

## 5. Serving without the Docker group

Research-only. The Docker group is root-equivalent; removing it from the
model-container path does not remove root from the host work (apt, NetworkManager,
ConnectX restarts, `vm.drop_caches`, CDI refresh), which already runs through
fixed `sparkring node` verbs.

What the container specification already does right: `--device /dev/infiniband`,
`--network host`, `--ipc host`, `--ulimit memlock=-1:-1`, bind mounts, health
checks, seccomp — all portable between Docker and Podman. What is
Docker-specific: `--gpus all|N` (the CDI name `nvidia.com/gpu=all` is the
portable replacement, and `ensure_gpu_cdi_spec()` already enables the CDI
refresh service), `--runtime runc`, the `--context default` pinning, the
Docker inspect schema that `expected_inspection()` and `owned()` compare
against, and the `Requires=docker.service` units of the mesh supervisor and
memory guard. The docker binary string is hard-coded in roughly sixty files;
the real chokepoints are `installer_host.run()` and `compose.compose_command()`.

Two rootless options, in order of effort:

- **Rootless Docker ≥ 29.5** (which runs the daemon in the host network
  namespace, so `--net=host` is real): install `docker-ce-rootless-extras`,
  run `dockerd-rootless-setuptool.sh install` under a service account, set
  `DOCKER_HOST` to its socket — the `--context default` pinning honours
  `DOCKER_HOST`, so most of the installer keeps working. The diff is the
  `sudo -n docker` call sites, the `/var/lib/docker` assumptions
  (image store, space checks) and the CDI device naming.
- **Rootless Podman** (Ubuntu 24.04 ships 4.9.3): `--device
  nvidia.com/gpu=all` with CDI spec files readable since container-toolkit
  1.13.3; `--network host` shares the host netns; inspect schema and compose
  labels differ, which breaks the adoption/equivalence checks and needs the
  `ContainerSpec` extensions listed below.

The decisive kernel facts, from the kernel sources (needs hardware
confirmation on GB10): `ib_umem_get` checks `CAP_IPC_LOCK` in the **init**
user namespace, so `--cap-add IPC_LOCK` is a no-op for RDMA in a rootless
container — the host user's hard `RLIMIT_MEMLOCK` is the only control
(`LimitMEMLOCK=infinity` on `user@UID.service` or limits.d); uverbs and
`rdma_cm` nodes are 0666 by default, so a rootless container can open them
unless a site udev rule tightens them; RC queue pairs, MR, CQ and PD have no
capability checks, while raw-packet QPs and mlx5 DEVX check `CAP_NET_RAW` in
the init namespace — and no container-side code was found to use them.

Changes the specification would need (design, not yet implemented): a
`runtime` field on `ContainerSpec` (`docker`/`podman`), a CDI GPU request
mode, `userns_mode` (`keep-id`) and `group_add` (`keep-groups`); the inspect
schema extended for CDI devices; the engine name parametrized at the two
chokepoints; per-user image stores and space checks; quadlet or user-manager
units for the mesh supervisor and memory guard with `loginctl enable-linger`.

Known risks, in order: memlock (make-or-break, mitigation above); the inspect
schema long tail; Ubuntu 24.04's `apparmor_restrict_unprivileged_userns` and
the stale Podman 4.9.3; per-user image stores duplicating multi-GB images and
the `/var/lib/docker` space logic; cgroup delegation for `--memory` and
`--cpuset-cpus`; and on GB10 specifically the untested CUDA-driver-in-userns
path plus the unresolved NVIDIA container-toolkit host-freeze report
(nvidia-container-toolkit#1910) — a rootful baseline is mandatory when
testing, so a hang is not misattributed. The full hardware test plan for one
pair of Sparks is in the agent research record and condensed in section 8.

## 6. The fabric tool

Branch `pi/sircl-fabric-tool` (`spark_transport/fabric-tool`), and the public
draft. Subcommands by privilege:

| Subcommand | Root? | Why |
|---|---|---|
| `plan` | no | offline computation from the topology |
| `facts`, `status` | no | read script: `ip -j`, sysfs, `tc show`, `pgrep`, `/run` record |
| `verify` | no (see below) | read script plus `ip -o route get` per lane; no `sudo` inside |
| `apply` (dry run) | no | read, diff, print scripts |
| `apply --execute` | yes | `sudo -n tc qdisc/filter` (ingress offload), `ip neigh/route replace`, `/run/fabric-table` record, marker start — NET_ADMIN; the marker needs raw QP/flow access |
| `remove --execute` | yes | the deletions, and killing the root-run marker |
| `install-marker` (dry) | no | compile and `install -m 0755` to `/var/tmp` work unprivileged |
| `install-boot --execute`, `remove-boot --execute` | yes, run as root itself | writes `/usr/local/lib/fabric-tool`, `/usr/local/bin/fabric-tool`, `/etc/systemd/system/fabric-table.service`, `/etc/fabric-table/*`, `systemctl enable` |
| net-sircl plugin | no | env-configured, opens uverbs via libibverbs; needs device access and memlock only |

`verify` does not need root for any single read: `RTM_GETQDISC` and
`RTM_GETTFILTER` dumps have no capability check (only the NEW/DEL operations
do), and route, neighbour, address and `ip route get` dumps are unprivileged.
Its reads are wrapped in `2>/dev/null || true`, so a permission failure would
today masquerade as drift. The non-root path: run `verify` unprivileged as-is,
add one hardware confirmation (`tc -j filter show dev X ingress` as a normal
user against offloaded rules), and make the `tc` sections fail loudly instead
of swallowing errors. Fallbacks, in order: `apply` (already privileged) writes
the final `tc -j filter show` snapshot into the existing
`/run/fabric-table/relay-state.json` for comparison; or a root timer dumps
`tc`/`ip` JSON to a group-readable file. A file-capability helper is neither
needed nor advisable: reads need no capability, and a `cap_net_admin` helper
would grant write power over the traffic rules.

## 7. Staged plan

| Stage | Scope | Status | Tests |
|---|---|---|---|
| 1 | The `sparkring` group: group-readable installation logs, the redacted operator status view, group-member `status`/`recover status`/`logs`, package group creation | **Implemented** (this branch) | `runtime/host/test_privilege.py`; existing `test_progress.py`, `test_dashboard_admin.py`, `test_recovery.py` |
| 2 | The privileged helper: allowlist, protocol, client, opt-in units, CLI flag | **Implemented** (this branch, opt-in and off by default) | `runtime/host/test_helper.py`; the unit wiring and one live `up --execute` need a Spark |
| 3 | Remaining read-only commands for the group: `fabric show`, `checkpoints`/`storage` listings, `up`/`down` plan review, `fabric tune` plan, `hairpin --plan` — either as further redacted views or by granting the group read access to the specific non-secret records | **Research-only** (design in sections 1.2 and 3; needs an owner decision on which record fields a group member may see) | none yet |
| 4 | Per-verb privilege tiers in `sparkring node` (read-only verbs drop the blanket euid gate; each verb declares its capability) and capability-bounded units (`sparkring-agent.service` bounds to NET_ADMIN+NET_RAW, `sparkring-ssh-mux.service` to none) | **Research-only** | none yet |
| 5 | Replace the `sudo -n python3 -c` SSH transport with fixed `sparkring node` verbs so the sudoers grant can stop being `ALL`; then the helper allowlist can widen to `install`/`setup` | **Research-only** (largest refactor; prerequisite for a narrow sudoers rule) | none yet |
| 6 | Rootless serving (section 5) | **Research-only** | hardware test plan, section 8 |
| 7 | Host `limits.d`/systemd memlock drop-ins shipped by the package, and the RDMA user setup in the operators' guide | **Research-only** (text written in section 2; the values need hardware sizing) | none yet |
| — | Removing root from setup, boot units, package installation and the ConnectX restarts | **Unsupported by design** | — |

Decisions the owner must make: which record fields the `sparkring` group may
see (the operator view redacts all IPv4 addresses; `fabric show` and plan
review would need the same policy); whether the helper ships enabled or
opt-in; whether rootless serving targets Docker-29.5 or Podman first; and
whether stage 5's transport refactor is scheduled, since every later
narrowing depends on it.

## 8. Hardware test plan (one pair of Sparks, 45 minutes)

Node A and Node B, directly cabled, with the package from this branch and the
fabric-tool branch's `net-sircl` where noted. Record versions first
(`uname -r`, `docker version`, `nvidia-ctk --version`, `dpkg -l | grep -E
'nvidia-container|rdma|ibverbs'`).

1. **Minutes 0-8 — group reads without sudo.** Create a test user, `sudo
   usermod -aG sparkring <user>`, re-login. As that user: `sparkring logs`,
   `sparkring status`, `sparkring recover status`, `sparkring node status`,
   `curl` the dashboard. Confirm the output shows no IPv4 address and that
   `ls -l /var/lib/sparkring/controller` denies traversal, `/etc/sparkring/control.json`
   is unreadable, and every attempted mutation (`sparkring down --execute`,
   `recover on`) fails without effect. Run the same commands once with `sudo`
   to show the live view beside the recorded one.
2. **Minutes 8-20 — non-root RDMA.** On both Sparks: `sudo groupadd -r rdma`,
   the limits.d file of section 2, add the test user. As the user:
   `ibv_devinfo`, then `ib_write_bw` pair (register ≥ 1 GiB), then
   nccl-tests `all_reduce_perf` at 2 GiB with `LD_PRELOAD` of libsircl (or
   `NCCL_NET` via net-sircl from the fabric-tool branch) and `NCCL_DEBUG=INFO`:
   pass bar is `NET/IB` (not `NET/Socket`) and bandwidth within 5 % of the
   rootful baseline. Negative control: repeat with the stock 64 MiB memlock and
   record the exact `ibv_reg_mr` ENOMEM.
3. **Minutes 20-30 — read-only fabric tool.** As the test user:
   `fabric-tool plan`, `status`, `verify` (confirm the `tc` reads do not
   silently degrade; compare against the root run's output), then
   `apply --execute` dry run. Root runs `apply --execute` once to have rules
   to verify.
4. **Minutes 30-42 — one rootless serving start.** With the rootless Docker
   service account prepared (subuid/subgid, linger, `LimitMEMLOCK=infinity`,
   cgroup delegation): start one small model's ranks rootless with
   `--device nvidia.com/gpu=all --device /dev/infiniband --network host --ipc
   host --ulimit memlock=-1:-1`; confirm `nvidia-smi -L` in the container, the
   model reaches `/health`, and one `sparkring check`-style functional request
   succeeds; record the receipts beside the rootful baseline from before the
   session. If the memlock or device step fails, record the errno and stop —
   that is the finding.
5. **Minutes 42-45 — record.** Journal audit lines from any helper use, memlock
   consumed (`VmPin`), device modes, and the comparison table of the three
   measured numbers (baseline, rootless, non-root RDMA).
