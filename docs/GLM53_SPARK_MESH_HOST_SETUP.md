# Prepare four DGX Sparks for the native-MTP3 mesh quickstart

This is the detailed managed-mesh extension to
[SparkRing prerequisites](PREREQUISITES.md#four-spark-managed-hardware-forwarded-mesh).
Use [the shared blank-cluster bootstrap](BOOTSTRAP.md) for SSH enrollment,
primary-interface netplans, host checks, and kernel routing/firewall setup.
This extension adds the second Socket Direct functions, mesh-specific driver
requirements, public model/runtime downloads, and then hands off to
the [model quickstart](GLM53_SPARK_MTP3_MESH_QUICKSTART.md), which pulls the
published image and starts GLM-5.3 Flash NVFP4-Spark with native MTP3.

Status: **research-only**. The recorded serving deployment passed the bounded
checks in the [managed functional record](../performance/records/glm53-flash/spark-mtp3-managed-mesh-functional-20260905.md).
The complete procedure below has **not** been rerun from four factory-reset
systems. Factory OS/driver versions can differ. Stop at a failed prerequisite;
do not treat this guide as permission to change arbitrary NICs or firmware.

Commands run in Bash on Linux. A Spark may also be the management host.
Keep a monitor/keyboard or independent management connection available while
changing data-network settings. Do not use a ConnectX data link as the only
way to administer a host being reconfigured.

## 1. Unbox, label, and onboard the hosts

Label the four systems **rank 0**, **rank 1**, **rank 2**, and **rank 3**.
Rank is the stable model-process position, not a performance ranking.
Attach power, the normal management-network connection, and any peripherals
needed for initial setup. Complete NVIDIA's first-boot wizard on every host,
creating a login account and allowing its critical software installation to
finish. Do not interrupt updates. NVIDIA documents both local-display and
network onboarding in [Initial Setup](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html).

Use a separate Ethernet management LAN, or a stable Wi-Fi management network,
for internet access and SSH. Reserve each Spark's management IP in the router
or DHCP server so it does not change during testing. This LAN can use a normal
switch; the four high-speed ConnectX cables below do not connect to it.

Apply supported DGX OS/component updates before configuring the mesh, and
reboot when requested. Keep all four hosts on compatible software versions.
Use NVIDIA's [OS and Component Update Guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html),
preferably DGX Dashboard on Founders Edition, or the partner vendor's update
procedure for its hardware. Do not use an unrelated desktop GPU driver
installer. Record the resulting versions:

```bash
cat /etc/os-release
uname -r
nvidia-smi
ip -br address
ip route show default
```

Write down the following inventory privately. The usernames may differ:

| Rank | Physical label | Login username | Management IP | Management netdev |
|---|---|---|---|---|
| 0 | Spark 0 | Your account | Actual LAN IP | Interface carrying the LAN default route |
| 1 | Spark 1 | Your account | Actual LAN IP | Interface carrying the LAN default route |
| 2 | Spark 2 | Your account | Actual LAN IP | Interface carrying the LAN default route |
| 3 | Spark 3 | Your account | Actual LAN IP | Interface carrying the LAN default route |

Never use the example's synthetic management addresses as if they identify
your hosts. Do not assign a second default route on the ConnectX links.

The inspected rank-zero software was Ubuntu 24.04.4, kernel
`6.17.0-1029-nvidia`, NVIDIA driver `580.173.02`, rdma-core 50,
NetworkManager 1.46, Docker 29.2.1, and NVIDIA Container Toolkit 1.19.1.
These identify a tested prepared host, not a universal factory image or a
requirement to downgrade other hosts blindly. Compare your inventory and
resolve driver-capability differences before starting the mesh.

## 2. Cable the four-node data ring

Use four compatible high-speed QSFP cables, one per row in this table:

| Cable | First end | Other end |
|---|---|---|
| 0–1 | Rank 0 physical port 0 | Rank 1 physical port 1 |
| 1–2 | Rank 1 physical port 0 | Rank 2 physical port 1 |
| 2–3 | Rank 2 physical port 0 | Rank 3 physical port 1 |
| 3–0 | Rank 3 physical port 0 | Rank 0 physical port 1 |

Port 0 points clockwise; port 1 points counter-clockwise. There are no
physical diagonal cables. NVIDIA's rear-view mapping identifies the left
ConnectX socket nearest Ethernet as port 0/f0 and the right socket as port
1/f1. Confirm that mapping using link state and the device inventory on your
hardware. The published transport contract requires this orientation.

Each physical port exposes two Socket Direct PCI functions. Therefore every
Spark needs four configured data netdevs, although it has only two cable
sockets. NVIDIA describes this distinction in
[ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).
This guide's four-host ring is the SparkRing profile. Do not ask NVIDIA
Sync's cluster assistant to configure a four-host direct ring: its documented
four-host topology uses a switch, not this cable cycle.

## 3. Complete shared bootstrap and check additional tools

Follow [Bootstrap a blank SparkRing cluster](BOOTSTRAP.md) from rank zero:
install the shared CLI, run `sparkring host check`, enroll four ranks with
`sparkring cluster init --size 4`, review/apply the primary-interface
netplans, and review the Ring Doctor plan. Keep its
`~/.config/sparkring/cluster.yaml` and management-safety checks.

The shared bootstrap configures **two primary data interfaces per host**.
It does not configure the two secondary Socket Direct functions or managed
hairpin services. Preserve its kernel routes, IPv4 forwarding, and scoped
fabric firewall rules: NCCL fallback and ordinary routed fabric traffic
still need them. ASIC-forwarded diagonals do not justify deleting those
baseline requirements.

Use `sparkring host check` on each host to identify missing components.
Install additional host utilities as needed:

```bash
sudo apt-get update
sudo apt-get install -y git openssh-client openssh-server rsync curl \
  ca-certificates openssl python3 python3-venv python3-pip iproute2 \
  ethtool rdma-core ibverbs-utils pciutils
sudo systemctl enable --now ssh
python3 --version
command -v devlink ibv_devinfo ip tc ethtool nmcli
systemctl is-active NetworkManager
```

Python must be 3.10 or later. The persistent network commands in section 6
require NetworkManager to manage the selected data interfaces. If `nmcli`
is absent, NetworkManager is inactive, or those interfaces are managed by
systemd-networkd/netplan instead, do not enable a second network manager over
the existing configuration. Resolve the backend mismatch using the supported
DGX OS configuration before continuing.

Docker and the NVIDIA Container Toolkit are supplied with DGX Spark's
supported software stack. Verify them before replacing packages:

```bash
sudo systemctl enable --now docker
sudo docker version
sudo docker info --format '{{json .Runtimes}}'
nvidia-ctk --version
```

The runtime list should include NVIDIA support. If Docker or the toolkit is
missing, repair the supported DGX installation using NVIDIA's
[Spark container-runtime guide](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
and [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Do not install a host CUDA toolkit or compile vLLM for this quickstart; the
published serving image carries the model runtime.

The scripts call Docker as the login user. Grant that user Docker access on
each dedicated test host and then log out and back in:

```bash
sudo usermod -aG docker "$(id -un)"
```

Confirm `docker info` works without sudo after the fresh login. Membership in
the Docker group effectively grants host-root authority; use a trusted
operator account, not an untrusted multiuser account.

On the host that will download weights, install the Hugging Face CLI in a
private virtual environment, not into the system Python:

```bash
python3 -m venv "$HOME/.venvs/sparkring-download"
"$HOME/.venvs/sparkring-download/bin/python" -m pip install --upgrade huggingface_hub
source "$HOME/.venvs/sparkring-download/bin/activate"
hf --help
```

The CLI is included in `huggingface_hub`; see its
[official CLI installation instructions](https://huggingface.co/docs/huggingface_hub/guides/cli#install-with-pip).
Activate this environment again in a shell that reports `hf: command not
found`. If a download requires authentication, run `hf auth login`
interactively; do not put tokens into repository files or command examples.

## 4. Configure management SSH and root authorization

Use rank zero as the management host after shared bootstrap. Its enrollment
step has already established verified self/worker SSH access using
`~/.ssh/id_ed25519`. Do not create a second enrollment workflow or overwrite
that key. If another management host is required, enroll its own key with
`ssh-copy-id` and verify fingerprints separately; do not copy rank zero's
private key.

Add four entries to your existing `~/.ssh/config` with a text editor. Replace
every angle-bracket placeholder with the inventory from section 1:

```sshconfig
Host spark-r0
    HostName <rank-0-management-ip>
    User <rank-0-username>
    IdentityFile ~/.ssh/id_ed25519
Host spark-r1
    HostName <rank-1-management-ip>
    User <rank-1-username>
    IdentityFile ~/.ssh/id_ed25519
Host spark-r2
    HostName <rank-2-management-ip>
    User <rank-2-username>
    IdentityFile ~/.ssh/id_ed25519
Host spark-r3
    HostName <rank-3-management-ip>
    User <rank-3-username>
    IdentityFile ~/.ssh/id_ed25519
```

The aliases must refer to the same management targets enrolled by bootstrap.
Verify them without weakening host-key checks:

```bash
for host in spark-r0 spark-r1 spark-r2 spark-r3; do
  ssh -o BatchMode=yes "$host" 'hostname; id -un; docker info >/dev/null'
done
```

The coordinator also requires `sudo -n` for systemd and the installed Python
supervisor. On each dedicated test host, use `sudo visudo -f
/etc/sudoers.d/sparkring-operator` to grant the trusted operator appropriate
noninteractive root access. For a personal dedicated cluster, the simple
policy below grants **full root authority**; replace `YOUR_USERNAME` and
accept that security scope explicitly:

```sudoers
YOUR_USERNAME ALL=(ALL:ALL) NOPASSWD: ALL
```

Use `visudo`, which checks syntax, rather than piping an unchecked line into
sudoers. On a managed network, obtain an administrator-approved policy or
root-operated workflow instead of weakening site security. Verify:

```bash
for host in spark-r0 spark-r1 spark-r2 spark-r3; do
  ssh -o BatchMode=yes "$host" 'sudo -n true'
done
```

Keep the management LAN trusted. Allow SSH between the management host and
each Spark and permit the four Sparks to reach each other's management TCP
port 9975. The model API port 8015 and liveness port 8016 need access only
from intended clients/operators. Do not expose these unauthenticated model
endpoints to the internet. Follow site firewall policy; do not flush rules
or disable the firewall globally.

## 5. Inventory each data function and save rollback information

On each Spark, identify the management interface before touching data links:

```bash
ip route show default
nmcli device status
nmcli -f NAME,UUID,TYPE,DEVICE connection show
rdma link show
```

Use the management LAN address of your administration computer to confirm
the return path, with `ip route get ACTUAL_MANAGEMENT_COMPUTER_IP`. Record
that `dev` as `MESH_MANAGEMENT_NETDEV`; never pass it to the data-configuration
commands. The profile expects these RDMA names, but netdev names must be
discovered locally:

| Direction/function | RDMA device | Common netdev name, verify before use |
|---|---|---|
| Clockwise primary | `rocep1s0f0` | `enp1s0f0np0` |
| Clockwise secondary | `roceP2p1s0f0` | `enP2p1s0f0np0` |
| Counter-clockwise primary | `rocep1s0f1` | `enp1s0f1np1` |
| Counter-clockwise secondary | `roceP2p1s0f1` | `enP2p1s0f1np1` |

Use `rdma link show` and `/sys/class/infiniband/DEVICE/device/net/` to match
each RDMA device to its actual netdev. Record each interface's real MAC from
`ip link show dev NETDEV`; do not copy the synthetic MACs in the JSON example.

Before changes, make an unused root-only backup directory on each host:

```bash
MESH_BACKUP="/var/tmp/sparkring-network-before-$(date +%Y%m%d-%H%M%S)"
sudo install -d -m 0700 "$MESH_BACKUP"
sudo tar -C / -czf "$MESH_BACKUP/network-config.tar.gz" \
  etc/NetworkManager/system-connections etc/netplan
ip -j addr show | sudo tee "$MESH_BACKUP/addresses.json" >/dev/null
ip -j route show table all | sudo tee "$MESH_BACKUP/routes.json" >/dev/null
ip -j neigh show | sudo tee "$MESH_BACKUP/neighbors.json" >/dev/null
sudo devlink dev param show | sudo tee "$MESH_BACKUP/devlink-params.txt" >/dev/null
nmcli -f NAME,UUID,TYPE,DEVICE connection show | sudo tee "$MESH_BACKUP/connections.txt" >/dev/null
```

If a listed configuration directory does not exist, inspect the actual host
backend and adjust the archive's explicit directory list; do not ignore a
failed backup. Preserve existing TC state for each verified data netdev with
`sudo tc -j -s filter show dev NETDEV ingress` and `sudo tc -j qdisc show dev
NETDEV`. Record the active NetworkManager profile UUID for each data netdev;
rollback uses those identifiers, not a guessed connection name.

## 6. Configure persistent data IPv4 and MTU

The addresses below retain the `198.18.*` benchmark-network values from
`fabric.example.json`. Verify that this isolated range does not overlap any
existing LAN/VPN route. Each cable has separate primary and secondary /24
subnets. Host /24 masks create connected routes to adjacent peers.
The fabric JSON's `ipv4_cidr` must remain **/32** because that field is a
single endpoint locator, not the host's subnet configuration. For example,
use `198.18.101.1/24` on the netdev and `198.18.101.1/32` in the JSON.
The renderer rejects /24 endpoint locators.

| Rank | Clockwise primary | Clockwise secondary | Counter-clockwise primary | Counter-clockwise secondary |
|---|---|---|---|---|
| 0 | `198.18.1.1/24` | `198.18.101.1/24` | `198.18.4.2/24` | `198.18.104.2/24` |
| 1 | `198.18.2.1/24` | `198.18.102.1/24` | `198.18.1.2/24` | `198.18.101.2/24` |
| 2 | `198.18.3.1/24` | `198.18.103.1/24` | `198.18.2.2/24` | `198.18.102.2/24` |
| 3 | `198.18.4.1/24` | `198.18.104.1/24` | `198.18.3.2/24` | `198.18.103.2/24` |

If shared bootstrap already configured the primary interfaces, preserve their
actual /24 addresses from `cluster.yaml`. Its default subnet numbering can
differ from this table. Copy those actual IPs into the private fabric's
primary endpoint locators as /32; do not replace working primary addresses
just to match an example. The secondary columns above use separate subnets.

Create one named NetworkManager profile per verified **secondary** data netdev.
The example below adds **only rank 0's clockwise secondary function**. Replace the
management and data netdev variables with the names you verified. All RDMA
workloads must be stopped. Keep the independent management connection open.

```bash
set -euo pipefail
MESH_MANAGEMENT_NETDEV='REPLACE_WITH_VERIFIED_MANAGEMENT_NETDEV'
MESH_NETDEV='REPLACE_WITH_RANK0_CLOCKWISE_SECONDARY_NETDEV'
MESH_CONNECTION='sparkring-r0-cw-secondary'
MESH_ADDRESS='198.18.101.1/24'
test "$MESH_NETDEV" != "$MESH_MANAGEMENT_NETDEV"
ip link show dev "$MESH_NETDEV"
nmcli -f GENERAL.CONNECTION,GENERAL.STATE device show "$MESH_NETDEV"
if nmcli -g connection.id connection show "$MESH_CONNECTION" >/dev/null 2>&1; then
  echo 'Connection name already exists; inspect it instead of overwriting.' >&2
  exit 1
fi
sudo nmcli connection add type ethernet ifname "$MESH_NETDEV" \
  con-name "$MESH_CONNECTION" connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  ipv4.method manual ipv4.addresses "$MESH_ADDRESS" \
  ipv4.never-default yes ipv4.ignore-auto-dns yes \
  ipv6.method link-local ipv6.never-default yes \
  802-3-ethernet.mtu 9000
sudo nmcli connection up "$MESH_CONNECTION"
ip -br addr show dev "$MESH_NETDEV"
ip route show dev "$MESH_NETDEV"
```

Repeat for each host's counter-clockwise secondary function and the other
hosts, choosing distinct profile names and the matching secondary address.
Do not create an additional NetworkManager primary profile over the shared
bootstrap's netplan-managed primary connection. Add no gateway or DNS server
to these data profiles. Activating a profile can displace an existing profile
on that **data** device; preserve the displaced UUID. Do not deactivate or
delete management profiles. NetworkManager persists the named configuration
across reboot; the command interface is described in the
[NetworkManager manual](https://networkmanager.dev/docs/api/latest/nmcli.html).

Test each direct neighbor after both ends of its cable are configured. For
rank 0's clockwise secondary example:

```bash
ping -c 3 -I "$MESH_NETDEV" 198.18.101.2
ping -c 3 -M do -s 8972 -I "$MESH_NETDEV" 198.18.101.2
```

The first ping checks address/cable reachability; the second checks a
9,000-byte IPv4 frame without fragmentation. Neither proves RDMA correctness.
Repeat on primary and secondary functions in both directions. Resolve link,
subnet, firewall, or MTU failures before continuing.

### Roll back only the profiles created here

With all RDMA users stopped and the independent management session intact,
use the exact created connection name and the saved former UUID:

```bash
sudo nmcli connection down "$MESH_CONNECTION"
sudo nmcli connection delete "$MESH_CONNECTION"
# Only if an earlier profile was recorded for this data netdev:
sudo nmcli connection up uuid 'REPLACE_WITH_RECORDED_PRIOR_UUID'
```

Never delete all NetworkManager connections or restore a whole network
archive blindly over an active host. For each function, verify the restored
address/MTU/route state against the saved inventory. The archive is an
operator-controlled recovery copy, not an automatic rollback program.

## 7. Verify GID index 3 and provision hardware hairpins

For each RDMA device, inspect GID index 3 and its associated netdev:

```bash
MESH_RDMA_DEVICE='rocep1s0f0'
cat "/sys/class/infiniband/$MESH_RDMA_DEVICE/ports/1/gids/3"
cat "/sys/class/infiniband/$MESH_RDMA_DEVICE/ports/1/gid_attrs/ndevs/3"
cat "/sys/class/infiniband/$MESH_RDMA_DEVICE/ports/1/gid_attrs/types/3"
ibv_devinfo -d "$MESH_RDMA_DEVICE" -i 1
```

Preserve IPv6 link-local addressing on the data netdevs. The example uses
`ipv6.method link-local`; disabling IPv6 can change GID allocation so an
IPv4 RoCE-v2 GID appears at index 1 instead of the required index 3. On the
netplan-managed primary profiles, retain IPv6 link-local support rather than
adding a global IPv6-disable setting.

Require index 3 to be the IPv4-mapped GID of the configured data address on
the expected netdev, with RoCE v2 type and active MTU 4,096. For example,
`::ffff:198.18.1.1` denotes the table's rank-zero primary example; substitute
the actual address if shared bootstrap allocated a different subnet.

If index 3 is absent, zero, mapped to another address, or has the wrong type:

1. Confirm the intended data profile is active and the cable link is up.
2. Check that the netdev has the single intended IPv4 address, rather than
   leftover addresses from another profile. Remove only configuration you
   explicitly created, never an unknown address or management profile.
3. Inspect all entries under that device's `gids/` and `gid_attrs/` directories
   to identify how the driver populated the table.
4. If the required mapping is still not at index 3, stop. This published
   profile fixes GID index 3; editing only a JSON value does not retarget all
   transport paths. Do not write GID sysfs files or claim a different index
   is equivalent without a validated transport change.

Complete the quickstart's
[ConnectX-7 driver configuration](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#connectx-7-driver-configuration-for-hardware-forwarding)
and [stopped-stack hairpin provisioning](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#optional-stopped-stack-hairpin-provisioning).
They discover PCI addresses, check the tested `hmfs`/legacy configuration,
set four hairpin queues of 1,024 packets only if needed, and explain driver
reload hazards. Do not reload through a data-link-only SSH connection. After
reload, bring up the four named data profiles, repeat pings/GID/MTU checks,
and verify management access before proceeding.

For primary interfaces managed by the shared bootstrap's netplan, restore
that configuration rather than creating competing connection profiles.
Run Ring Doctor read-only only after primary addresses and interface identity
are correct; inspect its repair plan before applying it. Do not discard the
shared IPv4-forwarding or Docker firewall requirements because mesh traffic
also has an ASIC-forwarded path.

The installer will not create the direct IP configuration, configure the
driver, or repair a failed GID check. It does install the subsequent exact
opposite-peer routes and hardware TC rules once these prerequisites pass.

## 8. Prepare storage and obtain the public runtime

Each host needs the **complete** model checkpoint on disk, not just its
tensor-parallel share. The inspected target occupies approximately 175 GiB on disk. Budget for
roughly 21 GB of unpacked image content, at least 40 GiB of persistent
SparkCache storage, and extra download/JIT/cache workspace; **at least
300 GiB free per host before downloading** is a practical planning target. The image is
shared-layer Docker content; archives can require additional tens of GiB.
The 300 GiB target is a planning allowance, not a qualified minimum.

The model uses shared CPU/GPU memory on each 128 GB Spark. The configured
24 GiB KV allocation is not the model's total memory requirement. Stop other
GPU/model workloads and check `free -h`, `df -h`, and `nvidia-smi` before
loading. Do not disable host memory protection or delete other workloads to
force the model to fit.

On each dedicated host, provision the named directories for the operator,
without recursively changing ownership of unrelated paths:

```bash
MESH_USER=$(id -un)
MESH_GROUP=$(id -gn)
sudo install -d -m 0755 -o "$MESH_USER" -g "$MESH_GROUP" \
  /srv/sparkring /srv/sparkring/site /srv/sparkring/artifacts \
  /srv/sparkring/receipts /srv/sparkring/glm53-mtp3-cache \
  /srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4fb16b1d37ae43d2cfd624de26ffbc832e
df -h /srv/sparkring /srv/models
```

Do not create `/opt/sparkring/managed-mesh` or
`/etc/sparkring/managed-mesh`; the installer requires those targets absent.
The root-only health-key directory is created separately by the managed guide.

Use the model quickstart's
[checkout instructions](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#preparation-order-and-command-locations)
to obtain the reviewed managed-profile revision; the shared bootstrap's CLI
checkout alone may not contain a draft-PR profile. Run subsequent repository
commands from that checkout. Then continue at
[Obtain the image and target](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#obtain-the-image-and-target).
The exact image is public and anonymously pullable; no local image build,
private image cache, or external DFlash model is needed. The quickstart
provides image-ID verification, checkpoint download and distribution, bundle
extraction, and model settings. After pulling the image, a device-access
smoke check on an otherwise idle Spark can use that same immutable image:

```bash
docker run --rm --gpus all --entrypoint nvidia-smi \
  ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:23f00af873ccc784cfb742b7be2a29c6d3c20ebec9741843c025320bb9c04685
```

This initializes GPU access but loads no model. Failure must be resolved
before creating the four serving containers.

## 9. Fill the private site and start serving

Copy the public site and fabric examples as described in the quickstart.
Replace management addresses, real MACs, netdev names, rank-specific model
and cache paths, and the extracted marker hash. Use each actual data IP with
a **/32 endpoint locator in the JSON**, while retaining the host's /24 subnet
mask. Keep the established physical-peer mapping and fixed RDMA
device roles; do not copy synthetic MACs or assume one host's interface names
apply to the other three.

Finish these steps in order using the linked command sections:

1. [Render and distribute the site](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#describe-and-render-the-site),
   together with the verified image receipt and public transport artifacts.
2. [Generate and distribute the shared key and epoch](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#prerequisites-and-identities)
   once; omit the source host from copying if it is rank 0.
3. [Pre-create stopped containers and install each rank](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#install-on-each-host).
   The installer requires explicit root authorization but starts no model.
4. Run managed `up`, then the quickstart's native all-rank RC correctness
   checks. Require zero completion errors and all correctness cases passing.
5. Run managed `start-model`, then `wait_managed_ready.py`. Wait for all four
   containers healthy plus rank-zero API/liveness HTTP 200, not merely a
   successful `systemctl start`.
6. Tail the model logs and submit a small request to model
   `glm-5.3-flash-spark` on rank zero's management IP, port 8015.

Use [managed stop/recovery](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#planned-stop-restart-and-recovery)
for subsequent maintenance. Do not change forwarding helpers beneath live
model RDMA connections. Keep this
as a supervised research deployment until the broader reliability gates
are qualified for your hosts and workload.
