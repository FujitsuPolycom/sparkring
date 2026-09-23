# Prepare the two-Spark data network

Use after [host preparation](host-preparation.md). This procedure targets the
published GLM/Qwen pair mapping: one cable from **physical cage p0 on rank 0 to
physical cage p0 on rank 1**, using both Socket Direct functions. Each cage has
two host-visible RDMA devices; one cable therefore needs two data addresses per
host. Confirm the physical mapping against link state on your hardware.

Status: **procedure awaiting hardware rehearsal**. Existing pair serving evidence
does not establish that this fresh-network procedure works on every factory image.
`sparkring cluster init` and Ring Doctor support rings, not this pair procedure.

## 1. Enroll management SSH

**Run on:** rank 0. **Shell:** Bash. Use the real management username/address of
rank 1. Inspect the presented host fingerprint against the worker before accepting
it. Generate a key only if you do not already have one; never overwrite a key.

```bash
test -f "$HOME/.ssh/id_ed25519.pub" || ssh-keygen -t ed25519
ssh-copy-id WORKER_USERNAME@WORKER_MANAGEMENT_IP
ssh -o BatchMode=yes WORKER_USERNAME@WORKER_MANAGEMENT_IP 'hostname; id -un'
```

Replace the uppercase placeholders before running. Save the target in your private
inventory. **Pass:** the second command can log in without a password. These
commands do not install the checkout on rank 1; host preparation does that.

## 2. Inventory the connected cage on both ranks

**Run on:** rank 0 and rank 1 separately. Keep an independent management session
open throughout preparation. Stop affected model/RDMA workloads before changing
data interfaces. This guide does not stop them automatically.

```bash
ip route show default
ip -br address
rdma link show
nmcli device status
nmcli -f NAME,UUID,DEVICE connection show
for device in rocep1s0f0 roceP2p1s0f0; do
  printf '%s: ' "$device"
  ls "/sys/class/infiniband/$device/device/net/"
done
```

Record the management interface and the one netdev corresponding to each RDMA
device. The two expected devices are `rocep1s0f0` (primary) and `roceP2p1s0f0`
(secondary). If either is absent, either netdev carries management, or the cable
is connected to another cage, stop and resolve the mapping. Do not rewrite the
profile's device list to hide a mismatch.

**Existing configured pair:** retain its actual IPs and NetworkManager UUIDs.
Skip section 3 and use those addresses in section 4. A prepared pair should not
be renumbered to match this example.

## 3. Configure an unused pair

**Run on:** each rank, only when the selected data interfaces are unused. Both
ends use the same two /24 subnets. Check `ip route show table all` on both hosts
first; the example ranges must not overlap management, VPN or existing data routes.

| Rank | Primary address | Secondary address |
|---|---|---|
| 0 | `198.18.20.1/24` | `198.18.21.1/24` |
| 1 | `198.18.20.2/24` | `198.18.21.2/24` |

Record existing data connection UUIDs and their settings before activation. Save
a private backup on each host:

```bash
BACKUP="$HOME/sparkring-network-before-$(date +%Y%m%d-%H%M%S)"
mkdir -m 700 "$BACKUP"
nmcli -f NAME,UUID,DEVICE connection show > "$BACKUP/connections.txt"
ip -j address > "$BACKUP/addresses.json"
ip -j route show table all > "$BACKUP/routes.json"
sudo tar -C / -czf "$BACKUP/network-config.tar.gz" \
  etc/NetworkManager/system-connections etc/netplan
sudo chmod 600 "$BACKUP/network-config.tar.gz"
```

If a backup input is absent, inspect the actual network backend and adjust the
explicit directory list. Require a successful backup before proceeding.

On rank 0 set `RANK=0`; on rank 1 set `RANK=1`. Fill the three netdev names from
section 2 on that host. This block refuses existing IPv4 assignments and duplicate
connection names; it never edits existing connections. Review any refusal.

```bash
set -euo pipefail
RANK=0
MANAGEMENT_NETDEV=REPLACE_WITH_MANAGEMENT_NETDEV
PRIMARY_NETDEV=REPLACE_WITH_PRIMARY_NETDEV
SECONDARY_NETDEV=REPLACE_WITH_SECONDARY_NETDEV
case "$RANK" in 0|1) ;; *) echo 'RANK must be 0 or 1' >&2; exit 1 ;; esac
systemctl is-active --quiet NetworkManager
test "$PRIMARY_NETDEV" != "$SECONDARY_NETDEV"
NETDEVS=("$PRIMARY_NETDEV" "$SECONDARY_NETDEV")
RDMA_DEVICES=(rocep1s0f0 roceP2p1s0f0)
ADDRESSES=("198.18.20.$((RANK + 1))/24" "198.18.21.$((RANK + 1))/24")
for i in 0 1; do
  dev=${NETDEVS[$i]}
  test "$dev" != "$MANAGEMENT_NETDEV"
  test -d "/sys/class/infiniband/${RDMA_DEVICES[$i]}/device/net/$dev"
  ip link show dev "$dev"
  if ip route show default | grep -qw "$dev"; then
    echo "Refusing default-route interface $dev" >&2; exit 1
  fi
  if test -n "$(ip -4 -o address show dev "$dev")"; then
    echo "Existing IPv4 address on $dev; use the existing-pair path" >&2; exit 1
  fi
  if nmcli connection show "sparkring-pair-r${RANK}-${i}" >/dev/null 2>&1; then
    echo 'Connection already exists; inspect or resume it explicitly' >&2; exit 1
  fi
done
# Both interfaces passed the guards. Review names and addresses before activation.
printf '%s\n' "rank=$RANK" "primary=$PRIMARY_NETDEV ${ADDRESSES[0]}" \
  "secondary=$SECONDARY_NETDEV ${ADDRESSES[1]}"
read -r -p 'Configure these unused data interfaces? Type yes: ' answer
test "$answer" = yes
for i in 0 1; do
  sudo nmcli connection add type ethernet ifname "${NETDEVS[$i]}" \
    con-name "sparkring-pair-r${RANK}-${i}" connection.autoconnect yes \
    ipv4.method manual ipv4.addresses "${ADDRESSES[$i]}" \
    ipv4.never-default yes ipv4.ignore-auto-dns yes \
    ipv6.method link-local ipv6.never-default yes 802-3-ethernet.mtu 9000
  sudo nmcli connection up "sparkring-pair-r${RANK}-${i}"
done
```

Repeat on rank 1 with its own variables. No gateway, DNS, forwarding or ring
routes are needed for this direct pair. The commands persist these data profiles.
Activation can displace an earlier profile; preserve its UUID for rollback.
If interrupted, inspect which connections were created before resuming. Do not
rerun the whole block or delete unrelated connections to bypass a refusal.

## 4. Verify both paths on both ranks

Set the local netdevs from section 2. On rank 0, the peer addresses are rank 1's;
on rank 1 they are rank 0's. For an existing pair, use its actual addresses.

```bash
PRIMARY_NETDEV=REPLACE_WITH_PRIMARY_NETDEV
SECONDARY_NETDEV=REPLACE_WITH_SECONDARY_NETDEV
PEER_PRIMARY=198.18.20.2
PEER_SECONDARY=198.18.21.2
ping -c 3 -I "$PRIMARY_NETDEV" "$PEER_PRIMARY"
ping -c 3 -M do -s 8972 -I "$PRIMARY_NETDEV" "$PEER_PRIMARY"
ping -c 3 -I "$SECONDARY_NETDEV" "$PEER_SECONDARY"
ping -c 3 -M do -s 8972 -I "$SECONDARY_NETDEV" "$PEER_SECONDARY"
for device in rocep1s0f0 roceP2p1s0f0; do
  printf '\n%s\n' "$device"
  cat "/sys/class/infiniband/$device/ports/1/gids/3"
  cat "/sys/class/infiniband/$device/ports/1/gid_attrs/ndevs/3"
  cat "/sys/class/infiniband/$device/ports/1/gid_attrs/types/3"
  ibv_devinfo -d "$device" -i 1
done
```

**Pass on each rank:** all four pings succeed, both ports are active, RDMA MTU is
4096, and GID index 3 is RoCE v2 with the intended local IPv4-mapped address and
matching netdev. For example `::ffff:198.18.20.1` corresponds to rank 0's primary
address. Preserve IPv6 link-local support; disabling it can change GID ordering.
If index 3 differs, stop and inspect addressing/driver state. Do not edit only a
profile GID value or write GID sysfs files.

Recheck after reboot before model startup. These observations establish link,
IP/MTU and GID configuration; actual RDMA collectives still require the selected
runtime's startup checks. Save output privately for the rehearsal.

## 5. Hand off to the selected model

Use the primary fabric address for `MASTER`/`MASTER_ADDR` (rank 0 on both hosts),
the local primary fabric address for `VLLM_HOST_IP`/`HOST_IP`, and its netdev for
the socket interface. Client API traffic can use rank 0's management address.

- [GLM pair: select the image and checkpoint](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md#select-the-image-and-checkpoint).
- [Qwen pair: image and checkpoint](../../profiles/qwen38-flash-next-tp2/README.md#image-and-checkpoint).

## Roll back a connection created here

Stop the pair's RDMA workloads first. On the affected rank, use the exact name
created in section 3 and the previous UUID recorded in the backup:

```bash
sudo nmcli connection down CREATED_CONNECTION_NAME
sudo nmcli connection delete CREATED_CONNECTION_NAME
# Only if a previous connection was recorded for this data interface:
sudo nmcli connection up uuid RECORDED_PREVIOUS_UUID
```

Replace placeholders with the recorded values. Restore one affected data
connection at a time and verify management and data addresses. Do not restore
the entire archive over an active system or remove all NetworkManager profiles.
