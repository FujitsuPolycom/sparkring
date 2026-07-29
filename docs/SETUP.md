# SparkRing Bring-Up Guide

**GLM-5.2 on 4x NVIDIA DGX Spark (GB10), tensor-parallel 4, switchless direct-cable 200GbE RoCE ring.**

This is a reconstruction of the complete reference deployment, plus the
publicly runnable subset: cabling, OS prerequisites, fabric network, patched
NCCL, model download, and native transport build/probes. It is **not** a
fresh-clone end-to-end public serving recipe; the launch/runtime gaps are
called out below. Follow any runnable stages **in order** because later stages
hard-depend on earlier ones.

---

## 0. Read this first

> **Snapshot scope — what this tree can and cannot execute.**
>
> This public snapshot does **not** include the launch/orchestration layer. The root `scripts/` orchestrator (`run-glm52-graph-window.ps1` and friends), the serving entrypoint `serve-glm52-trace.sh` (which applies the in-container vLLM source patches), and `glm52_load_format_preflight.py` all live in the maintainer's private archive, because they are coupled to a private vLLM fork build (Section 0.1).
>
> Concretely: **Stages 1-4, 6, and 7 are fully executable from this tree. Stages 5, 8, and 9 document the deployed system for transparency** and require either the maintainer's artifacts or your own adaptation to your vLLM build. Those stages are kept — marked, not deleted — so the full deployed procedure stays on record. Any path in this guide marked *(private archive, not in this snapshot)* is not in this repository.
>
> **SparkCache exception:** the complete current Python implementation, native
> placement source, GPU-free tests, and its two independently written
> upstream-vLLM compatibility patches are published under `sparkcache/` and
> `runtime/patches/vllm/`. Those patches apply fail-closed to the official
> pinned vLLM commit. This narrows the private-runtime gap for SparkCache; it
> does not publish the broader GLM/SM121 serving overlay or orchestration.

### 0.1 What was actually deployed (honesty statement)

1. **The deployed runtime was a private vLLM fork build, not stock vLLM.** The vLLM inside the production container identifies itself as:

   ```
   0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626
   ```

   That version string is pinned verbatim by the SparkRing overlay's fail-closed installers (`spark_transport/experiments/adaptive_mtp_controller/runtime_installer.py`, `spark_transport/experiments/q2r_phase_timing/live_installer.py`, and siblings). A sibling image in the same family measured `0.21.1rc1.dev339+g1967a5627bc3`. Neither is an upstream release. The fork's source and commit history are **not** part of this repository.

2. **The overlay pins upstream interfaces by SHA-256 and fails closed.** SparkRing does not vendor vLLM; it patches and wraps whatever vLLM is installed in the container. To protect itself, every adapter checks the exact vLLM version string and the SHA-256 of each upstream source region it touches (e.g. `vllm.v1.engine.core`, the scheduler modules, `sample_tokens`), and **refuses to install on any mismatch**. Practical consequence for you: if you run a stock upstream vLLM, the overlay will (correctly) refuse to load until you either obtain the same fork lineage or port the pinned interfaces and re-pin the hashes yourself.

3. **End-to-end clean-room reproduction from this snapshot has not been verified.** This guide was reconstructed from the working cluster's repository (scripts, launchers, orchestrator attestation payloads, and deliverable reports). Nobody has yet taken this document alone, on fresh hardware, and reached a serving cluster. Treat it as the best available map, not a proven recipe.

### 0.2 DOCUMENTED vs INFERRED

Every step in this guide carries one of two tags:

- **[DOCUMENTED]** — stated explicitly in the source repository (script, README, or deliverable report). The source file is named. Sources prefixed **private archive** are files in the maintainer's private working repository (Appendix C) and are **not in this snapshot**; unprefixed paths exist in this tree.
- **[INFERRED — unverified, reconstructed from configuration]** — not written down anywhere as a procedure; reconstructed from code, pinned hashes, or attested launch state. These steps are plausible and consistent with the evidence, but **verify them yourself before relying on them**. Appendix B lists all of them in one place.

### 0.3 Security and credentials

An early revision of the cluster's provisioning notes contained a local account credential (since removed). Do not repeat that mistake: complete Ubuntu first-boot setup **interactively**, choose your own username and password, and **never commit credentials, SSIDs, or passphrases** to any repository. Keep `sudo` password-protected; if unattended automation needs `sudo`, grant only the exact commands it needs (never `NOPASSWD:ALL`).

### 0.4 How to use this guide

- Do the stages strictly in order. Do not skip a "Verify before continuing" block — each one gates the next stage.
- Every command is copy-paste ready **after** you substitute your values for the `<angle-bracket>` placeholders defined in Section 1.
- Fresh builds of the patched NCCL and the transport library produce **new SHA-256 hashes**. The launcher and orchestrator pin artifact hashes and refuse to run on mismatch — this is by design. Whenever you build an artifact, record its hash and substitute it everywhere this guide says to.

Safety classes used by this repository:

| Stages / command | Safety class |
|---|---|
| Local validation, source inspection, and local builds | **OFFLINE** |
| `scripts/preflight.py --print-plan` | **OFFLINE** |
| `scripts/preflight.py` against a filled site config | **READ-ONLY REMOTE** |
| Stages 1-4 and 6-7 | **MUTATES HOST** — power, packages, networking, large downloads, builds, or staged files |
| Reference stages 5, 8, and 9 | **STOPS SERVING** and unavailable without an independently supplied runtime/launcher |

Do not let an automation agent run **MUTATES HOST** or **STOPS SERVING**
commands without explicit authorization for the named machines and action.

---

## 1. Placeholders

Substitute these everywhere they appear. **No real hostnames, usernames, IPs, or account-scoped image names appear in this guide** — everything site-specific is a placeholder.

| Placeholder | What it is | Example shape |
|---|---|---|
| `<node0>`..`<node3>` | Hostnames of the four DGX Sparks, in **ring rank order** 0..3 | `spark-a4f2` |
| `<user0>`..`<user3>` | Linux username on each node (they may differ per node) | `mladmin` |
| `<MGMT_IP_0>`..`<MGMT_IP_3>` | Each node's **management-network** IP (LAN / Wi-Fi / Tailscale). Never a fabric IP. | `10.0.5.21` |
| `<HEAD_MGMT_IP>` | Rank 0's management IP. Used as `HEAD_IP`, `--master-addr`, and the API base. | = `<MGMT_IP_0>` |
| `<MGMT_IFNAME>` | The management NIC name on the nodes (the reference cluster used its Wi-Fi interface) | `wlP9s9` |
| `<SUBNET_01>` `<SUBNET_12>` `<SUBNET_23>` `<SUBNET_30>` | Four **distinct** /24 subnet prefixes, one per cable (see Stage 1.3). Must stay distinct /24s — the patched NCCL matches peers by /24. | `10.10.1` / `10.10.2` / `10.10.3` / `10.10.4` |
| `<CAGE0_IP_n>` / `<CAGE1_IP_n>` | Node *n*'s fabric IP on its cage-0 link / cage-1 link (derived from the subnet table in Stage 1.3) | `10.10.1.10` |
| `<PEER0_IP_n>` / `<PEER1_IP_n>` | The **neighbor's** fabric IP reachable from node *n* via cage 0 / cage 1 | `10.10.1.11` |
| `<BASE_IMAGE>` | The community-built GB10 "sparkrun" vLLM Docker image for DGX Spark, `production-hybrid-1.3` lineage (related to the `eugr/spark-vllm-docker` project per `THIRD_PARTY_NOTICES.md`). Ask the maintainer for the exact public tag. | `<registry-account>/sparkrun-vllm-...:production-hybrid-1.3` |
| `<SERVING_IMAGE>` | Your locally derived serving image: `<BASE_IMAGE>` + pinned `instanttensor==0.1.9` wheel (Stage 5.2) | `glm52-serving:local` |
| `<MODEL_REPO>` | The GLM-5.2 MXFP4-experts GPTQ hybrid checkpoint (~382 GiB), public on Hugging Face: **`aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`**. Identity check: its `config.json` SHA-256 is `ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69`. | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ` |
| `<MODEL_DIR_n>` | Node *n*'s local model directory | `/home/<usern>/.cache/huggingface/glm52-hybrid` |
| `<JIT_CACHE_n>` | Node *n*'s JIT/compile cache directory | `/home/<usern>/glm-jit-cache` |
| `<SPARKRING_REPO>` | Path to your checkout of the SparkRing source tree (contains `spark_transport/`; the root `scripts/` orchestration directory referenced later is private archive, not in this snapshot) | `~/sparkring` |
| `<NCCL_SO_SHA256>` | SHA-256 of **your** patched NCCL build (Stage 4) | 64 hex chars |
| `<TRANSPORT_SO_SHA256>` | SHA-256 of **your** `libspark_transport_capi.so` build (Stage 7) | 64 hex chars |
| `<TRACE_SOURCE>` | The staged, manifested source-bundle directory on each node (Stage 7) | `/tmp/spark-vllm-bundle-<date>/source` |
| `<RANK>` | The node's ring rank, 0..3 | `0` |

Ring adjacency reference (used throughout):

| Rank | Cage-0 neighbor (round 0, `rocep1s0f0`) | Cage-1 neighbor (round 1, `rocep1s0f1`) |
|---|---|---|
| 0 | rank 1 (link 0-1) | rank 3 (link 0-3) |
| 1 | rank 0 (link 0-1) | rank 2 (link 1-2) |
| 2 | rank 3 (link 2-3) | rank 1 (link 1-2) |
| 3 | rank 2 (link 2-3) | rank 0 (link 0-3) |

---

## Stage 1 — Hardware cabling

### 1.1 What each node has **[DOCUMENTED: private archive new-node-provisioning.md §5; private archive APPROACH.md]**

Each DGX Spark (GB10) has one dual-cage ConnectX-7 complex:

- Cage 0 = netdev `enp1s0f0np0` / RDMA device `rocep1s0f0`
- Cage 1 = netdev `enp1s0f1np1` / RDMA device `rocep1s0f1`

Each 200G cage is internally a *pair* of 100G MACs on PCIe 5.0 x4. Alias interfaces named `enP2*` also appear — **never assign addresses to them** (they cause duplicate-subnet errors).

Parts: **four QSFP28 200GbE DAC cables** total. The public cable qualifier can
also test a direct 10GbE link, and SparkCache documents a proposed diagonal
replication carrier, but the reference cluster's diagonal topology/recovery
tooling remains private. The 10GbE diagonals are not part of the serving
fabric or the public-functional matrix.

### 1.2 Cable plan — the 4-cycle **[DOCUMENTED: private archive deliverables/fabric-inventory.md; spark_transport/README.md]**

Cable a ring where **every cable joins the same cage index on both ends** (cage0-to-cage0, cage1-to-cage1):

```
rank0 <node0>  cage0 <==DAC==> cage0  <node1> rank1     (link "0-1")
rank1 <node1>  cage1 <==DAC==> cage1  <node2> rank2     (link "1-2")
rank2 <node2>  cage0 <==DAC==> cage0  <node3> rank3     (link "2-3")
rank3 <node3>  cage1 <==DAC==> cage1  <node0> rank0     (link "0-3")
```

This yields the two simultaneous perfect-matching rounds the custom transport schedules:

```
round 0 (port f0 / cage 0 / rocep1s0f0):  0 <-> 1   and   2 <-> 3
round 1 (port f1 / cage 1 / rocep1s0f1):  0 <-> 3   and   1 <-> 2
```

**Do not trust cage labels.** The reference cluster's original plan assumed a cage0-to-cage1 pairing and was wrong; the real topology was discovered *empirically* by assigning a unique temporary IP to every cage on every node and pinging all pairs. Do the same discovery after cabling (Stage 3.1 note) and correct your cable map to match reality.

### 1.3 Per-link addressing plan **[DOCUMENTED: private archive fabric-inventory.md and scripts/netplan-template.sh]**

One dedicated point-to-point **/24 per physical cable**. Do **not** use 169.254 link-local addressing (RoCEv1 link-local GIDs were tried and rejected — see Stage 3.2). The subnets are yours to choose but **must be four distinct /24s**:

| Link | Round/port | Subnet | End A | End B |
|---|---|---|---|---|
| 0-1 | f0 | `<SUBNET_01>.0/24` | rank0 = `<SUBNET_01>.10` | rank1 = `<SUBNET_01>.11` |
| 1-2 | f1 | `<SUBNET_12>.0/24` | rank1 = `<SUBNET_12>.10` | rank2 = `<SUBNET_12>.11` |
| 2-3 | f0 | `<SUBNET_23>.0/24` | rank2 = `<SUBNET_23>.10` | rank3 = `<SUBNET_23>.11` |
| 0-3 | f1 | `<SUBNET_30>.0/24` | rank0 = `<SUBNET_30>.10` | rank3 = `<SUBNET_30>.11` |

From this table, each node *n* gets exactly two fabric IPs: `<CAGE0_IP_n>` (its address on its f0-round link) and `<CAGE1_IP_n>` (its address on its f1-round link), and two peer IPs `<PEER0_IP_n>` / `<PEER1_IP_n>` (the other end of each cable).

Keep the management network (LAN/Wi-Fi/Tailscale) **completely separate** — it carries SSH, NCCL bootstrap, and Gloo control traffic only. Never mix it with the RoCE data plane.

### 1.4 Physical install **[DOCUMENTED: private archive new-node-provisioning.md §5]**

Per node:

```bash
sudo shutdown now
# seat the QSFP28 DAC cable(s) in the cage(s) per the plan above, then power on
```

> **Verify before continuing (per node)**
>
> ```bash
> ibdev2netdev
> sudo ethtool enp1s0f0np0 | grep Speed
> sudo ethtool enp1s0f1np1 | grep Speed
> ```
>
> Expected: `ibdev2netdev` lists `rocep1s0f0 ... enp1s0f0np0 (Up)` and `rocep1s0f1 ... enp1s0f1np1 (Up)`; both `Speed:` lines read exactly `200000Mb/s`. If a link is down or slower, see Troubleshooting T1.

---

## Stage 2 — OS / driver / CUDA prerequisites (per node)

All steps **[DOCUMENTED: private archive new-node-provisioning.md §§1-4 and fabric-inventory.md]** unless tagged otherwise.

1. **First boot.** Complete the Ubuntu (DGX OS) first-boot setup interactively; set your own credentials; join the management network. (Deliberately generic — see Section 0.3.)

2. **Verify the baseline.**

   ```bash
   nvidia-smi
   uname -a
   df -h /
   free -h
   ```

   Expected: GB10 GPU, driver **580.x** (reference cluster: 580.173.02), **CUDA 13.0**; Ubuntu 24.04 **aarch64** with NVIDIA kernel **6.17**; ~3.7 TB NVMe root; ~120-121 GiB unified memory. The GPU architecture is **sm_121 / sm_121a** (the containers set `CUTE_DSL_ARCH=sm_121a` and `TORCH_CUDA_ARCH_LIST=12.1a`).

3. **Hostname.**

   ```bash
   sudo hostnamectl set-hostname <nodeN>
   ```

4. **SSH mesh.**

   ```bash
   ssh-keygen -t ed25519
   ssh-copy-id <userM>@<MGMT_IP_M>   # run from each node/host, once per peer  [INFERRED: standard procedure]
   ```

   Exchange public keys bidirectionally among all four nodes **and** with the control host that will run the orchestrator (Stage 8). Verify passwordless `ssh <userN>@<MGMT_IP_N>` from everywhere that needs it.

5. **System packages.**

   ```bash
   sudo apt-get update && sudo apt-get install -y \
     libnccl2 libnccl-dev libopenmpi-dev openmpi-bin \
     ethtool iproute2 rdma-core ibverbs-utils
   ```

6. **Docker + NVIDIA runtime.** Install Docker CE from the official apt repo (Ubuntu 24.04 arm64) **[INFERRED: standard procedure]**:

   ```bash
   sudo apt-get update && sudo apt-get install -y ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
     https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update && sudo apt-get install -y \
     docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

   Then `nvidia-container-toolkit` from NVIDIA's official apt repo **[INFERRED: standard procedure]**:

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   ```

   Then wire the runtime into Docker:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   sudo usermod -aG docker <userN>   # log out/in afterwards
   ```

7. **Host-side build tools** for the transport library: CMake >= 3.24 and the verbs headers (from `rdma-core`/`ibverbs-utils` above) must exist on the DGX host. Ubuntu 24.04's packaged CMake satisfies this: `sudo apt-get install -y cmake` **[INFERRED: standard procedure]**. `nvcc` is NOT needed on the host — it comes from the container (Stage 7). **[DOCUMENTED: spark_transport/README.md "Build"]**

8. **Control host.** **[INFERRED — unverified, reconstructed from configuration]** The reference cluster drove orchestration from a Windows machine with PowerShell 5.1, an OpenSSH client, and Python 3 (the `scripts/*.ps1` orchestrators — private archive, not in this snapshot — run there and shell into the nodes over SSH). Any host that can run PowerShell and SSH to all four nodes should work.

> **Verify before continuing (per node)**
>
> ```bash
> strings /usr/lib/aarch64-linux-gnu/libnccl.so.2 | grep "^2\."
> ibdev2netdev | wc -l
> docker run --rm --gpus all ubuntu:24.04 nvidia-smi
> ```
>
> Expected: system NCCL version >= **2.30.4**; `ibdev2netdev` shows **4** RoCE devices (2 cages x 2 MACs each); the Docker GPU test prints the same `nvidia-smi` table as the host.

---

## Stage 3 — Fabric network configuration

### 3.1 Netplan (per node) **[DOCUMENTED: private archive scripts/netplan-template.sh and new-node-provisioning.md §6]**

Create `/etc/netplan/40-cx7-ring.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [<CAGE0_IP_n>/24]   # this node's IP on its f0-round link
      dhcp4: no
      mtu: 9000
    enp1s0f1np1:
      addresses: [<CAGE1_IP_n>/24]   # this node's IP on its f1-round link
      dhcp4: no
      mtu: 9000
```

```bash
sudo chmod 600 /etc/netplan/40-cx7-ring.yaml
sudo netplan apply
```

Do **not** assign IPs to the `enP2*` alias interfaces.

**Topology discovery note (do this once, now):** if you have any doubt which cage is cabled to which (you should — the reference cluster's labels were wrong), temporarily give every cage on every node a unique address in a scratch subnet, `ping -I` from every interface to every other address, record which pairs answer, and only then commit the Stage 1.3 addressing to the *actual* topology. **[DOCUMENTED: private archive fabric-inventory.md "Actual Cable Topology (empirically discovered)"]**

> **Verify before continuing (per node)**
>
> ```bash
> ip link show enp1s0f0np0 | grep mtu
> ip link show enp1s0f1np1 | grep mtu
> ping -c 3 -I enp1s0f0np0 <PEER0_IP_n>
> ping -c 3 -I enp1s0f1np1 <PEER1_IP_n>
> ```
>
> Expected: both MTUs are **9000**; both pings report **0% packet loss** with sub-millisecond times.

### 3.2 RoCE / GID contract **[DOCUMENTED: private archive fabric-inventory.md "GID Index" and phase2-nccl-ring-findings.md; spark_transport/CABLE_QUALIFICATION.md]**

Use **GID index 3 (IPv4 RoCEv2)** bound to the netdev, everywhere a GID index is asked for. GID 0 (link-local RoCEv1) is **proven wrong** on this topology: packets go to the physical neighbor regardless of destination address. Every tool and env var in this guide that takes a GID uses `3`.

### 3.3 sysctl / iptables — you need none of it **[DOCUMENTED: spark_transport/ROUTED_QSFP_NCCL_BOOTSTRAP.md]**

The production switchless path requires **no** IP forwarding, rp_filter, or iptables changes. There is an optional legacy "routed-QSFP NCCL Socket bootstrap" (a diagnostic fallback, not the production transport) that needs `net.ipv4.ip_forward=1`, `rp_filter=2` on both fabric NICs, and one tagged FORWARD rule per rank; it is managed reboot-recoverably by `scripts/routed_qsfp_nccl_bootstrap.py` (private archive, not in this snapshot). Skip it unless you are debugging.

### 3.4 Per-edge cable qualification — before ANY model work **[DOCUMENTED: spark_transport/CABLE_QUALIFICATION.md; private archive new-node-provisioning.md §8; private archive APPROACH.md Phase 1]**

You will need the `spark_transport_probe` binary from Stage 7. It is fine (and normal) to jump ahead, build only the probes, and come back — cable qualification must pass before NCCL or model work begins.

**Breaking the circular dependency:** Stage 7's documented build wrapper runs inside the Stage 5 serving image, which this snapshot cannot hand you (see Section 0 snapshot scope) — but the probes do not need it. They build in **any CUDA 13.x devel image** targeting sm_121. Sanctioned alternative for probe-building, using the same image Stage 4.2 already uses:

```bash
docker run --rm \
  -v <SPARKRING_REPO>/spark_transport:/src -v <SPARKRING_REPO>/build:/build \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -lc 'apt-get update && apt-get install -y cmake libibverbs-dev &&
    cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 &&
    cmake --build /build --parallel --target spark_transport_probe'
```

(Drop `--target spark_transport_probe` to build every probe and the full library the same way.)

Per edge (all four cables):

1. Install the **identical** probe binary at `/tmp/spark_transport_probe` on both ends (SHA-256 equality is enforced by the harness).
2. Run the qualifier from a host that can SSH to both ends:

   ```bash
   python3 spark_transport/scripts/qualify_direct_cable.py \
     --tier roce200 \
     --left  <userA>@<MGMT_IP_A> --left-interface  enp1s0f0np0 \
     --right <userB>@<MGMT_IP_B> --right-interface enp1s0f0np0 \
     --left-ip <CAGE0_IP_A> --right-ip <CAGE0_IP_B> \
     --left-rdma-device rocep1s0f0 --right-rdma-device rocep1s0f0 \
     --gid-index 3 --expected-mtu 9000 \
     --probe-binary /tmp/spark_transport_probe \
     --iterations 10000 --strict-latency \
     --output results/cable-A-B.json
   ```

   (Substitute the correct IP, interface, and RDMA device per the link's round:
   f0 links use the cage-0 IP, `enp1s0f0np0`, and `rocep1s0f0`; f1 links use
   the cage-1 IP, `enp1s0f1np1`, and `rocep1s0f1`.)

   Gates enforced: exactly 200,000 Mb/s on both ports; expected IP and route; active RDMA ports; GID 3 = RoCE v2; bidirectional verified RC writes at 12,288 and 16,384 bytes; zero new PHY/CRC error counters; p99 latency <= 20 us.

3. Bandwidth check, both directions per edge:

   ```bash
   ib_write_bw -d rocep1s0f0                    # server side of the edge
   ib_write_bw -d rocep1s0f0 <PEER0_IP_n>       # client side, run on the neighbor  [INFERRED: standard procedure]
   ```

   Target: **>= 180 Gb/s per link**. Also confirm zero errors/drops in `ip -s link show enp1s0f0np0` (and the f1 twin).

4. Save the qualifier's JSON result per edge. **Re-run qualification after any cable reseat, firmware, MTU, GID, or driver change.**

> **Verify before continuing**
>
> All four edges have a passing qualification JSON, >= 180 Gb/s each direction, and zero error counters. Do not proceed to Stage 4 without this.

---

## Stage 4 — Build and install the patched switchless NCCL

### 4.1 Why stock NCCL cannot work here **[DOCUMENTED: private archive phase2-nccl-ring-findings.md; spark_transport/experiments/nccl_switchless_ring/README.md]**

Stock NCCL creates Tree/PAT channels between **all** rank pairs at init. On a switchless ring, the non-adjacent pairs (0-2 and 1-3) cannot form RDMA queue pairs — RoCEv2 connection management needs L2 adjacency — so init fails or silently degrades. Two patches fix this, in the `josephdrose/nccl-spark-switchless` lineage (see `THIRD_PARTY_NOTICES.md`):

1. `nccl-2.30.7-skip-tree-pat.patch` — never create Tree/PAT connections (ring-feasible neighbor pairs only), gated by `NCCL_SKIP_TREE_CONNECT=1`.
2. `nccl-2.30.7-advertise-all-listener-gids.patch` — SparkRing's NET/IB fix: the listener advertises the GIDs of BOTH eligible local RoCE devices, so the subnet-aware connector (`NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_SUBNET_PREFIX_LEN=24`) picks the device whose /24 contains the peer. No packet ever transits an intermediate Spark.

Both patch files live at `spark_transport/experiments/nccl_switchless_ring/`.

### 4.2 Pinned build **[DOCUMENTED: nccl_switchless_ring/README.md]**

On any of the nodes (arm64, Docker working):

```bash
git clone --branch v2.30.7-1 --depth 1 https://github.com/NVIDIA/nccl.git nccl-switchless-2.30.7
cd nccl-switchless-2.30.7
git apply <SPARKRING_REPO>/spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-skip-tree-pat.patch
git apply <SPARKRING_REPO>/spark_transport/experiments/nccl_switchless_ring/nccl-2.30.7-advertise-all-listener-gids.patch
docker run --rm --gpus all -v "$PWD:/src" -w /src \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -lc 'make -j"$(nproc)" src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"'
```

### 4.3 Install and pin **[DOCUMENTED: private archive scripts/run-glm52-graph-window.ps1 and scripts/launch-glm52-trace-4node.sh]**

1. Compute your build's hash and record it as `<NCCL_SO_SHA256>`:

   ```bash
   sha256sum build/lib/libnccl.so.2.30.7
   ```

2. Stage the library at the **same path on all four nodes**:

   ```bash
   scp build/lib/libnccl.so.2.30.7 <userN>@<MGMT_IP_N>:/tmp/libnccl-spark-switchless-2.30.7-gidfix.so
   ```

3. **Re-pin the hash everywhere it is enforced.** The reference cluster's binary hashed to `106150aebf7ef9d997f4dcab5edea13082d4bc72fd55b1160a030dbc05c60202` — **yours will differ**, and the stack fails closed on artifact identity. Substitute `<NCCL_SO_SHA256>` in:
   - the launcher env `VLLM_SPARK_SWITCHLESS_NCCL_SHA256` (Stage 8.3), and
   - the orchestrator's `$switchlessNcclSha256` variable in `scripts/run-glm52-graph-window.ps1` (private archive, not in this snapshot — skip if you are not using the orchestrator).

   At launch, the library is bind-mounted **read-only** at `/opt/sparkring/libnccl-switchless.so.2`; the orchestrator attests the mount, checksum, command line, environment, and image on every rank, and any mismatch is a hard stop.

> **Verify before continuing**
>
> ```bash
> for h in <user0>@<MGMT_IP_0> <user1>@<MGMT_IP_1> <user2>@<MGMT_IP_2> <user3>@<MGMT_IP_3>; do
>   ssh "$h" sha256sum /tmp/libnccl-spark-switchless-2.30.7-gidfix.so
> done
> ```
>
> Expected: the **same** `<NCCL_SO_SHA256>` on all four nodes.

---

## Stage 5 — Container image and the SparkRing overlay

> **Snapshot scope (see Section 0):** this stage documents the deployed system for transparency. The base image contains the **private vLLM fork build** (Section 0.1), and the launch-time overlay mechanism in Stage 5.3 depends on the serve entrypoint `serve-glm52-trace.sh` and `glm52_load_format_preflight.py` — both private archive, not in this snapshot. Executing Stage 5 requires either the maintainer's artifacts or your own adaptation to your vLLM build.

### 5.1 Base image **[DOCUMENTED: private archive new-node-provisioning.md §4, scripts/download-model.sh, and phase2-nccl-ring-findings.md]**

On every node:

```bash
docker pull <BASE_IMAGE>
```

Known contents of the base image:

- Python 3.12 venv at `/opt/venv`; vLLM installed at `/opt/venv/lib/python3.12/site-packages/vllm`, served via `/opt/venv/bin/vllm`.
- The **private B12X-patched vLLM fork** described in Section 0.1: it exposes `--attention-backend B12X_MLA_SPARSE`, B12X MoE (`VLLM_USE_B12X_MOE`), the B12X sparse indexer, the `nvfp4_ds_mla` KV dtype, `--decode-context-parallel-size` / `--dcp-comm-backend ag_rs`, MTP speculative config, and the `glm45`/`glm47` parsers. The in-container `vllm.__version__` must match the string in Section 0.1 or the overlay will refuse to install.
- NCCL **2.30.4** and a broken v9 "mesh" NCCL plugin at `/tmp/nccl-mesh-plugin/libnccl-net.so` (API-incompatible; unused — Stage 4's library replaces NCCL via `LD_PRELOAD`).
- `huggingface_hub` (used for the model download in Stage 6).

### 5.2 Derived serving image **[image identities DOCUMENTED: private archive deliverables/glm52-instanttensor-mmap-acceptance-gate.md and glm52-instanttensor-loader-optin.md; construction INFERRED — unverified, reconstructed from configuration]**

The production image is the base image plus **one layer** adding the pinned ARM64 `instanttensor==0.1.9` wheel (wheel SHA-256 `3c59b24f1f636932bc74b819a16fdd3bbf9f9b2d038d97ff97279d8592f823f4`). The repository pins the resulting image identity (base-config SHA-256 `bb3e87c5b74aaca6214cdee5161b1eab789e8ce73944fd165d550a2339ac90ff`, derived-config `4e60945927c3d435b06819fd75c9e1340f06da89b224b90fa26c861d7bb0cde7`, added layer `16b0523fdb79c209ee004ec7407107a041a3045331866ce3e12ad16cfbff22c7`) **but does not contain the Dockerfile**. The reconstruction:

```dockerfile
# Dockerfile.serving  [INFERRED — unverified, reconstructed from configuration]
FROM <BASE_IMAGE>
# install the pinned instanttensor 0.1.9 ARM64 wheel; verify its SHA-256 first
COPY instanttensor-0.1.9-*.whl /tmp/
RUN /opt/venv/bin/pip install /tmp/instanttensor-0.1.9-*.whl
```

```bash
sha256sum instanttensor-0.1.9-*.whl   # must be 3c59b24f...f823f4
docker build -f Dockerfile.serving -t <SERVING_IMAGE> .
```

Build or load `<SERVING_IMAGE>` on **all four nodes**. Note: per the acceptance-gate deliverable, whole-image IDs may legitimately differ across nodes (timestamp-only dummy layers in the pre-existing base images); what must match are the derived runtime configuration and added-layer hashes, the wheel hash, and the runtime preflight — do not gate on image-ID equality across nodes.

**Important:** the default and only correctness-proven load format is **safetensors**. InstantTensor direct-AIO loading is *disabled* after an MTP-acceptance-collapse failure — the wheel is opt-in machinery only. Leave `VLLM_SPARK_LOAD_FORMAT=safetensors`. **[DOCUMENTED: private archive README.md "Loader checkpoint" and CURRENT_STATUS.md]**

### 5.3 How SparkRing code enters the container **[DOCUMENTED: private archive scripts/launch-glm52-trace-4node.sh and serve-glm52-trace.sh; spark_transport/integrations/vllm/README.md]**

No image rebuild carries SparkRing code. Two mechanisms, applied per rank at launch:

1. **Read-only bind mounts:** the SHA-256-manifested source bundle (`<TRACE_SOURCE>`, contents = `spark_transport/integrations/vllm/*.py` plus experiment packages) mounts at `/opt/spark-vllm:ro`; the serve script at `/opt/spark/serve-glm52-trace.sh:ro`; a load-format preflight at `/opt/spark/glm52_load_format_preflight.py:ro`; the transport library at `/opt/spark-transport/libspark_transport_capi.so:ro`; the patched NCCL at `/opt/sparkring/libnccl-switchless.so.2:ro`.
2. **In-place source patching at container start:** the serve script's embedded `replace_once` Python patches the *installed* vLLM before anything imports it — multiproc-executor follower `collective_rpc` fix; `kv_cache_utils` empty-config guard; the B12X DCP1 logical-to-physical top-k remap (Triton kernel injected into `sparse_attn_indexer.py` + `deepseek_v2.py`); the shared-capture-stream patch (`parallel_state.py` + `cudagraph_utils.py`, gated by `VLLM_SPARK_SHARED_CAPTURE_STREAM=1`); capture-size synthesis in `config/compilation.py`; and the FULL-graph whole-request dispatch guard in `cudagraph_dispatcher.py`. **Patches are idempotent and refuse unexpected source** (this is where the pinned-interface fail-closed behavior from Section 0.1 bites stock-vLLM users). Then `PYTHONPATH=/opt/spark-vllm` puts the bundle's `sitecustomize.py` first, which installs the SparkRing adapters (custom all-reduce, generic all-gather, vocabulary all-gather, DCP query/combine, flight recorder, graph status reporter) according to the `VLLM_SPARK_*` env flags, before `exec vllm serve`.

> **Verify before continuing (per node)**
>
> ```bash
> docker run --rm --entrypoint bash <SERVING_IMAGE> -c \
>   '/opt/venv/bin/python -c "import vllm, instanttensor; print(vllm.__version__)" && /opt/venv/bin/pip show instanttensor | grep Version'
> ```
>
> Expected: the fork version string from Section 0.1, and `Version: 0.1.9`. If the vLLM version differs, stop — the overlay will fail closed at launch (see Section 0.1 and Troubleshooting T6).

---

## Stage 6 — Model download and placement

**[DOCUMENTED: private archive new-node-provisioning.md §7, scripts/download-model.sh, and scripts/download_model.py; spark_transport/README.md "GLM checkpoint precision"]**

Checkpoint: `<MODEL_REPO>` = **`aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`** at immutable revision `<MODEL_REVISION>` = **`46537e0e16fcd156627800139b41b9c497fc7ee2`**, public on Hugging Face, ~**382 GiB**. Declared precision: BF16 model/output dtype, MXFP4 routed experts, FP8 attention + shared expert, runtime NVFP4 MLA KV cache. Verify identity with both the revision and `config.json` SHA-256 pins below.

1. Create the directories on every node:

   ```bash
   mkdir -p <MODEL_DIR_n> <JIT_CACHE_n>
   ```

2. Download **once**, on one node, inside the container (the image ships `huggingface_hub`):

   ```bash
   docker run --rm -v /home/<usern>/.cache/huggingface:/root/.cache/huggingface \
     --entrypoint bash <BASE_IMAGE> \
      -c "python3 -c \"from huggingface_hub import snapshot_download; \
          print(snapshot_download('<MODEL_REPO>', revision='<MODEL_REVISION>', \
          local_dir='/root/.cache/huggingface/glm52-hybrid'))\""
   ```

3. **rsync the tree to the other three nodes over the QSFP fabric** (the reference cluster measured roughly 6 seconds per 150 GB at 200G) rather than re-downloading:

   ```bash
   rsync -a --info=progress2 <MODEL_DIR_src>/ <userN>@<PEER_FABRIC_IP>:<MODEL_DIR_n>/
   ```

   **Diagonal-node gap:** node0 has fabric adjacency to ranks 1 and 3 only (Stage 1.2), so there is no direct fabric path from node0 to rank 2. Either copy to rank 2 over the management network (`rsync` to `<MGMT_IP_2>` — slower, but one hop), or relay in two fabric hops (node0 -> node1 over link 0-1, then node1 -> node2 over link 1-2).

4. On every rank, write the immutable identity sidecars consumed by the public
   acceptance gate:

   ```bash
   printf '%s\n' '<MODEL_REPO>' > <MODEL_DIR_n>/.sparkring-model-repository
   printf '%s\n' '<MODEL_REVISION>' > <MODEL_DIR_n>/.sparkring-model-revision
   ```

5. Placement contract: every rank bind-mounts its local copy read-only at `/hybridmodel`; the MTP draft model is the checkpoint's `mtp-draft/` subdirectory (referenced as `/hybridmodel/mtp-draft` in the speculative config). With that mount, `scripts/config/gate.example.json` verifies `/hybridmodel/config.json` plus both sidecars independently on every rank.

> **Verify before continuing (per node)**
>
> ```bash
> du -sh <MODEL_DIR_n>
> sha256sum <MODEL_DIR_n>/config.json
> cat <MODEL_DIR_n>/.sparkring-model-repository
> cat <MODEL_DIR_n>/.sparkring-model-revision
> ls <MODEL_DIR_n>/mtp-draft/
> ```
>
> Expected: ~382 GiB; `config.json` SHA-256 exactly `ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69` and **identical on all four nodes**; the sidecars print the repository and 40-hex revision above; `mtp-draft/` exists and is non-empty.

---

## Stage 7 — Build `libspark_transport_capi.so` and the probe binaries

**[DOCUMENTED: spark_transport/README.md "Build"; spark_transport/CMakeLists.txt]**

### 7.1 Build

Toolchain split: the DGX host has CMake and the verbs headers; `nvcc` comes from the GLM container. The documented invocation uses in-container paths `/src` and `/build`, which implies the build ran inside the serving image with the source tree mounted. The exact `docker run` wrapper is **[INFERRED — unverified, reconstructed from configuration]**; the following is consistent with the documented paths. If you do not have `<SERVING_IMAGE>` (see the Stage 5 snapshot-scope note), the Stage 3.4 `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` build works for the full library and all probes — this stage is fully executable from this tree either way.

```bash
docker run --rm --gpus all \
  -v <SPARKRING_REPO>/spark_transport:/src -v <SPARKRING_REPO>/build:/build \
  --entrypoint bash <SERVING_IMAGE> -lc '
    cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 &&
    cmake --build /build --parallel &&
    ctest --test-dir /build --output-on-failure'
```

Products: the shared library `libspark_transport_capi.so` (verbs endpoints, TP4 two-matching schedule, DCP query/combine, vocab all-gather, graph command ring) and the probe executables: `spark_transport_probe` (used in Stage 3.4), `spark_tp2_probe`, `spark_tp4_probe`, `spark_tp4_tensor_probe`, `spark_tp4_graph_q1_probe`, `spark_tp4_vocab_graph_probe`, `spark_tp4_dcp_graph_probe`, `spark_tp4_dcp_sequence67_probe`, and others. The CTest suite (statistics, wire-protocol, topology, layout, C-API tests) must pass.

To run the Python test suite from the repository root (`python -m pytest spark_transport -q`), install `pytest`, `numpy`, and `torch` first — a CPU-only torch wheel suffices; none of these ship in the stock CUDA devel image. **[DOCUMENTED: verified during the 2026-07-27 clean-room build — 20/20 native tests and 821 Python tests passed from this tree in `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` on DGX Spark hardware.]** `librdmacm-dev` is not required; `cmake` and `libibverbs-dev` are sufficient.

### 7.2 Staging contract **[DOCUMENTED: private archive scripts/run-glm52-graph-window.ps1 `$artifacts`]**

Stage identical, versioned copies of every artifact on **all four nodes** and record every SHA-256; the orchestrator verifies hash and executability per rank before any mutation. The production layout (your version suffixes and hashes will differ — that is expected and enforced):

- `/tmp/libspark_transport_capi-<yourver>.so` — record `<TRANSPORT_SO_SHA256>`
- `/tmp/spark_tp4_vocab_graph_probe-<yourver>`, `/tmp/spark_tp4_graph_q1_probe-<yourver>`, `/tmp/spark_tp4_dcp_graph_probe-<yourver>` (the DCP probe only for the custom-DCP arm)
- `/tmp/serve-glm52-<yourver>.sh` (the serve entrypoint, `serve-glm52-trace.sh` lineage), `/tmp/launch-glm52-<yourver>.sh` (the per-rank launcher, `scripts/launch-glm52-trace-4node.sh` lineage), `/tmp/glm52_load_format_preflight-<yourver>.py` — all three source scripts are private archive, not in this snapshot (see Section 0 snapshot scope); only the launch/serve stage needs them
- the source bundle `<TRACE_SOURCE>` with its sibling `manifest.sha256` (per-file SHA-256 plus a manifest hash, verified on every rank and locally)
- `/tmp/libnccl-spark-switchless-2.30.7-gidfix.so` (Stage 4)

**Wire compatibility rule:** every rank in a session must run the **same** transport-library hash. Newer Q512-capable doorbell packing is wire-incompatible with older builds. **[DOCUMENTED: private archive HANDOFF.md]**

> **Verify before continuing**
>
> ```bash
> # ctest summary from 7.1 shows 100% tests passed, and on every node:
> for h in <user0>@<MGMT_IP_0> <user1>@<MGMT_IP_1> <user2>@<MGMT_IP_2> <user3>@<MGMT_IP_3>; do
>   ssh "$h" 'sha256sum /tmp/libspark_transport_capi-*.so /tmp/spark_tp4_*probe* 2>/dev/null'
> done
> ```
>
> Expected: identical hashes for every artifact across all four nodes, matching your recorded values.

---

## Stage 8 — Launch

> **Snapshot scope (see Section 0):** every launch-layer file this stage documents — the orchestrator `scripts/run-glm52-graph-window.ps1` and its siblings, the per-rank launcher `scripts/launch-glm52-trace-4node.sh`, the serve entrypoint `serve-glm52-trace.sh` (which applies the in-container vLLM source patches), and `glm52_load_format_preflight.py` — is **private archive, not in this snapshot**, because it is coupled to the private vLLM fork build. Stages 8 and 9 document the deployed launch and verification procedure for transparency; executing them requires the maintainer's artifacts or your own equivalents adapted to your vLLM build.

### 8.1 The orchestrator (recommended path) **[DOCUMENTED: private archive scripts/run-glm52-graph-window.ps1 and scripts/start-glm52-graph-window-detached.ps1]**

First edit the rank table at the top of `scripts/run-glm52-graph-window.ps1` to map each rank to `<userN>@<MGMT_IP_N>`, and re-pin every artifact hash (Stages 4 and 7). Then, from the control host:

```powershell
# read-only preflight (verifies SSH, artifacts, hashes, API idle) — run this first, always
powershell -ExecutionPolicy Bypass -File scripts/run-glm52-graph-window.ps1 -Mode Preflight

# execute the production DCP4 window (RC1-style switchless arm)
powershell -ExecutionPolicy Bypass -File scripts/run-glm52-graph-window.ps1 `
  -Mode Execute -Confirmation STOP-GLM52-TRACE-ON-ALL-FOUR `
  -DcpSize 4 -NcclTransportMode switchless_ib `
  -MtpTokens 4 -MtpMode adaptive-2-4 -MaxNumSeqs 8 -KvScaleMode per-token
```

Every mutation requires the exact confirmation string `STOP-GLM52-TRACE-ON-ALL-FOUR`. The execution sequence is fail-closed with a single-instance mutex and automatic container rollback:

preflight -> `docker stop glm52-trace` on all ranks -> **offline probe gates** (vocab stream-switch, mixed-Q all-reduce graph, DCP graph probe when custom DCP, vocab MTP4+MTP5 probes — all run model-down against the staged binaries and image) -> **model-down memory gate** (`MemAvailable >= 116,000,000 KiB` per rank) -> rename the old container as backup -> launch all four ranks in parallel over SSH (`env RANK=<n> ... bash /tmp/launch-...sh`) -> **runtime attestation** -> **live gate** (Stage 9). The API base is `http://<HEAD_MGMT_IP>:8210`.

**First bring-up recommendation [DOCUMENTED: private archive APPROACH.md phases; private archive serve script defaults]:** run eager mode first — set `VLLM_SPARK_ENABLE_CUDAGRAPH=0`, which makes the serve script pass `--enforce-eager` — and only enable CUDA graphs after eager serving is verified end-to-end. Likewise, run NCCL Socket-mode two-rank sanity tests before any four-rank switchless attempt. Be aware that eager bring-up changes the expected-environment attestation set (Stage 9.2) relative to what is documented here, and the orchestrator invocation shown above assumes graph mode — an eager window needs correspondingly adjusted attestation expectations.

### 8.2 Per-rank container invocation **[DOCUMENTED: private archive scripts/launch-glm52-trace-4node.sh]**

What the launcher runs on each node (for understanding and for manual bring-up):

```bash
docker run -d --name glm52-trace --network host --ipc host --shm-size 10gb \
  --gpus all --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v <MODEL_DIR_n>:/hybridmodel:ro \
  -v <JIT_CACHE_n>:/cache/jit \
  -v <TRACE_SOURCE>:/opt/spark-vllm:ro \
  -v /tmp/serve-glm52-<yourver>.sh:/opt/spark/serve-glm52-trace.sh:ro \
  -v /tmp/glm52_load_format_preflight-<yourver>.py:/opt/spark/glm52_load_format_preflight.py:ro \
  -v /tmp/libspark_transport_capi-<yourver>.so:/opt/spark-transport/libspark_transport_capi.so:ro \
  -v /tmp/libnccl-spark-switchless-2.30.7-gidfix.so:/opt/sparkring/libnccl-switchless.so.2:ro \
  <...the ~90 -e environment variables from 8.3 and 8.4...> \
  <SERVING_IMAGE> bash /opt/spark/serve-glm52-trace.sh
```

Note `--network host`: the API endpoint, master port (29501), and all transport control ports live in the **host** network namespace. The management NIC carries NCCL bootstrap and Gloo only — fabric payloads must appear exclusively on the RoCE devices.

### 8.3 NCCL environment (switchless_ib mode) **[DOCUMENTED: private archive launcher `nccl_args`; nccl_switchless_ring/README.md]**

```bash
LD_PRELOAD=/opt/sparkring/libnccl-switchless.so.2
VLLM_NCCL_SO_PATH=/opt/sparkring/libnccl-switchless.so.2
VLLM_SPARK_NCCL_TRANSPORT_MODE=switchless_ib
VLLM_SPARK_SWITCHLESS_NCCL_SHA256=<NCCL_SO_SHA256>
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_IB_GID_INDEX=3
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_IB_MERGE_NICS=0
NCCL_CROSS_NIC=1
NCCL_ALGO=Ring
NCCL_SKIP_TREE_CONNECT=1
NCCL_SOCKET_IFNAME=<MGMT_IFNAME>     # bootstrap/management only; payloads ride RoCE
NCCL_MAX_NCHANNELS=4
NCCL_MIN_NCHANNELS=4
NCCL_CUMEM_ENABLE=0
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
# NCCL_PROTO deliberately UNSET (the serve script unsets it in switchless mode)
# LD_LIBRARY_PATH, NCCL_P2P_LEVEL, NCCL_LOCAL_INFERENCE_PATH: explicitly cleared/empty
```

Fallback `socket` mode (diagnostics only) instead sets `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_PROTO=Simple`, `NCCL_SOCKET_NTHREADS=4`, and clears all IB/preload variables.

### 8.4 SparkRing / vLLM environment (attested per rank; RC1 DCP4 values) **[DOCUMENTED: private archive run-glm52-graph-window.ps1 `$expectedEnvironment` + launcher]**

Core:

```bash
MODEL_PATH=/hybridmodel
HEAD_IP=<HEAD_MGMT_IP>
PORT=8210
VLLM_SPARK_LOAD_FORMAT=safetensors
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DSV4_INDEXER_SP=1                  # 0 when DCP1
VLLM_B12X_MLA_CKV_GATHER=1
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=458752
VLLM_B12X_MLA_DECODE_SPARSE_GATHER=0
VLLM_B12X_MLA_DECODE_GATHER_V2=0
B12X_NSA_CONTIGUOUS_PREFILL_BLOCK_K=auto
VLLM_ENGINE_READY_TIMEOUT_S=3600
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CUTE_DSL_ARCH=sm_121a
TORCH_CUDA_ARCH_LIST=12.1a
XDG_CACHE_HOME=/cache/jit
GLOO_SOCKET_IFNAME=<MGMT_IFNAME>
```

MTP / speculative decode:

```bash
VLLM_SPARK_MTP_TOKENS=4
VLLM_SPARK_MTP_ADAPTIVE_WINDOW=32
VLLM_ADAPTIVE_SPEC_DEPTHS=2,4
VLLM_SPARK_MTP_MODE_ID=adaptive-mtp2-4-window32
SPARK_GLM52_MTP_INDEX_REUSE=1
# SPARK_ADAPTIVE_MTP_CONTROL / VLLM_SPARK_TRUE_ADAPTIVE_DRAFT: per experiment arm
SPARK_B12X_DCP1_PHYSICAL_REMAP=0        # MUST be 1 if and only if DCP size is 1
```

Sizing and CUDA graphs:

```bash
VLLM_SPARK_DCP_SIZE=4
VLLM_SPARK_MAX_MODEL_LEN=458752
VLLM_SPARK_KV_CACHE_MEMORY_BYTES=4000000000
VLLM_SPARK_MAX_NUM_BATCHED_TOKENS=4096
VLLM_SPARK_MAX_NUM_SEQS=8
VLLM_SPARK_MAX_QUERY_ROWS=40            # = seqs * (K+1); ABI cap Q40
VLLM_SPARK_ENABLE_CUDAGRAPH=1           # set 0 for first bring-up (eager)
VLLM_SPARK_SHARED_CAPTURE_STREAM=1
VLLM_SPARK_DECODE_CAPTURE_SIZES=1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40
# (equal to VLLM_SPARK_GRAPH_CAPTURE_SIZES unless the Q512 prefill opt-in adds
#  48,72,144,224,288,352,432,512 plus VLLM_SPARK_FULL_DECODE_CAPTURE_SIZES=5,10,15,20,25,30,35,40)
VLLM_SPARK_TP4_PREFILL_Q512=0
VLLM_SPARK_KV_SCALE_MODE=per-token
VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1        # present iff per-token; absent for legacy
```

Custom-transport selectors:

```bash
VLLM_SPARK_TP4_MODE=custom
VLLM_SPARK_TP4_GRAPH_Q1=1
VLLM_SPARK_TP4_ALLGATHER_MODE=custom
VLLM_SPARK_TP4_ALLGATHER_POLICY=spark-custom
VLLM_SPARK_TP4_VOCAB_MODE=custom
SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so
SPARK_TP4_MAX_INFLIGHT=64
# DCP family — custom arm only (RC1 keeps the DCP trio on patched NCCL-IB, i.e. these unset):
#   VLLM_SPARK_TP4_DCP_MODE=custom
#   VLLM_SPARK_TP4_DCP_QUERY_ENABLED=1 / VLLM_SPARK_TP4_DCP_COMBINE_ENABLED=1
#   VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM=1
#   SPARK_TP4_DCP_GRAPH_POLL_POLICY=adaptive-yield   # or dedicated-spin
SPARK_TP4_DCP_COLLECTIVE_AUDIT=1        # whenever DCP > 1
```

Note for reproducers: the README's sealed C1 single-stream decode headline (20.83 / 19.28 / 21.43 tok/s at 8K/16K/32K) was produced with the custom DCP trio **enabled** — the env vars shown commented-out above for RC1 — so that custom-DCP arm, not the RC1 configuration, is the one that produced the flagship number.

**Transport peer addressing — you MUST set this.** The integration backends
read per-rank peer IPs from `SPARK_TP4_PEER0` / `SPARK_TP4_PEER1`. Their
`_DEFAULT_PEERS` tables intentionally contain non-routable RFC 5737
documentation placeholders, not the reference cluster's addresses. They
cannot work on a live fabric. Set the overrides explicitly on every rank:

```bash
SPARK_TP4_PEER0=<PEER0_IP_n>    # neighbor IP reachable via SPARK_TP4_DEVICE0
SPARK_TP4_PEER1=<PEER1_IP_n>    # neighbor IP reachable via SPARK_TP4_DEVICE1
SPARK_TP4_DEVICE0=rocep1s0f0    # default
SPARK_TP4_DEVICE1=rocep1s0f1    # default
SPARK_TP4_GID0=3                # default
SPARK_TP4_GID1=3                # default
```

Control ports and CPU pinning (host-network TCP endpoint exchange; all attested; CPUs must be distinct and inside the container cpuset):

| Function | Ports | Pinned CPU(s) |
|---|---|---|
| TP graph | 9970 / 9971 | submit 10, progress 11 |
| Vocab graph | 10110 / 10111 | progress 12 |
| Vocab eager | 9990 / 9991 | - |
| DCP eager | 9890 / 9891 | - |
| DCP graph | 9892 / 9893 | progress 13 |
| Indexer graph | 9462 / 9463 | 14 |
| All-gather | base 9490 (slots +10) | - |

Optional Q2R probe instrumentation: `SPARK_Q2R_PROBE=1` plus session/manifest/config/image fingerprints (`SPARK_Q2R_SOURCE_BUNDLE_MANIFEST`, `SPARK_Q2R_CONFIG_SHA256`, `SPARK_Q2R_IMAGE_FINGERPRINT=<exact docker image ID>`, `SPARK_Q2R_RUNNER_SAMPLE_SHA256`); mutually exclusive with true-adaptive drafting.

### 8.5 The `vllm serve` command **[DOCUMENTED: private archive serve-glm52-trace.sh; attested in private archive run-glm52-graph-window.ps1]**

Identical on all ranks except where noted:

```bash
vllm serve /hybridmodel \
  --served-model-name glm-5.2 --host 0.0.0.0 --port 8210 \
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-auto-tool-choice --enable-prefix-caching \
  --load-format safetensors \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 4 --dcp-comm-backend ag_rs \
  --attention-backend B12X_MLA_SPARSE \
  --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  --kv-cache-dtype nvfp4_ds_mla --kv-cache-memory-bytes 4000000000 \
  --max-model-len 458752 --max-num-batched-tokens 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.89 \
  --distributed-timeout-seconds 3600 --cpu-distributed-timeout-seconds 3600 \
  --distributed-executor-backend mp \
  --nnodes 4 --node-rank <RANK> --master-addr <HEAD_MGMT_IP> --master-port 29501 \
  --speculative-config '{"model":"/hybridmodel/mtp-draft","method":"mtp","num_speculative_tokens":4,"draft_attention_backend":"B12X_MLA_SPARSE","adaptive_speculative_tokens_window":32}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40]}' \
  --headless        # ranks 1-3 ONLY — rank 0 omits this; it hosts the API and is --master-addr
```

Per-rank differences: `RANK` / `--node-rank` 0..3; `--headless` on ranks 1-3; per-rank `<MODEL_DIR_n>` / `<JIT_CACHE_n>` mount sources; per-rank graph-status file `/cache/jit/spark-graph-status-rank<N>.json`; per-rank `SPARK_Q2R_IMAGE_FINGERPRINT` (image IDs may differ across nodes; config and layer hashes must match).

> **Verify before continuing**
>
> ```bash
> docker ps --filter name=glm52-trace          # on each node: container Up
> curl -s -o /dev/null -w "%{http_code}\n" http://<HEAD_MGMT_IP>:8210/health
> ```
>
> Expected: all four containers running; `/health` eventually returns `200` (model load takes minutes; the validator polls with a 1200 s default timeout). Then complete ALL of Stage 9 before sending traffic.

---

## Stage 9 — Verification

### 9.1 Model-down transport probes (BEFORE serving) **[DOCUMENTED: spark_transport/experiments/nccl_switchless_ring/{probe_dcp4_collectives.py, probe_dcp4_entrypoint.sh, README.md}; private archive deliverables/glm52-dcp4-switchless-result-20260727.md]**

Run `probe_dcp4_collectives.py` via `probe_dcp4_entrypoint.sh` on all four ranks (the entrypoint refuses to start unless `NCCL_SKIP_TREE_CONNECT` and the checksum-pinned `LD_PRELOAD` are set). It creates the same communicator scopes vLLM will create (one 4-rank TP group plus DCP groups) and validates **exact values and rank-major layout** — not just latency — for: TP4 all-reduce `[6144]` BF16; owner top-k `[1,2,2048]` INT32; query `[Q,16,576]` BF16 at Q1/Q40/Q4096; LSE `[1,32]` FP32; output reduce-scatter; plus CUDA-graph capture and replay of the collectives.

Required log evidence: NCCL version `2.30.7+cudaXX`; `NET/IB` on the data path (**never** `NET/Socket`); the ring-only skip messages; subnet-aware device selection per cable; and `Connected all rings` on every rank.

The orchestrator additionally runs the **offline probe gates** while the model is down: vocab graph stream-switch probe (32 iterations), mixed-Q all-reduce graph probe (Q up to 512, multi-graph plus mixed-Q validation), DCP graph probe (custom arm), vocab MTP4 and MTP5 probes — each pinned to the documented CPUs/ports and hash-pinned binaries. Any nonzero exit aborts the window and rolls back.

### 9.2 Runtime attestation (after launch, before traffic) **[DOCUMENTED: private archive run-glm52-graph-window.ps1 `Invoke-RuntimeAttestation`]**

For every rank, the orchestrator reads `/proc/1/cmdline` and `/proc/1/environ` inside the container (retrying up to 120 x 2 s until `/opt/venv/bin/vllm serve` appears) and asserts EVERY expected argument and environment variable from Stage 8, the `--headless` contract, the exact speculative and compilation JSON, the container image ID, a container start time after the window opened, and (switchless mode) the read-only, checksum-pinned NCCL mount. The result is written as `runtime-attestation.json` evidence. Any mismatch is a hard stop.

### 9.3 Health and idleness **[DOCUMENTED: private archive run-glm52-graph-window.ps1 and scripts/validate_glm52_graph_live.py]**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://<HEAD_MGMT_IP>:8210/health   # 200
curl -s http://<HEAD_MGMT_IP>:8210/metrics | grep -E "num_requests_(running|waiting)"
```

Expected before a controlled window: `vllm:num_requests_running == 0` and `vllm:num_requests_waiting == 0`.

### 9.4 Expected startup log lines **[DOCUMENTED: private archive serve script echo, CURRENT_STATUS.md, and deliverables/glm52-release-candidate-1.md]**

In `docker logs glm52-trace` on each rank:

- serve entrypoint: one `patched <file>` line per source patch, then `starting traced GLM-5.2 rank=<N>; trace=... mtp_tokens=4 ... dcp_size=4 ...`;
- NCCL: `NCCL ... 2.30.7+cuda13.x`, `NET/IB`, `Connected all rings` on every rank; **zero** `NET/Socket` data-path lines;
- graph capture completion: **26/26 PIECEWISE and 16/16 FULL** captures (the RC1 DCP4 contract);
- model + MTP draft load completed on all four ranks; engine-reported KV pool of **500,224 logical tokens** (4 GB/rank, per-token scale; the earlier 3 GB/rank DCP4 config reported 375,040).

### 9.5 Graph-capture census and live request gate **[DOCUMENTED: private archive scripts/validate_glm52_graph_live.py and run-glm52-graph-window.ps1 `Invoke-LiveGate`]**

1. `validate_glm52_graph_live.py --discover-census` reads each rank's `<JIT_CACHE_n>/spark-graph-status-rank<N>.json` and requires a **rank-synchronous, positive** custom-node census. Reference values: DCP1/Q40 baseline = 5,464 custom all-reduce + 24 custom vocabulary + **zero** stock captures; DCP4/switchless = 6,744 custom all-reduce + 24 vocabulary + 2,904 attested stock NCCL-IB captures per rank. Stock captures are admitted ONLY with `--allow-attested-switchless-nccl-ib` after the Stage 4 attestation.
2. The live gate then issues a real request against `--base-url http://<HEAD_MGMT_IP>:8210` and requires the census counters to ADVANCE identically on all ranks (`published == consumed == completed`, zero overflow). For custom-DCP arms, `validate_glm52_dcp_graph_live.py` (private archive, not in this snapshot) must prove a before/after replay delta per collective family.
3. Post-gate sanity: the API stays healthy, and a sustained-decode benchmark cell (e.g. `llm-decode-bench`, concurrency 1 at 8K/16K/32K, 30 s active decode) should land near the published RC1 tables: **~19-21 tok/s** single-stream decode, **~856-873 tok/s** prefill.

### 9.6 Ongoing gates **[DOCUMENTED: private archive APPROACH.md "Quality and Stability Gates"]**

Repeated health checks plus clean stop/start; fixed smoke outputs; no NCCL/RDMA error-counter growth; no unbounded memory growth; per-edge requalification (Stage 3.4) after ANY physical change.

---

## Troubleshooting

### T1. A fabric link is down, or not 200G

- Reseat the DAC and re-check `ibdev2netdev` (both `rocep1s0f0`/`rocep1s0f1` must be Up) and `ethtool ... | grep Speed` (exactly `200000Mb/s`).
- **Do not trust cage labels.** Re-run the empirical discovery from Stage 3.1: unique temporary IP on every cage, ping all pairs, rebuild your cable map from what actually answers.
- Confirm no `enP2*` alias interface carries an address (duplicate-subnet routing silently breaks pings).
- Confirm MTU 9000 on both ends and that each ping is forced out the correct interface (`ping -I`).
- After ANY reseat or change: re-run the full Stage 3.4 qualification for that edge. Zero new PHY/CRC counters is a gate, not a suggestion.

### T2. NCCL falls back to Socket (`NET/Socket` in the logs)

A `NET/Socket` line on the data path means the IB path was not admitted — throughput collapses and the run is invalid. Check, in order:

1. `LD_PRELOAD=/opt/sparkring/libnccl-switchless.so.2` is set, the mount exists read-only, and its SHA-256 matches `<NCCL_SO_SHA256>` (the probe entrypoint and orchestrator refuse to run otherwise — if you launched manually, you bypassed that protection).
2. The log shows NCCL `2.30.7` (the patched build), not the image's stock 2.30.4.
3. `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`, `NCCL_IB_GID_INDEX=3`.
4. `NCCL_SKIP_TREE_CONNECT=1` is present — without it, stock Tree/PAT setup tries to connect non-adjacent pairs (0-2, 1-3), which can never work switchless, and init fails or degrades.
5. `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_IB_SUBNET_PREFIX_LEN=24`, and your four link subnets really are four **distinct** /24s (Stage 1.3). If two links share a /24, device selection is ambiguous and the connector picks wrong.
6. GID 3 on each device is actually IPv4 RoCEv2 bound to the netdev (check `/sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3`).
7. `NCCL_SOCKET_IFNAME=<MGMT_IFNAME>` refers to the management NIC — it is *supposed* to carry bootstrap traffic; only *data-path* `NET/Socket` transport lines are a failure.

### T3. `ibv_reg_mr` fails on `cudaMalloc` buffers

**Expected on GB10 — not a bug you can fix.** GPUDirect RDMA registration of `cudaMalloc` and `cudaMallocManaged` pointers fails on both Sparks; the failed registrations and mapped-memory measurements are recorded in [`spark_transport/README.md`](../spark_transport/README.md#results) and [RESULTS T1](RESULTS.md#3-transport-microbenchmarks). **The supported path is mapped host memory:** `cudaHostAlloc(cudaHostAllocMapped)` arenas registered with `ibv_reg_mr` (measured ~4.5-4.8 us GPU-to-GPU one-way; on this unified-memory part the "host" allocation is the same physical memory). The [SIRCL data path](SIRCL.md) uses this deliberately; do not attempt to force a GDR path, and leave `NCCL_CUMEM_ENABLE=0` as specified.

Related trap: **allocation failures while the model is loaded.** Once the model has committed the Spark's unified-memory budget, even ordinary `cudaMalloc` and `cudaHostAllocMapped` calls from a *separate* process fail. This is why all offline probes run model-down and why the orchestrator enforces the model-down memory gate (`MemAvailable >= 116,000,000 KiB` per rank). If a probe fails to allocate or register memory, check that no model container is running.

### T4. CUDA graph capture failures, or a census mismatch

- First bring-up should be **eager** (`VLLM_SPARK_ENABLE_CUDAGRAPH=0` -> `--enforce-eager`). Only enable graphs once eager serving passes Stage 9.3/9.4.
- The shared-capture-stream source patch must be active: `VLLM_SPARK_SHARED_CAPTURE_STREAM=1` and a corresponding `patched parallel_state.py` / `cudagraph_utils.py` line in the startup log. Capture across the custom collectives fails without it.
- Expect exactly **26/26 PIECEWISE + 16/16 FULL** captures per rank (RC1 DCP4). A shortfall on any rank means that rank's capture aborted — check its log for the first CUDA error, and confirm the capture-size lists in env and `--compilation-config` are identical on all ranks.
- The census in `spark-graph-status-rank<N>.json` must be **rank-synchronous** (identical counts on all four ranks) and positive. Stock NCCL-IB captures are only acceptable in switchless mode with the attestation flag; any *unattested* stock captures mean an adapter failed to install — check for the overlay's fail-closed refusal messages (see T6).
- Control-port collisions or CPU pins outside the container cpuset abort graph sessions at startup — the port/CPU table in Stage 8.4 must hold, with all pinned CPUs distinct.
- Any nonzero probe-gate exit in the orchestrator rolls the window back automatically — read the probe's own log before retrying.

### T5. MTP acceptance collapses (accepted-token rate far below the published tables)

Confirm `--load-format safetensors` / `VLLM_SPARK_LOAD_FORMAT=safetensors`. InstantTensor direct-AIO loading silently corrupts MTP acceptance and is disabled for that reason; safetensors is the only correctness-proven loader.

### T6. The overlay refuses to install / "unsupported vLLM version" / patch refuses unexpected source

This is the fail-closed design working (Section 0.1). The adapters and `replace_once` patches verified either the vLLM version string or a pinned SHA-256 of an upstream source region and found a mismatch. You are running a different vLLM than the pinned private fork build. Options: obtain the same fork image lineage; or port the pinned interfaces to your vLLM and re-pin the version string and hashes in the overlay sources. Do not weaken the checks — they are the only thing standing between you and silently wrong collectives.

### T7. Attestation or hash-pin failures at launch

Every artifact (NCCL library, transport library, probes, serve/launch scripts, source bundle) is hash-pinned per rank. A mismatch means either you rebuilt an artifact and forgot to re-pin (Stages 4.3, 7.2, and the orchestrator variables), or the ranks are not running identical artifacts (forbidden — see the wire-compatibility rule in Stage 7.2). Re-stage identical copies, re-record hashes, re-run `-Mode Preflight`.

### T8. DCP1-specific wrong output

`switchless_ib` is only admitted for validated TP4/DCP2 or TP4/DCP4 layouts. Running DCP1 requires acknowledging the B12X logical-to-physical top-k remap with `SPARK_B12X_DCP1_PHYSICAL_REMAP=1` (and `VLLM_DSV4_INDEXER_SP=0`). The launcher validates this pairing; do not override it.

---

## Appendix A — Sequencing constraints and traps

1. **Order matters:** cables + netplan + qualification (Stages 1-3) MUST precede any NCCL/model work; NCCL Socket-mode two-rank tests before four-rank; eager-mode serving before CUDA graphs.
2. `switchless_ib` is admitted only for validated TP4/DCP2 or TP4/DCP4 layouts (DCP1: see T8).
3. Every rank in a session must run the SAME transport-library hash (wire compatibility).
4. Load format stays `safetensors` (see T5).
5. The API endpoint, master port 29501, and all transport control ports ride the host network namespace; the management NIC carries NCCL bootstrap/Gloo only — fabric payloads must appear exclusively on the RoCE devices.
6. All orchestrator mutations require the exact confirmation string `STOP-GLM52-TRACE-ON-ALL-FOUR` and roll back automatically to the renamed backup container on any gate failure.
7. Fresh builds produce new SHA-256s; re-pin every enforced hash (Stages 4.3, 7.2).

## Appendix B — Unverified (INFERRED) items checklist

These are the steps this guide reconstructs from configuration rather than documented procedure. Verify before relying on them; corrections welcome.

| # | Item | Status |
|---|---|---|
| 1 | Control-host stack (Windows + PowerShell 5.1 + OpenSSH + Python 3) | Inferred from private archive `scripts/*.ps1` conventions |
| 2 | Dockerfile for the derived serving image (Stage 5.2) | Reconstructed; only the base-config/derived-config/added-layer/wheel hashes are documented |
| 3 | Exact `docker run` wrapper for the in-container transport build (Stage 7.1) | Reconstructed from the documented `/src`,`/build` CMake paths |
| 4 | The vLLM fork's source/commit | **Not available.** The version string and the overlay's per-module SHA-256 pins are documented; the fork itself is not in this repository |
| 5 | `SPARK_TP4_PEER0/1` semantics (Stage 8.4) | The env-var override path is documented in code and is mandatory in the public source because its defaults are non-routable placeholders. The historical reference launcher used its own private site mapping; the public explicit-override path has not completed end-to-end acceptance. |

Resolved since the first reconstruction (no longer inferred): the deployed vLLM fork **version string** is pinned verbatim in `spark_transport/experiments/adaptive_mtp_controller/runtime_installer.py` and `spark_transport/experiments/q2r_phase_timing/live_installer.py` (Section 0.1); the derived-image identity hashes are documented in `deliverables/glm52-instanttensor-mmap-acceptance-gate.md`.

## Appendix C — Source documents

The private working repository ("private archive") this guide was reconstructed from (paths relative to its root). The public snapshot now includes `spark_transport/`, `THIRD_PARTY_NOTICES.md`, and a clean-room `scripts/` orchestration layer. The historical scripts named below, `APPROACH.md`, `serve-glm52-trace.sh`, the deliverable reports, and the private archive's own `README.md`/`CURRENT_STATUS.md`/`HANDOFF.md` remain private-archive-only:

- `README.md`, `APPROACH.md`, `HANDOFF.md`, `CURRENT_STATUS.md`, `THIRD_PARTY_NOTICES.md`
- `new-node-provisioning.md` (provisioning; credential-bearing early history — nothing reproduced from old revisions)
- `phase2-nccl-ring-findings.md` (why stock NCCL fails switchless)
- `deliverables/fabric-inventory.md`, `deliverables/glm52-release-candidate-1.md`, `deliverables/glm52-dcp4-switchless-result-20260727.md`, `deliverables/glm52-instanttensor-loader-optin.md`, `deliverables/glm52-instanttensor-mmap-acceptance-gate.md`
- `scripts/run-glm52-graph-window.ps1`, `scripts/start-glm52-graph-window-detached.ps1`, `scripts/launch-glm52-trace-4node.sh`, `scripts/netplan-template.sh`, `scripts/download-model.sh`, `scripts/download_model.py`, `scripts/validate_glm52_graph_live.py`, `scripts/validate_glm52_dcp_graph_live.py`, `scripts/routed_qsfp_nccl_bootstrap.py`
- `serve-glm52-trace.sh` (the serving entrypoint), `glm52_load_format_preflight.py`
- `spark_transport/README.md`, `spark_transport/CMakeLists.txt`, `spark_transport/CABLE_QUALIFICATION.md`, `spark_transport/ROUTED_QSFP_NCCL_BOOTSTRAP.md`, `spark_transport/integrations/vllm/README.md`, `spark_transport/experiments/nccl_switchless_ring/` (patches, probes, README)
