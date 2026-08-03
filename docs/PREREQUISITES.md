# SparkRing prerequisites

This page is the exhaustive pre-model checklist for a four-node SparkRing
deployment. Complete it before downloading GLM-5.2 or building the runtime.

SparkRing does not require an Ethernet or InfiniBand switch for inference.
It does require a separate management path and a correctly configured
four-cable 200 GbE cycle.

## Physical and management topology

```text
                 operator or bot
                       |
          Wi-Fi / LAN / USB / Tailscale
             |      |      |      |
             S0     S1     S2     S3

                   200 Gb/s
              S0 ========== S1
              ||             ||
     200 Gb/s ||             || 200 Gb/s
              ||             ||
              S3 ========== S2
                   200 Gb/s

       Management: SSH, downloads, launch, API
       200 GbE ring: RDMA inference traffic only
```

The four required fabric edges are:

```text
S0 <-> S1
S1 <-> S2
S2 <-> S3
S3 <-> S0
```

Ranks 0 and 2 are not directly connected. Ranks 1 and 3 are not directly
connected. Do not add or assume those fabric paths.

## Hardware

- Four NVIDIA DGX Sparks with their ConnectX-7 200 GbE ports available.
- Four qualified 200 GbE DACs, one for each edge above.
- Both 200 GbE ports detected and link-capable on every Spark.
- A separate management connection on every Spark: wired LAN, Wi-Fi, USB
  Ethernet, or Tailscale.
- Adequate cooling and power for four sustained full-load systems.
- About 450 GB of usable storage per rank for the checkpoint, MTP draft,
  images, caches, and working headroom.
- At least 60 GB free in both Docker storage and the bootstrap cache on rank 0;
  the builder enforces this minimum.
- Additional rank-0 space for the temporary image archive used to fan one
  exact image ID to ranks 1-3.

## Operating system and container runtime

Every rank needs:

- a supported DGX OS/Ubuntu installation for DGX Spark;
- a working NVIDIA driver and CUDA-capable GB10 device;
- Docker Engine;
- NVIDIA Container Toolkit configured for Docker;
- permission for the deployment account to run Docker noninteractively;
- `git`, Python 3, `bash`, `ssh`, `scp`, `curl`, `tar`, and standard GNU/Linux
  utilities, including `rsync` and `sha256sum` on **all four ranks**;
- Python's `yaml` module (`python3 -c 'import yaml'` must succeed);
- working DNS, system time synchronization, and internet access during
  download/build.

Basic checks:

```bash
nvidia-smi
docker version
docker info
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:13.0.2-base-ubuntu24.04 \
  nvidia-smi
python3 -c 'import yaml; print(yaml.__version__)'
rsync --version
sha256sum --version
```

Do not continue until a CUDA container can see the GPU on every rank.

## Management and SSH

The management network carries SSH orchestration, Internet downloads, process
launch, rendezvous/bootstrap traffic, and the API. Multi-gigabyte image
archives do **not** cross management: the public bootstrap moves them over the
direct 200 GbE ring tree.

Required:

- stable management IP addresses or resolvable hostnames for all four ranks;
- `sshd` running on every rank;
- one known deployment username per rank;
- passwordless public-key SSH from the operator machine to rank 0;
- passwordless public-key SSH from rank 0 to ranks 1-3;
- noninteractive Docker access on every rank;
- `sudo` access when initial networking, package, or service repair is needed;
- no credentials, passwords, or tokens committed to `site.yaml`.

After filling the management targets in `site.yaml`, check them:

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope bootstrap

python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope image-fanout
```

With explicit operator approval, the verifier can install missing public-key
edges:

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope bootstrap \
  --fix

python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope image-fanout \
  --fix
```

`--fix` distributes public keys only. It does not discover passwords, bypass
authentication, or repair an unreachable management network.

The bootstrap checks both scopes. `bootstrap` covers orchestration: the
controller must reach every management `ssh_target`, and rank 0 must reach its
own configured management target plus ranks 1-3. `image-fanout` covers the
exact bulk-data tree: rank 0 sends to both direct neighbors in parallel, then
the lower-ID neighbor relays to the opposite rank. Management SSH starts and
attests these operations, but archive bytes travel only on direct-ring
addresses. Use `--scope all-adjacent` only to audit every optional direction.

Run `--fix` from a trusted operator/controller that already has authenticated
management SSH to all four ranks. Rank 0 can be that controller only if it
already has the same access. After this scope passes, copy the untracked
`site.yaml` to an exact checkout of the same commit on rank 0; the model
bootstrap itself runs there.

## 200 GbE and RoCE

Each physical cable must have:

- one unique point-to-point IPv4 subnet; use four distinct `/24` subnets;
- one configured address on each endpoint;
- MTU 9000 end to end;
- a negotiated speed of 200,000 Mb/s;
- an ACTIVE RDMA port with Ethernet link layer;
- a RoCEv2 GID whose IPv4-mapped value matches that interface's ring address;
- no default route, DNS, or ordinary management workload;
- no requirement for IP forwarding, bridging, or intermediate-node routing.

Do not assume interface names, RDMA-device names, cage labels, or GID index 3.
Discover and verify the values on the actual machines.

Useful read-only commands:

```bash
ip -br link
ip -o -4 addr show
ip route
rdma link show
ls -l /sys/class/infiniband/*/device/net/
cat /sys/class/net/<interface>/speed
cat /sys/class/net/<interface>/mtu
ls /sys/class/infiniband/<device>/ports/
```

Inspect the GID table:

```bash
dev=<rdma-device>
port=<rdma-port>
for i in $(seq 0 15); do
  printf '%2s %s %s\n' "$i" \
    "$(cat /sys/class/infiniband/$dev/ports/$port/gids/$i)" \
    "$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/types/$i)"
done
```

To discover actual cable neighbors, temporarily assign unique test addresses
to all eight fabric interfaces and use interface-bound pings. Build the edge
map from the pairs that answer. Do not trust printed cage labels.

The detailed netplan, address, GID, MTU, and cable-qualification procedure is
in [SETUP.md](SETUP.md#stage-1--hardware-cabling).

## Network and firewall

- Rank 0 must expose the configured API port to intended clients.
- The management network must permit the configured SSH and master/rendezvous
  ports.
- The host network namespace must permit SparkRing's generated control ports.
- Direct neighbors must exchange RoCEv2/RDMA traffic on their dedicated
  fabric subnets.
- No unrelated process may already own the API, master, or generated control
  ports.
- NetworkManager/netplan must not automatically remove the fabric addresses,
  lower MTU, or add default routes to the 200 GbE interfaces.

The current example uses API port `8000` and master port `29500`. Treat
`site.yaml` and the generated launch plan as authoritative for a real site.

## Storage and downloads

- Choose an absolute model directory and JIT-cache directory on every rank.
- The same logical paths should exist and be writable on all ranks.
- Verify free space in the model filesystem, Docker root, and bootstrap cache.
- Ensure GitHub, Hugging Face, and the pinned container registry are reachable.
- Supply a Hugging Face token only if the pinned repositories require one.
- Interrupted downloads are safe to resume with the same bootstrap command.
- Existing model/draft files are reused only after their pinned hashes pass.
- Existing images are reused only when their receipts and image identity pass.

Useful checks:

```bash
df -h
docker info --format '{{.DockerRootDir}}'
docker system df
```

Do not run broad Docker cleanup commands on a shared host merely to satisfy
space checks. Resolve exact targets with the operator.

## What the operator must provide

A bot cannot safely infer:

- usernames, passwords, SSH private keys, or other credentials;
- which physical Spark the operator wants to call rank 0;
- permission to install packages, alter networking, reboot, or stop workloads;
- whether moving/reseating a cable is physically safe;
- storage-retention policy for existing models and images.

Give the bot:

- four reachable management SSH targets;
- the desired rank numbering, or permission to assign it;
- permission boundaries for `sudo`, network changes, and stopping containers;
- any required Hugging Face authentication through a secure mechanism.

## What the bot can discover

After management SSH works, a bot can determine:

| Site field | Discovery source |
|---|---|
| management interface/address | `ip -o -4 addr show`, `ip route` |
| two 200 GbE interfaces | `ip -br link`, `/sys/class/net/*/speed` |
| netdev-to-RDMA mapping | `rdma link show`, `/sys/class/infiniband/*/device/net/` |
| RDMA port number | `/sys/class/infiniband/<device>/ports/` |
| RoCEv2 GID index | GID values plus `gid_attrs/types/<i>` |
| physical cable neighbors | temporary unique addresses plus bound pings |
| MTU/link state/speed | sysfs, `ip`, and preflight |
| model/Docker/cache free space | `df`, `docker info`, `docker system df` |

The commands are also embedded beside each field in
[`site.example.yaml`](../scripts/config/site.example.yaml) and summarized in
[`scripts/config/README.md`](../scripts/config/README.md#deriving-the-non-obvious-values).

## Required validation sequence

From a clean SparkRing checkout on a trusted operator/controller:

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
```

Then run, in order:

```bash
# 1. Offline schema/topology validation.
python scripts/sparkring_site.py scripts/config/site.yaml

# 2. Read-only SSH-path validation.
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope bootstrap
```

If repair is required, run the same command with `--fix` here. The controller
must already authenticate to all four ranks. After the SSH scope passes, put
the exact same commit and this untracked `site.yaml` on rank 0, then continue
there:

```bash

# 3. Show and run the image-independent fabric/RDMA gate.
python scripts/preflight.py \
  --site scripts/config/site.yaml \
  --scope fabric \
  --print-plan

python scripts/preflight.py \
  --site scripts/config/site.yaml \
  --scope fabric

# 4. Read-only model/bootstrap plan after the live fabric gate passes.
python scripts/bootstrap_nf3.py plan \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8
```

The `fabric` scope is fail-closed for SSH, management identity, ring
link/speed/MTU/address, RDMA state/link layer, RoCEv2 GID, jumbo DF path, and
peer reachability. It omits image, artifact, disk-path, and final-port checks.
The default full scope needs the exact image ID and generated artifact identity,
which do not exist before the first build; do not require it to pass against
the unresolved input site. The bootstrap repeats `--scope fabric` before any
model download or image build.

After steps 1-4 pass, the safest first mutation is preparation without launch:

```bash
python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8 \
  --no-launch \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

This downloads/verifies the model, builds and fans out the exact image, writes
`.sparkring/bootstrap/site.yaml`, and runs the full read-only preflight against
that resolved site. It does not start the model.

## Ready-to-bootstrap checklist

- [ ] Four DGX Sparks are assigned ranks 0-3.
- [ ] Four 200 GbE cables form exactly `0-1-2-3-0`.
- [ ] Management connectivity remains available if all fabric links are down.
- [ ] Passwordless management SSH works.
- [ ] Docker and NVIDIA Container Toolkit pass on every rank.
- [ ] All eight fabric interfaces report 200 Gb/s and MTU 9000.
- [ ] All eight RDMA mappings and RoCEv2 GIDs are known.
- [ ] Four unique point-to-point fabric subnets are configured.
- [ ] Required ports are free.
- [ ] Model, Docker, bootstrap-cache, and archive space are sufficient.
- [ ] `sparkring_site.py` passes.
- [ ] `verify_ssh_mesh.py` passes.
- [ ] The offline preflight plan has been reviewed.
- [ ] The NVFP4/FP8-RoPE bootstrap plan is reviewed.
- [ ] `bootstrap_nf3.py execute --no-launch` passes, including its resolved
      full preflight.

Once every box is true, continue with
[the four-Spark NF3 quickstart](QUICKSTART.md#5-build-verify-distribute-and-launch).
