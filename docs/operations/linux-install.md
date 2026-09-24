# Linux installation

**Development:** the package and control network have local/simulated tests.
DGX Spark installation, reboot recovery and serving qualification are pending.

Choose any Spark as Node A. Connect its 10GbE port to your network. Connect a
pair with p0↔p0, or a four-Spark ring with each p0 connected to the next p1.
The nodes can be in any order; setup discovers ranks from the cables.

On Node A:

```bash
sudo apt install ./sparkring_<version>_arm64.deb
sudo sparkring setup
sudo sparkring status
sparkring models
sudo sparkring up qwen38-flash-next-qad-tp4  # select a profile for your node count
```

Follow the installation from another terminal:

```bash
sudo sparkring logs --follow
```

Concise timestamped progress is appended to `/var/log/sparkring/install.log`.
Long steps report that they are still working. Verbose command output goes to
`install-details.log`; use `sparkring logs --details --follow` when investigating
an error. Credentials entered through SSH are not recorded.

Interactive terminals show colored results and a spinner. Saved logs and piped
output stay plain; `--plain` or `NO_COLOR=1` disables terminal decoration. From
Windows PowerShell, use `ssh -t spark-r0 "sudo sparkring logs --follow"` (replace
`spark-r0` with Node A's SSH address). The follower's spinner means it is waiting
for log lines; it does not indicate model readiness.

An explicit development image can accompany the existing Qwen profile:

```bash
sudo sparkring up qwen38-flash-next-qad-tp4 --instance candidate --image-lock image-lock.json --plan
```

The `sparkring-installer-image/v1` file pins the image configuration, optional
registry manifest, external software receipt, toolchain receipt, composition,
prepared transport and status package. This adapter supports the ARM64 CUDA
13.4.2/NCCL 2.32.3 external-image composition with status 0.3.x and SparkCache off.
It preserves the model profile and verifies the installed image contents before
launching through the image's toolchain entrypoint. Published release selections
stay intact; their qualification does not transfer to the development image.

For a private image without a reachable registry, set `image_reference` to its
exact `image_id` and preload it on every rank. The installer stops with a clear
message if it is absent. A different `--instance` gives the image its own
deployment and compilation cache. Preview works while the existing model runs;
execution requires completing `sparkring down` for that deployment first.
`sparkring export --share` retains the image lock and per-rank Compose files.

`sparkring status --json --refresh` reports the saved deployment/image IDs and
separate host and container observations. Host observations include persistent
node ID, boot ID and their own `observed_at`; cached observations retain their
original time and become stale after 90 seconds. Container observations include
the inspected container ID, start time and actual image ID. Missing identities
are `null`, with a reason, rather than guessed from hostname or rank. Network
observations do not establish model readiness or serving qualification.

Setup finds neighbors over IPv6 link-local addresses, asks for SSH login and host
key confirmation, copies SparkRing and its Debian dependencies through the fabric,
and shows the proposed network changes. Workers need no separate Ethernet cable
or Internet connection. The logged-in Spark becomes Node A.

Workers need SSH enabled and an existing root login or a login with sudo. Ubuntu
24.04 ARM64, NetworkManager, NVIDIA drivers, Docker and NVIDIA Container Toolkit
must already work on each Spark. Setup does not replace the host driver or
firmware. Node A and workers must use the same OS distribution/version.

Existing compatible fabric addresses are kept. Incompatible addresses trigger
a separate replacement confirmation. Setup saves backups and receipts. It keeps
the active NetworkManager connection identity and IPv6 address generation while changing
fabric IPv4/MTU settings, so its administration path survives renumbering.
Six-node and arbitrary port orientations are unsupported by this installer.

## If a worker has no SSH

A machine without remote access needs one local preparation step. On Node A:

```bash
sudo sparkring setup --worker-bundle
```

Copy the printed archive to a USB drive, extract it on each worker, then run:

```bash
sudo python3 install.py --apply --prepare
```

This installs the bundled packages and enables access with Node A's public key.
Return to Node A and run `sudo sparkring setup --ssh-port 2222`.
Private keys and passwords are not copied to workers. The preparation listener
is disabled after permanent administration access works on every node.

## Optional preferences

Interactive setup needs no `.env`. Repeated installations can use:

```dotenv
SPARKRING_NAME=home
SPARKRING_SSH_USER=cody
SPARKRING_SHARE_INTERNET=yes
SPARKRING_LINK_POLICY=keep
```

Run `sudo sparkring setup --env /path/to/settings.env`. Values are parsed as
literal settings, never sourced as shell code. SSH handles credential prompts.
`--reset-links` requests reviewed replacement of fabric IPv4 settings.
`--plan` discovers/reviews through existing access without configuring hosts.

## What is installed

The Debian package contains the CLI, host services, profile/deployment code,
an immutable source bundle and a file manifest. It contains **no model weights,
CUDA stack or inference image**. [Images](images.md) supply the serving software;
[profile adapters](installer.md) still own image admission and model startup.
The [standalone Compose files](compose.md) remain usable independently.

Workers use a private WireGuard administration tree over their existing IPv6
link-local fabric addresses. Only Node A's key is admitted by a separate SSH
service on that network. Optional Internet sharing routes downloads and DNS
through Node A. It does not move inference collectives into WireGuard.

Configuration is in `/etc/sparkring/`; private controller state and receipts are
in `/var/lib/sparkring/controller/`. `sparkring status` works from any directory.
It distinguishes cached/stale observations, network configuration and saved model
progress; `--refresh` contacts the enrolled nodes and observes the model containers.
Network configuration is not proof of working RDMA collectives or model accuracy.

Host services restore approved configuration after reboot. Models start only
when requested. `sparkring down` stops the selected deployment. Package removal
stops SparkRing host services but retains configuration, weights, caches, receipts
and already-installed network state; it does not stop a running model deployment.

`sparkring models` lists exact model/version/quantization/topology profiles and
marks which support automated installation. Family names such as `qwen` are
ambiguous and are rejected. Guide-only profiles remain listed with their guides.

Pair GLM/Qwen, managed GLM TP4 and Qwen TP4 use the existing deployment engines.
Qwen TP4 discovers and verifies an installed native mesh automatically. If none
exists, it prepares the pinned host marker, creates stopped model containers,
installs supervised mesh services, waits for every rank, and starts the model.
An unhealthy or partly installed mesh stops the plan for inspection.
`--fresh-mesh` prints an explicit replacement plan for an existing native mesh.
Use a separate `--instance fresh` when rehearsing this alongside an existing
deployment. Old container/source/weight directories are retained.

The installer checks existing container model mounts and standard model
directories for the selected checkpoint. A complete metadata match is proposed
for reuse, then every pinned shard is verified during preparation. On Linux,
later gates reuse that checksum receipt only while the complete file list, device,
inode, size, modification time and change time match; changes trigger checksum
verification again. Missing checkpoints
are downloaded; mismatched or corrupt caches are never overwritten silently.
`--model-path /absolute/checkpoint` selects a cache explicitly when it is stored
elsewhere. This avoids downloading weights already present on the ranks.
Arbitrary upstream images still need their own compatible transport adapter.

## Local build and tests

From a clean committed checkout on Linux:

```bash
python3 scripts/build_deb.py
python3 -m pytest runtime/host runtime/common/test_distribution.py -q
```

The builder produces an ARM64 `.deb` and SHA-256 file under `.sparkring/dist/`.
The package has no compiled host payload, so its assembly can run on x86 Linux.
An optional Linux namespace rehearsal exercises real WireGuard between two and
four simulated hosts, with no physical interfaces:

```bash
sudo env SPARKRING_LINUX_LAB=1 python3 -m pytest scripts/test_appliance_linux.py -q
```

If an execution receipt says `running` or `uncertain`, inspect it and host state
before recovery. Do not delete receipts to force a blind retry. Driver reloads
require `--allow-driver-reload`, stopped containers/GPU work, and no RDMA users.
