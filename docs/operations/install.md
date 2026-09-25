# Install SparkRing

**Development:** one-command installation on a configured four-Spark ring,
automatic recovery after a failed switch, and profile switches on the shared
image are hardware-tested. Blank-host setup, reboot recovery and
cable-reordering tests remain pending. See the
[acceptance record](../development/installer-acceptance.md).

Choose any Spark as Node A. Connect its 10GbE port to your network. Connect a
pair with p0↔p0, or a four-Spark ring with each p0 connected to the next p1.
The nodes can be in any order; setup discovers ranks from the cables.

On Node A:

```bash
sudo apt install ./sparkring_<version>_arm64.deb
sudo sparkring install
```

Choose an exact model profile when prompted. The command discovers/configures
the cluster on first use, updates workers from Node A's package, reuses cached
images and weights, and copies missing assets over verified fabric paths. It
prepares assets before stopping the previous managed model. A failed switch
attempts recovery from the retained deployment and records the outcome.

For an LLM or a repeatable installation:

```bash
sudo sparkring install --profile qwen38-flash-next-qad-tp4 --plan --json
sudo sparkring install --profile qwen38-flash-next-qad-tp4 --yes --json
sudo sparkring status --refresh --json
```

`--json` writes one result to stdout; progress stays on stderr and in the log.
Exit codes are 0 for success/planning, 3 for missing input, and 2 for failure.
`needs_input` identifies the required choice, such as profile, approval or
storage. `--yes` approves model replacement and first-use setup; it does not
trust unknown SSH keys or authorize stopping unrelated workloads. A configured
ring is inspected without changing links. Use `sparkring setup` to review cable
or network changes separately. `sparkring models` lists the exact profiles.

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

Every installer profile runs on one shared serving image, pinned by the
[installer image lock](../../runtime/releases/dev-20260924-cuda1342-nccl2323-status031/installer-image.json)
(`sparkring-installer-image/v2`). It is the ARM64
`ghcr.io/fujitsupolycom/sparkring:dev-20260924-cuda1342-nccl2323-status031`
image: eugr's `spark-vllm-b12x` nightly base with CUDA 13.4.2, NCCL 2.32.3 and the
runtime-status dashboard 0.3.1. The lock lists the admitted profiles and pins the
image configuration, registry manifest, external software receipt, toolchain
receipt, composition, prepared transport and status package. `sparkring models`
marks only these profiles as installer-supported:

| Profile | Status |
|---|---|
| `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2` | Qwen3.8 Flash Next; TP4 installed and served on this image |
| `glm53-flash-nvfp4-spark-tp4`, `glm53-flash-nvfp4-spark-tp2` | GLM-5.3-Flash NVFP4-Spark with MTP3; see each profile's evidence scope |
| `mimo-v26-flash-rl-tp4`, `mimo-v26-flash-rl-tp2` | MiMo-V2.6-Flash-RL with DFlash5; see each profile's evidence scope |

Each profile states its own evidence scope; published qualifications of other
images do not transfer. All run with SparkCache off and vLLM's native prefix
cache on. Profiles tied to earlier per-release images keep their guides and are
not installed by `sparkring install`. `--image-lock FILE` replaces the shared
lock for a development rehearsal and must list the selected profile.

The external image's B12X checkpoint loader requires `io_uring`. Its container
uses the [pinned loader policy](../../third_party/moby_seccomp/README.md), which
adds only the three `io_uring` calls to the Moby 29.2.1 default profile. A CPU-only
probe checks them during asset preparation, before the previous model stops.
Docker daemon defaults and host kernel policy are unchanged.

After creating each stopped container, the installer supplies a read-only
`sparkring-runtime-binding/v1` file with deployment, node, container, image and
rank identities. It checks the file before starting the model and never rewrites
it beneath a running container. This is an installer assertion for compatible
dashboard consumers; boot identity and observation times remain independently
observed. It is not attestation or serving qualification.

For a private image without a reachable registry, set `image_reference` to its
exact `image_id` and load it on one enrolled Spark. The installer finds that
copy and streams it to missing peers without creating another export archive.
A registry-backed image is downloaded once on Node A. Source/image/profile
choices automatically select a separate deployment and compilation cache;
there is no instance name or manual stop command to supply.
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

Before changing anything, `sparkring install` confirms noninteractive SSH and
`sudo` on every enrolled Spark. A missing grant returns `needs_input` with field
`access` and, per Spark, the one-time command a person runs there; it prompts
for that Spark's password. SparkRing never accepts passwords as options or
settings.

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

Run `sudo sparkring install --env /path/to/settings.env` for first-use setup.
Values are parsed as
literal settings, never sourced as shell code. SSH handles credential prompts.
`--reset-links` requests reviewed replacement of fabric IPv4 settings.
`--plan` discovers/reviews through existing access without configuring hosts.

## What is installed

The Debian package contains the CLI, host services, profile/deployment code,
an immutable source bundle and a file manifest. It contains **no model weights,
CUDA stack or inference image**. [Images](images.md) supply the serving software;
profile adapters still own image admission and model startup.
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
verification again. A missing checkpoint is copied from a verified peer, or
downloaded once on Node A if none has it. Copies are checksum-verified before
launch. Mismatched or corrupt unowned directories are not overwritten.
`--model-path /absolute/checkpoint` selects a cache explicitly when it is stored
elsewhere. This avoids downloading weights already present on the ranks.
`--cache-path /absolute/cache` chooses the writable compilation cache. Image
imports and checkpoint copies check storage before the model switch; insufficient
space leaves the old model running. The installer does not delete model weights
or unrelated archives to make room.
Profiles with a `SHA256SUMS` file pin every checkpoint file. A reused copy whose
files differ from those pins is synchronized in place from the pinned hub
revision, which downloads only the differing files, then verified again.
Hub-style folders named `<owner>--<name>` are found with or without a revision
subfolder; metadata hashes decide a match, not folder names.
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

## Lower-level commands and Compose sharing

`sparkring install` is the supported entry point. The underlying steps remain
available for inspection and rehearsals:

```bash
python3 scripts/sparkring.py init --model glm53 --host spark0 --host spark1
python3 scripts/sparkring.py up            # review the stages
python3 scripts/sparkring.py up --execute
python3 scripts/sparkring.py status --refresh
python3 scripts/sparkring.py down --execute
```

`init` discovers addresses read-only and saves `.sparkring/deployment` with the
shared installer image. `--model` accepts `glm53`, `mimo26` or `qwen38`; host
count selects the profile. For an offline site, fill in
[the site example](../../profiles/install-site.example.json) and pass
`--site YOUR_FILE` instead of `--host`. No live command runs without `--execute`
except discovery and status refresh.

`sparkring export --share --output profile-template.zip` writes a portable
template: the pinned profile, the image lock, an example site and per-rank
Compose files. It excludes private addresses, paths, receipts and source
bundles. `sparkring export --output private-deployment.zip` keeps the actual
configuration. Compose is the container format, not a multi-host scheduler:
each host runs its own rank, and Compose alone does not configure RDMA.

Validate a Compose file locally, without a Docker daemon, images or GPUs:

```bash
python3 scripts/sparkring.py validate-compose compose.yaml
python3 scripts/sparkring.py validate-compose --all --output .sparkring/compose-validation.json
```

It checks profile identity and settings, resolves every rank with example
inputs, tests missing-variable guards and simulates rank registration. A pass
does not establish GPU/RDMA or inference behavior. [Compose](compose.md) covers
the standalone recipes.
