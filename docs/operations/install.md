# Install SparkRing

Status: **implemented**. Hardware runs of `sparkring install` cover
installation on a configured four-Spark ring, recovery of the previous
deployment after a failed model switch, and profile switches between installer
profiles. Installer package revision `eb8ec2ba3b17` installed each of the six
installer profiles once, on one pair and one four-Spark ring, on the parent
image `dev-20260925-cuda1342-nccl2323-status031`; its Qwen profiles pinned
checkpoint revision `60215d26cf5e`, whereas the Qwen profiles in this source
tree pin `629bc3218833`. These conditions have no hardware evidence:

- setup from factory-reset hosts, recovery after a reboot, and re-cabled rings;
- a first installation on Sparks that hold neither the serving image nor the
  checkpoint, where checkpoint preparation waits for image distribution;
- repeating `sparkring install` after a failed or interrupted asset
  preparation, including continuing a partial checkpoint download;
- the four-Spark driver step, `sudo sparkring setup --allow-driver-reload`,
  which four-Spark rings need before the first installation and after every
  reboot; see [Four-Spark rings](#four-spark-rings);
- Docker's containerd image store; see
  [Image distribution and caches](#image-distribution-and-caches).

Offline tests cover the second and third conditions. The
[acceptance record](../development/installer-acceptance.md) states the
conditions of each run.

Choose any Spark as Node A. Connect its 10GbE port to your network. Connect a
pair with one cable from port p0 to port p0, or a four-Spark ring with each
Spark's p0 connected to the next Spark's p1. The nodes can be in any order;
setup discovers ranks from the cables. Six-Spark rings and other port
orientations are unsupported by this installer.

Every Spark needs Ubuntu 24.04 ARM64 (DGX OS) with working NVIDIA drivers,
Docker, NVIDIA Container Toolkit and NetworkManager, SSH enabled, and an
existing root login or a login with sudo. Check Docker's image store before
installing; the containerd image store is untested (see
[Image distribution and caches](#image-distribution-and-caches)). All Sparks
must use the same OS distribution and version. Setup does not replace the host
driver or firmware.
Workers need no Ethernet cable or Internet connection of their own. Before the
first installation, read [Security and host exposure](#security-and-host-exposure)
and [Downloads, storage and outbound hosts](#downloads-storage-and-outbound-hosts).

## Get the package

SparkRing installs from one Debian package on Node A; setup copies it to the
workers. A Git checkout cannot replace the package: `sparkring install` runs
only from an installed package. Obtain the package in one of three ways.

**Build, install and run in one command.** On Node A, as a user with sudo,
[`install.sh`](../../install.sh) clones the `one-command-installer` branch in
full, builds its package, installs it with `apt`, then runs
`sudo sparkring install` with every option you pass it:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

`--ref BRANCH_TAG_OR_COMMIT` selects another source and `--repository URL`
another repository or a local Git bundle; the script's own copy must come from
the same ref. It builds in a temporary directory under `/var/tmp` and removes
it afterwards. When the script arrives through a pipe, the installer's
questions are read from the terminal.

**Download a published build.** A GitHub prerelease of
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring/releases)
carries a `sparkring_*_arm64.deb` asset and its `.sha256` file for the commit
the prerelease names. Download both into one directory on Node A, then check
the file; the command prints `OK` when it matches the published checksum:

```bash
sha256sum --check sparkring_*_arm64.deb.sha256
```

**Build from a full clone.** The build needs `git`, `dpkg-deb` and Python 3.12
or later, all present on DGX OS, and runs on any Linux host that provides
them. It requires a
full clone (not `--depth`) with no uncommitted or untracked files, because the
package embeds the commit's history as a Git bundle. Replace `BRANCH` with the
branch or tag to install:

```bash
git clone --branch BRANCH https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
python3 scripts/build_deb.py
```

To pin an exact commit, run `git checkout --detach COMMIT` before the build.
The builder prints the package path, its SHA-256, the source `revision` and the
package `version`, and writes the package and a `.sha256` file to
`.sparkring/dist/`; it refuses to overwrite an existing package there
(`--output DIR` selects another directory). Packages built from separate clones
of one commit can have different SHA-256 values, because the embedded bundle
depends on how the clone is packed. Compare the source revision instead.

**Check the source revision.** The package version ends in `+git` followed by
the first 12 hexadecimal digits of the source commit:

```bash
dpkg-deb --field sparkring_*_arm64.deb Version
```

For a local build, pass `.sparkring/dist/sparkring_*_arm64.deb`. After
installation, `/usr/lib/sparkring/distribution.json` records the full commit as
`revision`.

**Install the package on Node A only:**

```bash
sudo apt install ./sparkring_*_arm64.deb
python3 -c 'import json; print(json.load(open("/usr/lib/sparkring/distribution.json"))["revision"])'
sparkring models
```

For a local build, install `./.sparkring/dist/sparkring_*_arm64.deb`. `apt` can
report that the download is performed unsandboxed as root; for a local file
that notice is harmless. On a Spark where [bootstrap.sh](bootstrap.md)
installed a Git checkout, `~/.local/bin/sparkring` runs that checkout for
commands without `sudo`; run `/usr/bin/sparkring` to use the package.

## Install a model

On Node A, for a pair:

```bash
sudo sparkring install --profile qwen38-flash-next-tp2
```

On a four-Spark ring, complete the [driver step](#four-spark-rings) first and
use `qwen38-flash-next-qad-tp4`. Without `--profile`, the command lists the
installer profiles for the cabled node count and asks for one.

On first use the command lists everything automated setup will do and asks one
question, `Proceed? [Y/n]`; Enter approves. That approval covers discovery,
trusting each cabled Spark's SSH host key on first contact (setup prints the
fingerprints it recorded), package installation, the administration network,
fabric addressing, including replacement of incompatible fabric IPv4 settings
(see [Setup and access](#setup-and-access)), and the first model installation.
SSH still asks for each worker's password when no key login exists, and a
worker signed in as a non-root user asks for its sudo password twice. Stopping
a running GPU container always needs its own answer or `--stop-workloads`.
Setup signs in to each Spark once: other fabric functions and return paths are
recognized from that Spark's inventory.

The command updates workers from Node A's package, reuses cached images and
weights, and copies missing assets over verified fabric paths. It prepares
assets before stopping the previous managed model. A failed switch attempts
recovery from the retained deployment and records the outcome. A successful
installation ends with `Model ready:` and the model's API URL on Node A.

Asset preparation never creates, starts or stops a model container. When it
fails or is interrupted before the model switch, for example by a storage
check, a download error or Ctrl-C, repeating `sudo sparkring install` with the
same choices starts preparation again, re-verifies what the earlier attempt
left and continues an unfinished checkpoint download. Offline tests cover this
repetition; it has no hardware evidence.

For an LLM or a repeatable installation:

```bash
sudo sparkring install --profile qwen38-flash-next-tp2 --plan --json
sudo sparkring install --profile qwen38-flash-next-tp2 --yes --json
sudo sparkring status --refresh --json
```

`--json` writes one result to stdout; progress stays on stderr and in the log.
Exit codes are 0 for success/planning, 3 for missing input, and 2 for failure.
`needs_input` identifies the required choice, such as profile, approval or
storage. `--yes` approves model replacement and first-use setup without a
terminal; it does not trust unknown SSH host keys or authorize stopping
unrelated workloads. A configured ring is inspected without changing links.
Use `sparkring setup` to review cable or network changes separately.
`sparkring models` lists the exact profiles.

Follow the installation from another terminal:

```bash
sudo sparkring logs --follow
```

Concise timestamped progress is appended to `/var/log/sparkring/install.log`.
Long steps report that they are still working. Verbose command output goes to
`install-details.log`; use `sparkring logs --details --follow` when investigating
an error. Credentials entered through SSH are not recorded.

Interactive terminals show colored results and a spinner. Saved logs and piped
output stay plain. `sparkring logs --plain` disables terminal decoration for the
follower. For `sparkring install` itself, set `NO_COLOR` through `sudo`, which
does not pass the caller's environment: `sudo env NO_COLOR=1 sparkring install
...`. From Windows PowerShell, use `ssh -t spark-r0 "sudo sparkring logs --follow"`
(replace `spark-r0` with Node A's SSH address). The follower's spinner means it
is waiting for log lines; it does not indicate model readiness.

## Serving image and profiles

Every installer profile runs on one shared serving image, pinned by the
[installer image lock](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/installer-image.json)
(`sparkring-installer-image/v2`). It is the ARM64 image
`ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f`
(tag `dev-20260925-qwendecode-cuda1342-nccl2323-status031`, configuration
`sha256:4100e1d2bd038f885d92f8c0021d482b23f9a38a003cd7bfc3e700e7e0afa971`),
built on the `eugr/spark-vllm-b12x:nightly-20260924` base image with CUDA
13.4.2, NCCL 2.32.3, the runtime-status dashboard 0.3.1, the paced RoCEnante
transport, whose forwarded-path send window bounds traffic that a ring node
relays for its neighbours, and the Qwen decode kernels described below. The
lock lists the admitted profiles and pins the image configuration, registry
manifest, external software receipt, toolchain receipt, composition, prepared
transport and status package. The image's
[publication record](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/publication.json)
lists its parent image, `dev-20260925-cuda1342-nccl2323-status031`, and every
file its derived layer replaces. `sparkring models` marks only these profiles
as installer-supported:

| Profile | Checkpoint | Status and evidence |
|---|---|---|
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | Qwen3.8 Flash Next NVFP4 QAD, revision `629bc3218833` | **Development** (`implemented`): installed with `sparkring install` on this image, with every setting these profiles make including probabilistic drafting, on one pair (source revision `e75451a671a3`) and one four-Spark ring (`f0ce5bea531f`); counting, arithmetic and code checks passed on both. Those installations' decode and prefill rates, and the measurements behind each setting, are in the [installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md). Serving is not qualified. |
| `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash NVFP4-Spark, revision `a608241037e4`, with MTP3 | **Experimental** (`research-only`): one installation each, on the parent image `dev-20260925-cuda1342-nccl2323-status031` with installer package revision `eb8ec2ba3b17`, passed counting, arithmetic and code checks ([record](../../performance/records/images/dev-20260925-installer-profiles-20260925.md)). No installation on this image, or with an installer revision after `eb8ec2ba3b17`, is recorded. |
| `mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4` | MiMo-V2.6-Flash-RL, revision `5711b2681699`, with DFlash5 | **Experimental** (`research-only`): as for GLM. |

Each profile's `profile.json` states its own evidence scope; published
qualifications of other images do not transfer. The installer image differs
from its parent image only in Qwen model files and one vLLM setting,
`VLLM_QWEN4_EXP_MXFP8_HC`, which is off unless a profile sets it. All installer profiles run with SparkCache off and
vLLM's native prefix cache on. Profiles that select other images, such as the
`shared-2026.09.3` release, keep their own guides and are not installed by
`sparkring install`. `--image-lock FILE` replaces the shared lock for a
development rehearsal and must list the selected profile.

Both Qwen profiles use one prefill recipe on TP2 and TP4: each rank owns a
share of the token rows in the hyper-connection (HC) prefill path
(`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`), which excludes HC projection sharding,
and the image's `qwen-collectives` collective policy and `qwen4-prefill` hooks
are active. Both quantize the target LM head (`VLLM_MXFP8_LM_HEAD=1`) and the
hyper-connection down/injection projections (`VLLM_QWEN4_EXP_MXFP8_HC=1`) from
BF16 to MXFP8 at load. Every rank reads these weights in full at each decode
step, so halving their bytes shortens the step; batches above 16 rows, such as
prefill chunks, keep using the BF16 hyper-connection weights. The image runs
the remaining BF16 projections, such as the MoE router, through skinny-GEMM
plans measured on GB10, and the profiles select vLLM's fused rotary-embedding
op. Decode all-reduces of up to 64 rows (`QWEN_DISPATCH_AR_BYTES=327680`) run on
RoCEnante rather than NCCL. The TP4 profile sets
`NCCL_IB_EXTENDED_IPV4_GIDS=1`, which lets the image's NCCL use all four ring
NIC functions for prefill collectives.

Both Qwen profiles speculate three tokens with the checkpoint's MTP head and
sample drafts from the draft distribution
(`"draft_sample_method": "probabilistic"` in `--speculative-config`). The
rejection test then uses the full draft/target probability ratio, so outputs
follow the target model's sampling distribution. At the checkpoint's default
sampling (temperature 1.0, top-k 20, top-p 0.95), tokens per decode step for
prose, which count the accepted draft tokens plus the one token the target
model adds, rose from 1.97 to 2.16 on TP2 and from 1.99 to 2.14 on TP4
compared with greedy drafting; on TP4, step time and KV-cache capacity were
unchanged. These values come from serving containers derived from installer
deployments, not from an installation of this configuration. The
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md)
gives these measurements, and the
[decode A/B](../../performance/records/qwen38-flash-next/decode-ab-20260925.md)
covers the checkpoint and LM-head choices. Before any serving container is
created, admission reads the image's external software receipt and refuses a
profile whose HC mode is not listed for its node count or whose features the
image does not provide.

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

## Image distribution and caches

A registry-backed image reaches every Spark that lacks it through a registry
relay on Node A. The relay serves only the pinned repository on Node A's
loopback address. Each worker reaches it through an SSH remote forward on the
worker's own loopback address. It downloads each layer from the registry once,
in parallel byte ranges, verifies it against its digest, and serves the
verified copy to every node. Nodes pull concurrently, so the Internet link
carries the image once and every node unpacks it in parallel. Pulled images
keep the reference `127.0.0.1:5255/<repository>@<digest>`.

Docker's pull skips a layer the node already holds only if Docker recorded that
layer's registry digest; layers that arrived through `docker load` or a local
build lack that record and would download again. The relay therefore reads the
pinned manifest and image configuration first, and each node reports how many
of the image's leading layers its Docker image store already holds. A node
that holds some of them receives an archive with the configuration and only
its missing compressed layers; `docker load` reuses the held layers, checks
each loaded layer against the configuration, and tags the image
`127.0.0.1:5255/<repository>:<image-lock name>`. A node that holds none of the
layers pulls. The relay reads anonymous pulls, as public GHCR and Docker Hub
repositories allow. Node A needs room for the relay's layer cache, which is
removed after distribution, in addition to its own pull. Image distribution
starts first and overlaps the prerequisite and source checks. Checkpoint
preparation waits until every node holds the image, because a checkpoint
download or repair runs the image's own Hugging Face client; image admission
follows. Offline tests cover this ordering; it has no hardware evidence.

Docker's containerd image store is untested and probably unsupported: the
installer identifies an image by comparing Docker's image ID with the pinned
configuration digest, and that store can report a different digest as the
image ID. Check each Spark before installing:

```bash
docker info --format '{{json .DriverStatus}}'
```

Output that contains `io.containerd.snapshotter` indicates the containerd
image store. Report that output in an issue before installing on such a Spark.

When the relay cannot reach the registry, or the registry requires credentials,
a Spark that already holds the image streams it to the others with
`docker save` and `docker load`. For a private image without a reachable
registry, set `image_reference` to its exact `image_id` and load it on one
enrolled Spark; the installer uses that stream, without creating another export
archive. Streaming is slower than the relay: with Docker's overlay2 image
store, `docker save` writes the whole image to a temporary directory before
sending its first byte, and `docker load` unpacks only after receiving the
whole image.

Source/image/profile choices automatically select a separate deployment; there
is no instance name or manual stop command to supply. Compile and B12X tuning
results share one cache per cluster, `/srv/sparkring/<cluster>/cache`, in
subdirectories keyed by model family, image and checkpoint revision, so
reinstalling or switching back to a profile reuses its earlier tuning. Compiled
B12X kernels are the exception: B12X keys each one by its package source,
Python, torch, CUTLASS DSL and CUDA binding versions, compile environment and
GPU device UUID, so installer profiles keep them in a subdirectory named by
model family, CUDA toolkit version and checkpoint revision, and an image update
that leaves B12X unchanged reuses them. The first start of a profile on a Spark
still includes that Spark's tuning. With installer package revision
`eb8ec2ba3b17` on the parent image `dev-20260925-cuda1342-nccl2323-status031`,
the first start of each Qwen profile with empty caches reached API readiness in
546.8 s on TP2 and 476.9 s on TP4
([record](../../performance/records/images/dev-20260925-installer-profiles-20260925.md)).
At that revision the Qwen profiles pinned checkpoint revision `60215d26cf5e`
(Hugging Face branch `qad-step5500-ple1000`) without the MXFP8 LM-head and
hyper-connection quantization or probabilistic drafting described above. No
first-start time is recorded for the pinned image with checkpoint
`629bc3218833`.
`sparkring export --share` retains the image lock and per-rank Compose files.

## Status observations

`sparkring status --json --refresh` reports the saved deployment/image IDs and
separate host and container observations. Host observations include persistent
node ID, boot ID and their own `observed_at`; cached observations retain their
original time and become stale after 90 seconds. Container observations include
the inspected container ID, start time and actual image ID. Missing identities
are `null`, with a reason, rather than guessed from hostname or rank. Network
observations do not establish model readiness or serving qualification.

## Setup and access

Setup finds neighbors over IPv6 link-local addresses and signs in to each
worker as the account that ran `sudo`. `sparkring setup --ssh-user USER`, or
`SPARKRING_SSH_USER` in the settings file passed to `sparkring install --env`,
selects another account; when a settings file is given, its
`SPARKRING_SSH_USER` applies and defaults to `root`. Setup copies SparkRing and
its Debian dependencies through the fabric and shows the proposed network
changes. The logged-in Spark becomes Node A.

Before changing anything, `sparkring install` confirms noninteractive SSH and
`sudo` on every enrolled Spark. A missing grant returns `needs_input` with field
`access` and, per Spark, the one-time command a person runs there; it prompts
for that Spark's password. SparkRing never accepts passwords as options or
settings. [Security and host exposure](#security-and-host-exposure) describes
what that command grants.

Existing compatible fabric addresses are kept. When existing addresses are
incompatible, interactive setup prints the problem and replaces the fabric
IPv4 settings under the single `Proceed?` approval, whose list names this step;
it asks no further question. A run without a terminal stops at that point
instead, unless `SPARKRING_LINK_POLICY=reset` in the settings file or
`sparkring setup --reset-links` requested the replacement. Setup saves
connection backups and receipts. It keeps the active NetworkManager connection
identity and IPv6 address generation while changing fabric IPv4/MTU settings,
so its administration path survives renumbering.

## Four-Spark rings

Status: four-Spark installation is **implemented**. Its hardware runs used a
ring whose ConnectX functions already held the hairpin setting described
below. Applying that setting through `sudo sparkring setup
--allow-driver-reload` (step 3) has no hardware evidence. Setup reaches the
workers through its WireGuard administration network, which runs over the
fabric links that each reload takes down, and a reload changes the interface
index that the network's link-local endpoints depend on. Use the steps below
only with console access or an independent management connection to every
Spark; start with a pair otherwise.

Every four-Spark installer profile (`qwen38-flash-next-qad-tp4`,
`glm53-flash-nvfp4-spark-tp4` and `mimo-v26-flash-rl-tp4`) relays traffic
between nonadjacent Sparks through ConnectX hardware forwarding. Forwarding
requires a hairpin queue size (`hairpin_queue_size`) of 8192 on every ConnectX
function; the driver starts with 1024 at every boot. Applying 8192 reloads the
NIC driver (`devlink dev reload ... action driver_reinit`), which
`sparkring install` cannot authorize. Without the step below, a first
installation stops with `Driver reload required; review with
--allow-driver-reload on an idle cluster`, and an installation after a reboot
stops with `persistent addresses or driver settings do not match the plan`.
SparkRing's boot services do not restore the setting. Pairs never need this
step.

Before the first four-Spark installation, and again after any Spark in the
ring reboots:

**1. Keep an independent path to every Spark.** Connect to every Spark
through a console or a management network that does not use the ring cables.
The reload takes each ConnectX function's links down, and a worker without its
own Ethernet connection reaches Node A only through those links. Setup does not
check that an independent path exists.

**2. Stop every GPU and RDMA user on all four Sparks.** List SparkRing's active
services and stop each unit whose name ends in `mesh.service`, such as
`sparkring-mesh.service`; its model unit stops with it. Stop any other GPU or
RDMA workload the same way it was started. `nvidia-smi` must then list no
compute process, and `rdma resource show qp` must list no queue pair with a
`pid`; queue pairs without one belong to the kernel.

```bash
systemctl list-units --type=service --state=active 'sparkring-*'
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
rdma resource show qp
```

**3. Review, then apply the driver setting from Node A**, through that
console or management connection:

```bash
sudo sparkring setup --plan
sudo sparkring setup --allow-driver-reload
```

`--plan` inspects every Spark without changing a host. It prints each rank's
action, `configure` when its fabric settings or driver setting differ from the
plan, and any blocker, such as a GPU process or an RDMA user. It saves the plan
as `plan.json` in the newest directory under
`/var/lib/sparkring/controller/setups/`; the `driver_steps` of each host in its
`network` section list the `devlink` commands, including every driver reload.
Do not continue while a rank shows a blocker.

`--allow-driver-reload` asks one question, `Proceed? [Y/n]`, before it prints
the plan. The list shown with that question does not mention the driver
reload, and the one answer approves everything that follows: setup prints the
plan and, without asking again, reloads one ConnectX function per step,
inspects every Spark again after each reload, and saves the resulting network.

On Sparks without SparkRing setup, `--plan` only discovers them. Run
`sudo sparkring setup` first: it completes first-use setup, then stops with
`Driver reload required` before changing fabric IPv4 addresses or driver
settings. Then review with `--plan` and apply as above.

**4. Install the model.** For a first installation, run
`sudo sparkring install --profile qwen38-flash-next-qad-tp4`. Restarting an
installed four-Spark model after a reboot has no hardware evidence; complete
steps 1–3 before any attempt to start it.

The installer renders each rank's container from the profile's shared
container specification. Every four-Spark installer profile runs on a native
mesh. The installer discovers and verifies an installed native mesh. If none
exists, it downloads and verifies the pinned host marker on every rank, creates
stopped model containers, installs supervised mesh services, waits for every
rank, and starts the model. An unhealthy or partly installed mesh stops the
plan for inspection. `sparkring install` does not replace an existing mesh:
`sparkring up PROFILE --fresh-mesh --plan` prints an explicit replacement plan,
and `sparkring up PROFILE --instance fresh --fresh-mesh` rehearses the
replacement beside an existing deployment. Container, source and weight
directories of a replaced deployment are retained.

## Security and host exposure

The installer changes each Spark's network exposure as follows. Review this
list before approving `Proceed? [Y/n]`.

- **Open model API.** The model's OpenAI-compatible API listens on all of
  Node A's interfaces with no API key: port 8000 for `qwen38-flash-next-tp2`
  and `glm53-flash-nvfp4-spark-tp2`, 8015 for `qwen38-flash-next-qad-tp4` and
  `glm53-flash-nvfp4-spark-tp4`, and 8020 for `mimo-v26-flash-rl-tp2` and
  `mimo-v26-flash-rl-tp4`. Containers use host networking, and the
  runtime-status dashboard (`/v1/sparkring/status/view`) answers on the same
  port. Anyone who can reach that port can use the model. Keep Node A on a
  trusted network or restrict the port with a firewall.
- **Passwordless sudo.** When noninteractive `sudo` is missing, the command
  that `needs_input` prints for a Spark writes
  `USER ALL=(ALL) NOPASSWD:ALL` to `/etc/sudoers.d/USER` for the SSH account.
  It gives that account passwordless root on that Spark; review it before
  running it.
- **Administration network.** Setup creates a WireGuard network, interface
  `sr-control` on UDP port 51871 with addresses in `10.253.255.0/29` by
  default, over the fabric links' IPv6 link-local addresses. A separate SSH
  service (`sparkring-access.service`) listens on TCP port 2222 of each Spark's
  administration address and admits only root with Node A's controller key,
  `/var/lib/sparkring/controller/controller_ed25519`.
- **Setup SSH service.** During first-use setup, `sparkring-seed.service` runs
  an SSH server on TCP port 2222 on all IPv6 addresses of Node A, and of any
  worker prepared with `sparkring setup --worker-bundle`. It admits only root
  with Node A's controller key. Setup disables it after the administration
  network works on every Spark. If setup stops earlier, it stays enabled;
  `sudo systemctl disable --now sparkring-seed.service` removes it.
- **Internet sharing.** Sharing is on by default
  (`SPARKRING_SHARE_INTERNET=yes`). Node A forwards and masquerades traffic
  from the administration network and answers workers' DNS with `dnsmasq`.
  Each worker accepts every IPv4 destination through Node A in its WireGuard
  configuration (`AllowedIPs 0.0.0.0/0`) and sends DNS for all domains to
  Node A. This behavior is read from the setup code; it has not been observed
  on hardware. When workers have their own network connection, put
  `SPARKRING_SHARE_INTERNET=no` in a settings file and pass it with `--env` on
  the first installation. Four-Spark rings whose workers have no connection of
  their own need sharing; see
  [Downloads, storage and outbound hosts](#downloads-storage-and-outbound-hosts).
- **Host announcement.** The package enables `avahi-daemon` (mDNS) and
  `lldpd` (LLDP) with their default configuration, which announces on every
  interface, and publishes a SparkRing mDNS service,
  `/etc/avahi/services/sparkring.service`.

Removing the package (`sudo apt remove sparkring`) on a Spark stops and
disables SparkRing's services there, including both SSH services on port 2222
and the worker DNS service, and deletes the SparkRing mDNS service file. It
does not remove the passwordless sudo file or disable `avahi-daemon` and
`lldpd`. The WireGuard interface, forwarding settings and iptables rules
already applied stay until the Spark reboots, and a running model keeps
serving its open API. See [What is installed](#what-is-installed).

## Downloads, storage and outbound hosts

Node A needs outbound HTTPS to these hosts. Image and checkpoints are read
anonymously; no registry or Hugging Face account is used.

| Host | Used for |
|---|---|
| `ghcr.io`, which redirects to `pkg-containers.githubusercontent.com` | The serving image, downloaded once by Node A's relay |
| `huggingface.co`, which redirects to its download CDN | The model checkpoint, downloaded once when no Spark holds a matching copy |
| Your Ubuntu package mirror | The package's dependencies during `apt install` on Node A |
| `github.com` and its release-asset download host | Four-Spark rings only: every rank downloads the native mesh host marker from `https://github.com/FujitsuPolycom/sparkring/releases/download/r33-host-tools-c8646b0/mlx5-rdma-tx-marker` when the installer creates a mesh |

Workers receive the package, image and checkpoint from Node A. A Spark that
repairs a reused checkpoint copy (see [Checkpoints](#checkpoints)) downloads the
differing files itself, and each rank of a four-Spark ring downloads the mesh
marker itself; workers reach the Internet through Node A's sharing unless they
have their own connection.

| Asset | Size |
|---|---:|
| Serving image `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | 14.2 GiB download, 29.5 GiB unpacked |
| Qwen checkpoint, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `629bc3218833` | 98.6 GiB |
| MiMo checkpoint, `XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b2681699` | 165.6 GiB |
| GLM checkpoint, `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` @ `a608241037e4` | 174.8 GiB |

The installer checks free space before each download or copy and before the
model switch. It does not delete anything to make room, and a failed check
leaves the running model in place:

- Image: each Spark without the image needs the unpacked size plus the
  download size plus 8 GiB (51.7 GiB) free in Docker's data root. Node A needs
  a further 14.2 GiB there for the relay's layer cache.
- Checkpoint and caches: each Spark needs the planning allowances in
  [storage planning](../../profiles/storage-planning.json) on the filesystems
  that hold the checkpoint, Docker's data root (while the image is absent) and
  the compile cache: a checkpoint allowance of 120 GiB (Qwen), 190 GiB (MiMo)
  or 200 GiB (GLM), 68 GiB for the image and 32 GiB for caches.

With the checkpoint, Docker and the cache on one filesystem, keep at least
220 GiB (Qwen), 290 GiB (MiMo) or 300 GiB (GLM) free on every Spark, and about
15 GiB more on Node A. Checkpoints default to
`/srv/sparkring/<cluster>/<profile>-<instance>/models/<revision>` and caches to
`/srv/sparkring/<cluster>/cache`; `--model-path` and `--cache-path` choose other
locations.

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

Interactive setup needs no settings file. Repeated installations can use one:

```dotenv
SPARKRING_NAME=home
SPARKRING_SSH_USER=operator
SPARKRING_SHARE_INTERNET=yes
SPARKRING_LINK_POLICY=keep
```

`sudo sparkring install --env /path/to/settings.env` reads the file only for
first-use setup, before the cluster is configured; `sudo sparkring setup --env
/path/to/settings.env` reads it on every run. Values are parsed as literal
settings, never sourced as shell code, and SSH handles credential prompts.
The supported keys are:

| Key | Default | Accepted values |
|---|---|---|
| `SPARKRING_NAME` | `sparkring` | Lowercase letters, digits and `-`, starting with a letter, at most 35 characters |
| `SPARKRING_SSH_USER` | `root` | A Linux account name |
| `SPARKRING_SSH_PORT` | `22` | `22` or `2222` |
| `SPARKRING_SHARE_INTERNET` | `yes` | `yes` or `no` |
| `SPARKRING_CONTROL_CIDR` | `10.253.255.0/29` | An IPv4 `/29` network |
| `SPARKRING_FABRIC_CIDR` | `198.18.0.0/21` | An IPv4 `/16` through `/21` network that does not overlap the control network |
| `SPARKRING_LINK_POLICY` | `keep` | `keep` or `reset` |

The `SPARKRING_SSH_USER` default applies when a settings file is given;
without one, setup signs in as the account that ran `sudo`
(see [Setup and access](#setup-and-access)). `SPARKRING_LINK_POLICY=reset`, or
`sparkring setup --reset-links`, requests reviewed replacement of fabric IPv4
settings. `sparkring setup` accepts each setting as an option (`--name`,
`--ssh-user`, `--ssh-port`, `--control-cidr`, `--fabric-cidr`,
`--no-share-internet`, `--reset-links`); `sparkring install` passes only
`--env`, `--yes` and `--stop-workloads` to setup. `sparkring setup --plan`
discovers and reviews through existing access without configuring hosts;
`sparkring install --plan` saves an installation plan without updating workers
or models.

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

At boot, SparkRing's enabled host services start the administration network
and its SSH service, and `sparkring-fabric.service` restores the approved
fabric routes, per-interface IPv4 forwarding and forwarding rules;
NetworkManager keeps the fabric addresses. The ConnectX hairpin setting that
four-Spark rings require is not restored; see [Four-Spark rings](#four-spark-rings).
Models start only when requested. `sparkring down` stops the selected
deployment.

Package removal (`sudo apt remove sparkring`) records which SparkRing host
services are enabled, then disables and stops them. It retains configuration,
weights, caches, receipts, network state and any running model deployment.
Installing the package again re-enables the recorded services and starts the
administration services; the fabric service is enabled for the next boot and
not started. No command reverts setup's network, SSH, sudo or service changes.

`sparkring models` lists exact model/version/quantization/topology profiles and
marks which support automated installation. Family names such as `qwen` are
ambiguous and are rejected. Guide-only profiles remain listed with their guides.

## Checkpoints

The installer checks existing container model mounts and standard model
directories under `/srv/models`, `/models`, `/var/tmp/models` and
`/srv/sparkring` for the selected checkpoint. A directory whose
configuration and index hashes match, with every indexed weight file present,
is proposed for reuse, then every pinned shard is verified during preparation.
On Linux, each host records the verified hashes per checkpoint path in
`/var/lib/sparkring/checkpoints/`; later gates and later installations reuse
them only while the complete file list, device, inode, size, modification time
and change time match; changes trigger checksum verification again. A missing
checkpoint is copied from a verified peer, or downloaded once on Node A if none
has it. Copies travel over the fabric cables outward from the Spark that holds
the checkpoint, ring neighbors in parallel: plain TCP between the two ends of
each cable, one stream per shared fabric function. The receiver binds only its
fabric addresses, accepts only the sender's fabric address and a one-time token
delivered over administration SSH, and writes only files named in the verified
manifest; files that already match are kept. When a direct copy fails, the
remaining Sparks are filled with rsync over administration SSH. Copies are
checksum-verified before launch.

Each installer profile's `SHA256SUMS` file pins the checkpoint files it lists.
A reused copy, whether discovered or given with `--model-path`, whose files
differ from those pins is repaired in place: a root container synchronizes the
directory from the pinned hub revision, replacing only the differing files,
and the copy is verified again. Discovered directories need not belong to
SparkRing, and the installer does not ask before repairing them. To keep such a
directory unchanged, move it out of the searched locations or pass
`--model-path` with a copy you own. Hub-style folders named `<owner>--<name>`
are found with or without a revision subfolder; metadata hashes decide a match,
not folder names.

Before a download writes its first file, the installer records the destination
as that deployment's download. A nonempty directory at the installer's default
model path is accepted only as that recorded, unfinished download; repeating
`sudo sparkring install` continues it, keeping completed files. Any other
nonempty directory without an installer receipt is refused. Offline tests
cover continuing a download; it has no hardware evidence.

`--model-path /absolute/checkpoint` selects a cache explicitly when it is stored
elsewhere. This avoids downloading weights already present on the ranks.
`--cache-path /absolute/cache` chooses another writable compilation cache. Image
imports and checkpoint copies check storage before the model switch;
insufficient space leaves the running model in place. The installer does not
delete model weights or unrelated archives to make room. Arbitrary upstream
images still need their own compatible transport adapter.

## Local build and tests

[Get the package](#get-the-package) builds the Debian package. The package has
no compiled host payload, so its assembly can run on x86 Linux. The tests need
pytest and the other packages in
[requirements-dev.txt](../../requirements-dev.txt), which DGX OS does not ship;
install them in a virtual environment. On Ubuntu 24.04, including DGX OS,
`python3 -m venv` needs the `python3-venv` package
(`sudo apt-get install python3-venv`). From a clean committed checkout on Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest runtime/host runtime/common/test_distribution.py -q
```

An optional Linux namespace rehearsal exercises real WireGuard between two and
four simulated hosts, with no physical interfaces:

```bash
sudo env SPARKRING_LINUX_LAB=1 .venv/bin/python -m pytest scripts/test_appliance_linux.py -q
```

A failed asset preparation can be repeated; see
[Install a model](#install-a-model). For any other operation, if an execution
receipt says `running` or `uncertain`, inspect it and host state before
recovery. Do not delete receipts to force a blind retry. A driver reload
requires `sudo sparkring setup --allow-driver-reload`, stopped containers and
GPU work, and no RDMA users; see [Four-Spark rings](#four-spark-rings).

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
