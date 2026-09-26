# Install SparkRing: reference

This page holds the details behind [Install SparkRing](install.md): package
builds, approvals, the serving image, image and checkpoint handling, host
exposure and lower-level commands. Every command and flag is listed in
[SparkRing commands](commands.md).

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

`sudo sparkring install --profile PROFILE` on Node A installs one installer
profile ([commands](install.md#install)). On a four-Spark ring the approval
also covers the [ConnectX hairpin setting](#four-spark-rings). Without
`--profile`, the command lists the installer profiles for the cabled node count
and asks for one.

On first use the command lists everything automated setup will do and asks
`Proceed? [Y/n]`; Enter approves. That approval covers discovery, trusting each
cabled Spark's SSH host key on first contact (setup prints the fingerprints it
recorded), package installation, the administration network, fabric
addressing, including replacement of incompatible fabric IPv4 settings (see
[Setup and access](#setup-and-access)), the ConnectX driver restarts that
four-Spark rings need, and the first model installation, unless the checkpoint
plan downloads more than 1 GiB or a Spark's search stopped early or failed.
Then the command prints that plan and asks
`Proceed with this checkpoint plan? [y/N]`; answer `y`. Enter cancels; setup
stays complete, and the same command asks again.
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

Running the command again for the installed model leaves that model running
while it serves: its container runs on every Spark and, on four Sparks, every
ring check passes. Otherwise, for example after one Spark restarted, the
command stops the model on every Spark and starts it again, because the Sparks
that stayed up keep a model that waits for the restarted one.

Asset preparation never creates, starts or stops a model container. When it
fails or is interrupted before the model switch, for example by a storage
check, a download error or Ctrl-C, repeating `sudo sparkring install` with the
same choices starts preparation again, re-verifies what the earlier attempt
left and continues an unfinished checkpoint download.

For an LLM or a repeatable installation:

```bash
sudo sparkring install --profile qwen38-flash-next-tp2 --plan --json
sudo sparkring install --profile qwen38-flash-next-tp2 --yes --json
sudo sparkring status --refresh --json
```

`--json` writes one result to stdout; progress stays on stderr and in the log.
Exit codes are 0 for success/planning, 3 for missing input, and 2 for failure.
`needs_input` identifies the required choice, such as profile, approval,
storage or checkpoint; `driver` means a four-Spark ConnectX driver restart
cannot run, and its details list what to stop or what did not complete.
`--yes` approves model replacement, first-use setup and, on an idle four-Spark
ring, the ConnectX driver restarts, without a terminal; it does not trust
unknown SSH host keys or authorize stopping unrelated workloads. When a
four-Spark ring needs the ConnectX step, `--plan` lists it as
`apply-hairpin-setting`. A configured ring is inspected without changing links.
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

| Profiles | Checkpoint | Speculative decoding |
|---|---|---|
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | Qwen3.8 Flash Next NVFP4 QAD step 5500, revision `60215d26cf5e` (branch `qad-step5500-ple1000`) | MTP, three tokens, probabilistic drafting |
| `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash NVFP4-Spark, revision `a608241037e4` | MTP3 |
| `mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4` | MiMo-V2.6-Flash-RL, revision `5711b2681699` | DFlash5 |

The installer image differs from its parent image only in Qwen model files and
one vLLM setting, `VLLM_QWEN4_EXP_MXFP8_HC`, which is off unless a profile sets
it. All installer profiles run with SparkCache off and vLLM's native prefix
cache on. Profiles that select other images, such as the
`shared-2026.09.3` release, keep their own guides and are not installed by
`sparkring install`. `--image-lock FILE` replaces the shared lock for a
development rehearsal and must list the selected profile.

Both Qwen profiles use one prefill recipe on TP2 and TP4: each rank owns a
share of the token rows in the hyper-connection (HC) prefill path
(`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`), which excludes HC projection sharding,
and the image's `qwen-collectives` collective policy and `qwen4-prefill` hooks
are active. Both keep the BF16 target LM head (`VLLM_MXFP8_LM_HEAD=0`) and
quantize the hyper-connection down/injection projections
(`VLLM_QWEN4_EXP_MXFP8_HC=1`) from BF16 to MXFP8 at load. Every rank reads these
projections in full at each decode step, so halving their bytes shortens the
step; batches above 16 rows, such as prefill chunks, keep using the BF16
hyper-connection weights. The image runs
the remaining BF16 projections, such as the MoE router, through skinny-GEMM
plans measured on GB10, and the profiles select vLLM's fused rotary-embedding
op. Decode all-reduces of up to 64 rows (`QWEN_DISPATCH_AR_BYTES=327680`) run on
RoCEnante rather than NCCL. The TP4 profile sets
`NCCL_IB_EXTENDED_IPV4_GIDS=1`, which lets the image's NCCL use all four ring
NIC functions for prefill collectives.

Both Qwen profiles speculate three tokens with the checkpoint's MTP head and
sample drafts from the draft distribution
(`"draft_sample_method": "probabilistic"` in `--speculative-config`). The MTP
head's routed experts are MXFP8, which the B12X MoE backend does not implement,
so the draft runs them on the `humming` backend (`"moe_backend": "humming"`). The
rejection test then uses the full draft/target probability ratio, so outputs
follow the target model's sampling distribution. At the checkpoint's default
sampling (temperature 1.0, top-k 20, top-p 0.95), probabilistic drafting raises
tokens per decode step for prose (accepted draft tokens plus the one token the
target model adds) from 1.97 to 2.16 on TP2 and from 1.99 to 2.14 on TP4
compared with greedy drafting, with unchanged TP4 step time and KV-cache
capacity. The
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md)
gives these measurements, the decode and prefill rates and the measurement
behind each setting; the
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
it beneath a running container. Dashboards read it to label ranks; boot
identity and observation times are observed separately, and the file is not an
attestation.

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
preparation waits until every node holds the image, because a download runs
the image's own Hugging Face client and every checkpoint write is checked
against free space after the image is in place; image admission follows.

Not supported: Docker's containerd image store. The installer identifies an
image by comparing Docker's image ID with the pinned configuration digest, and
that store can report a different digest as the image ID. Check each Spark
before installing:

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
still includes that Spark's tuning: with empty caches, the Qwen profiles reached
API readiness in 546.8 s on TP2 and 476.9 s on TP4, on the parent image
`dev-20260925-cuda1342-nccl2323-status031` with checkpoint revision
`60215d26cf5e` and without the MXFP8 quantization or probabilistic drafting
described above
([record](../../performance/records/images/dev-20260925-installer-profiles-20260925.md)).
`sparkring export --share` retains the image lock and per-rank Compose files.

## Status observations

`sparkring status --json --refresh` reports the saved deployment/image IDs and
separate host and container observations. Host observations include persistent
node ID, boot ID and their own `observed_at`; cached observations retain their
original time and become stale after 90 seconds. Container observations include
the inspected container ID, start time and actual image ID. Missing identities
are `null`, with a reason, rather than guessed from hostname or rank. Network
observations do not show whether the model is ready.

On a four-Spark ring, each Spark's host observation also covers the
[ConnectX hairpin setting](#four-spark-rings). A function without the setting
makes that Spark `needs-attention`, with an error naming the function and its
value and the next action `on Node A: sudo sparkring hairpin`. The
observation's `warnings` report a setting that is in effect but not applied at
boot, boot restarts suspended after a failed restart (naming the function and
the time), a boot started with `sparkring.hairpin=off`, and a mesh unit without
the hairpin start check.

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
what that command grants. Installer operations on a Spark run as root: when a
Spark's SSH account is not `root`, they run through `sudo -n`.

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

Every four-Spark installer profile (`qwen38-flash-next-qad-tp4`,
`glm53-flash-nvfp4-spark-tp4` and `mimo-v26-flash-rl-tp4`) relays traffic
between nonadjacent Sparks through ConnectX hardware forwarding. That needs the
ConnectX hairpin setting on each of a Spark's four ConnectX functions: a
hairpin queue of 8192 packets (`hairpin_queue_size`), four hairpin queues
(`hairpin_num_queues`) and hardware TC offload. The driver starts every boot
with 1024 packets and uses the larger queue only after that function's driver
restarts (`devlink dev reload … action driver_reinit`). A restart takes the
function's link down for about 8 seconds. Pairs do not use the setting.

SparkRing applies the setting itself; a first installation needs no flag or
separate step.

- **First installation.** The approval question of the first
  `sudo sparkring install` on four Sparks lists the driver restarts. After
  fabric addressing is configured, SparkRing restarts each function once:
  Node A first, then one worker at a time, about 30 seconds per Spark.
- **Every boot.** `sparkring-hairpin.service` restarts each function before
  NetworkManager starts, which adds about 30 seconds to each boot. SparkRing
  enables the service on a Spark after a run on it in which every restart
  succeeded; from then on, a reboot needs no manual step for the setting.

A ring set up by a SparkRing package without `sparkring-hairpin.service` has
no boot record for the setting, and a reboot leaves its mesh services stopped by
their start check. For any installed ring whose Sparks lack the setting or its
boot service, run this on Node A with this package installed:

```bash
sudo sparkring hairpin
```

It lists what it will change, including a SparkRing package update on workers
that run another revision, and asks once. `--plan` only prints the list, and
`--yes` approves without asking. When the setting is already in effect, it only
records the approval and enables the boot service, without a restart. When an
updated worker then needs a driver restart that the list did not show, it stops
and asks for that restart first. `sudo sparkring install` includes the same
step in its question when a Spark needs it. With `--json`,
`sudo sparkring hairpin` prints one `sparkring-hairpin-result/v1` document; its
exit codes are those of `sparkring install`, and it keeps a receipt of each run
below `/var/lib/sparkring/controller/hairpin/`.

SparkRing never restarts a driver while a model, a mesh service, mesh
forwarding rules or another RDMA program is present on the ring. It lists what
to stop and restarts nothing.

When the setting is not in effect on a Spark, that Spark's mesh service does
not start and its log names the function, `sparkring status` reports the Spark
as `needs-attention` with the function and its value, and
`sparkring up --execute` starts nothing. Run `sudo sparkring hairpin` on
Node A: once every Spark has the setting, it starts each enabled mesh service
that the check refused. `sudo sparkring install` starts the mesh its
installation uses, as described below.

If a restart fails during a boot, or a boot ends during a restart, later boots
of that Spark restart nothing and `sparkring status` warns about it, until
`sudo sparkring hairpin` succeeds. A Spark that is unreachable after a failed
restart becomes reachable after a power cycle, without the setting. A failed
restart of a function that carries the administration network to other Sparks
cuts those Sparks off from Node A, while the Spark itself stays reachable over
its other links; `sparkring status` then names the Spark to reboot. Its next
boot restarts no function, and `sudo sparkring hairpin` then retries it. To
boot once without the restarts, add `sparkring.hairpin=off` to the kernel
command line. On a Spark, `journalctl -b -u sparkring-hairpin.service` shows
the boot run, `sudo sparkring node hairpin status` prints each function's
setting, and `sudo sparkring node hairpin apply --dry-run --boot` prints the
restarts that the next boot performs, in order, or none while the service is
not enabled.

`sudo sparkring hairpin --revoke` on Node A stops applying the setting at boot
on every Spark; the setting stays in effect until each Spark reboots, and after
that reboot the start check keeps each mesh service stopped until the setting
is applied again. `--allow-driver-reload` is accepted by `sparkring setup` and
`sparkring install` and is not needed.

The installer renders each rank's container from the profile's shared
container specification. Every four-Spark installer profile runs on a native
mesh. The installer reuses the mesh service that every Spark has enabled or
running. If no Spark has one, it downloads and verifies the pinned host marker
on every rank, creates stopped model containers, installs supervised mesh
services, waits for every rank, and starts the model. A mesh that only some
Sparks have, or that differs between Sparks, stops the plan for inspection.

SparkRing enables the mesh service, so it starts at each boot after the
hairpin setting. Before an installation starts the model, each Spark starts its
mesh service when it is stopped, restarts it when its routes or forwarding
rules are missing, and re-adds a port's IPv4 address when its RoCE GID has left
the pinned GID index, which happens on the neighbors of a Spark that restarted.
The installation then waits up to four minutes for the ring check on every
Spark. `sparkring install` does not replace an existing mesh:
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
  Node A. When workers have their own network connection, put
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
| `huggingface.co`, which redirects to its download CDN | Checkpoint files that no Spark holds, downloaded once by Node A |
| Your Ubuntu package mirror | The package's dependencies during `apt install` on Node A |
| `github.com` and its release-asset download host | Four-Spark rings only: every rank downloads the native mesh host marker from `https://github.com/FujitsuPolycom/sparkring/releases/download/r33-host-tools-c8646b0/mlx5-rdma-tx-marker` when the installer creates a mesh |

Workers receive the package, image and checkpoint from Node A. In `sparkring
install`, only Node A contacts huggingface.co, and only for checkpoint files
that no Spark holds. Each
rank of a four-Spark ring downloads the mesh marker itself; workers reach the
Internet through Node A's sharing unless they have their own connection.

| Asset | Size |
|---|---:|
| Serving image `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | 14.2 GiB download, 29.5 GiB unpacked |
| Qwen checkpoint, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `60215d26cf5e` | 102.6 GiB |
| MiMo checkpoint, `XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b2681699` | 165.6 GiB |
| GLM checkpoint, `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` @ `a608241037e4` | 174.8 GiB |

The installer checks free space before each download or copy and before the
model switch. It does not delete anything to make room, and a failed check
leaves the running model in place:

- Image: each Spark without the image needs the unpacked size plus the
  download size plus 8 GiB (51.7 GiB) free in Docker's data root. Node A needs
  a further 14.2 GiB there for the relay's layer cache.
- Checkpoint: the bytes a Spark copies, receives or downloads, plus the largest
  of those files once more for its staging copy. Hard-linked files need no
  space. The printed plan shows each Spark's total, which also counts 32 GiB
  for the compile cache when the cache shares that filesystem and, on a Spark
  without the image, 68 GiB for the image. `sparkring up`, whose image step
  pulls the image itself, reserves the full checkpoint allowance of
  [storage planning](../../profiles/storage-planning.json), 120 GiB (Qwen),
  190 GiB (MiMo) or 200 GiB (GLM), unless the Spark holds a verified
  checkpoint.
- Caches: 32 GiB for the compile cache, and 68 GiB in Docker's data root while
  the image is absent.

With the checkpoint, Docker and the cache on one filesystem, the plan asks a
Spark that holds neither the image nor any checkpoint file for 202.9 GiB
(Qwen), 278.0 GiB (MiMo) or 279.5 GiB (GLM); Node A needs 14.2 GiB more in
Docker's data root for the relay's layer cache. Checkpoints are kept in
`/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>` and caches in
`/srv/sparkring/<cluster>/cache`; `--cache-path` chooses another cache
location.

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

At boot on a four-Spark ring, `sparkring-hairpin.service` first applies the
approved [ConnectX hairpin setting](#four-spark-rings), before NetworkManager
starts. SparkRing's enabled host services then start the administration
network and its SSH service, also over the remaining links when one
administration link fails, and `sparkring-fabric.service` restores the
approved fabric routes, per-interface IPv4 forwarding and forwarding rules;
NetworkManager keeps the fabric addresses. Models start only when requested.
`sparkring down` stops the selected deployment.

The package's systemd generator,
`/usr/lib/systemd/system-generators/sparkring-hairpin-mesh-check`, adds a
start check to each mesh unit in `/etc/systemd/system`
(`sparkring-mesh.service` and `sparkring-*-mesh.service`) without changing the
unit file. On a four-Spark ring the unit then starts only when the ConnectX
hairpin setting is in effect; elsewhere the check passes. Each Spark of a
four-Spark ring records its approval of the setting in
`/etc/sparkring/hairpin.json`.

Package removal (`sudo apt remove sparkring`) records which SparkRing host
services are enabled, then disables and stops them. It retains configuration,
weights, caches, receipts, network state and any running model deployment.
Installing the package again re-enables the recorded services and starts the
administration services; the fabric and hairpin services are enabled for the
next boot and not started. Installing or updating the package never restarts a
ConnectX driver. No command reverts setup's network, SSH, sudo or service
changes.

`sparkring models` lists exact model/version/quantization/topology profiles and
marks which support automated installation. Family names such as `qwen` are
ambiguous and are rejected. Guide-only profiles remain listed with their guides.

## Checkpoints

SparkRing keeps one checkpoint directory per cluster and checkpoint revision,
`/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>`, shared by
every deployment of that revision. It holds exactly the files that the pin
manifest in [`profiles/checkpoints/`](../../profiles/checkpoints) requires, 53
for Qwen; the profile's `SHA256SUMS` lists the same files.

**Where the installer looks.** Before it prints the plan, `sudo sparkring
install` searches every Spark at once, for up to 20 s each; a Spark whose search
runs out of time searches once more for up to 40 s, and Docker queries take up
to 15 s, so one Spark's search ends within 75 s:

- SparkRing's own checkpoint directories and records;
- Hugging Face caches in every account's home, and those that `HF_HOME`,
  `HF_HUB_CACHE` and similar variables name in system files, systemd units and
  root's and your own shell files;
- Docker containers' model mounts, volumes and Hugging Face caches;
- folders with any name, such as `hf download --local-dir` folders and git-lfs
  clones, under `/var/tmp/models`, `/models`, home directories, `/data`,
  `/srv`, `/mnt`, `/opt` and other local disks.

The search runs as root and reads directory listings, file sizes, download
metadata and SparkRing's records, home directories included; it hashes only
files up to 64 MiB and does not read other accounts' shell files. It never
enters network storage or automounts. The plan names what was not searched.

**What it does with a copy.** Files are identified by SHA-256 against the pin
manifest, not by names, so a copy of the repository's `main` branch supplies
every Qwen file except `config.json`. SparkRing hashes each file before using
it. Weight files on the same filesystem as SparkRing's directory are hard-linked
into it and take no extra space; the other files (configuration, tokenizer, chat
template) are copied, so later edits to your copy do not reach the served model.
**SparkRing never writes, moves or deletes files it did not create.**
Hard-linking changes only the link count and change time of your weight files;
a backup tool that compares change times reads them once more. A copy in
another account's home is listed but used only when named with `--model-path`.
Files that no Spark holds are downloaded once by Node A.

Copies between Sparks travel over the fabric cables outward from a Spark that
holds the checkpoint. The receiver binds only its fabric addresses and accepts
only the sender's address and a one-time token sent over administration SSH;
when a direct copy fails, rsync over administration SSH fills in. rsync sends
only the files the receiver lacks, into an empty staging directory beside
SparkRing's directory, and reads them without changing their access times
(`--open-noatime`, which needs rsync 3.2.3 or later on both ends; Ubuntu 24.04
ships 3.2.7). Every received file is placed only after its SHA-256
matches. Later starts compare each file's recorded device, inode, size and
times, and re-hash a file that changed. The last step of a model switch,
`Confirm checkpoint unchanged during loading`, compares them again on every
Spark once the model has loaded; a change fails the switch, and the previous
deployment is restored.

**Approval.** The plan shows, for each Spark, its sources and their owners, how
many files it hard-links, and the bytes it copies, receives or downloads. A plan
reviewed with `--plan`, or approved at a terminal prompt, bounds every later
`--yes` run: such a run searches again and proceeds only while its plan stays
within the reviewed one, with no new downloads, at most 1 GiB more written per
Spark, the same mode and no new source folders on each Spark. A run whose plan
leaves it stops, prints the difference and keeps the reviewed plan as the bound,
so repeating `--yes` stops again; review the changed plan with `--plan`, then
repeat `--yes`. The refused plan is saved as `checkpoint-plan.refused.json`
beside the reviewed one. A download or write that the approved plan does not
include stops the installation with `needs_input` (field `checkpoint`) before it
starts; the running model is not changed. Setup's approval of a first
installation does not cover a download over 1 GiB or a search that stopped early
or failed: a terminal asks `Proceed with this checkpoint plan? [y/N]`, and a
run without one stops. A plan whose problems stop the installation, such as a
named path that exists on no Spark or too little free space, records no
deployment. Commands that messages suggest repeat your `--profile`,
`--model-path`, `--cache-path` and `--image-lock` options, which identify the
deployment. With `--json`, the result carries a summary of the plan as
`checkpoint`; the full plan, with each file's action, is `checkpoint-plan.json`
in the result's `deployment` directory.

**Options.**

- `--model-path PATH` names a copy for every Spark, `--model-path N=PATH` one
  for Node N; repeat it as needed. Named copies are used first and the search
  still runs. A named copy on another filesystem than SparkRing's directory that
  holds exactly the pinned files is served in place, read-only, and the model
  does not start while that folder is changed or missing. SparkRing never
  creates or writes a named path. Named paths are part of the deployment:
  repeat every command with the same `--model-path` options, because other
  named paths plan another deployment. Whether a named copy is served in place
  is decided when its deployment is first planned; if that copy later changes,
  the installation stops and says how to continue.
- `--ignore-local-copies` uses only SparkRing's own checkpoint directories and
  named copies.
- A file named `.sparkring-ignore` keeps SparkRing out of its folder and
  everything below, even a named path. For a Hugging Face cache, put it in
  `$HF_HOME` (the parent of `hub`) or in a `models--*` folder; in `hub` itself
  it makes `hf cache scan` report an error.
- `--cache-path /absolute/cache` chooses another writable compilation cache.

`sparkring up PROFILE` does not search: each Spark uses SparkRing's checkpoint
directory and downloads the files it lacks itself, with the serving image's
Hugging Face client, so every Spark must already hold the image (`sudo
sparkring install` distributes it). `sparkring up --model-path PATH` serves a
complete copy in place.

**Disk space held by links.** While SparkRing's directory links a copy's weight
files, deleting that copy or pruning the Hugging Face cache frees nothing.
`sudo sparkring checkpoints` lists SparkRing's checkpoint directories on every
Spark, the deployments that use each, the copies it shares files with and the
space a release frees. `sudo sparkring checkpoints --release PATH` removes one
from every Spark: it refuses a directory that the active deployment or the
rollback target uses, also one served in place, and a directory that a running
container mounts; it names other deployments that use it (they need `sudo
sparkring install` again) and asks first (`--yes` in scripts). It removes only
the names and directories SparkRing placed and never writes, moves or deletes
the copies they were linked from.

**If SparkRing did not find your copy,** read the plan's `Not searched` line and
whether a Spark's search stopped at its time limit, and repeat the plan; a
repeated search is faster. Otherwise name the copy with `--model-path N=PATH`.

**If deleting your copy did not free space,** run `sudo sparkring checkpoints`,
then `sudo sparkring checkpoints --release PATH` for a directory that no running
deployment uses.

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
recovery. Do not delete receipts to force a blind retry. A ConnectX driver
restart requires stopped models, mesh services and other RDMA users;
`sudo sparkring hairpin` checks this before it restarts anything; see
[Four-Spark rings](#four-spark-rings).

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

A site's checkpoint path (`<workspace>/models/<revision>` unless a row names
another) becomes a SparkRing checkpoint directory. Docker's download mount and
rsync's destination name its staging directory by path, so SparkRing claims the
path only when every existing directory above it is writable by its owner alone
or carries the sticky bit, as `/tmp` does, and an existing directory at the
path is empty, owned by root and writable by root alone. Keep it below
directories that only root or the operator can change: an account that owns a
directory above it could still rename what is inside.

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
inputs, tests missing-variable guards and simulates rank registration. It runs
no GPU, RDMA or inference code. [Compose](compose.md) covers the standalone
recipes.
