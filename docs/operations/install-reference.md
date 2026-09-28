# Install SparkRing: reference

The details behind [Install SparkRing](install.md): package builds, approvals,
the serving image, image and checkpoint handling, host exposure and
lower-level commands. [SparkRing commands](commands.md) lists every command and
flag.

## Get the package

SparkRing runs from one Debian package, installed on Node A only; setup copies
it to the workers. `sparkring install` runs only from an installed package,
not from a Git checkout.

### One command

On Node A, as a user with sudo:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile PROFILE
```

[`install.sh`](../../install.sh) clones `main` in full, builds the package,
installs it with `apt` and runs `sudo sparkring install` with your options.
Installing the package initializes the Spark's SparkRing node, enables and
starts `avahi-daemon` and `lldpd` for discovery and restarts
`sparkring-agent`, so the script first shows the installed and built versions
and asks `Install the package? [Y/n]`. Its flags, including `--yes`, `--plan`,
`--package-only` and `--json`, are in [SparkRing commands](commands.md#installsh).

### Pin a commit

The command above installs whatever `main` holds when it runs. To repeat an
installation exactly, fetch the script and source at the commit it printed as
`Source revision:` (the full 40-character ID):

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/COMMIT/install.sh | bash -s -- --ref COMMIT --profile PROFILE
```

`--ref BRANCH_TAG_OR_COMMIT` selects another source and `--repository URL`
another repository or a local Git bundle; fetch the script from the same ref.
Changes reach the `one-command-installer` branch before `main`; to test them,
fetch the script from that branch and pass `--ref one-command-installer`.

### Download a published build

A GitHub prerelease of
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring/releases)
carries `sparkring_*_arm64.deb` and its `.sha256` file for the commit it
names. Download both into one directory on Node A; this prints `OK` when the
package matches the published checksum:

```bash
sha256sum --check sparkring_*_arm64.deb.sha256
```

### Build from a full clone

Any Linux host with `git`, `dpkg-deb` and Python 3.12 or later can build; DGX
OS has all three. The package embeds the commit's history as a Git bundle, so
use a full clone (not `--depth`) with no uncommitted or untracked files.
`BRANCH` is a branch or tag:

```bash
git clone --branch BRANCH https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
python3 scripts/build_deb.py
```

- To pin a commit, run `git checkout --detach COMMIT` before building.
- The builder writes the package and a `.sha256` file to `.sparkring/dist/`
  (`--output DIR` for another directory), refuses to overwrite a package
  there, and prints the package path, its SHA-256, the source `revision` and
  the package `version`.
- Clones of one commit can produce packages with different SHA-256 values,
  because the embedded bundle depends on how the clone is packed; compare
  source revisions instead.

### Check the source revision

The package version ends in `+git` and the first 12 hexadecimal digits of the
source commit. For a local build, pass `.sparkring/dist/sparkring_*_arm64.deb`:

```bash
dpkg-deb --field sparkring_*_arm64.deb Version
```

Once installed, `/usr/lib/sparkring/distribution.json` records the full
commit as `revision`.

### Install the package on Node A

```bash
sudo apt install ./sparkring_*_arm64.deb
python3 -c 'import json; print(json.load(open("/usr/lib/sparkring/distribution.json"))["revision"])'
sparkring models
```

- For a local build, install `./.sparkring/dist/sparkring_*_arm64.deb`.
- `apt` may report that the download is performed unsandboxed as root;
  for a local file that is harmless.
- Where [bootstrap.sh](bootstrap.md) installed a Git checkout,
  `~/.local/bin/sparkring` runs that checkout for commands without `sudo`;
  `/usr/bin/sparkring` runs the package.

## Install a model

`sudo sparkring install --profile PROFILE` on Node A installs one installer
profile ([commands](install.md#install)) and ends with `Model ready:` and the
model's API URL on Node A. Below it a short card names the served model, the
API and [dashboard](dashboard.md) addresses, a sample `curl` request, and the
commands that switch back to the model it replaced, stop the model and remove
the package. Without `--profile`, it lists the installer profiles for the
cabled node count and asks for one.

### Questions and approvals

First use lists everything setup will do and asks `Proceed? [Y/n]` (Enter
approves). That covers:

- discovery, and trusting each cabled Spark's SSH host key on first contact
  (setup prints the recorded fingerprints);
- package installation and the administration network;
- fabric addressing, including replacing incompatible fabric IPv4 settings
  ([Setup and access](#setup-and-access));
- on four Sparks, the ConnectX driver restarts for the
  [hairpin setting](#four-spark-rings);
- the first model installation, unless the checkpoint plan downloads more than
  1 GiB or a Spark's search stopped early or failed. Then it prints the plan
  and asks `Proceed with this checkpoint plan? [y/N]`; answer `y`. Enter
  cancels; setup stays complete and the same command asks again.

Asked separately: each worker's SSH password when no key login exists, a
non-root worker account's sudo password (twice), and stopping any running GPU
container, which always needs its own answer or `--stop-workloads`. Setup
signs in to each Spark once; that Spark's inventory identifies its other
fabric functions and return paths.

### What a run does

It updates workers from Node A's package, reuses cached images and weights,
copies missing assets over verified fabric paths, and prepares assets before
stopping the running managed model; preparation never creates, starts or
stops a model container. If the switch fails, it attempts recovery from the
retained deployment and records the outcome. Before each model container
starts:

- **GPU access.** Docker must give the container its GPU through the NVIDIA
  CDI specification, `/var/run/cdi/nvidia.yaml`. Otherwise Docker falls back
  to the NVIDIA runtime hook, and the next systemd reload on the host (snapd
  performs them on its own) removes the container's GPU access; a starting
  model then stops with NVML `Unknown Error`. Each Spark therefore enables
  NVIDIA's `nvidia-cdi-refresh.service`, which DGX OS ships disabled, so the
  specification is written at every boot, and starts it when the
  specification is missing.
- **Memory.** Immediately before start, each Spark writes back dirty pages,
  drops its clean page cache and reclaimable kernel caches, and compacts free
  memory. A GB10's GPU allocates from the same memory, so the model starts
  from cleared memory whatever the Spark read before, and reads its weights
  from disk.

### Running the command again

- **Same package, model serving:** it keeps running. Serving means its
  container runs on every Spark and every ring check passes (four Sparks) or
  every fabric address is in RoCE GID index 3 (pair).
- **Model not serving**, for example after one Spark restarted: it stops the
  model on every Spark and starts it again, because the Sparks that stayed up
  keep a model waiting for the restarted one.
- **Package from another source revision**, even one differing only in
  documentation or profile text: a separate deployment, because the
  deployment lock and ID cover the source revision and the source bundle's
  SHA-256, and its directory, workspace and container names derive from the
  source revision. It is prepared, then replaces the running model, which
  restarts the model.
- **Preparation failed or was interrupted** before the switch, for example by
  a storage check, a download error or Ctrl-C: the same choices restart
  preparation, re-verify what the earlier attempt left and continue an
  unfinished checkpoint download.
- **A file missing from, or changed in, SparkRing's own checkpoint
  directory:** the run copies it from another Spark or downloads it again.
  A copy you named with `--model-path` and that SparkRing serves in place is
  never repaired; restore its files, or install without naming it.

For any other operation, inspect an execution receipt that says `running` or
`uncertain`, and the host state, before recovery; do not delete receipts to
force a blind retry.

### RoCE GID index 3

Installer containers use RoCE GID index 3 for every fabric function: profiles
set `NCCL_IB_GID_INDEX=3`, and the image's B12X RoCE transport reads one
`B12X_ROCE_GID_INDEX` (or `NCCL_IB_GID_INDEX`) for all of a rank's HCAs.

The index is not a fixed property of an address. When a link drops while a
model holds the address's GID entry, as on the Sparks cabled to one that
restarts, the address's RoCE v2 GID returns at another index, and a Spark's
ports facing the restarted Spark can differ from its other ports. Before a
pair's model starts, each Spark therefore re-adds a fabric address whose RoCE
GID has left index 3, as the [ring step](#the-rings-mesh) does on four
Sparks; that keeps one index valid for every HCA.

The pair's GID check and the ring step locate each address's RoCE v2 GID with
the resolver in
[`spark_roce_gid.py`](../../integrations/vllm/spark_roce_gid.py), which reads
the host's GID table. A failed pair check names the index the GID moved to, or
the RoCE v2 entries present when the address has none. To see the index on a
Spark:

```bash
python3 /usr/lib/sparkring/integrations/vllm/spark_roce_gid.py DEVICE ADDRESS
```

### Scripts and JSON

For an LLM or a repeatable installation:

```bash
sudo sparkring install --profile PROFILE --plan --json
sudo sparkring install --profile PROFILE --yes --json
sudo sparkring status --refresh --json
```

- `--json` writes one result to stdout; progress stays on stderr and in the
  log. Exit codes: 0 success or plan, 3 missing input, 2 failure.
- `needs_input` names the required choice, such as profile, approval, storage
  or checkpoint. `driver` means a four-Spark ConnectX driver restart cannot
  run; its details list what to stop or what did not complete.
- `--yes` approves, without a terminal, model replacement, first-use setup
  and, on an idle four-Spark ring, the ConnectX driver restarts. It does not
  trust unknown SSH host keys or authorize stopping unrelated workloads.
- `--plan` lists a needed four-Spark ConnectX step as `apply-hairpin-setting`.
- A configured ring is inspected without changing links; review cable or
  network changes separately with `sparkring setup`.
- With a [download limit](#limit-the-download-rate), the result holds it in
  `download_limit_bps`, in bits per second.
- A `complete` result also holds the card's fields: `dashboard_url` (null
  when the image has no dashboard), `example_request` (a `curl` command for
  `/v1/chat/completions` with the served model name) and `commands` with
  `switch_back` (null when no other model ran before), `stop` and `remove`.
- `--events FILE` adds a line-by-line progress stream
  ([Event stream](#event-stream)); the result on stdout stays the same.

### Event stream

`sudo sparkring install ... --events FILE` writes one JSON object per line to
`FILE` as the installation runs, replacing what `FILE` held. Stdout keeps its
single result and stderr its human progress, so the stream goes to its own
file. A reader follows it with `tail -f FILE`, or reads a named pipe made with
`mkfifo FILE`; the installation then waits until the reader opens the pipe.
The first event starts the run (`phase` `install`); the last carries its
result.

| Field | Present | Meaning |
|---|---|---|
| `schema` | always | `sparkring-install-event/v1` |
| `time` | always | ISO 8601 time with UTC offset |
| `state` | always | `start`, `working` (every 30 s while a step runs), `done`, `failed` or `error` (a failure's message); in the last event, the result's `complete`, `planned`, `needs_input` or `failed` |
| `label` | always | The progress text, such as `Node 0: Wait for API readiness` |
| `node` | always | The Spark's node number, or null for work that is not one Spark's |
| `phase` | always | A step key, such as `install`, `ready`, `start` or `model-fetch`, or null |
| `elapsed_s` | steps | Seconds since the step started |
| `detail` | `working` | The text appended to the "Still working" line |
| `bytes_done`, `bytes_total`, `percent` | download | Bytes of the Hugging Face download in place so far and in total |
| `rate_bps`, `eta_s` | download | Recent rate in bits per second and seconds left, once bytes move |
| `log_quiet_s` | readiness | Seconds since the model's log last changed |
| `message` | `error`, result | The failure or question text |
| `field` | `needs_input` result | The input the installation needs |
| `api_url` | `complete` result | The model's API address |

Fields may be added; readers ignore keys they do not know.

### Logs

Follow the installation from another terminal:

```bash
sudo sparkring logs --follow
```

- Concise timestamped progress is appended to
  `/var/log/sparkring/install.log`; long steps report that they are still
  working every 30 seconds.
- A Hugging Face download adds its progress to that line, for example
  `121 of 166 GiB, 850 Mb/s, about 7 min left`, measured from the bytes in its
  staging directory on Node A.
- While Node 0 waits for the API, the line names the model's startup step from
  the last 200 lines of its container log: `loading weights (shard 42/131)`,
  `weights loaded`, `compiling kernels`, `tuning kernels`,
  `setting up the KV cache`, `capturing CUDA graphs`, `warming up` or
  `starting the API server`. Without a recognized line it names none. After 5
  minutes without a new log line it adds, for example,
  `no new model log output for 6 min`. The wait ends after 30 minutes.
- Verbose command output goes to `install-details.log`;
  `sparkring logs --details --follow` shows it when investigating an error.
- Credentials entered through SSH are not recorded.
- Interactive terminals show colored results and a spinner; saved logs and
  piped output stay plain. The follower's spinner means it awaits log lines,
  not that the model is ready.
- `sparkring logs --plain` drops terminal decoration from the follower. For
  `sparkring install` itself, set `NO_COLOR` through `sudo`, which does not
  pass the caller's environment: `sudo env NO_COLOR=1 sparkring install ...`.
- From Windows PowerShell: `ssh -t NODE_A "sudo sparkring logs --follow"`
  (`NODE_A` is Node A's SSH address).

## Serving image and profiles

Every installer profile runs on one shared ARM64 serving image, pinned by the
[installer image lock](../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/installer-image.json)
(`sparkring-installer-image/v2`):

| Item | Identifier |
|---|---|
| Image | `ghcr.io/fujitsupolycom/sparkring@sha256:c977a2d2efb7ecf9ea856cd0379fdd93a8913f0770f4627a3ae00a084cfaf582` |
| Tag | `dev-20260928-plainstatus-cuda1342-nccl2323-status033` |
| Configuration | `sha256:4b7049d1e00f263c65713b62247a4497eba72fb38977087941830cec38609a8c` |
| Base image | `eugr/spark-vllm-b12x:nightly-20260924` |
| Parent image | `dev-20260928-toolchoice-cuda1342-nccl2323-status032` |

The image stacks SparkRing's layers on that base, from the CUDA 13.4.2 and
NCCL 2.32.3 toolchain up to the runtime-status dashboard 0.3.3. The
[installer image builders](../../runtime/images/installer-images.md) list each
layer, what it adds and what the lock pins; the image's
[publication record](../../runtime/releases/dev-20260928-plainstatus-cuda1342-nccl2323-status033/publication.json)
describes the layer it adds to its parent.

`sparkring models` lists exact model/version/quantization/topology profiles,
including guide-only ones with their guides, and marks only these as
installer-supported; family names such as `qwen` are ambiguous and rejected:

| Profiles | Checkpoint | Speculative decoding |
|---|---|---|
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | Qwen3.8 Flash Next NVFP4 QAD step 5500, revision `60215d26cf5e` (branch `qad-step5500-ple1000`) | MTP, three tokens, probabilistic drafting |
| `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash NVFP4-Spark, revision `a608241037e4` | MTP3 |
| `mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4` | MiMo-V2.6-Flash-RL, revision `5711b2681699` | DFlash5 |
| `deepseek-v41-flash-tp4` | DeepSeek-V4.1-Flash, revision `dba1be0a40aa` | DSpark, five tokens, probabilistic drafting, adaptive verification |
| `swift15-qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp4` | Swift 1.5 Qwen3.8-Flash-Next NVFP4, revision `3ff0520224f2` | MTP, three tokens, probabilistic drafting |

- All installer profiles run with SparkCache off and vLLM's native prefix
  cache on.
- Profiles that select other images, such as the `shared-2026.09.3` release,
  keep their own guides; `sparkring install` does not install them.
- `--image-lock FILE` replaces the shared lock for a development rehearsal and
  must list the selected profile.

### Tool-result contract

Installer containers set `SPARKRING_TOOL_CHOICE_CONTRACT=1`, and the installer
image carries the
[layer](../../integrations/vllm/tool_choice_contract/README.md#installer-images)
that reads it: a Chat Completions request whose `tool_choice` is `required` or
names a function fails with an error, instead of returning an empty result,
when generation ends without a complete call. A profile opts out with `0`; the
[tool-result contract](../../integrations/vllm/tool_choice_contract/README.md)
gives each check and the error it returns.

### Qwen profiles

Each Qwen profile's Settings table lists its prefill, decode, collective and
drafting settings
([two Sparks](../../profiles/qwen38-flash-next-tp2/README.md#settings),
[four Sparks](../../profiles/qwen38-flash-next-qad-tp4/README.md#settings)).
The
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md)
measures them; the
[step 5500 record](../../performance/records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md)
and the
[decode A/B](../../performance/records/qwen38-flash-next/decode-ab-20260925.md)
cover the checkpoint, LM-head and draft-backend choices.

Token-row ownership (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`) excludes HC
projection sharding. Before creating any serving container, admission reads
the image's external software receipt and refuses a profile whose HC mode is
not listed for its node count or whose features the image does not provide.

### Checkpoint loader and runtime binding

The external image's B12X checkpoint loader requires `io_uring`, so its
container uses the [pinned loader policy](../../third_party/moby_seccomp/README.md):
the Moby 29.2.1 default profile plus only the three `io_uring` calls. A
CPU-only probe checks them during asset preparation, before the running model
stops. Docker daemon defaults and host kernel policy are unchanged.

After creating each stopped container, the installer supplies a read-only
`sparkring-runtime-binding/v1` file with deployment, node, container, image
and rank identities, checks it before starting the model and never rewrites
it beneath a running container. Dashboards read it to label ranks. It is not
an attestation; boot identity and observation times are observed separately.

## Image distribution and caches

Node A downloads the serving image once and passes it to every Spark lacking
it. Docker's containerd image store is not supported.

### Check for the containerd image store

The installer identifies an image by comparing Docker's image ID with the
pinned configuration digest, and the containerd image store can report a
different digest as the image ID. Check each Spark before installing:

```bash
docker info --format '{{json .DriverStatus}}'
```

Output containing `io.containerd.snapshotter` means the containerd image
store; report that output in an issue before installing on that Spark.

### Registry relay

A registry-backed image reaches Sparks that lack it through a registry relay
on Node A:

- The relay serves only the pinned repository, on Node A's loopback address;
  workers reach it through an SSH remote forward on their own loopback
  address. It reads the registry anonymously, as public GHCR and Docker Hub
  repositories allow.
- It downloads each layer once, in parallel byte ranges, verifies it against
  its digest and serves the verified copy to every node. Nodes pull
  concurrently, so the Internet link carries the image once and every node
  unpacks it in parallel.
- Pulled images keep the reference `127.0.0.1:5255/<repository>@<digest>`.
- Node A needs room for the relay's layer cache, removed after distribution,
  besides its own pull.

Docker's pull skips a held layer only if Docker recorded its registry digest;
layers from `docker load` or a local build lack that record and would
download again. So the relay first reads the pinned manifest and image
configuration, and each node reports how many of the image's leading layers
its Docker image store holds. A node holding some receives an archive of the
configuration and only its missing compressed layers; `docker load` reuses
the held layers, checks each loaded layer against the configuration and tags
the image `127.0.0.1:5255/<repository>:<image-lock name>`. A node holding
none pulls.

Image distribution starts first, overlapping the prerequisite and source
checks. Checkpoint preparation waits until every node holds the image,
because downloads run the image's own Hugging Face client and every
checkpoint write is checked against free space after the image is in place;
image admission follows.

### Without a reachable registry

If the relay cannot reach the registry, or the registry requires credentials,
a Spark holding the image streams it to the others with `docker save` and
`docker load`. For a private image without a reachable registry, set
`image_reference` to its exact `image_id` and load it on one enrolled Spark;
the installer uses that stream without creating another export archive.
Streaming is slower than the relay: with Docker's overlay2 image store,
`docker save` writes the whole image to a temporary directory before sending
its first byte, and `docker load` unpacks only after receiving all of it.

### Compile and tuning caches

Source, image and profile choices select a separate deployment automatically;
there is no instance name or manual stop command to supply. Compile and B12X
tuning results share one cache per cluster, `/srv/sparkring/<cluster>/cache`,
in subdirectories keyed by model family, image and checkpoint revision, so
reinstalling or returning to a profile reuses its tuning.

Compiled B12X kernels are the exception: B12X keys each by its package source,
Python, torch, CUTLASS DSL and CUDA binding versions, compile environment and
GPU device UUID, so installer profiles keep them in a subdirectory named by
model family, CUDA toolkit version and checkpoint revision, and an image
update that leaves B12X unchanged reuses them.

A profile's first start on a Spark still includes that Spark's tuning: with
empty caches, the Qwen profiles reached API readiness in 546.8 s on TP2 and
476.9 s on TP4, on image `dev-20260925-cuda1342-nccl2323-status031`
with checkpoint revision `60215d26cf5e` and without the profiles' MXFP8
hyper-connection quantization or probabilistic drafting
([record](../../performance/records/images/dev-20260925-installer-profiles-20260925.md)).

## Status observations

`sparkring status` works from any directory and distinguishes cached or stale
observations, network configuration and saved model progress; `--refresh`
contacts the enrolled nodes and observes the model containers.
`sparkring status --json --refresh` reports:

- the saved deployment and image IDs, the deployment's checkpoint
  (`checkpoint`, `model_repository`, `model_revision`) and image release
  (`image_release`);
- host observations: persistent node ID, boot ID and their own `observed_at`;
  cached ones keep their original time and go stale after 90 seconds;
- container observations: the inspected container ID, start time and actual
  image ID.

Missing identities are `null` with a reason, never guessed from hostname or
rank. Network observations do not show whether the model is ready.

On a four-Spark ring, host observations also cover the
[ConnectX hairpin setting](#four-spark-rings). A function without it makes
the Spark `needs-attention`, with an error naming the function and its value
and the next action `on Node A: sudo sparkring hairpin`. `warnings` report a
setting in effect but not applied at boot, boot restarts suspended after a
failed restart (naming the function and time), a boot started with
`sparkring.hairpin=off`, and a mesh unit without the hairpin start check.

## When a model stops serving

A multi-Spark model stops serving when any rank stops, and its containers do
not restart themselves (`restart: 'no'`). To recover:

1. Save each Spark's model container log: `sudo docker ps -a` names the
   container; keep `sudo docker logs CONTAINER`.
2. Run the command that installed the model again,
   `sudo sparkring install --profile PROFILE`; it stops the model on every
   Spark and starts it again.

Saved-log lines beginning `RoCEnante rank` tell which rank was late and why.
`sudo sparkring status --refresh --json` shows rank 0's container with
`running: false` while the others still run.

Ranks exchange every all-reduce and all-gather over RoCE with the prepared
RoCEnante transport. Its
[supervised peer wait](../../integrations/vllm/rocenante_prepared/README.md#peer-wait)
waits up to `B12X_ROCE_PEER_TIMEOUT_S` seconds (300 by default) for a late
peer whose queue pairs still answer, logging each wait over 5 s. It stops only
when a peer is unreachable, sends a stop notice or reports an inconsistent
sequence, or at that limit; it then tells the other ranks, whose logs name the
cause, and the stopping rank poisons its runtime. Rank 0's container exits;
the other Sparks' containers keep running without serving. A Spark that
restarts, or whose model container exits, stops the model the same way.

Containers do not restart themselves because the installer starts every rank
together, right after clearing each Spark's page cache, and a single
restarted rank cannot rejoin the others.

## Setup and access

The Spark you run setup on becomes Node A. Setup finds neighbors over IPv6
link-local addresses, copies SparkRing and its Debian dependencies through
the fabric and shows the proposed network changes.

**Accounts.** Setup signs in to each worker as the account that ran `sudo`;
`sparkring setup --ssh-user USER`, or `SPARKRING_SSH_USER` in the settings
file passed to `sparkring install --env`, selects another. With a settings
file, its `SPARKRING_SSH_USER` applies and defaults to `root`. Installer
operations on a Spark run as root, through `sudo -n` when its SSH account is
not `root`.

**Access check.** Before changing anything, `sparkring install` confirms
noninteractive SSH and `sudo` on every enrolled Spark. A missing grant
returns `needs_input` with field `access` and, per Spark, a one-time command a
person runs there, which prompts for that Spark's password
([what it grants](#security-and-host-exposure)). SparkRing never accepts
passwords as options or settings.

**Fabric addresses.** Compatible existing addresses are kept. For
incompatible ones, interactive setup prints the problem and replaces the
fabric IPv4 settings under the single `Proceed?` approval, whose list names
this step, with no further question. A run without a terminal stops there
unless `SPARKRING_LINK_POLICY=reset` in the settings file or
`sparkring setup --reset-links` requested the replacement. Setup saves
connection backups and receipts, and keeps the active NetworkManager
connection identity and IPv6 address generation while changing fabric
IPv4/MTU settings, so its administration path survives renumbering.

## Four-Spark rings

Four-Spark rings need the ConnectX hairpin setting on every Spark; pairs do
not use it. SparkRing applies it itself; a first installation needs no flag or
separate step.

### The hairpin setting

Every four-Spark installer profile (`qwen38-flash-next-qad-tp4`,
`glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-rl-tp4`,
`deepseek-v41-flash-tp4` and `swift15-qwen38-flash-next-tp4`) relays traffic
between nonadjacent Sparks through ConnectX hardware forwarding. That needs,
on each of a Spark's four ConnectX functions, a hairpin queue of 8192 packets
(`hairpin_queue_size`), four hairpin queues (`hairpin_num_queues`) and
hardware TC offload.

The driver starts each boot at 1024 packets and uses the larger queue only
after that function's driver restarts
(`devlink dev reload … action driver_reinit`), which takes the function's link
down for about 8 seconds. SparkRing restarts the functions:

- **At the first installation**, whose approval question lists the restarts:
  each function once, after fabric addressing is configured, Node A first,
  then one worker at a time, about 30 seconds per Spark.
- **At every boot**: `sparkring-hairpin.service` restarts each function before
  NetworkManager starts, adding about 30 seconds per boot. SparkRing enables
  it on a Spark after a run there in which every restart succeeded; from then
  on, a reboot needs no manual step for the setting.

SparkRing never restarts a driver while a model, a mesh service, mesh
forwarding rules or another RDMA program is present on the ring; it lists
what to stop and restarts nothing.

### Apply or repair the setting

On Node A, with this package installed, run this for any installed ring whose
Sparks lack the setting or its boot service:

```bash
sudo sparkring hairpin
```

That includes a ring set up by a SparkRing package without
`sparkring-hairpin.service`: it has no boot record for the setting, and a
reboot leaves its mesh services stopped by their start check.

- It lists what it will change, including a SparkRing package update on
  workers running another revision, and asks once. `--plan` only prints the
  list; `--yes` approves without asking.
- If the setting is already in effect, it only records the approval and
  enables the boot service, without a restart.
- If an updated worker then needs a restart the list did not show, it stops
  and asks for that restart first.
- Once every Spark has the setting, it starts each enabled mesh service the
  start check refused.
- `--json` prints one `sparkring-hairpin-result/v1` document; exit codes are
  those of `sparkring install`. Each run leaves a receipt below
  `/var/lib/sparkring/controller/hairpin/`.
- `sudo sparkring hairpin --revoke` stops applying the setting at boot on
  every Spark. The setting stays in effect until each Spark reboots; after
  that, the start check keeps each mesh service stopped until the setting is
  applied again.

`sudo sparkring install` includes the same step in its question when a Spark
needs it. `sparkring setup` and `sparkring install` accept
`--allow-driver-reload`, which is not needed.

### Missing setting or failed restart

Without the setting in effect, a Spark's mesh service does not start and its
log names the function, `sparkring status` reports the Spark as
[`needs-attention`](#status-observations), and `sparkring up --execute` starts
nothing. Run `sudo sparkring hairpin` on Node A; `sudo sparkring install`
starts the mesh its installation uses ([The ring's mesh](#the-rings-mesh)).

- If a restart fails during a boot, or a boot ends during a restart, later
  boots of that Spark restart nothing, and `sparkring status` warns, until
  `sudo sparkring hairpin` succeeds.
- A Spark unreachable after a failed restart becomes reachable after a power
  cycle, without the setting.
- A failed restart of a function carrying the administration network to
  other Sparks cuts them off from Node A, while the Spark itself stays
  reachable over its other links; `sparkring status` then names the Spark to
  reboot. Its next boot restarts no function, and `sudo sparkring hairpin`
  then retries it.
- To boot once without the restarts, add `sparkring.hairpin=off` to the
  kernel command line.

On a Spark:

| Command | Shows |
|---|---|
| `journalctl -b -u sparkring-hairpin.service` | The boot run |
| `sudo sparkring node hairpin status` | Each function's setting |
| `sudo sparkring node hairpin apply --dry-run --boot` | The restarts that the next boot performs, in order, or none while the service is not enabled |

### The ring's mesh

Every four-Spark installer profile runs on a native mesh; the installer
renders each rank's container from the profile's shared container
specification.

- It reuses the mesh service every Spark has enabled or running.
- If no Spark has one, it downloads and verifies the pinned host marker on
  every rank, creates stopped model containers, installs supervised mesh
  services, waits for every rank and starts the model.
- A mesh that only some Sparks have, or that differs between them, stops the
  plan for inspection.

SparkRing enables the mesh service, so it starts at each boot after the
hairpin setting. Before an installation starts the model, each Spark starts
its mesh service if stopped, restarts it if its routes or forwarding rules are
missing, and re-adds a port's IPv4 address whose RoCE v2 GID is not at the
pinned [GID index 3](#roce-gid-index-3). The installation then waits up to
four minutes for the ring check on every Spark.

`sparkring install` does not replace an existing mesh:
`sparkring up PROFILE --fresh-mesh --plan` prints an explicit replacement
plan, and `sparkring up PROFILE --instance fresh --fresh-mesh` rehearses the
replacement beside an existing deployment. A replaced deployment's container,
source and weight directories are retained.

## Security and host exposure

Review this list before approving `Proceed? [Y/n]`; it is how the installer
changes each Spark's network exposure.

**Open model API.** The OpenAI-compatible API listens on all of Node A's
interfaces with no API key, so anyone who can reach its port can use the
model. Keep Node A on a trusted network or firewall the port. Containers use
host networking, and the runtime-status dashboard
(`/v1/sparkring/status/view`) answers on the same port:

| Port | Profiles |
|---|---|
| 8000 | `qwen38-flash-next-tp2`, `glm53-flash-nvfp4-spark-tp2`, `swift15-qwen38-flash-next-tp2` |
| 8015 | `qwen38-flash-next-qad-tp4`, `glm53-flash-nvfp4-spark-tp4`, `deepseek-v41-flash-tp4`, `swift15-qwen38-flash-next-tp4` |
| 8020 | `mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4` |

**Passwordless sudo.** When noninteractive `sudo` is missing, the command
`needs_input` prints for a Spark writes `USER ALL=(ALL) NOPASSWD:ALL` to
`/etc/sudoers.d/USER` for the SSH account, giving it passwordless root on that
Spark. Review it before running it.

**Administration network.** Setup creates a WireGuard network over the fabric
links' IPv6 link-local addresses: interface `sr-control`, UDP port 51871,
addresses in `10.253.255.0/29` by default. A separate SSH service
(`sparkring-access.service`) listens on TCP port 2222 of each Spark's
administration address and admits only root with Node A's controller key,
`/var/lib/sparkring/controller/controller_ed25519`.

**Setup SSH service.** During first-use setup, `sparkring-seed.service` runs
an SSH server on TCP port 2222 on all IPv6 addresses of Node A and of any
worker prepared with `sparkring setup --worker-bundle`, admitting only root
with Node A's controller key. Setup disables it once the administration
network works on every Spark. If setup stops earlier, it stays enabled;
`sudo systemctl disable --now sparkring-seed.service` removes it.

**Internet sharing.** On by default (`SPARKRING_SHARE_INTERNET=yes`). Node A
forwards and masquerades traffic from the administration network and answers
workers' DNS with `dnsmasq`; each worker accepts every IPv4 destination
through Node A in its WireGuard configuration (`AllowedIPs 0.0.0.0/0`) and
sends DNS for all domains to Node A. If workers have their own network
connection, put `SPARKRING_SHARE_INTERNET=no` in a settings file and pass it
with `--env` on the first installation. Four-Spark rings whose workers have
no connection of their own need sharing
([outbound hosts](#downloads-storage-and-outbound-hosts)).

**Host announcement.** The package enables `avahi-daemon` (mDNS) and `lldpd`
(LLDP) with their default configuration, which announces on every interface,
and publishes a SparkRing mDNS service, `/etc/avahi/services/sparkring.service`.

Package removal leaves some of these in place
([What is installed](#what-is-installed)).

## Downloads, storage and outbound hosts

Node A needs outbound HTTPS to these hosts. Image and checkpoints are read
anonymously, with no registry or Hugging Face account.

| Host | Used for |
|---|---|
| `raw.githubusercontent.com` and `github.com` | The one-line command: `install.sh`, then a full clone of the branch, on Node A |
| `ghcr.io`, which redirects to `pkg-containers.githubusercontent.com` | The serving image, downloaded once by Node A's relay |
| `huggingface.co`, which redirects to its download CDN | Checkpoint files that no Spark holds, downloaded once by Node A |
| Your Ubuntu package mirror | The package's dependencies during `apt install` on Node A |
| `github.com` and its release-asset download host | Four-Spark rings only: every rank downloads the native mesh host marker from `https://github.com/FujitsuPolycom/sparkring/releases/download/r33-host-tools-c8646b0/mlx5-rdma-tx-marker` when the installer creates a mesh |

Workers receive the package, image and checkpoint from Node A; in
`sparkring install`, only Node A contacts huggingface.co. Each rank of a
four-Spark ring downloads the mesh marker itself; workers reach the Internet
through Node A's sharing unless they have their own connection.

The installer checks free space before each download or copy and before the
model switch. It deletes nothing to make room, and a failed check leaves the
running model in place; the message names `sudo sparkring storage`
([Finding and freeing space](#finding-and-freeing-space)). It requires:

- **Image:** on each Spark without the image, the unpacked size plus the
  download size plus 8 GiB (51.7 GiB) free in Docker's data root. Node A needs
  a further 14.2 GiB there for the relay's layer cache.
- **Checkpoint:** the bytes a Spark copies, receives or downloads, plus the
  largest of those files again for its staging copy. Hard-linked files need
  no space.
- **Caches:** 32 GiB for the compile cache, and 68 GiB in Docker's data root
  while the image is absent.

The printed plan shows each Spark's total, which also counts the 32 GiB
compile cache when it shares the checkpoint's filesystem and, on a Spark
without the image, 68 GiB for the image. `sparkring up`, whose image step pulls the image itself,
reserves the full checkpoint allowance of
[storage planning](../../profiles/storage-planning.json) unless the Spark
holds a verified checkpoint.

The serving image `dev-20260928-plainstatus-cuda1342-nccl2323-status033` is a
14.2 GiB download, 29.5 GiB unpacked. The last column below is the plan's
total for a Spark holding neither the image nor any checkpoint file, with the
checkpoint, Docker and the cache on one filesystem.

| Checkpoint | Size | `sparkring up` allowance | Plan total, empty Spark |
|---|---:|---:|---:|
| Qwen, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `60215d26cf5e` | 102.6 GiB | 120 GiB | 206.6 GiB |
| MiMo, `XiaomiMiMo/MiMo-V2.6-Flash-RL` @ `5711b2681699` | 165.6 GiB | 190 GiB | 278.0 GiB |
| GLM, `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` @ `a608241037e4` | 174.8 GiB | 200 GiB | 279.5 GiB |
| DeepSeek, `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa` | 475.3 GiB | 500 GiB | 669.8 GiB |
| Swift, `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` @ `3ff0520224f2` | 173.7 GiB | 200 GiB | 281.7 GiB |

### Limit the download rate

```bash
sudo sparkring install --profile PROFILE --download-limit 850Mbit
```

`RATE` is megabits or gigabits per second, as network links are rated:
`850Mbit`, `1.5Gbit` or `2Gbit` (any letter case, at least `1Mbit`), or
`none`. Without the flag, the [preference](#optional-preferences)
`SPARKRING_DOWNLOAD_LIMIT` of an `--env` file applies; the default is `none`.
The plan prints the limit when it downloads checkpoint files.

- It caps only checkpoint files that Node A downloads from huggingface.co.
  Image pulls, copies between Sparks and `sparkring up` are not limited.
- The download container, which runs the serving image's Hugging Face client,
  paces every TLS read to the limit, so TCP flow control slows the sender. No
  host network setting changes.
- With a limit, the client's parallel downloaders (Xet and `hf_transfer`) are
  turned off and each file arrives over one HTTPS stream. A limit above what
  one stream reaches has no further effect.

### Finding and freeing space

`sudo sparkring storage` reports each Spark's filesystem use and everything
SparkRing keeps there, counting hard-linked files once:

- checkpoint directories;
- compile caches, one directory per model family, image and checkpoint
  revision (`/srv/sparkring/<cluster>/cache/<family>-<image>-<revision>`, and
  `<family>-cuda<version>-<revision>` for kernels that the installer images
  share). A cache of an image or checkpoint that no installer profile selects
  stays until it is released;
- deployment workspaces, one per installation request
  (`/srv/sparkring/<cluster>/<profile>-i<identity>`); only the installed
  deployment's is in use;
- Docker images, and every other entry of `/srv/sparkring`, such as
  directories you created there.

Each item is `installed`, `profile`, `unreferenced` or `unmanaged`
([classes](commands.md#storage)), and the report ends with the release
commands for the unreferenced ones.

- `sudo sparkring storage --release PATH` removes one unreferenced cache
  directory or workspace from every Spark that holds it, after asking
  (`--yes` in scripts). Each Spark first checks again that no installed
  deployment or running container uses it.
- `sudo sparkring checkpoints --release PATH` releases checkpoint directories
  ([Checkpoints](#checkpoints)).
- Docker images and directories that SparkRing's installer did not create are
  never removed.

## If a worker has no SSH

A machine without remote access needs one local preparation step:

1. On Node A, run `sudo sparkring setup --worker-bundle`.
2. Copy the printed archive to a USB drive and extract it on each worker.
3. On each worker, run `sudo python3 install.py --apply --prepare`. It
   installs the bundled packages and enables access with Node A's public key.
4. On Node A, run `sudo sparkring setup --ssh-port 2222`.

Private keys and passwords are not copied to workers. The preparation listener
is the [setup SSH service](#security-and-host-exposure), disabled once
administration access works on every node.

## Optional preferences

Interactive setup needs no settings file. Repeated installations can use one:

```dotenv
SPARKRING_NAME=home
SPARKRING_SSH_USER=operator
SPARKRING_SHARE_INTERNET=yes
SPARKRING_LINK_POLICY=keep
```

`sudo sparkring install --env /path/to/settings.env` reads the setup keys
only for first-use setup, before the cluster is configured, and
`SPARKRING_DOWNLOAD_LIMIT` on every run;
`sudo sparkring setup --env /path/to/settings.env` reads it on every run.
Values are parsed as literal settings, never sourced as shell code, and SSH
handles credential prompts.

| Key | Default | Accepted values | `sparkring setup` option |
|---|---|---|---|
| `SPARKRING_NAME` | `sparkring` | Lowercase letters, digits and `-`, starting with a letter, at most 35 characters | `--name` |
| `SPARKRING_SSH_USER` | `root`; without a settings file, the account that ran `sudo` ([Setup and access](#setup-and-access)) | A Linux account name | `--ssh-user` |
| `SPARKRING_SSH_PORT` | `22` | `22` or `2222` | `--ssh-port` |
| `SPARKRING_SHARE_INTERNET` | `yes` | `yes` or `no` | `--no-share-internet` |
| `SPARKRING_CONTROL_CIDR` | `10.253.255.0/29` | An IPv4 `/29` network | `--control-cidr` |
| `SPARKRING_FABRIC_CIDR` | `198.18.0.0/21` | An IPv4 `/16` through `/21` network that does not overlap the control network | `--fabric-cidr` |
| `SPARKRING_LINK_POLICY` | `keep` | `keep`, or `reset` to request reviewed replacement of fabric IPv4 settings | `--reset-links` |
| `SPARKRING_DOWNLOAD_LIMIT` | `none` | `none`, or a rate such as `850Mbit` or `2Gbit` ([details](#limit-the-download-rate)) | `sparkring install --download-limit`, which takes precedence |

`sparkring install` passes only `--env`, `--yes` and `--stop-workloads` to
setup. `sparkring setup --plan` discovers and reviews through existing access
without configuring hosts; `sparkring install --plan` saves an installation
plan without updating workers or models.

## What is installed

The Debian package contains the CLI, host services, profile and deployment
code, an immutable source bundle and a file manifest, but **no model weights,
CUDA stack or inference image**: [images](images.md) supply the serving
software, and profile adapters own image admission and model startup. The
[standalone Compose files](compose.md) are usable on their own.

- Configuration is in `/etc/sparkring/`; private controller state and
  receipts are in `/var/lib/sparkring/controller/`.
- Workers use a private WireGuard administration tree over their existing
  IPv6 link-local fabric addresses
  ([Security and host exposure](#security-and-host-exposure)). Optional
  Internet sharing routes downloads and DNS through Node A; it does not move
  inference collectives into WireGuard.

**At boot**, after the [hairpin setting](#four-spark-rings) on four-Spark
rings, SparkRing's enabled host services start the administration network and
its SSH service, also over the remaining links if one administration link
fails, and `sparkring-fabric.service` restores the approved fabric routes,
per-interface IPv4 forwarding and forwarding rules; NetworkManager keeps the
fabric addresses. Models start only when requested; `sparkring down` stops
the selected deployment.

**Mesh start check.** The package's systemd generator,
`/usr/lib/systemd/system-generators/sparkring-hairpin-mesh-check`, adds a
start check to each mesh unit in `/etc/systemd/system`
(`sparkring-mesh.service` and `sparkring-*-mesh.service`) without changing
the unit file: on a four-Spark ring the unit starts only when the hairpin
setting is in effect; elsewhere the check passes. Each Spark of a four-Spark
ring records its approval of the setting in `/etc/sparkring/hairpin.json`.

**Removal.** `sudo apt remove sparkring` records which SparkRing host
services are enabled, then disables and stops them, including both SSH
services on port 2222 and the worker DNS service, and deletes the SparkRing
mDNS service file. It keeps:

- configuration, weights, caches, receipts and network state;
- any running model deployment, which keeps serving its open API;
- the passwordless sudo file, and `avahi-daemon` and `lldpd` enabled;
- until the Spark reboots, the WireGuard interface, forwarding settings and
  iptables rules already applied.

Reinstalling the package re-enables the recorded services and starts the
administration services; the fabric and hairpin services are enabled for the
next boot, not started. Installing or updating the package never restarts a
ConnectX driver. No command reverts setup's network, SSH, sudo or service
changes.

## Checkpoints

SparkRing keeps one checkpoint directory per cluster and checkpoint revision,
`/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>`, shared by
every deployment of that revision. It holds exactly the files the pin
manifest in [`profiles/checkpoints/`](../../profiles/checkpoints) requires (53
for Qwen); the profile's `SHA256SUMS` lists the same files.

### Another checkpoint of a profile

The Qwen profiles list two checkpoints by Hugging Face branch:
`qad-step5500-ple1000`, the default, also named `qad-step-5500`, and
`qad-step-4000`. `--checkpoint NAME` installs a listed one with the settings
it needs, as its own deployment with its own pinned revision and checkpoint
directory; an unlisted name changes nothing. Installing again without
`--checkpoint` switches back to the default.

```bash
sudo sparkring install --profile qwen38-flash-next-tp2 --checkpoint qad-step-4000
```

### Where the installer looks

Before printing the plan, `sudo sparkring install` searches every Spark at
once:

- SparkRing's own checkpoint directories and records;
- Hugging Face caches in every account's home, and those that `HF_HOME`,
  `HF_HUB_CACHE` and similar variables name in system files, systemd units and
  root's and your own shell files;
- Docker containers' model mounts, volumes and Hugging Face caches;
- folders with any name, such as `hf download --local-dir` folders and git-lfs
  clones, under `/var/tmp/models`, `/models`, home directories, `/data`,
  `/srv`, `/mnt`, `/opt` and other local disks.

Each Spark searches for up to 20 s, and once more for up to 40 s if that runs
out; Docker queries take up to 15 s, so a Spark's search ends within 75 s. It
runs as root and reads directory listings, file sizes, download metadata and
SparkRing's records, home directories included; it hashes only files up to
64 MiB, does not read other accounts' shell files and never enters network
storage or automounts. The plan names what was not searched.

If the search missed your copy, check the plan's `Not searched` line and
whether a Spark's search hit its time limit, then repeat the plan; a repeated
search is faster. Otherwise name the copy with `--model-path N=PATH`.

### What it does with a copy

Files are identified by SHA-256 against the pin manifest, not by name: a copy
of the repository's `main` branch supplies every file of Qwen checkpoint step
4000 (`--checkpoint qad-step-4000`) except `config.json`. SparkRing hashes
each file before using it.

- Weight files on the same filesystem as SparkRing's directory are hard-linked
  into it and take no extra space. That changes only the link count and change
  time of your weight files; a backup tool comparing change times reads them
  once more.
- Other files (configuration, tokenizer, chat template) are copied, so later
  edits to your copy do not reach the served model.
- A copy in another account's home is listed but used only when named with
  `--model-path`.
- Files no Spark holds are downloaded once by Node A.

**SparkRing never writes, moves or deletes files it did not create.**

Copies between Sparks travel over the fabric cables outward from a Spark that
holds the checkpoint. The receiver binds only its fabric addresses and
accepts only the sender's address and a one-time token sent over
administration SSH. If a direct copy fails, rsync over administration SSH
fills in: it sends only files the receiver lacks, into an empty staging
directory beside SparkRing's directory, reading them without changing their
access times (`--open-noatime`, which needs rsync 3.2.3 or later on both ends;
Ubuntu 24.04 ships 3.2.7). Each received file is placed only after its
SHA-256 matches.

Later starts compare each file's recorded device, inode, size and times, and
re-hash a changed file. The model switch's last step,
`Confirm checkpoint unchanged during loading`, compares them again on every
Spark once the model has loaded; a change fails the switch and restores the
previous deployment.

### Plan approval

The plan shows, per Spark, its sources and their owners, how many files it
hard-links, and the bytes it copies, receives or downloads.

A plan reviewed with `--plan`, or approved at a terminal prompt, bounds every
later `--yes` run. Such a run searches again and proceeds only while its plan
stays within the reviewed one: no new downloads, at most 1 GiB more written
per Spark, and the same mode and no new source folders on each Spark.
Otherwise it stops, prints the difference and keeps the reviewed plan as the
bound, so repeating `--yes` stops again: review the changed plan with
`--plan`, then repeat `--yes`. The refused plan is saved as
`checkpoint-plan.refused.json` beside the reviewed one.

- A download or write the approved plan does not include stops the
  installation with `needs_input` (field `checkpoint`) before it starts; the
  running model is unchanged.
- Setup's approval of a first installation does not cover a download over
  1 GiB or a search that stopped early or failed: a terminal asks
  `Proceed with this checkpoint plan? [y/N]`
  ([Questions and approvals](#questions-and-approvals)), and a run without
  one stops.
- A plan whose problems stop the installation, such as a named path that
  exists on no Spark or too little free space, records no deployment.
- Commands suggested in messages repeat your `--profile`, `--model-path`,
  `--cache-path` and `--image-lock` options, which identify the deployment.
- With `--json`, the result carries a plan summary as `checkpoint`; the full
  plan, with each file's action, is `checkpoint-plan.json` in the result's
  `deployment` directory.

### Options

- `--model-path PATH` names a copy for every Spark, `--model-path N=PATH` one
  for Node N; repeat as needed. Named copies are used first; the search still
  runs.
  - A named copy on another filesystem than SparkRing's directory that holds
    exactly the pinned files is served in place, read-only; the model does not
    start while that folder is changed or missing. Serving in place is decided
    when the deployment is first planned; if the copy later changes, the
    installation stops and says how to continue.
  - SparkRing never creates or writes a named path.
  - Named paths are part of the deployment: other named paths plan another
    deployment, so repeat every command with the same `--model-path` options.
- `--ignore-local-copies` uses only SparkRing's own checkpoint directories and
  named copies.
- A `.sparkring-ignore` file keeps SparkRing out of its folder and everything
  below, even a named path. For a Hugging Face cache, put it in `$HF_HOME`
  (the parent of `hub`) or in a `models--*` folder; in `hub` itself it makes
  `hf cache scan` report an error.
- `--cache-path /absolute/cache` chooses another writable compilation cache.

`sparkring up PROFILE` does not search: each Spark uses SparkRing's checkpoint
directory and downloads the files it lacks itself, with the serving image's
Hugging Face client. That download runs before `up` pulls the image, so a
Spark that lacks checkpoint files must already hold the image
(`sudo sparkring install` distributes it). `sparkring up --model-path PATH`
serves a complete copy in place.

### Disk space held by links

While SparkRing's directory links a copy's weight files, deleting that copy or
pruning the Hugging Face cache frees nothing; release SparkRing's directory
instead.

- `sudo sparkring checkpoints` lists SparkRing's checkpoint directories on
  every Spark, the deployments using each, the copies it shares files with and
  the space a release frees.
- `sudo sparkring checkpoints --release PATH` removes one from every Spark
  after asking (`--yes` in scripts). It refuses a directory that the active
  deployment or the rollback target uses, also one served in place, and one a
  running container mounts. It names other deployments using it; they need
  `sudo sparkring install` again. It removes only the names and directories
  SparkRing placed and never writes, moves or deletes the copies they were
  linked from.

## Local build and tests

[Get the package](#get-the-package) builds the Debian package. The package has
no compiled host payload, so it can also be assembled on x86 Linux.

The tests need pytest and the other packages in
[requirements-dev.txt](../../requirements-dev.txt), which DGX OS does not
ship; install them in a virtual environment. On Ubuntu 24.04, including DGX
OS, `python3 -m venv` needs `python3-venv`
(`sudo apt-get install python3-venv`). From a clean committed checkout on
Linux:

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

## Lower-level commands and Compose sharing

`sparkring install` is the supported entry point. The underlying steps are
available for inspection and rehearsals:

```bash
python3 scripts/sparkring.py init --model MODEL --host spark0 --host spark1
python3 scripts/sparkring.py up            # review the stages
python3 scripts/sparkring.py up --execute
python3 scripts/sparkring.py status --refresh
python3 scripts/sparkring.py down --execute
```

- `init` discovers addresses read-only and saves `.sparkring/deployment` with
  the shared installer image. `--model` accepts `glm53`, `mimo26` or `qwen38`;
  the host count selects the profile.
- For an offline site, fill in
  [the site example](../../profiles/install-site.example.json) and pass
  `--site YOUR_FILE` instead of `--host`.
- No live command runs without `--execute` except discovery and status
  refresh.

A site's checkpoint path (`<workspace>/models/<revision>` unless a row names
another) becomes a SparkRing checkpoint directory. Docker's download mount and
rsync's destination name its staging directory by path, so SparkRing claims
the path only when:

- every existing directory above it is writable by its owner alone or carries
  the sticky bit, as `/tmp` does; and
- an existing directory at the path is empty, owned by root and writable by
  root alone.

Keep it below directories that only root or the operator can change: an
account that owns a directory above it could still rename what is inside.

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
