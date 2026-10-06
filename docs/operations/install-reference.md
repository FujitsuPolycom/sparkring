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
model's API URL ([API endpoint](#api-endpoint)). Below it a short card names the served model, the
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
container, which always needs its own answer or `--stop-workloads`. In a
terminal, a run without `--yes` and without the API options also asks, before
it searches the Sparks, which address the model's API listens on and which
port it uses ([API endpoint](#api-endpoint)); Enter keeps both. Setup
signs in to each Spark once; that Spark's inventory identifies its other
fabric functions and return paths.

### What a run does

It updates workers from Node A's package through a bundle of the package and
its dependencies. Each worker's copy in `/var/tmp` is removed once that
worker runs Node A's revision, and Node A's copy when the update ends. It
reuses cached images and weights, copies missing assets over verified fabric
paths, and prepares assets before stopping the running managed model;
preparation never creates, starts or stops a model container. If the switch fails, it attempts recovery from the
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

Once the model serves, the run releases what older deployments hold on the
Sparks ([automatic release](#automatic-release)).

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
- **Switch interrupted** while it stopped the running model or started the
  selected one, for example because a Spark restarted: the same command
  resumes the switch. Another profile or checkpoint replaces it instead: after
  the GPU check, the installer stops the unfinished model through its own
  deployment and switches from the last model that served.
- **Selected model's last start did not complete**, for example when a first
  installation failed its readiness check or a restart of the installed model
  failed: the same command first stops that model on every Spark, then
  prepares and starts it again. The stop checks ownership labels and stops
  only that model's containers, so another model that serves keeps serving
  until the switch. A stop of the selected model that did not complete
  finishes first; a stop action whose outcome is uncertain, for example one
  that timed out, stops the command before any preparation until its receipt
  and the Spark are inspected.
- **A file missing from, or changed in, SparkRing's own checkpoint
  directory:** the run copies it from another Spark or downloads it again.
  A copy you named with `--model-path` and that SparkRing serves in place is
  never repaired; restore its files, or install without naming it.

For any other operation, inspect an execution receipt that says `running` or
`uncertain`, and the host state, before recovery; do not delete receipts to
force a blind retry.

### A start that does not become ready

Once every Spark's model container runs, Node A waits for the API on Node 0
(`Node 0: Wait for API readiness`). Node N is rank N. The start fails, and
the console prints the error after `Error:`, when:

| Error | Cause |
|---|---|
| `API rank exited during startup` | Node 0's model container exited. |
| `API health check failed; inspect this rank's logs` | Node 0's container health check reported `unhealthy`. |
| `Readiness exceeded 30 minutes; inspect logs before restarting` | The API was not ready after 30 minutes. |
| `Rank N's model container on HOST stopped while rank 0 was loading (exit code CODE)` | Docker on Node N reports its model container stopped. |
| `Rank N's model container on HOST was removed while rank 0 was loading` | Docker on Node N has no model container. |

During the wait, Node A asks every other Spark every 5 seconds whether its
model container runs, with the check that confirmed the container started.
Only Docker's answer counts: a check that cannot reach the Spark, takes
longer than 60 seconds or fails in any other way is repeated 5 seconds later,
and a Spark still loading the model answers that its container runs. A
stopped container on another Spark therefore fails the start within about 10
seconds; Node 0's model itself notices a missing rank only when its
collective operations time out, about 11 minutes on two Sparks.

- `install-details.log` (`sudo sparkring logs --details`) holds the last 5
  lines of the stopped container's log, above the error.
- Node 0's model container keeps running until the model is stopped or it
  gives up on the missing rank.
- The start is recorded as incomplete, as for every error in the table.

### RoCE GID index 3

Installer containers run NCCL on RoCE GID index 3 for every fabric function:
profiles set `NCCL_IB_GID_INDEX=3`. The installer image's RoCEnante transport
instead reads each HCA's index at startup and uses index 3 only for a device
whose GID table does not identify its fabric address
([GID index per port](../../integrations/vllm/rocenante_prepared/README.md#gid-index-per-port)).

The index is not a fixed property of an address. When a link drops while a
model holds the address's GID entry, as on the Sparks cabled to one that
restarts, the address's RoCE v2 GID returns at another index, and a Spark's
ports facing the restarted Spark can differ from its other ports. Before a
pair's model starts, each Spark therefore re-adds a fabric address whose RoCE
GID has left index 3, as the [ring step](#the-rings-mesh) does on four
Sparks, because NCCL needs index 3 to hold the address's GID on every HCA.

The pair's GID check and the ring step locate each address's RoCE v2 GID with
the resolver in
[`spark_roce_gid.py`](../../integrations/vllm/spark_roce_gid.py), which reads
the host's GID table. A failed pair check names the index the GID moved to, or
the RoCE v2 entries present when the address has none. To see the index on a
Spark:

```bash
python3 /usr/lib/sparkring/integrations/vllm/spark_roce_gid.py DEVICE ADDRESS
```

Index 3 also needs exactly one IPv6 link-local address per fabric function,
derived from its hardware address. SparkRing's fabric connections use
NetworkManager's `ipv6.addr-gen-mode eui64`. A connection made by hand often
uses `default` or `stable-privacy`, which adds another link-local address and
moves the IPv4 GID to a higher index. Setup then stops, names the connection
and the fix:

```bash
nmcli connection modify CONNECTION ipv6.addr-gen-mode eui64
nmcli connection up CONNECTION
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
- With `--api-address`, `api_url`, `dashboard_url` and `example_request` use
  that address, and `check_url` holds the URL that SparkRing's own checks use
  ([API endpoint](#api-endpoint)). `serving` lists `api_port` and `api_bind`
  with the other serving settings.
- `retention` reports [automatic release](#automatic-release): `state` is
  `released`, `nothing`, `off` or `failed` (with `message`); a release lists
  the `released` deployments, `freed_bytes` and each Spark's results, with
  refusals in `errors`.
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
  `no new model log output for 6 min`. The wait ends after 30 minutes, or
  sooner when a rank stops ([errors](#a-start-that-does-not-become-ready)).
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
[installer image lock](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json)
(`sparkring-installer-image/v2`):

| Item | Identifier |
|---|---|
| Image | `ghcr.io/fujitsupolycom/sparkring@sha256:71d410571407fef3ce2959c6d392f5a2c3f44b757b856e853a71f6e3295620ad` |
| Tag | `dev-20261004-kraken-cuda1342-nccl2323-status034` |
| Configuration | `sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f` |
| Base image | `eugr/spark-vllm-b12x` nightly-20261001, `sha256:141f46a4a2c3751798f16759cc859648784be430be852a84a21f0c4c427b4052` |
| vLLM and B12X | Local Inference Lab's Karmic Kraken beta branches with SparkRing's changes, branches `sparkring/kraken-beta-20261004` |

The image adds two layers to that base: SparkRing's vLLM and B12X sources
with its transports, features, SparkCache assets and runtime-status dashboard
0.3.4, then the CUDA 13.4.2 and NCCL 2.32.3 toolchain. The
[composition record](../../runtime/images/compositions/external-kraken-20261004/README.md)
lists the source commits and pinned inputs; the image's
[publication record](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/publication.json)
describes both layers. The rollback image,
`dev-20261001-kraken-cuda1342-nccl2323-status034`, has the same base,
integration assets and toolchain, and the vLLM and B12X sources of branches
`sparkring/kraken-beta-20261001`; `--image 2026.10.0` selects it
([Another image](#another-image)).

`sparkring models` lists exact model/version/quantization/topology profiles,
including guide-only ones with their guides, and marks only these as
installer-supported; family names such as `qwen` are ambiguous and rejected:

| Profiles | Checkpoint | Speculative decoding |
|---|---|---|
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | Qwen3.8 Flash Next NVFP4 QAD step 5500, revision `60215d26cf5e` (branch `qad-step5500-ple1000`) | MTP, three tokens, probabilistic drafting |
| `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash NVFP4-Spark, revision `a608241037e4` | MTP3 |
| `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4` | MiMo-V2.6-Flash-MOPD, revision `2479e2d0029e` | DFlash5 |
| `deepseek-v41-flash-tp4` | DeepSeek-V4.1-Flash, revision `dba1be0a40aa` | DSpark, five tokens, probabilistic drafting, adaptive verification |
| `swift15-qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp4` | Swift 1.5 Qwen3.8-Flash-Next NVFP4, revision `3ff0520224f2` | MTP, three tokens, probabilistic drafting |

- The table names each profile's default checkpoint. The Qwen profiles and
  `glm53-flash-nvfp4-spark-tp4` also install other checkpoints
  ([Another checkpoint of a profile](#another-checkpoint-of-a-profile)).
- All installer profiles run with SparkCache off and vLLM's native prefix
  cache on.
- Profiles that select other images, such as the `shared-2026.09.3` release,
  keep their own guides; `sparkring install` does not install them.
- `--image NAME` runs a profile on another installer image this package
  records ([Another image](#another-image)). `--image-lock FILE` replaces the
  shared lock for a development rehearsal and must list the selected profile.

### Another image

`sparkring images` lists the installer images this package records, the
default first, with the GitHub release that published each and the profiles
each runs. `--image NAME` installs a profile on one of them:

```bash
sudo sparkring install --profile qwen38-flash-next-tp2 --image statusrows
```

`NAME` is an image's full name
(`dev-20261001-statusrows-cuda1342-nccl2323-status034`), the release tag that
published it (`2026.10.0`), or a part of the name that only one image has
(`statusrows`, `portgid`). `sparkring images --profile PROFILE` lists only the
images that run that profile.

- The image is part of the deployment: another image installs a separate
  deployment, and the one it replaces becomes the rollback target, as with
  another checkpoint. Running the command without `--image` returns to the
  default image; naming the default image is the same as leaving `--image`
  out.
- The profiles' measurements and checks were made on the default image. On
  another image, each Spark checks before the model starts that the image
  has what the profile needs, and the installation stops if it does not.
- A Spark downloads an image it does not hold; an image built on one the
  Spark holds downloads only its added layers. Compile caches are kept per
  image, so a profile's first start on an image compiles its kernels again.
- `sudo sparkring up PROFILE --image NAME` selects an image for a deployment
  the same way.

### Serving settings

`sudo sparkring install`, `sudo sparkring up PROFILE`, `sparkring init` and
`sparkring compose render` accept optional settings, each of which replaces
one value of the profile's vLLM configuration for that deployment. Without a
flag, the profile's value applies.

| Flag | vLLM value it replaces | Unit |
|---|---|---|
| `--max-images N` | the image count in `--limit-mm-per-prompt` | images per request; 0 accepts none |
| `--max-videos N` | the video count in `--limit-mm-per-prompt` | videos per request; 0 accepts none |
| `--context-length N` | `--max-model-len` | tokens, at least 1,024 |
| `--max-concurrency N` | `--max-num-seqs` | requests served at the same time |
| `--kv-cache-gib N` | `--kv-cache-memory-bytes` | GiB of KV cache on each Spark, at most a tenth above the profile's value |
| `--save-cpu` | vLLM's shared-memory reader window (container variable `SPARKRING_SHM_BUSY_LOOP_S`) | a switch: readers poll 2 ms after a read instead of one second |
| `--reasoning-effort LEVEL` | the model's default effort level, in `--default-chat-template-kwargs` on Node A | one of the model's [levels](#thinking) |
| `--thinking off` | the model's default thinking, in `--default-chat-template-kwargs` on Node A | thinking stays off unless a request turns it on |
| `--api-port N` | `--port` | the TCP port of the model's API, 1024 to 65535 ([API endpoint](#api-endpoint)) |
| `--api-bind ADDRESS` | `--host` of the Spark that serves the API | one IPv4 address of that Spark, which the API alone listens on |

- A setting for a value the profile does not set is refused.
  `deepseek-v41-flash-tp4` accepts no videos and sizes its KV cache as a
  fraction of GPU memory, so `--max-videos` and `--kv-cache-gib` do not apply
  to it. The two [thinking](#thinking) settings are the exception: the
  profiles leave thinking to the model, and these settings set its default.
- A deployment records its settings, so other settings make another
  deployment. `sparkring install` with other values installs that deployment
  and replaces the running model, restarting it once; the same values select
  the same deployment again. `sparkring up PROFILE` refuses settings that
  differ from an existing deployment's; `--instance NAME` names a separate
  one.
- `sparkring install --plan` lists each setting beside the profile's value,
  and `sparkring status` shows the settings of a deployment.
- `sparkring export` and `sparkring export --share` write a deployment's
  settings into every rank's Compose file. The shared template's README names
  them in its `sparkring init` command.
- A profile's evidence (its records and memory measurements) covers the
  profile's own values. Other values are not measured: more images or videos
  use more memory on each Spark while a request runs, and a longer context or
  more concurrent requests share the same KV cache. vLLM refuses to start with
  a context length that its KV cache cannot hold.
- `--save-cpu` lets vLLM's waiting processes sleep between decode steps
  instead of polling for one second after each step. On two Sparks it freed
  about 1.7 CPU cores on Node A while a model decoded and cost about 1% of
  decode speed with one request and 2% with eight
  ([record](../../performance/records/qwen38-flash-next/shm-spin-window-20260930.md)).
  Without requests, the processes sleep either way. It needs an image whose
  vLLM reads `SPARKRING_SHM_BUSY_LOOP_S`: the default image, and
  `dev-20260930-spinwait-cuda1342-nccl2323-status033` and the images derived
  from it. On another image, such as `--image plainstatus`, `sparkring
  install`, `sparkring up`, `sparkring init` and `sparkring compose render`
  refuse `--save-cpu` and name the image.
  [installer-capabilities.json](../../runtime/releases/installer-capabilities.json)
  records the images whose own layer adds it; an image derived from one of
  them has it too.
- `--kv-cache-gib` accepts up to a tenth above the profile's value, and at
  least 1 GiB above it (11 for a profile of 10, 26 for 24, 44 for 40), and
  prints a warning for a value above the profile's: each Spark keeps that much
  less memory for images and long requests, and the value has not been
  validated as stable. A larger value is refused. vLLM
  allocates the KV cache when the model starts, and a Spark's GPU and CPU
  share one memory: a cache far above the profile's can exhaust it, the kernel
  then stops processes, and the Spark stops answering until it recovers, too
  late for the installation to restore the previous model.
  `sudo sparkring install --profile PROFILE` restores a deployment after such
  a failure.

### Thinking

A reasoning model can think before it answers. When a request doesn't say
whether it should, or how hard, the model's chat template decides; for
DeepSeek-V4.1-Flash, whose checkpoint has no chat template, vLLM's prompt
encoder decides. The installer profiles set no default of their own:

| Profiles | Without a choice | Effort levels (`reasoning_effort`) | Turns thinking off |
|---|---|---|---|
| `qwen38-flash-next-*`, `swift15-qwen38-flash-next-*` | on, `xhigh` | `low`, `medium`, `xhigh` | `"enable_thinking": false` |
| `glm53-flash-nvfp4-spark-*` | always, `max` | `low`, `high`, `max` | nothing: the model always thinks |
| `mimo-v26-flash-mopd-*` | on | none | `"enable_thinking": false` |
| `deepseek-v41-flash-tp4` | on, `high` | `low` (50), `high` (75), `max` (100), or a whole number from 1 to 100; `xhigh` is 75 like `high` | `"thinking": false`, or `"reasoning_effort": "none"` |

Every checkpoint that these profiles' `--checkpoint` choices install behaves
like its profile's default checkpoint. `sparkring models` shows each installer
profile's default and levels.

A request chooses with `chat_template_kwargs`:

```bash
curl http://NODE_A:PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "MODEL", "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

- `--reasoning-effort LEVEL` and `--thinking off` (serving settings above)
  change the default of one deployment. The installer writes the model's own
  argument into vLLM's `--default-chat-template-kwargs` on Node A's model
  container, which serves the API: `--reasoning-effort low` writes
  `{"reasoning_effort":"low"}`; `--thinking off` writes
  `{"enable_thinking":false}` for Qwen, Swift and MiMo and
  `{"thinking":false}` for DeepSeek. The other Sparks' containers serve no
  API and are unchanged.
- A level the model doesn't accept is refused with the ones it does.
  `--reasoning-effort` is refused for MiMo, which has no levels, and
  `--thinking off` for GLM, which always thinks; the two settings can't be
  combined.
- A request's own `chat_template_kwargs` and its `reasoning_effort` field
  take precedence over the deployment's default. vLLM also turns thinking on
  for a request that names a `reasoning_effort` other than `none`, unless the
  request sets `enable_thinking`. Qwen's template refuses a level outside its
  own.
- `sudo sparkring install --plan` lists the setting beside the model's
  default, `sparkring status` names the deployment's default, and the
  [dashboard](dashboard.md) shows the deployment's arguments in its
  `Default chat template arguments` row.
- Earlier turns' reasoning: GLM keeps it in the prompt unless a request sets
  `"clear_thinking": true`; DeepSeek drops it unless a request sets
  `"drop_thinking": false` or the conversation has tools.
- Evidence: [`profiles/thinking.json`](../../profiles/thinking.json) records
  each behaviour with the SHA-256 of every chat template read for it, which
  the checkpoints' pin manifests list, and for DeepSeek the installer image
  and the digests of vLLM's encoder modules. The records cover the default
  installer image. Status: implemented. The defaults come from the templates
  and vLLM source; no measurement covers a deployment with these settings.

### API endpoint

vLLM serves the model's API on the deployment's first Spark: Node A for a
pair, a four-Spark model and a model on Sparks 0 and 1, and Spark 2 for a
model on Sparks 2 and 3. Without options it listens on every address of that
Spark at the profile's port ([ports](#security-and-host-exposure)), and
SparkRing shows it at the LAN address setup recorded for Node A, or at Spark
2's own ([where it serves](#two-models-on-one-ring)).

| Option | Sets | Part of the deployment |
|---|---|---|
| `--api-port N` | The API's TCP port, 1024 to 65535 | Yes, a serving setting |
| `--api-bind ADDRESS` | The one IPv4 address of that Spark the API listens on | Yes, a serving setting |
| `--api-address ADDRESS` | The address shown for the model: a DNS name, a Tailscale address or another network's address | No, shown only |

- `--api-port` and `--api-bind` are [serving settings](#serving-settings):
  the same values select the same deployment, and other values another one.
  `--api-port` replaces `--port` on every Spark; `--api-bind` replaces
  `--host 0.0.0.0` on the API's Spark only. The container's health check,
  which the installation waits for, the short test request and the dashboard
  check ask the API at that address and port.
- `--api-port` refuses the ports SparkRing uses: 2222 (administration SSH),
  5255 (image relay), 29500 (vLLM's default master port) and the profile's
  `--master-port`.
- Before it plans a deployment with either setting, and before
  `sparkring up PROFILE` creates one, Node A reads the addresses and the
  listening TCP ports of the API's Spark (`ip` and `ss` over SSH). Before any
  Spark is searched, it refuses a listen address that is not one of that
  Spark's, and names the Spark's addresses; a loopback address such as
  127.0.0.1, unless `--allow-loopback-bind` accepts it for a new deployment;
  a loopback address on Spark 2 in any case, because Node A's checks reach the
  API over the network; and a port that another program holds on that
  address or on every address. The port may be held by the deployment itself
  and by the running models that the installation replaces or stops. The
  API's Spark checks the address and the port again before the model starts.
- `--api-address` changes only what SparkRing shows: the plan's `Model API:`
  line, `Model ready:`, the card's dashboard and `curl` lines, `api_url` in
  the JSON results and `sparkring status`. The readiness wait, the short
  test request, `sparkring status --refresh` and
  [automatic recovery](#automatic-recovery) keep using the listen address,
  or the automatic address, which Node A reaches. Status prints that URL in
  parentheses, and the JSON results hold it as `check_url`. The address is
  not part of the deployment: the same request with another address
  installs the same deployment. An approved installation records it in the
  deployment's `api-address.json`; one without `--api-address` shows the
  automatic address again, and `--plan` records nothing. The commands that
  plans suggest repeat it.
- In a terminal, a run without `--yes`, `--json` and these options asks
  before it searches the Sparks:

  ```text
  Model API (Enter keeps the first choice):
    1. Every address of Node A, shown as http://198.51.100.10:8000/v1
    2. Only 198.51.100.10 (enP7s7)
    3. Only 198.51.100.11 (wlP9s9)
  API address number [1]:
  API port [8000]:
  ```

  The list leaves out the fabric (the cluster's fabric CIDR), the
  administration network, loopback and link-local addresses and links that
  are down. A listed address becomes `--api-bind` and a port `--api-port`.
  When the Spark's addresses cannot be read, the question is skipped and the
  endpoint stays automatic.
- `sparkring setup` takes no `--api-address`. It records Node A's automatic
  address, which every deployment's site includes, and a model on Sparks 2
  and 3 serves on Spark 2, which that address does not name.

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

### GLM-5.3-Flash profiles

| Setting | Two Sparks | Four Sparks |
|---|---|---|
| Context window per request | 1,048,576 tokens | 1,048,576 tokens |
| KV cache per Spark | 10 GiB | 40 GiB |
| KV capacity reported at startup | 1,530,566 tokens | 6,128,169 tokens |
| KV cache page | 2,048 tokens | 1,024 tokens |
| Requests running at once | 8 | 16 |
| Images per request | 8 | 32 |
| Videos per request | 1 | 1 |

- Each image is resized to at most 4,096 tokens. Each video is sampled at 16
  frames and resized to at most 8,192 tokens. The model's vision processor
  accepts one video per request; a request with more fails with
  `At most 1 video(s) may be provided in one prompt.`
- The prefix cache reuses whole pages. With pages of B tokens, a repeated
  prompt of P tokens reuses (⌊(P − 1) / B⌋ − 1) × B tokens: nothing of a
  2,458-token prompt with 2,048-token pages, and about 90% of a 38,703-token
  prompt. On two Sparks, 2,048-token pages hold 60% more tokens per GiB than
  1,024-token pages; on four Sparks both sizes hold the same, so the ring
  uses the smaller page.
- Node A decodes and preprocesses every image and keeps its pixel data until
  the request finishes, including while the request waits for a free slot.
  Node A's free memory therefore falls with the number of images in flight
  across all requests. Exhausting it stops the Spark until it is
  power-cycled. The
  [GLM memory record](../../performance/records/glm53-flash/installer-memory-20260929.md)
  gives the measured lows for each profile.

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
- `containers`: each rank's host and model container name
  (`sr-<site>-r<rank>`) and its labels: `io.sparkring.deployment` holds the
  deployment ID and `io.sparkring.rank` the rank. A deployment keeps its names;
  a new deployment of the same profile, such as one `sparkring install` makes
  for another package, gets another site name, so a controller that follows
  deployments across installations selects containers by the deployment ID
  label, for example `docker ps --filter label=io.sparkring.deployment=ID`;
- host observations: persistent node ID, boot ID and their own `observed_at`;
  cached ones keep their original time and go stale after 90 seconds;
- container observations: the inspected container ID, start time and actual
  image ID, and for a stopped container its `exit_code` and `finished_at`;
- `api_url`, the model's API as SparkRing shows it, and `check_url`, the URL
  its checks use, when an [address is shown](#api-endpoint) in its place;
- `model`, when the last operation is a completed `up`: `state` (`serving`,
  `stopped`, `partial`, `api-failing`, `rebooted`, `mesh-failed`,
  `mesh-cleanup`, `unreachable` or `unknown`), `summary`, `next_action` and
  `details`, as the first lines of the text output show them;
- `recovery`: the active deployment's
  [automatic recovery](#automatic-recovery) record and whether its timer is
  enabled;
- `fabric_bandwidth`: the saved [cable speed](#cable-speed) result with
  `state` `measured` and `age_seconds`, or only `state` `never-measured` or
  `unreadable`;
- per Spark, `control`: each administration tunnel peer's primary link,
  carrier, endpoint, the [path](#admin-tunnel) that endpoint names (`path`:
  `via` `cable` or `lan`, `netdev`, `address`, `primary`), its number of
  fallback paths (`fallbacks`) and seconds since its latest handshake, with a
  warning when that is over 200 seconds or the peer runs over a fallback
  path; on a four-Spark ring, `mesh`: active and failed mesh
  services, each failure's systemd result and last log line, and running mesh
  marker processes.

Missing identities are `null` with a reason, never guessed from hostname or
rank. Network observations do not show whether the model is ready.

On a four-Spark ring, host observations also cover the
[ConnectX hairpin setting](#four-spark-rings). A function without it makes
the Spark `needs-attention`, with an error naming the function and its value
and the next action `on Node A: sudo sparkring hairpin`. `warnings` report a
setting in effect but not applied at boot, boot restarts suspended after a
failed restart (naming the function and time), a boot started with
`sparkring.hairpin=off`, a mesh unit without the hairpin start check, and
mesh code that differs from the deployment's
([when mesh code changes take effect](#the-rings-mesh)).

A fabric link without carrier makes the Spark `needs-attention`, with an
error naming the link, the rank at its other end and any fabric routes
through it, which return with the link. On a four-Spark ring, an
[approved route](#fabric-addresses-and-routes) missing while its link is up
is named with its reason: another route in its place, a failed addition, or
`sparkring-fabric.service` not active; otherwise `sparkring-agent` adds it
within 30 seconds. An approved per-interface setting that differs is named
with its value, such as `net.ipv4.conf.enp1s0f1np1.rp_filter is 2, approved 0`,
and the same reasons.

## When a model stops serving

A multi-Spark model stops serving when any rank stops, and its containers do
not restart themselves (`restart: 'no'`). `sudo sparkring status --refresh`
then starts with what is wrong and the command that fixes it; the lines
below it name each Spark's container, exit code and time:

| First line | Cause | Next step |
|---|---|---|
| `The model is not running on any Spark` | Every rank's container stopped, for example after the Sparks restarted | `sudo sparkring up --execute` |
| `The model runs on rank 0 (…) but stopped on rank 1 (…)` | The Sparks that stayed up keep a model that waits for the others | `sudo sparkring down --execute`, then `sudo sparkring up --execute` |
| `The model's containers run, but its API fails` | Rank 0's `/health` answers 503 (the engine stopped), or neither `/health` nor `/v1/models` answers | the same |
| `rank N (…) restarted after the model started` | The Spark's boot ID changed since the model started | the same |
| `The mesh service failed on …` | A four-Spark mesh service failed; its last log line follows | `sudo sparkring up --execute` |
| `SparkRing cannot reach rank N (…)` | The Spark does not answer | Power it on, or reconnect the [admin tunnel](#admin-tunnel) cable that the line names |

`up` repeats every step when the container runs on no Spark, including
restoring the RoCE GID index, a ring's mesh step and the NVIDIA CDI
specification. While the container still runs on some Sparks, it holds the
RoCE GID entries that step repairs, so `up` refuses until `down` has stopped
it everywhere. `sudo sparkring install --profile PROFILE` does both.

Before restarting by hand, save each Spark's model container log:
`sudo docker ps -a` names the container; keep `sudo docker logs CONTAINER`.
Log lines beginning `RoCEnante rank` tell which rank was late and why.

### Automatic recovery

Node A restarts the model by itself. Once a minute
(`sparkring-recover.timer`), it checks the active model, or each half's model
on a [ring that serves two](#two-models-on-one-ring), and, for the first four
rows of the table, runs the same `up`, or `down` and then `up`, that the
table names. It acts only when all of these hold:

- The last model operation was a completed `sudo sparkring up` or
  `sudo sparkring install`. After `sudo sparkring down --execute` it does
  nothing until the next `up` or `install`.
- No `install`, `setup`, `up`, `down` or `hairpin` is running.
- Every Spark answers. While one does not, it only waits, and `status` names
  the Spark it waits for.
- Two checks in a row found the model not serving. An API that gives no
  answer at all, rather than a 503, must stay silent for 5 minutes.
- For a four-Spark model: every Spark reports its mesh services, the
  [hairpin setting](#four-spark-rings) is in effect, and no mesh forwarding
  process runs without its service ([below](#mesh-forwarding-without-its-service)).

Before it acts it takes the install lock and checks everything again, so a
`recover off`, another command or a model that came back in the meantime
stops it.

After a failed attempt it waits 2 minutes, then 5, then 15. After 4 failed
attempts in a row, about 25 minutes of trying, it stops. It also stops,
instead of restarting a fourth time, when it has already restarted the model
3 times within 6 hours: a model that keeps stopping has a cause a restart
does not fix. Either way `status` says why, and it resumes after
`sudo sparkring up --execute`, `sudo sparkring install` or
`sudo sparkring recover on`. Attempts are logged to
`/var/log/sparkring/install.log` (`sudo sparkring logs`), and `status` shows
the last attempt and the next.

- Turn it off for the active model: `sudo sparkring recover off`; on again:
  `sudo sparkring recover on`. `sparkring up` keeps that choice.
- Install a model without it:
  `sudo sparkring install --profile PROFILE --no-auto-recover`. Each
  `install` sets the choice again: on unless you pass the flag.
- Stop the model and keep it stopped: `sudo sparkring down --execute`.

After a package upgrade, a model started by an earlier package is checked
once its timer runs: the next `sudo sparkring up --execute`,
`sudo sparkring install` or `sudo sparkring recover on` enables it. Until
then `status` says `sparkring-recover.timer is not enabled`. Installing the
package never enables the timer, so it never restarts a model by itself.

It covers deployments that `sparkring install` and `sparkring up` start with
Compose. For managed GLM deployments `status` says recovery is not
available. Its record is `/var/lib/sparkring/recovery.json`; an unreadable
record is moved aside to `recovery.json.unreadable-*` and recovery starts
again from empty records.

### Mesh forwarding without its service

On four Sparks, a mesh service whose own process was killed (systemd result
`watchdog` or `signal`) leaves its forwarding processes running, and a new
mesh service refuses to start while they run. `status` then says
`The mesh on rank N (…) stopped without stopping its forwarding processes`,
and automatic recovery leaves it to you. On that Spark, with its model
container stopped, run the mesh's own cleanup, which stops only the processes
the service recorded and refuses when others remain:

```bash
sudo python3 /opt/sparkring/deployments/NAME/runtime/glm53-spark-mtp3-mesh/managed_service.py \
  cleanup --config /etc/sparkring/deployments/NAME/service.json
```

`NAME` is the middle of the unit name `sparkring-NAME-mesh.service`; for
`sparkring-mesh.service` the directories are `/opt/sparkring/managed-mesh` and
`/etc/sparkring/managed-mesh`. Then, on Node A,
`sudo sparkring down --execute` and `sudo sparkring up --execute`.

### Admin tunnel

Node A reaches each worker through the WireGuard administration tunnel
`sr-control`. Each link of the tunnel has a primary path, the fabric cable
and ConnectX function that setup recorded for it, and fallback paths in this
order: each other fabric cable between the same two Sparks, once per ConnectX
function of its port; the primary cable's other function; the two Sparks'
LAN addresses. On a pair the other cable is the model's port, which then
also carries the tunnel. A fallback changes only where the tunnel's packets
go; the tunnel's addresses, keys and routes stay the same. Every 20 seconds
each Spark checks its paths (`sparkring-control-refresh.timer`):

- When the primary cable has no link, the tunnel moves to the first fallback
  whose link is up, within about 20 seconds. A fallback that does not answer
  within one check gives way to the fallback after it in the order.
- A path without a WireGuard handshake for 200 seconds is left the same way.
  A reachable Spark renews its session within 180 seconds, because every
  peer sends a keepalive every 15 seconds.
- The other Spark follows without a check of its own: WireGuard sends to
  wherever the peer's last authenticated packet came from.
- The Spark that moved the tunnel returns it to the primary cable once that
  cable's link has been up for two checks. If the other Spark does not answer
  there, the tunnel stays on the fallback for 10 minutes, then 20, 40 and at
  most 60 minutes between tries; unplugging and reconnecting the cable
  retries at once.

While a fallback carries the tunnel, `status` shows under that worker, for
example, `admin tunnel: over LAN 198.51.100.137 (primary cable enp1s0f1np1: no link)`,
and `install`, `up` and `down` keep working. Checkpoint and package copies
between Sparks still need every primary cable and stop naming the failed
link. When no path answers, `status` shows
`admin tunnel: no recent handshake (last 46 min ago); Node A's enp1s0f1np1: no link`,
naming the fallback it tries before the semicolon (`... ago) over LAN
198.51.100.137; ...`), and `install`, `up` and `down` fail on that worker.
Each Spark's journal names every move:
`journalctl -u sparkring-control-refresh.service`.

A cluster whose tunnel lists no fallback paths (`fallbacks: 0` in
`sudo sparkring status --json`) gains them on Node A with
`sudo sparkring setup --admin-fallback`; add `--plan` to list each Spark's
fallback paths without changing anything. It asks for one approval, updates
workers that run another SparkRing revision to Node A's, and changes nothing
else: each Spark accepts a configuration that differs from its own only in
its fallback paths. Run it again after a Spark's LAN address changes.

On a four-Spark ring the tunnel is a tree through the cables, so Node A
reaches one worker through another. Each tree link gets the other ConnectX
function of its cable and the two Sparks' LAN addresses as fallbacks: the
cable's other function carries the tunnel past a failed function, and the
LAN past a failed cable when both Sparks have a LAN address. The ring's
fourth cable is not in the tree, and moving the Sparks behind a powered-off
worker onto it, under another worker, is unsupported: they stay unreachable
until that worker runs again.

### Why a rank stops the others

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

**Finding the other Sparks.** Setup pings each fabric link and signs in only
to neighbors that answer there; it skips cached neighbor addresses that do
not answer. It names each Spark by its ConnectX hardware, not by
`/etc/machine-id`, which Sparks flashed from one factory image share. When
two Sparks share it, setup prints a note, because other software on them,
such as DHCP, may still confuse them. To give a Spark its own ID:
`sudo rm -f /etc/machine-id && sudo systemd-machine-id-setup && sudo reboot`.
A failed sign-in names its cause: a password or account the other Spark did
not accept, no SSH answer over the cable, SSH refused on its port, or a
changed host key.

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

### Cabling

- **Pair:** a cable between port 0 (p0) of both Sparks. Pair profiles use
  port 0 on both Sparks. A second cable between the two ports 1 only carries
  the [admin tunnel's](#admin-tunnel) fallback path.
- **Four-Spark ring:** one loop in which every cable runs from port 0 of one
  Spark to port 1 (p1) of the next. Node A is rank 0; the Spark on its port 0
  is rank 1, and so on.

The serving images rely on these ports, so setup never remaps them. When the
cables differ, setup stops and names the change: a cable end to move, or a
Spark whose two cables to swap, and the ring order afterwards.
`sudo sparkring cabling` prints the same advice without setting anything up
([command](commands.md#cabling)). For a loop it names the fewest swaps;
when two choices tie, it leaves Node A's cables alone.

### Cable speed

A cable can stay up and count no errors while it carries far less than it
should. Models still decode at their usual speed, but prompt processing
(prefill) over the cable is slower. Status: **implemented**. The measurement
comes from the [cable speed record](../../performance/records/transport/fabric-cable-bandwidth-20261002.md);
no hardware run of the command itself is on record.

**What is measured.** Each port appears as two network functions that share
its cable; the check's output calls each one a link. A function reaches
about 109 Gb/s each way, the limit of the PCIe Gen5 x4 connection behind it,
and each such connection carries one function of each port.
`sudo sparkring cabling --bandwidth`, and setup as its last step, send RDMA
writes in both directions at once over each link: `ib_write_bw -b` with
1 MiB messages for 5 seconds, from the Debian package `perftest`, which the
SparkRing package depends on. A healthy link measures
about 213 Gb/s; 190 Gb/s or more counts as healthy. A one-way test does not
show the fault, and neither do RoCE retransmit or FEC counters: corrected-bit
counts differ by cable model.

**Which cables.** A pair's cable between the ports 0, and each ring cable
from port 0 of rank r to port 1 of rank r+1. The addresses come from the
setup record. A pair's second cable between the ports 1 has no fabric
addresses and is not measured. Links run one at a time, because two links
share each PCIe connection and parallel tests would disturb each other.

**Before each test.** RoCE GID index 3 of both ends must hold the link's
fabric address ([RoCE GID index 3](#roce-gid-index-3)); otherwise the link
reports that instead of a speed. A link that cannot be tested names the
reason, such as `ib_write_bw` missing on a Spark or a test that did not
finish. Each test server runs under a time limit and stops when Node A is
done with it; it listens on a free TCP port from 18620 to 18639.

**Serving models.** The test fills a cable for several seconds per link and
slows a model that uses it. Cables that touch a serving model's Sparks are
skipped, also those of a [ring half](#two-models-on-one-ring);
`--while-serving` measures them anyway and notes that model traffic can lower
the result. Setup measures nothing while a model serves. The on-demand check
holds the installation lock, so no model starts meanwhile.

**Repair.** Reboot both Sparks on a degraded cable, then measure again.
Restarting the link (`ip link set down/up`) or the driver
(`devlink dev reload`) does not clear it. If the cable is still degraded
after the reboot, reseat it at both ends. Plugging cables in while the
Sparks run is a likely cause; it is not confirmed.

**Result.** The latest result is saved as
`/var/lib/sparkring/controller/fabric-bandwidth.json` (schema
`sparkring-fabric-bandwidth/v1`, the document `--json` prints). A check that
measured nothing, because a model served on every cable, keeps the saved
result. `sparkring status` shows its verdict and age and lists every cable
that is not healthy, with the repair steps for a degraded one; a result saved
for another setup of the Sparks counts as never measured.

## Re-form Sparks into another pair or ring

Sparks that belonged to other SparkRing clusters can form another pair or ring:

1. Cable them as a [pair or ring](#cabling); `sudo sparkring cabling` names
   any cable to move.
2. Stop their models: on each Spark that was a Node A,
   `sudo sparkring down --execute`.
3. On the Spark that becomes Node A, review, then set up:

   ```bash
   sudo sparkring setup --name NAME --plan
   sudo sparkring setup --name NAME
   ```

Setup re-forms the Sparks when the Sparks cabled to Node A differ from its
cluster record, or when a cabled Spark keeps another cluster's setup. It
reaches each Spark over the LAN or the cables, as in
[Setup and access](#setup-and-access), and its plan lists by Spark what it
moves aside:

- the records of a Spark that was a Node A, in
  `/var/lib/sparkring/controller`; Node A keeps its SSH key;
- the admin network (`sr-control`) configuration and its services;
- the fabric record, its boot service and its routes;
- automatic recovery, mesh services and the ConnectX hairpin approval;
- fabric IPv4 addresses that SparkRing did not set. Setup replaces them after
  backing up their NetworkManager connections;
- the IPv6 link-local addresses of fabric connections made by hand: setup
  copies each connection's file aside and sets the hardware-derived form, so
  [RoCE GID index 3](#roce-gid-index-3) holds the port's IPv4 address.

After the one `Proceed?` approval, or `--yes`, setup moves that state to
`/var/lib/sparkring/retired/STAMP/` on each Spark and keeps it there, with a
`receipt.json` that lists how to restore it. Node A's `reform.json` there
collects every Spark's receipt. Setup installs Node A's SparkRing on each
worker, which asks for that worker's `sudo` password once. Then it sets the
Sparks up as on a first setup, with renumbered fabric addresses.

- Setup stops while a SparkRing model runs on one of the Sparks and prints
  the command that stops it. It never stops a model itself.
- Checkpoints, images and caches in `/srv/sparkring` stay; `sparkring install`
  finds checkpoint copies there.
- Each Spark keeps its own identity, `/etc/sparkring/node.json`.
- Run setup again after an interruption; it skips finished steps.

## Four-Spark rings

Four-Spark rings need the ConnectX hairpin setting on every Spark; pairs do
not use it. SparkRing applies it itself; a first installation needs no flag or
separate step.

### The hairpin setting

Every four-Spark installer profile (`qwen38-flash-next-qad-tp4`,
`glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp4`,
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
  other Sparks cuts them off from Node A unless a
  [fallback path](#admin-tunnel) carries them, while the Spark itself stays
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

### Fabric addresses and routes

Each cable carries two `/24` subnets, one per ConnectX function of its
ports. With the default fabric network (`--fabric-cidr 198.18.0.0/21`), the
cable from port 0 of rank e to port 1 of rank e+1 (rank 0 after rank 3)
uses `198.18.(2e).0/24` and `198.18.(2e+1).0/24`; compatible addresses that
setup kept may differ. Each Spark reaches the two cables it is not on
through a neighbor: one static route per subnet, over the shorter side of
the ring, for example `198.18.4.0/24 via 198.18.6.1 dev enp1s0f1np1` on
rank 0. Setup records these approved routes in `/etc/sparkring/fabric.json`;
a pair has none.

- **At boot**, `sparkring-fabric.service` adds the routes, sets each fabric
  function's per-interface settings (`net.ipv4.conf.IFACE.forwarding=1` and
  `net.ipv4.conf.IFACE.rp_filter=0`) and adds the forwarding rules.
- **When a link returns.** A link goes down when the Spark at its other end
  reboots or restarts a ConnectX function, or when the cable is out.
  NetworkManager then removes the function's fabric address, and the kernel
  deletes the routes through it. NetworkManager adds the address again when
  the link returns, and `sparkring-agent`, which checks every 30 seconds,
  adds each missing approved route once its function has its link and
  address.
- **After a driver restart** of a ConnectX function, its netdev returns with
  the kernel's default settings, such as `rp_filter` 2. The agent sets the
  approved settings that differ again on every fabric function that exists,
  and logs each change.
- The agent changes only approved routes and settings, never another
  interface or a global setting, and never removes or replaces a route:
  another route to the same subnet stays, and `sparkring status` names it.
  It acts while `sparkring-fabric.service` is active, as it is after a
  successful boot restoration; stopping that service stops it until the
  next boot. The forwarding rules come from `sparkring-fabric.service`
  alone: they match interfaces by name, so a link going down or a restarted
  function leaves them in place.
- **Status.** While a link is down, `sparkring status` names the link and
  the rank at its other end; the routes need no step
  ([Status observations](#status-observations)).
- The routes stay while [two models serve on the ring](#two-models-on-one-ring);
  the halves do not use them.
- Installing the package restarts `sparkring-agent`, so an updated Spark
  restores routes and settings this way without setup or a reboot.

### The ring's mesh

Every four-Spark installer profile runs on a native mesh; the installer
renders each rank's container from the profile's shared container
specification.

- It reuses the mesh service every Spark has enabled or running, or stopped
  while [two models served on the ring](#two-models-on-one-ring).
- If no Spark has one, it downloads and verifies the pinned host marker on
  every rank, creates stopped model containers, installs supervised mesh
  services, waits for every rank and starts the model.
- A mesh that only some Sparks have, or that differs between them, stops the
  plan for inspection.

SparkRing enables the mesh service, so it starts at each boot after the
hairpin setting. Before an installation starts the model, every Spark first
stops its mesh service where it must start again: where the mesh's routes or
forwarding rules are missing, and where a port's IPv4 address must be re-added
because its RoCE v2 GID is not at the pinned
[GID index 3](#roce-gid-index-3). Then each Spark starts its mesh service if
it is stopped, and the installation waits up to four minutes for the ring
check on every Spark.

**When mesh code changes take effect.** Each Spark runs its mesh from the
supervisor code and systemd units installed for that mesh:
`/opt/sparkring/managed-mesh` and the `sparkring-mesh*` units for the default
mesh, `/opt/sparkring/deployments/NAME` and the `sparkring-NAME-*` units for a
named one. Installing a package, `sparkring install` and `sparkring up` never
change the code of a running mesh service, and a boot starts the installed
code. When `sparkring install` or `sparkring up` starts a mesh service, each
Spark first compares the installed files with the deployment's own SparkRing
source. Where they differ, it installs the deployment's files, updates the
mesh's installation receipt (`source-hashes.json` or `installation.json` in
`/etc/sparkring/managed-mesh` or `/etc/sparkring/deployments/NAME`), reloads
systemd if a unit file changed, and then starts the mesh. It does so only when
all of these hold:

- the mesh, model and liveness services on that Spark have stopped;
- the deployment's code accepts the mesh's installed configuration;
- every mesh supervisor running on another Spark runs the deployment's code.
  The four supervisors form a mesh only when they run the same code, so the
  four Sparks change code together.

A fix to the mesh supervisor therefore reaches a ring at the first
`sparkring install` or `sparkring up` of a deployment that includes it, once
the mesh service has stopped on all four Sparks: for example after the mesh
failed on every Spark, or after `sudo sparkring down --execute` and
`sudo systemctl stop sparkring-mesh.service` on each Spark (a named mesh's
unit is `sparkring-NAME-mesh.service`). The `up` of a deployment created from
other SparkRing source installs that source's mesh code in the same way.
Files that the receipt lists but the deployment's code does not include are
removed; other files in the code directory stay.

Until the files match, `sparkring status` reports on each such Spark, for
example, `mesh code of sparkring-mesh.service installed 2026-09-21 differs
from this deployment's; it refreshes when sparkring up next starts the mesh on
all four Sparks`. The deployment it compares with is the last one whose
`install` or `up` started or checked that mesh.

`sparkring install` does not replace an existing mesh:
`sparkring up PROFILE --fresh-mesh --plan` prints an explicit replacement
plan, and `sparkring up PROFILE --instance fresh --fresh-mesh` rehearses the
replacement beside an existing deployment. A replaced deployment's container,
source and weight directories are retained.

A mesh takes its name from the cluster, profile and instance. Changing a
deployment's package or settings means creating another deployment:
`sparkring install` names each deployment by its request, and
`sparkring up PROFILE --instance NAME` takes an unused name. Each instance has
its own workspace, containers and mesh, and `sparkring status --json` lists
its containers and their labels. A deployment created under an earlier
deployment's instance needs that deployment's workspace released and its model
containers removed; it installs its mesh where the earlier mesh is. While a
mesh that the earlier deployment created is installed on the Sparks, its host
marker keeps that workspace in use and `sudo sparkring storage` does not
release it ([Finding and freeing space](#finding-and-freeing-space)); create
the deployment under an unused instance name instead. With
`--fresh-mesh`, the installation takes over that earlier mesh on each Spark
when the reviewed plan listed its service, its site file is
unchanged since that review, and its model container on that Spark is stopped
(by `sparkring down`) or removed. It stops the earlier mesh, model and
liveness services and moves their configuration, code and unit files to
`/var/lib/sparkring/replaced-meshes/`; nothing is deleted. Every Spark is
checked before any changes, so a refusal on one Spark leaves all four as they
were. A mesh service that is installed but neither enabled nor running is not
listed by the plan and is not taken over.

The mesh services start and check the exact model containers that the
deployment created. When those containers are created again, for example
after `docker container prune` removed them while the model was stopped, the
deployment's next `up` installs its mesh services again for the new
containers, in the same way.

## Two models on one ring

A four-Spark ring serves one four-Spark model, or two two-Spark models: one
on Sparks 0 and 1, one on Sparks 2 and 3. Switch between them with
`sudo sparkring install`, as you switch models.

```bash
sudo sparkring install --profile TWO_SPARK_PROFILE --on 0,1   # Sparks 0 and 1
sudo sparkring install --profile TWO_SPARK_PROFILE --on 2,3   # Sparks 2 and 3
sudo sparkring install --profile FOUR_SPARK_PROFILE           # all four again
```

- **Which half.** `--on` takes `0,1` or `2,3`: each pair shares a cable.
  Without `--on`, a two-Spark profile goes on the half that serves no model,
  and the run says which. When both halves serve, or neither does, it asks
  for `--on`. `--on` is refused on a pair and for four-Spark profiles.
- **What stops.** A model on a half replaces that half's model and leaves the
  other half running. It stops a running four-Spark model. A four-Spark
  model stops the models on both halves. The plan lists every model the
  switch stops (`It stops PROFILE on Sparks 2 and 3.`), and the run asks
  before it stops one outside the half you named; `--yes` approves that.
- **Where it serves.** A half serves on its first Spark, at the profile's
  port: Sparks 0 and 1 on Node A's address, Sparks 2 and 3 on Spark 2's own
  LAN address. `Model ready:` and `sudo sparkring status` print each URL. A
  Spark 2 without its own LAN connection serves on its administration
  address, which only Node A reaches; the plan says so. For Sparks 2 and 3,
  the [API options](#api-endpoint) and the question name Spark 2's
  addresses.
- **Its own steps.** A half's checkpoint plan and model steps call its two
  Sparks Node 0 and Node 1 and name their host names. Package updates and the
  serving image go to every Spark; they restart no model.

### The ring's mesh while halves serve

Before a half's model starts, every Spark stops and disables its mesh
service, and the half's own start checks that no mesh runs on its two Sparks.
A two-Spark model uses its cable directly, as a pair does, so its two Sparks
restore their fabric addresses in [GID index 3](#roce-gid-index-3) first.
Fabric addresses, [routes](#fabric-addresses-and-routes) and the
[hairpin setting](#the-hairpin-setting) stay as setup left them, and a
half's installation applies no hairpin change.

Each Spark records the mesh services it stopped in
`/etc/sparkring/mesh-parked.json`, and such a service counts as enabled for
the next four-Spark installation, which reuses it. That installation's
[mesh step](#the-rings-mesh) enables and starts it again on every Spark and
removes the record. `sudo sparkring node mesh-park` stops and disables the
mesh services of one Spark by hand.

### Status, stop and recovery

- `sudo sparkring status` lists each half's model, its checkpoint, its API
  URL and its automatic recovery.
- `sudo sparkring down --on 2,3 --execute` stops one half's model, and
  `sudo sparkring up --on 2,3 --execute` starts it again. With models on
  both halves, `up` and `down` without a profile ask for `--on`.
- `sudo sparkring up` refuses a four-Spark model while a half's model runs,
  and a half's model while the four-Spark model runs, and names the `down`
  command that frees the Sparks.
- [Automatic recovery](#automatic-recovery) checks each half's model and
  restarts only the half that stopped serving. `recover on` and `recover off`
  apply to both halves.
- [Automatic release](#automatic-release) keeps each half's active model and
  the model its last switch replaced.

One installation lock covers the whole ring. While one half installs, the
other half's model keeps serving, and its recovery check reports `busy` and
runs again a minute later.

Node A keeps each half's records in
`/var/lib/sparkring/controller/slots/0-1/` and `slots/2-3/`
(`active.json`, `transaction.json`); the whole ring's stay in
`/var/lib/sparkring/controller/`.

## Security and host exposure

Review this list before approving `Proceed? [Y/n]`; it is how the installer
changes each Spark's network exposure.

**Open model API.** The OpenAI-compatible API listens with no API key on all
interfaces of the Spark that serves it: Node A, or Spark 2 for a model on
Sparks 2 and 3 ([where each half serves](#two-models-on-one-ring)). Anyone who
can reach its port can use the model, so keep that Spark on a trusted network
or firewall the port. `--api-bind` lets it listen on one address only, and
`--api-port` moves it ([API endpoint](#api-endpoint)). Containers use
host networking, and the runtime-status dashboard
(`/v1/sparkring/status/view`) answers on the same port:

| Port | Profiles |
|---|---|
| 8000 | `qwen38-flash-next-tp2`, `glm53-flash-nvfp4-spark-tp2`, `swift15-qwen38-flash-next-tp2` |
| 8015 | `qwen38-flash-next-qad-tp4`, `glm53-flash-nvfp4-spark-tp4`, `deepseek-v41-flash-tp4`, `swift15-qwen38-flash-next-tp4` |
| 8020 | `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4` |

**Passwordless sudo.** When noninteractive `sudo` is missing, the command
`needs_input` prints for a Spark writes `USER ALL=(ALL) NOPASSWD:ALL` to
`/etc/sudoers.d/USER` for the SSH account, giving it passwordless root on that
Spark. Review it before running it.

**Administration network.** Setup creates a WireGuard network over the fabric
links' IPv6 link-local addresses: interface `sr-control`, UDP port 51871,
addresses in `10.253.255.0/29` by default. The firewall accepts that UDP port
from link-local addresses on each primary cable's interface, and on each
[fallback path](#admin-tunnel)'s interface only from that peer's address: its
link-local address on another cable, or its LAN IPv4 address on the LAN
interface. A Spark's fallback paths therefore expose UDP port 51871 on its
LAN interface to its tunnel peers' LAN addresses; WireGuard accepts
sessions only from the configured peer keys. A separate SSH service
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
([Finding and freeing space](#finding-and-freeing-space)).

The printed plan states, for each Spark, the free space it needs on the
checkpoint directory's filesystem, and each part of that figure with the
reason it is reserved:

```text
needs 55.9 GiB free on /: 51.9 GiB for the whole image (no image it derives from is present), 4 GiB for the compile cache; 62 GiB free
```

The parts are:

- **Checkpoint files:** the bytes a Spark copies, receives or downloads, plus
  the largest of those files again as headroom, at most 16 GiB. Hard-linked
  files need no space. Each file is written once: staging shares the
  filesystem and SparkRing hard-links the verified file into place, so the
  headroom need not grow with the file. It is for writes the plan does not
  itemize while a transfer that can take hours runs, among them up to 1 GiB
  of unplanned writes per Spark, which an approved plan tolerates. Only
  DeepSeek's two 94.6 GiB files exceed the cap; the plan then reads
  `16.0 GiB of headroom (the largest file, 94.6 GiB, capped)`.
- **Image**, when Docker's data root shares the filesystem. Each installer
  release's `publication.json` names the release it derives from, and a
  derived image keeps its parent's layers
  ([installer images](../../runtime/images/installer-images.md)). On a Spark
  that holds:
  - the image: nothing;
  - an image it derives from, in Docker's graph-driver image store: the
    missing layers. SparkRing loads them with `docker load`, whose own check
    needs four times their download size plus 4 GiB; the plan bounds that
    download by the larger of the two releases' unpacked-size and
    download-size differences. The default image, `dev-20261004-kraken-cuda1342-nccl2323-status034`,
    names no release it derives from, so this case does not arise for it;
  - no image it derives from: the unpacked size plus the download size plus
    8 GiB for Docker's metadata and allocation, 51.9 GiB for the default image. The
    same applies on Docker's containerd image store, into which SparkRing
    loads no single layers, and on a Spark whose images could not be listed.

  An image lock that records no sizes needs the 68 GiB image allowance of
  [storage planning](../../profiles/storage-planning.json).
- **Relay copy**, on Node A: the registry relay keeps one compressed copy of
  every layer that some Spark lacks until the image is distributed, up to the
  image's 14.2 GiB download.
- **Compile cache**, when it shares the filesystem: 4 GiB, and nothing when
  every cache directory of this image and checkpoint already holds files.

Before downloading the image, the relay's check repeats the image and relay
figures in each Spark's Docker data root, from the registry's layer list and
the leading layers each Spark holds. Before writing checkpoint files, each
Spark repeats the checkpoint and compile cache figures.

**Compile cache allowance.** Installer containers write Triton,
TorchInductor, vLLM torch.compile, FlashInfer, CuTe DSL, TileLang and TVM FFI
output into one directory per model family, image and checkpoint revision,
and B12X kernels into one per family, CUDA version and checkpoint revision
([Finding and freeing space](#finding-and-freeing-space)). Measurement: the
disk use of these directories on three clusters. Result:

| Cluster | Directories | Total |
|---|---:|---:|
| Spark pair | 5 | 720 MB |
| Another Spark pair | 30 | 4.2 GB |
| Four-Spark ring | 33 | 4.7 GB |

A directory held 0.15 to 0.5 GB, and none exceeded about 0.5 GB. Caches of
GLM containers started by hand, outside the installer, reached 2.6 GiB. The
measurements do not separate profiles, so one allowance applies to every
profile. Conclusion: 4 GiB covers the largest cache seen with 1.4 GiB to
spare, and the largest installer directory eight times over. The margin
remains because a cache grows with the CUDA graph sizes, tuned kernel shapes
and serving settings of each image and profile, and is written during the
first start after the model switch, where running out of space fails that
start.

`sparkring up`, whose image step pulls the whole image itself, reserves the
image's whole pull, the compile cache allowance and, unless the Spark holds a
verified checkpoint, the checkpoint's files and headroom from its pin manifest
(the third column below). `sparkring setup storage` reserves the same
checkpoint figure, the 68 GiB image allowance, because a profile selection
records no image sizes, and the compile cache allowance. The per-repository
checkpoint allowances of storage planning apply only to a revision without a
pin manifest.

The serving image `dev-20261004-kraken-cuda1342-nccl2323-status034` is a
14.2 GiB download, 29.7 GiB unpacked. The last column below adds the whole
image, 51.9 GiB, and the 4 GiB compile cache allowance to the checkpoint
figure: the need of a Spark holding neither the image nor a checkpoint file,
with the checkpoint, Docker and the cache on one filesystem. Node A needs
14.2 GiB more for the relay copy.

| Checkpoint | Size | Files and headroom | Plan total, empty Spark |
|---|---:|---:|---:|
| Qwen, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `60215d26cf5e` | 102.6 GiB | 106.6 GiB | 162.5 GiB |
| MiMo, `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `2479e2d0029e` | 165.6 GiB | 176.9 GiB | 232.8 GiB |
| GLM, `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` @ `a608241037e4` | 174.8 GiB | 179.5 GiB | 235.4 GiB |
| GLM `--checkpoint nvfp4-qad`, `local-inference-lab/GLM-5.3-Flash-NVFP4` @ `175ae8ce3b5a` | 185.7 GiB | 190.7 GiB | 246.6 GiB |
| GLM `--checkpoint nvidia-nvfp4`, `nvidia/GLM-5.3-Flash-NVFP4` @ `da920bb0b9f4` | 190.4 GiB | 198.8 GiB | 254.7 GiB |
| GLM `--checkpoint nvfp4-mxfp8-csf-qad`, `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` @ `fd660d51d1fc` | 166.3 GiB | 171.0 GiB | 226.9 GiB |
| Qwen `--checkpoint jmni-qad5500-hybrid`, `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` @ `87c8f2fb738b` | 99.1 GiB | 103.0 GiB | 158.9 GiB |
| DeepSeek, `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa` | 475.3 GiB | 491.3 GiB | 547.2 GiB |
| Swift, `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` @ `3ff0520224f2` | 173.7 GiB | 181.7 GiB | 237.6 GiB |

The derived Qwen checkpoint, `--checkpoint qad-step5500-mxfp8-attention`,
adds to the Qwen row the 5.6 GiB of files its recipe writes on every Spark
and, on Node A, the 2.6 GiB of step-4000 files the recipe reads
([derived checkpoints](#derived-checkpoints)): 170.5 GiB on an empty Node A
and 167.9 GiB on the other Sparks. `sparkring up` and `sparkring setup
storage` reserve the Node A figure.

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
  (`/srv/sparkring/<cluster>/<profile>-i<identity>`). The installed
  deployment's is in use. On a four-Spark ring, so is the workspace of the
  deployment that created the ring's mesh: it holds the mesh's host marker,
  and later deployments reuse that mesh;
- Docker images, and every other entry of `/srv/sparkring`, such as
  directories you created there.

Each item is `installed`, `profile`, `unreferenced` or `unmanaged`
([classes](commands.md#storage)), and the report ends with the release
commands for the unreferenced ones.

- `sudo sparkring storage --release PATH` removes one unreferenced cache
  directory or workspace from every Spark that holds it, after asking
  (`--yes` in scripts). Each Spark first checks again that no installed
  deployment, no mesh installed on it and no running container uses it.
- A mesh counts while its configuration is in `/etc/sparkring/managed-mesh`
  or `/etc/sparkring/deployments/<name>`, running or not: SparkRing starts a
  stopped or disabled mesh again for a deployment that uses it. A mesh whose
  site file cannot be read keeps every cache directory and workspace on its
  Spark.
- `sudo sparkring checkpoints --release PATH` releases checkpoint directories
  ([Checkpoints](#checkpoints)). A derived checkpoint's directory is
  `installed` while its deployment is, and otherwise `profile` while an
  installer profile lists it; releasing it frees only the files its recipe
  wrote ([derived checkpoints](#derived-checkpoints)).
- Docker images and directories that SparkRing's installer did not create are
  never removed.
- [Automatic release](#automatic-release) removes what older deployments hold
  after each installation and `up`.

### Automatic release

Each installation request with another package revision, image, checkpoint or
serving setting is a separate deployment, with its own directory in
`/var/lib/sparkring/controller/deployments`. Once it has run, it holds on
every Spark its workspace (the source checkout, its Compose and runtime-binding
files and its checkpoint receipt, about 100 MB), its stopped model container,
whose writable layer Docker counts as reclaimable, and compile caches for its
image and checkpoint revision. After every `sudo sparkring install` and
`sudo sparkring up` that completes, Node A keeps a deployment when:

- it is the active deployment, the rollback target or the candidate of an
  unfinished model switch, on the whole ring or pair or on either half of a
  ring that serves [two models](#two-models-on-one-ring);
- its model container runs on a Spark;
- a mesh installed on a Spark uses its workspace. On a four-Spark ring, the
  deployment that created the mesh holds the mesh's host marker, and the mesh
  starts that deployment's container;
- its last operation did not complete, or its state cannot be read, since it
  may need recovery from its own receipts;
- its last operation is a completed `up`: it was started and not stopped
  since; or a completed preparation, which a repeated preparation verifies
  through its workspace;
- it has a managed GLM backend, whose services own its containers;
- it is one of the 2 most recent deployments of its profile, by the time of
  its last operation. For each model these are the deployment that runs or ran
  last and the one before it, usually the previous package revision or another
  checkpoint or serving setting, which then start without preparing a
  workspace, container or compile cache.

For every other deployment whose last operation is a completed `down`, each
Spark removes:

- its stopped model containers (`sr-<site>-r<rank>`, labelled with the
  deployment's ID);
- its workspace;
- each compile cache that no kept deployment uses and no installer profile of
  the installed package references (class `unreferenced`);
- remainders of interrupted `sudo sparkring storage --release` runs.

Node A removes its checkout of a SparkRing source revision
(`/var/lib/sparkring/controller/retained-sources/<revision>`, about 80 MB
with the current source tree) once every deployment of that revision is
released. An operation on such a deployment clones it again from the
deployment's source bundle.

Nothing is removed while a Spark cannot be listed, its Docker cannot be read,
or it runs another package revision than Node A. Each Spark checks again before
it removes anything. A container must be stopped, carry the deployment's label
and not be the one that an installed mesh starts. Each workspace and cache
passes the same checks as `sudo sparkring storage --release`, with the paths
that the kept deployments name: no installed mesh, model files, mount point or
running container. A refusal is listed under the summary line, and the rest
continues. Checkpoint
directories, Docker images, the deployment directories on Node A (lock, source
bundle and receipts) and anything SparkRing's installer did not create are
never removed. The release prints one line, for example:

```text
Released 7 older deployments' containers, workspaces and caches: 9.8 GiB
```

A released deployment starts again with
`sudo sparkring up PROFILE --instance i<hash>` once the running model is
stopped, or with an installation of the same choices. Its `source` step copies
the deployment's source bundle from Node A into a new workspace. Its `model`
step writes the checkpoint receipt again from the checkpoint directory's
journal and SparkRing's path records, hashing only files whose size or times
changed. Its `create` step creates the container. A first start without its
compile cache compiles and tunes kernels again. `sudo sparkring down` of a
released deployment reports that it is stopped and changes nothing, because
its stop would read the removed workspace.

`sudo sparkring storage` lists the kept deployments with their reasons and
what the next release frees. `sudo sparkring storage --retain-deployments N`
keeps the N most recent deployments of each profile; `0` keeps only the
deployments that the other rules keep, and `off` turns automatic release off.
The preference `SPARKRING_RETAIN_DEPLOYMENTS` of
`sudo sparkring install --env FILE` sets the same value
([Optional preferences](#optional-preferences)). Both are saved on Node A
and apply to every later installation and `up`.

The first release on a cluster with many earlier deployments removes all of
them at once. Afterwards a release contacts the Sparks only when a deployment
leaves the kept set, or while a released deployment still keeps a workspace or
container that a check refused, for example a workspace holding model files;
`sudo sparkring storage` names such deployments.

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
`SPARKRING_DOWNLOAD_LIMIT` and `SPARKRING_RETAIN_DEPLOYMENTS` on every run;
`sudo sparkring setup --env /path/to/settings.env` reads it on every run.
An installation that completes saves `SPARKRING_RETAIN_DEPLOYMENTS` on
Node A, where later installations and `sparkring up` read it.
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
| `SPARKRING_RETAIN_DEPLOYMENTS` | the value saved on Node A, else `2` | `off`, or the number of recent deployments of each profile to keep, `0` to `99` ([details](#automatic-release)) | `sparkring storage --retain-deployments` |

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
its SSH service, also over the remaining links and the
[fallback paths](#admin-tunnel) if one administration link fails, and
`sparkring-fabric.service` restores the approved fabric routes,
per-interface IPv4 forwarding and forwarding rules; NetworkManager keeps the
fabric addresses. Afterwards `sparkring-agent` adds an approved route again
when the link it uses returns, and sets approved per-interface settings that
a driver restart reset
([Fabric addresses and routes](#fabric-addresses-and-routes)). Models start
only when requested, or when [automatic recovery](#automatic-recovery)
restarts the active model after its last `up` completed; `sparkring down`
stops the selected deployment.

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

Some profiles list more than one checkpoint. `--checkpoint NAME` installs a
listed one with the settings it needs, as its own deployment with its own
pinned revision and checkpoint directory; an unlisted name changes nothing.
Installing again without `--checkpoint` switches back to the default.

```bash
sudo sparkring install --profile qwen38-flash-next-tp2 --checkpoint qad-step-4000
sudo sparkring install --profile qwen38-flash-next-tp2 --checkpoint qad-step5500-mxfp8-attention
sudo sparkring install --profile qwen38-flash-next-tp2 --checkpoint jmni-qad5500-hybrid
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp4 --checkpoint nvidia-nvfp4
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp4 --checkpoint nvfp4-mxfp8-csf-qad
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp2 --checkpoint nvfp4-qad
```

| Profiles | `--checkpoint` | Checkpoint | Settings that differ from the default |
|---|---|---|---|
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | `qad-step5500-ple1000`, also `qad-step-5500` (default) | Branch `qad-step5500-ple1000` of Local Inference Lab's Qwen3.8-Flash-Next NVFP4, revision `60215d26cf5e` | — |
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | `qad-step-4000` | Branch `qad-step-4000` of the same repository, revision `629bc3218833` | MXFP8 target LM head; the draft's NVFP4 experts on B12X |
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | `qad-step5500-mxfp8-attention` | Step 5500 with its 240 text attention projections in MXFP8, which the installer derives on the Sparks from step 5500 and step 4000's MXFP8 tensors ([derived checkpoints](#derived-checkpoints)) | Served as `Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-TP2` or `-TP4`; other settings as step 5500 |
| `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` | `jmni-qad5500-hybrid` | [Qwen3.8-Flash-Next NVFP4 QAD-5500 Hybrid](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid/tree/87c8f2fb738b597de99bf9a885130f4a18a94f3d) by JMNI Labs, revision `87c8f2fb738b` | The draft's NVFP4 experts on B12X; served as `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid-TP2` or `-TP4` |
| `glm53-flash-nvfp4-spark-tp4` | `nvfp4-spark` (default) | [GLM-5.3-Flash NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) by Local Inference Lab, revision `a608241037e4` | — |
| `glm53-flash-nvfp4-spark-tp4` | `nvfp4-qad` | [GLM-5.3-Flash NVFP4 QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/175ae8ce3b5af842b0d0140dbeb43e9cfc557c49) by Local Inference Lab, revision `175ae8ce3b5a` | The draft's MXFP8 experts on the Humming MoE backend; 37 GiB of KV cache per Spark; served as `GLM-5.3-Flash-NVFP4-QAD-TP4` |
| `glm53-flash-nvfp4-spark-tp2` | `nvfp4-spark` (default) | GLM-5.3-Flash NVFP4-Spark, as above | — |
| `glm53-flash-nvfp4-spark-tp2` | `nvfp4-qad` | GLM-5.3-Flash NVFP4 QAD, as above | 5 GiB of KV cache per Spark; a 524,288-token context window; served as `GLM-5.3-Flash-NVFP4-QAD-TP2`. The pair's draft already runs its experts on the Humming MoE backend |
| `glm53-flash-nvfp4-spark-tp4` | `nvidia-nvfp4` | [GLM-5.3-Flash NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) by NVIDIA (ModelOpt), revision `da920bb0b9f4` | `--quantization modelopt_fp4` and `--load-format safetensors`; the draft's BF16 experts on vLLM's unquantized MoE kernel; 36 GiB of KV cache per Spark; served as `GLM-5.3-Flash-NVFP4-NVIDIA-TP4` |
| `glm53-flash-nvfp4-spark-tp4` | `nvfp4-mxfp8-csf-qad` | [GLM-5.3-Flash NVFP4 MXFP8 CSF QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD/tree/fd660d51d1fc3caae26a4bf31b7451475bbb9bdc) by Local Inference Lab, revision `fd660d51d1fc`: QAD experts, MXFP8 attention, and NVFP4 scales stored compressed in an NVFP4-CSF container | `--quantization nvfp4_csf` and `--load-format nvfp4_csf`; serves the container's `metadata/` directory, with `--hf-overrides` naming the mounted container as the checkpoint root; routed experts as W4A16 (`VLLM_B12X_MOE_FP4_FORCE_A16=1`, `B12X_W4A16_FP32_TOPK_WEIGHTS=1`); served as `GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD-TP4` |

- The Qwen `qad-step5500-mxfp8-attention` entry is **implemented** on the
  installer image on two and four Sparks. On one pair and one four-Spark
  ring, `sudo sparkring install` derived it on the Sparks (on the pair,
  including the 2.6 GiB donor download in 31 s; the recipe took 32 s and
  39.5 s), every file matching the manifest. Served from it, each passed all
  7 functional checks and a 256-request correctness screen with no
  degenerate or failed response. Against stock step 5500 on the same
  cluster and image, two runs each, it ran more decode steps per second at
  1, 8 and 16 streams: 17%, 8% and 7% more on the pair (27.3 / 92.2 / 130.8
  against 23.3 / 85.5 / 122.1) and 8%, 4% and 1% more on the ring (36.8 /
  124.5 / 178.9 against 34.0 / 119.5 / 176.9), and prefilled 4.2 to 5.3% and
  1.9 to 2.3% faster ([pair](../../performance/records/images/dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002.md), [ring](../../performance/records/images/dev-20261001-kraken-qwen38-flash-next-qad-tp4-qad-step5500-mxfp8-attention-20261002.md)). It
  costs a little quality: the [research record](../../performance/records/qwen38-flash-next/mxfp8-attention-20261001.md)
  measured, over a 9,708-token log-likelihood check, a mean negative
  log-likelihood 0.0045 and 0.0063 nats per token above step 5500's, about
  0.5% in perplexity, against 0.0003 and 0.0015 between two runs of one
  checkpoint, and 1.19 GiB less weight memory on each Spark of a pair. It is
  never a profile's default.
- The Qwen `jmni-qad5500-hybrid` entry is **research-only** and third-party:
  JMNI Labs built it from Local Inference Lab's published tensors, and Local
  Inference Lab has not reviewed or qualified it. Its model card reports, on
  two Sparks with another vLLM build, 26.5 decode steps per second at one
  stream against 23.0 for step 5500 (with its draft on the Marlin MoE
  kernel), and 66.9% on 1,000 MMLU-Pro questions with direct answers against
  67.0% for step 5500. It needs a runtime whose `modelopt_mixed` method
  serves MXFP8 attention, W4A16 NVFP4 MTP experts and NVFP4 PLE; the installer
  image serves each of them in step 4000 or step 5500. A tensor-by-tensor
  comparison with the research checkpoint of the derived entry found its 480
  attention weight and scale tensors byte-identical (same dtype, shape and
  bytes); the shard that holds them, `hybrid-main-00002.safetensors`, has the
  SHA-256 of step 4000's `model-00035-of-00036.safetensors`. It differs from
  the derived checkpoint in its weights' publisher and in the MTP draft's
  routed experts, which it stores as W4A16 NVFP4 from Local Inference Lab's
  `main` revision `7c4f1bc1a2d6`, as step 4000 does, so its draft runs them on
  B12X; the derived checkpoint keeps step 5500's MXFP8 draft experts on
  Humming. Draft experts change how fast drafting runs, not the output
  distribution, so the card's MMLU-Pro result also describes the derived
  checkpoint's target weights. No installation has run on Sparks.
- The four-Spark `nvfp4-qad` entry is **implemented** on the installer
  image: on one four-Spark ring it passed all 7 functional checks and a
  256-request correctness screen with no degenerate or wrong response, and
  decoded 57.7 / 184 / 258 tok/s at 1 / 8 / 16 streams in one run
  ([record](../../performance/records/images/dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvfp4-qad-20261001.md)).
  Its host memory headroom was not measured.
- The GLM `nvidia-nvfp4` checkpoint is **implemented** on the installer
  image: on one four-Spark ring it passed all 7 functional checks and a
  256-request correctness screen with no degenerate or wrong response, and
  decoded 53.2 / 171 / 244 tok/s at 1 / 8 / 16 streams in one run
  ([record](../../performance/records/images/dev-20260928-plainstatus-glm53-flash-nvfp4-spark-tp4-nvidia-nvfp4-20261001.md)).
  Its host memory headroom and multi-turn tool calls were not checked.
- The GLM `nvfp4-mxfp8-csf-qad` checkpoint is **research-only**: no
  installation has run on Sparks. Its container format,
  `lil-nvfp4-csf-checkpoint/1`, keeps the served configuration, tokenizer,
  chat template and weight index under `metadata/` and the 44 weight shards
  under `tensors/`. The pin manifest covers both, and the entry serves
  `metadata/` while vLLM's `nvfp4_csf` loader reads the shards and checks the
  container's `manifest.json` and `build-contract.json` from the root that
  `--hf-overrides` names, `/models/target` in every container. The override
  carries the checkpoint's own ModelOpt quantization table, copied into
  `profiles/glm53-flash-nvfp4-spark-tp4/nvfp4-mxfp8-csf-qad.quantization.json`.
  The entry keeps the profile's 40 GiB of KV cache: its files are 8.5 GiB
  smaller than NVFP4-Spark's, but its loaded weight memory has not been
  measured.
  - The profile's installer image,
    `dev-20261004-kraken-cuda1342-nccl2323-status034`, has the `nvfp4_csf`
    loader, but not the later vLLM and B12X changes for CSF scale prefetch
    and W4A16 CSF performance; an image built after them is expected to
    decode faster.
  - The checkpoint is published under the LIL License 1.0, which forbids
    re-uploading it and requires its attribution notice at the top of any
    README or landing page of a service that runs it. Read the licence on the
    model page before serving it.
- The four-Spark GLM profile's evidence and the
  [GLM memory record](../../performance/records/glm53-flash/installer-memory-20260929.md)
  cover NVFP4-Spark only.
- By their pin manifests, the QAD and NVIDIA weights take 2.4 and 3.9 GiB more
  than NVFP4-Spark's on each Spark of a ring. Their KV caches are smaller than
  the profile's 40 GiB by that much, rounded up to whole GiB, so each Spark
  keeps the free memory that the memory record measured for images, videos and
  long requests. The ring's KV cache held 5,668,802 tokens at 37 GiB with
  NVFP4-Spark.
- The two-Spark GLM profile offers the QAD checkpoint with 5 GiB of KV cache
  per Spark and a 524,288-token context window, and does not list NVIDIA's.
  On a pair, the QAD and NVIDIA weights take 4.9 and 7.8 GiB more on each
  Spark than NVFP4-Spark's. The QAD entry's KV cache is the pair's 10 GiB less
  that 4.9 GiB, rounded down to whole GiB, so each Spark keeps the free memory
  the memory record measured. The pair's KV cache held about 137,000 tokens per
  GiB with a 262,144-token context window and 153,000 with a 1,048,576-token
  window, so 5 GiB holds 0.68 to 0.77 million tokens: one request of 524,288
  tokens needs 3.4 to 3.8 GiB. A 1,048,576-token request needs about 6.8 GiB,
  more than 5 GiB, and NVIDIA's weights would leave about 2 GiB. The entry
  keeps the pair's other settings, which do not size the KV cache: at most 8
  requests at a time, which share the smaller cache, 8,192 tokens per batch
  and 8 images per request.
- The pair's `nvfp4-qad` entry is **implemented**: on one pair it passed all 7
  functional checks and a 256-request correctness screen with no degenerate or
  wrong response, decoded 30.7 / 94 tok/s at 1 / 8 streams in one run, and one
  request with 8 images left Node A 2.19 GiB of memory, as NVFP4-Spark's pair
  profile does
  ([record](../../performance/records/images/dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001.md)).
- NVIDIA's revision `da920bb0b9f4` holds the same weights and weight index as
  revision `423acf37583782c51c142d145aef733d72943d93`, which the
  [manual NVIDIA target](../../profiles/glm53-nvidia-nvfp4.md) pins. Its
  `config.json` and `hf_quant_config.json` also exclude the BF16 MTP layer
  from quantization, the entries that the manual target adds with
  `--hf-overrides`.

### Derived checkpoints

A derived checkpoint is one that no repository publishes: the installer writes
it on the Sparks from pinned published files and a recipe in this repository,
then serves it from its own checkpoint directory. `--checkpoint
qad-step5500-mxfp8-attention` is one: step 5500 with its 240 text attention
projections stored as MXFP8 block 32 instead of BF16. Its recipe,
[`mxfp8_attention.py`](../../runtime/common/mxfp8_attention.py), takes those
tensors from step 4000, which stores the same frozen weights in MXFP8, only
after vLLM's MXFP8 quantization of each step-5500 BF16 weight reproduces step
4000's weight and scale bytes exactly. Its
[manifest](../../profiles/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/648b194a96e5f130ab62702113242d8e1ddd6e76.json)
pins the size and SHA-256 of all 54 files: 48 that are step 5500's, unchanged,
and 6 that the recipe writes (both shards that hold projections,
`config.json`, `hf_quant_config.json`, the weight index and
`derivation.json`). Its revision is the identity of the base revision, the
donor revision and files, and the recipe's SHA-256, so a changed input or
recipe is another checkpoint.

An installation:

1. acquires step 5500 into its own checkpoint directory on every Spark, as
   for `--checkpoint qad-step5500-ple1000`: from copies found on the Sparks,
   from another Spark or, failing those, from Hugging Face;
2. hard-links on every Spark the 48 unchanged files from step 5500's directory
   into the derived directory, so they take no space;
3. downloads on Node A the two step-4000 files the recipe reads, the weight
   index and `model-00035-of-00036.safetensors` (2.8 GB), unless step 4000's
   own checkpoint directory already holds them, and checks each against step
   4000's pin manifest;
4. runs the recipe on Node A in the deployment's installer image, CPU only
   and without network, with both checkpoints mounted read-only;
5. copies the 6 written files (5.6 GiB) to the other Sparks over the fabric.

Every written, downloaded or copied file is placed only after its SHA-256
equals the manifest's, and both directories are verified again before the
model starts. The plan lists each step with its sizes, and the download
counts toward the approval that a download of more than 1 GiB needs. A
repeated installation finds the derived directory complete and only verifies
it.

The derived directory is
`/srv/sparkring/<cluster>/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/<revision>`,
beside step 5500's; step 4000's files go to its own checkpoint directory. The
installation stops, naming what failed, when:

- step 5500 would be served in place from a named copy, or the derived
  directory lies on another filesystem than step 5500's directory, because
  the unchanged files are hard links;
- quantizing a step-5500 projection does not reproduce step 4000's bytes; the
  message names the projection;
- a written file differs from the manifest, or the recipe in the installed
  package differs from the one the manifest pins; the message names the file
  or the recipe.

Every Spark needs 5.6 GiB free beside step 5500 for the written files, and
Node A 2.6 GiB more for step 4000's files ([space](#downloads-storage-and-outbound-hosts)).
`sudo sparkring checkpoints` and `sudo sparkring storage` list the derived
directory with the base it is derived from; releasing it frees only the 5.6
GiB the recipe wrote, because its other files are hard links to step 5500's.
Step 5500's directory keeps its data whatever happens to the derived one, and
neither is released while a protected deployment uses it.

The derived files do not depend on where or when they are written: the recipe
writes each shard with its own safetensors writer, in the layout of the
safetensors library, so the manifest can pin every file. From step 5500's
`config.json`, `hf_quant_config.json`, weight index and shard headers, the
recipe reproduces the research checkpoint's `config.json`,
`hf_quant_config.json` and weight index byte for byte, and the index's total
size fixes the two rewritten shards' sizes; the shards' SHA-256 come from the
[research record](../../performance/records/qwen38-flash-next/mxfp8-attention-20261001.md), and an installation on one pair wrote all
6 files with the manifest's SHA-256 ([record](../../performance/records/images/dev-20261001-kraken-qwen38-flash-next-tp2-qad-step5500-mxfp8-attention-20261002.md)).

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
4000 (`--checkpoint qad-step-4000`) except `config.json`, and a copy of NVIDIA
GLM revision `423acf37` every file of `--checkpoint nvidia-nvfp4` except
`config.json` and `hf_quant_config.json`. SparkRing hashes each file before
using it.

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
Compose files with the deployment's [serving settings](#serving-settings). It
excludes private addresses, paths, receipts and source bundles.
`sparkring export --output private-deployment.zip` keeps the actual
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
