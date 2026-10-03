# SparkRing commands

Run `sparkring` on Node A, the Spark connected to your network, unless the
table says otherwise. `sparkring COMMAND --help` prints the same flags.

[Install SparkRing](install.md) · [Reference](install-reference.md)

## Common examples

Pick `PROFILE` for your model and number of Sparks from the
[profile table](../../README.md#profiles).

```bash
sudo sparkring install --profile PROFILE                     # set up the Sparks and start the model
sudo sparkring install --profile PROFILE --plan              # print the plan, change nothing
sudo sparkring install --profile PROFILE --model-path /data/models/my-model  # reuse a copy
sudo sparkring install --profile PROFILE --checkpoint NAME   # another checkpoint the profile lists
sudo sparkring install --profile PROFILE --image NAME        # another image from sparkring images
sudo sparkring install --profile PROFILE --on 2,3            # a two-Spark model on half of a four-Spark ring
sudo sparkring logs --follow                                 # follow progress
```

## All commands

| Command | Runs on | sudo | Does |
|---|---|---|---|
| [`install`](#install) | Node A | yes | Set up the Sparks and start one model |
| [`setup`](#setup) | Node A | yes | Set up the Sparks without a model |
| [`cabling`](#cabling) | Node A | yes | Show how the Sparks are cabled and what to move; changes nothing |
| [`models`](#models) | any | no | List profiles and mark those `install` supports |
| [`images`](#images) | any | no | List the installer images `install --image` can select |
| [`status`](#status) | Node A | yes | Show each Spark's state and the saved model |
| [`logs`](#logs) | Node A | yes | Show or follow the installation log |
| [`hairpin`](#hairpin) | Node A | yes | Apply the ConnectX setting that four-Spark rings need |
| [`checkpoints`](#checkpoints) | Node A | yes | List or release SparkRing's checkpoint directories |
| [`storage`](#storage) | Node A | yes | Report disk use; release caches and workspaces no deployment uses |
| [`up`, `down`](#up-and-down) | Node A | yes | Start or stop a model deployment |
| [`recover`](#recover) | Node A | yes | Show, or turn on or off, the automatic restart of the model |
| [`node`](#node) | each Spark | yes, except status reports | Per-Spark services; Node A calls most of them |
| [`init`, `export`](#init-and-export) | any | no | Save a deployment from hosts or a site file; export it as files |
| [`compose`](#compose-and-validate-compose) | a checkout | no | Render, check, start or stop profile Compose deployments |
| [`validate-compose`](#compose-and-validate-compose) | any | no | Check Compose files offline |
| [`deploy`](#deploy) | a checkout | no | Standalone discovery, network plans and staged model operations |
| [`cluster`, `doctor`, `host`](#cluster-doctor-and-host) | rank 0 or one Spark | no | Manual ring bootstrap and diagnosis |

With `--json`, `install`, `hairpin`, `checkpoints` and `storage` print one JSON
document. They exit with 0 on success or plan, 3 when input is needed and 2 on
failure.

## install

`sudo sparkring install [flags]` sets up the Sparks on first use, prepares the
image and checkpoint, then starts or switches the model. Once the model
serves, it releases what older deployments hold on the Sparks
([automatic release](#automatic-release)).

| Flag | Meaning |
|---|---|
| `--profile PROFILE` | Exact profile from `sparkring models`; asked in a terminal when omitted |
| `--on 0,1` or `--on 2,3` | Put a two-Spark profile on one half of a four-Spark ring ([two models on one ring](install-reference.md#two-models-on-one-ring)); default: the half that serves no model |
| `--plan` | Print and save the setup, checkpoint and model plan; change nothing. Before the first setup, use `sudo sparkring setup --plan` |
| `--yes` | Approve setup, the checkpoint plan, ConnectX restarts on an idle ring and the model switch; unknown SSH host keys still need confirmation |
| `--json` | One JSON result on stdout; progress on stderr |
| `--checkpoint NAME` | Another checkpoint the profile lists ([names](install-reference.md#another-checkpoint-of-a-profile)); default: the profile's own |
| `--model-path [N=]PATH` | A checkpoint copy to reuse, for every Spark or for Node N; repeatable; never written |
| `--ignore-local-copies` | Use only SparkRing's own checkpoint directories and named copies |
| `--cache-path PATH` | Another writable compile cache on each Spark |
| `--download-limit RATE` | Cap the checkpoint download from Hugging Face: `850Mbit`, `2Gbit` or `none` ([details](install-reference.md#limit-the-download-rate)) |
| `--events FILE` | Also write progress to FILE, one JSON object per line ([fields](install-reference.md#event-stream)) |
| `--env FILE` | Preferences file: setup keys on first installation, the download limit and the [retained deployments](#automatic-release) on every run ([keys](install-reference.md#optional-preferences)) |
| `--stop-workloads` | Stop (never remove) GPU containers that are not SparkRing's |
| `--no-auto-recover` | Do not restart this model by itself when a Spark stops serving ([automatic recovery](install-reference.md#automatic-recovery)) |
| `--image NAME` | Another installer image: a name or release tag from [`sparkring images`](#images) ([details](install-reference.md#another-image)); default: the installer's own image |
| `--image-lock FILE` | Development image lock that replaces the shared installer image |
| `--max-images N`, `--max-videos N`, `--context-length N`, `--max-concurrency N`, `--kv-cache-gib N`, `--save-cpu` | Replace one of the profile's serving values for this deployment ([serving settings](install-reference.md#serving-settings)) |
| `--allow-driver-reload` | Accepted and not needed; the approval covers ConnectX restarts |

## install.sh

[`install.sh`](../../install.sh) builds the SparkRing package from a full clone
of `main`, installs it with `apt` and runs `sudo sparkring install` with every
option you pass it. Run it on Node A as a user with sudo; the one-line command
is in [Get the package](install-reference.md#get-the-package). It builds in a
temporary directory under `/var/tmp` and removes it before `sparkring install`
starts, or after a plan from the extracted package ends. When piped, it reads
its questions and the installer's from the terminal. It skips `apt` if the
built version is installed.

| Flag | Meaning |
|---|---|
| `--yes` | Answer the package question and `sparkring install`'s questions; required without a terminal |
| `--plan` | Install nothing; plan with the built version (below) |
| `--package-only` | Ask the package question, install or keep the package and stop before `sparkring install`; not with `--plan` |
| `--json` | One `sparkring-install-result/v1` document on stdout (below); progress, questions and `apt` output on stderr |
| `--ref BRANCH_TAG_OR_COMMIT`, `--repository URL` | Build another ref or repository ([Pin a commit](install-reference.md#pin-a-commit)) |

`--plan` runs `sparkring install --plan` from the installed package if this
Spark has the built version, otherwise from the built package extracted into
the temporary directory; with no SparkRing package on this Spark, the script
stops with `needs_input` (field `package`). The plan inspects every Spark,
Node A included, through its installed package and compares their revisions
with the built one, which installation puts on Node A first. Like any
`sparkring install --plan`, it is saved on Node A and bounds a later `--yes`
run with the same source revision and options.

After `--package-only`, `sudo sparkring install --profile PROFILE --plan`
reviews the rest. With `--json`, `--package-only` reports `state`
`package-installed`, `version`, the `previous` version and whether it
`changed`.

The `--json` document is the installer's result, or the script's own when it
stops first or a plan from the extracted package ends without a result (stage
`plan`): `state` `failed` and the failed `stage` (exit status 2), or
`needs_input` and the needed `field` (exit status 3).

## setup

`sudo sparkring setup [flags]` discovers the cabled Sparks and configures them,
without a model. `sparkring install` runs it on first use. Sparks that belonged
to other SparkRing clusters are
[re-formed](install-reference.md#re-form-sparks-into-another-pair-or-ring)
into the pair or ring now cabled.

| Flag | Meaning |
|---|---|
| `--plan` | Discover and review over existing SSH access; configure nothing |
| `--yes` | Accept the listed changes; unknown SSH host keys still need confirmation |
| `--env FILE` | Literal preferences file; its keys set the defaults below ([keys](install-reference.md#optional-preferences)) |
| `--name NAME` | Cluster name: a lowercase letter, then lowercase letters, digits or `-`; at most 35 characters (default `sparkring`) |
| `--ssh-user USER` | Worker account (default: the account that ran `sudo`; `root` with `--env` or port 2222) |
| `--ssh-port 22\|2222` | Worker SSH port; 2222 reaches workers prepared with `--worker-bundle` |
| `--control-cidr CIDR` | Administration network, an IPv4 `/29` (default `10.253.255.0/29`) |
| `--fabric-cidr CIDR` | Fabric addresses, an IPv4 `/16` to `/21` (default `198.18.0.0/21`) |
| `--no-share-internet` | Do not route workers' downloads and DNS through Node A |
| `--reset-links` | Replace incompatible fabric IPv4 settings, also without a terminal |
| `--stop-workloads` | Stop (never remove) GPU containers that block fabric preparation |
| `--worker-bundle` | Build a USB bundle for [workers without SSH](install-reference.md#if-a-worker-has-no-ssh) |
| `--admin-fallback` | On a set-up cluster, add [fallback paths](install-reference.md#admin-tunnel) (the other cables, the LAN) to the admin tunnel and change nothing else; with `--plan`, list them |
| `--allow-driver-reload` | Accepted and not needed |

With `--node` or `--inventory`, setup uses listed targets instead of discovery
over the cables, and accepts `--name`, `--fabric-cidr`, `--plan`, `--yes`,
`--allow-driver-reload` and these flags:

| Flag | Meaning |
|---|---|
| `--node USER@IP` | A management target, Node A included; repeat for each Spark, each with the same package |
| `--apply` | Apply the reviewed plan |
| `--adopt` | Verify and record existing networking without changing links, routes or services |
| `--skip-enroll` | SSH keys and host trust are already configured |
| `--inventory FILE` | Offline node records; planning only, with `--plan` |
| `--head-id ID` | Node A's identity for `--inventory` |
| `--output DIR` | Directory for the setup receipt |

Two setup actions run offline without sudo and accept `--variant`:

- `sparkring setup show PROFILE [--format text|json|shell]` prints the
  profile's image and checkpoint.
- `sparkring setup storage PROFILE --model-path P --cache-path P --docker-path P [--reuse-model] [--reuse-image] [--json]`
  checks free space on those filesystems without writing.

## cabling

`sudo sparkring cabling [flags]` shows how the Sparks on this Spark's fabric
cables are cabled, and what to move for a pair or a four-Spark ring. It
changes nothing on any Spark. Setup stops with the same advice when the
cables do not fit ([cabling rules](install-reference.md#cabling)).

```text
Sparks read:
  spark-a: this Spark
  spark-b: the admin network at 10.253.255.2
  spark-c: the LAN at 192.0.2.13 as operator (LLDP not readable without sudo)
  spark-d: the LAN at 192.0.2.14 as operator (LLDP not readable without sudo)
Cables:
  spark-a port 0 ↔ spark-b port 1
  spark-b port 0 ↔ spark-c port 1
  spark-c port 0 ↔ spark-d port 0
  spark-d port 1 ↔ spark-a port 1
The four Sparks form a loop, but 2 cables join the same port number at both ends. In a ring, every cable runs from port 0 of one Spark to port 1 of the next.
To fix:
  On spark-d, swap its two cables (port 0 ↔ port 1).
Ring order after the fix: spark-a → spark-b → spark-c → spark-d
```

It reads this Spark, the Sparks of its recorded cluster over the admin
network, and the other Sparks on the cables. It signs in to those over the
LAN or the cables as the account that ran `sudo`, with Node A's setup key
where they accept it; SSH asks for a password elsewhere.

| Flag | Meaning |
|---|---|
| `--json` | One `sparkring-cabling/v1` document |
| `--ssh-user USER` | Account for signing in to the other Sparks (default: the account that ran `sudo`) |
| `--no-sign-in` | Read only this Spark and its recorded cluster's Sparks |

It exits with 0 when the cables form a pair or ring as SparkRing needs, 1
when a cable needs to move or not every cable could be seen, and 2 on
failure.

## models

`sparkring models [--json]` lists every profile (exact model, version,
quantization and topology) and marks those `sparkring install` supports.

## images

`sparkring images [--profile PROFILE] [--json]` lists the installer images
this package records, the default first, with the GitHub release that
published each, its download size and the profiles it runs. `--profile`
lists only the images that run that profile. Any listed name, release tag or
part of a name that only one image has selects that image in
`sudo sparkring install --image NAME`.

## status

`sudo sparkring status [PROFILE [--instance NAME]] [flags]` prints Node A's
state, one line per Spark with the next action for any Spark that needs
attention, the saved model (the active deployment, or the one named) and
automatic recovery. On a ring that serves
[two models](install-reference.md#two-models-on-one-ring) it prints each
half's model under `Sparks 0 and 1:` and `Sparks 2 and 3:`:

```text
Saved model operation: PROFILE | up complete
Checkpoint: NAME (REPOSITORY @ REVISION) | Image: RELEASE
Automatic recovery: on
```

A profile with one checkpoint shows only `REPOSITORY @ REVISION`; a
[derived checkpoint](install-reference.md#derived-checkpoints) adds
`, derived from REPOSITORY @ REVISION` of its base. The
recovery lines add the Spark it waits for, its last attempt and the next.
After `sparkring down` they read `on; idle until the next sudo sparkring up
--execute or sudo sparkring install`, and a deployment that recovery does not
restart, such as managed GLM, reads `not available`.

With `--refresh`, a line per model container follows the saved model. When
the model does not serve, the first line says why and gives the command that
fixes it ([every case](install-reference.md#when-a-model-stops-serving)):

```text
The model runs on rank 0 (spark-a) but stopped on rank 1 (spark-b) | next: sudo sparkring down --execute, then sudo sparkring up --execute
  rank 1 (spark-b): stopped, exit code 255 at 2026-09-30 17:02:11 UTC
```

| Flag | Meaning |
|---|---|
| `--refresh` | Contact every Spark, inspect the model containers and ask rank 0's API `/health` |
| `--on 0,1` or `--on 2,3` | Only that half's model |
| `--json` | Print the full observation as JSON; a ring with half models adds `slots`, one entry per half |

## logs

`sudo sparkring logs [flags]` prints the end of `/var/log/sparkring/install.log`.

| Flag | Meaning |
|---|---|
| `--follow` | Keep printing lines as they arrive, until Ctrl-C |
| `--details` | Read `install-details.log`, the full command output, instead |
| `--lines N` | Lines to show first, 1 to 10000 (default 40) |
| `--plain` | No colors or spinner |

## hairpin

`sudo sparkring hairpin [flags]` applies the ConnectX hairpin setting on every
Spark of a four-Spark ring and at every boot. It first updates any Spark that
runs a different SparkRing revision than Node A.
[More about the setting](install-reference.md#four-spark-rings).

| Flag | Meaning |
|---|---|
| `--plan` | Show what each Spark needs; change nothing |
| `--yes` | Approve the listed driver restarts or records on an idle ring |
| `--json` | Print one `sparkring-hairpin-result/v1` document |
| `--revoke` | Stop applying the setting at boot; values in effect stay until reboot |

## checkpoints

`sudo sparkring checkpoints [flags]` lists SparkRing's checkpoint directories on
every Spark, the deployments that use them and what a release frees. A
[derived checkpoint](install-reference.md#derived-checkpoints)'s directory
names the base it is derived from; a deployment of it uses both directories.

| Flag | Meaning |
|---|---|
| `--release PATH` | Remove that directory from every Spark; copies it links to are never touched |
| `--yes` | Approve the release without asking |
| `--json` | Print one JSON result |

## storage

`sudo sparkring storage [flags]` reports, for every Spark:

- the filesystems that hold `/srv/sparkring`, Docker's data root and `/`:
  size, free space and use;
- every item in `/srv/sparkring` and every Docker image, with its size and class;
- the releases it proposes;
- the deployments that [automatic release](#automatic-release) keeps and why,
  and what it releases after the next `install` or `up`.

Items are checkpoint directories, compile caches
(`/srv/sparkring/<cluster>/cache/<family>-<image>-<revision>`), deployment
workspaces (`/srv/sparkring/<cluster>/<profile>-i<identity>`), the remainder of
an interrupted release, and other entries. Sizes count hard-linked files once.
A size ending in `+` was still being measured when the Spark's 60-second limit
ran out.

| Class | Meaning |
|---|---|
| `installed` | Used by the active deployment, the rollback target, an unfinished model switch or a mesh installed on that Spark, such as the workspace holding the host marker of a four-Spark ring's mesh. The report names the mesh |
| `profile` | An installer profile of the installed package references it: a checkpoint the profile lists, derived ones included, the image the installer selects for it, or their compile cache. Kept for that profile's next installation |
| `unreferenced` | Neither. Proposed for release unless a running container uses it or it holds model files. Other retained deployments that use it are named; they need `sudo sparkring install` again after a release |
| `unmanaged` | Not created by SparkRing's installer, such as a directory you made in `/srv/sparkring`; never removed |

| Flag | Meaning |
|---|---|
| `--release PATH` | Remove this unreferenced cache directory or deployment workspace from every Spark that holds it |
| `--yes` | Approve the release without asking |
| `--json` | Print one JSON result |
| `--retain-deployments N` or `off` | Keep the N most recent deployments of each profile at each [automatic release](#automatic-release) (default 2), or turn it off |

A release asks first. Each Spark then checks again that no installed deployment,
no mesh installed on it and no running container uses the path. A mesh counts
while its configuration is in `/etc/sparkring/managed-mesh` or
`/etc/sparkring/deployments/<name>`, running or not
([Finding and freeing space](install-reference.md#finding-and-freeing-space)).
A release never removes:

- a checkpoint directory; `sudo sparkring checkpoints --release PATH` does;
- a workspace that holds model files;
- a Docker image; images are only listed, and `docker image rm ID` on that
  Spark removes one by hand. An image's size counts the layers it shares with
  other images, so removing it frees only the layers no other image uses.
  Each installer image adds a few megabytes to the one it was built on
  ([builders](../../runtime/images/installer-images.md));
- an `unmanaged` item.

`sparkring setup storage PROFILE` is a separate, local check of one profile's
space allowances before an installation.

### Automatic release

Every installation request with another package, checkpoint or serving setting
is a separate deployment, and each one that ran leaves its workspace, its
stopped model container and its compile caches on every Spark. After each
`install` and `up` that completes, SparkRing keeps:

- the active deployment, the rollback target and an unfinished model switch;
- a deployment whose model container runs, or whose workspace a mesh
  installed on a Spark uses;
- a deployment whose last operation did not complete, or that was started and
  not stopped, or prepared and not started;
- the 2 most recent deployments of each profile (`--retain-deployments`).

For the other deployments, which `down` stopped last, it removes on each
Spark their stopped model containers and workspaces, and compile caches that
no kept deployment and no installer profile of the installed package uses. It
prints one line, for example `Released 7 older deployments' containers,
workspaces and caches: 9.8 GiB`. Each Spark checks every container, workspace
and cache again as `--release` does. Checkpoint directories, images and the
deployments' records on Node A stay, so
`sudo sparkring up PROFILE --instance i<hash>` starts a released deployment
again once the running model is stopped: it copies its source and creates its
container again, and compiles kernels whose cache was removed.
`sudo sparkring down` of a released deployment says that it is stopped.

`sudo sparkring storage --retain-deployments off` turns it off;
`sudo sparkring storage` shows the setting. The preference
[`SPARKRING_RETAIN_DEPLOYMENTS`](install-reference.md#optional-preferences)
sets it too.

## up and down

`sudo sparkring up [PROFILE] [flags]` starts the named profile's deployment, or
the active one. `sudo sparkring down [PROFILE] [flags]` stops the named
deployment, or the active one, and `sparkring status [PROFILE]` reports it. Both
`up` and `down` print their steps and ask; `--execute` skips the question.
`sparkring install` is the usual way to start or switch a model.

A deployment keeps the SparkRing source that created it. `up`, `down` and
`status` plan and run each deployment with that source, so a deployment made by
an earlier package still starts and stops after the package changes. `up`
makes the deployment it starts the active one; stopping another deployment
leaves the active one unchanged. Without PROFILE, `up` and `down` act on the
deployment that was active when they printed their steps; when a
`sparkring install` makes another one active before the steps start, they stop
and ask for a new review. Repeating `up` re-checks a running deployment
and starts one whose containers stopped on every Spark, for example after a
restart ([when a model stops serving](install-reference.md#when-a-model-stops-serving)).
A completed `up` clears [automatic recovery's](#recover) failed attempts and
keeps its on or off choice, then runs [automatic release](#automatic-release);
after `down` the model stays stopped.
`sparkring install` names its deployments
with instances `i<hash>`: `sparkring down PROFILE --instance i<hash>` stops
one of them. The deployment directories are under
`/var/lib/sparkring/controller/deployments/`.

On a ring that serves [two models](install-reference.md#two-models-on-one-ring),
each half has its own active deployment: `sudo sparkring down --on 2,3
--execute` stops the model on Sparks 2 and 3. Without a profile or `--on`,
`up` and `down` act on the one recorded model and ask for `--on` when there
are several. `up` refuses to start a four-Spark model while a half's model
runs, or a half's model while the four-Spark model runs.

| Flag | Meaning |
|---|---|
| `--plan` | Print the steps; change nothing |
| `--execute` | Apply the printed steps without asking |
| `--json` | Print the result as JSON |
| `--model-path PATH` | `up PROFILE` only: serve this complete copy read-only on every Spark |
| `--instance NAME` | With PROFILE: a deployment beside the main one, for example a rehearsal |
| `--on 0,1` or `--on 2,3` | Without PROFILE: that half's model. With `up PROFILE`: a two-Spark profile on that half, as instance `on-0-1` or `on-2-3` unless `--instance` names another |
| `--fresh-mesh` | `up PROFILE` only: plan replacement of an existing four-Spark mesh |
| `--max-images N` and the other [serving settings](install-reference.md#serving-settings) | `up PROFILE` only: replace one of the profile's serving values for a new deployment; an existing deployment keeps its own |
| `--image NAME` | `up PROFILE` only: another installer image from [`sparkring images`](#images) |
| `--image-lock FILE` | `up PROFILE` only: another image lock, for a rehearsal. An existing deployment keeps the image it recorded, and naming another lock for it is refused |
| `--deployment DIR` | Use a deployment saved by `sparkring init` instead ([lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing)) |

## recover

`sudo sparkring recover [status|on|off]` shows or sets the automatic restart
of the active model when a Spark stops serving
([how it works](install-reference.md#automatic-recovery)). On a ring that
serves [two models](install-reference.md#two-models-on-one-ring), each
command covers both halves' models.

| Command | Does |
|---|---|
| `sudo sparkring recover` | Show whether it is on, the Spark it waits for, the last attempt and the next |
| `sudo sparkring recover off` | Stop restarting the active model by itself |
| `sudo sparkring recover on` | Restart it by itself again; clears failed attempts and the restart count |
| `--json` | Print one JSON document |

Each `sparkring install` sets the choice for the model it installs: on, or
off with `--no-auto-recover`, also after `recover off`. `sparkring up` keeps
the deployment's choice; a deployment without one starts with recovery on.
To stop the model and keep it stopped, use `sudo sparkring down --execute`.
`sparkring-recover.timer` runs `sparkring recover --auto` once a minute; you
do not run it by hand.

## node

`sparkring node ACTION` runs on one Spark. Node A and SparkRing's services call
most actions. Useful by hand:

| Command | Meaning |
|---|---|
| `sparkring node status [--refresh]` | This Spark's observation; `--refresh` needs sudo |
| `sparkring node hairpin status [--busy]` | Each ConnectX function's hairpin setting; `--busy` (needs sudo) adds what blocks a restart |
| `sudo sparkring node hairpin apply --dry-run --boot` | The restarts the next boot performs |
| `sudo sparkring node assets --profile PROFILE` | Where this Spark holds copies of the profile's checkpoint (read-only) |

## init and export

`sparkring init` saves a locked deployment in `.sparkring/deployment`:

- from SSH discovery (`--host`, repeated in rank order) or a site file
  (`--site`);
- for `--model glm53|mimo26|qwen38` or `--profile`;
- with optional `--variant`, `--image-lock`, `--name`, `--workspace`,
  `--output` and [serving settings](install-reference.md#serving-settings).

`sparkring export --output FILE` writes the deployment as a zip;
`--deployment DIR` selects another. Its Compose files carry the deployment's
serving settings.

- `--share` writes a portable template without private inputs.
- `--format compose` writes one standalone Compose file for the deployment's
  profile, or for the one that `--profile` and `--variant` name. It refuses a
  deployment with an image lock or serving settings, which the file would
  discard.

Both accept `--json`. See
[lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing).

## compose and validate-compose

- `sparkring compose render|check|start|stop` generates and coordinates
  profile-owned Compose deployments; see [Compose deployments](compose.md).
  `render` takes `--image NAME`, `--checkpoint NAME` and the serving-setting
  flags of `sparkring install`.
- `python scripts/generate_compose_builder.py --output DIR [--verify]` writes
  the [Compose builder](compose-builder.md) page for the checkout.
- `sparkring validate-compose FILE` (or `--all`, `--json`, `--output FILE`)
  checks Compose files without Docker or GPUs.

## deploy

`sparkring deploy discover|plan|network-plan|network-check|stage|runtime-plan|apply-plan`
is a standalone workflow, run from a Linux checkout, for one four-Spark GLM-5.3
profile. See [Prepare and operate a four-Spark deployment](deployment-suite.md).

## cluster, doctor and host

- `sparkring cluster init|show|configure` and `sparkring doctor` bootstrap and
  diagnose a ring by hand; see [Bootstrap a blank SparkRing cluster](bootstrap.md).
- `sparkring host check [--json] [--require-telemetry-disabled]` runs read-only
  checks on one Spark.
