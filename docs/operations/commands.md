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
sudo sparkring logs --follow                                 # follow progress
```

## All commands

| Command | Runs on | sudo | Does |
|---|---|---|---|
| [`install`](#install) | Node A | yes | Set up the Sparks and start one model |
| [`setup`](#setup) | Node A | yes | Set up the Sparks without a model |
| [`models`](#models) | any | no | List profiles and mark those `install` supports |
| [`status`](#status) | Node A | yes | Show each Spark's state and the saved model |
| [`logs`](#logs) | Node A | yes | Show or follow the installation log |
| [`hairpin`](#hairpin) | Node A | yes | Apply the ConnectX setting that four-Spark rings need |
| [`checkpoints`](#checkpoints) | Node A | yes | List or release SparkRing's checkpoint directories |
| [`storage`](#storage) | Node A | yes | Report disk use; release caches and workspaces no deployment uses |
| [`up`, `down`](#up-and-down) | Node A | yes | Start or stop a model deployment |
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
image and checkpoint, then starts or switches the model.

| Flag | Meaning |
|---|---|
| `--profile PROFILE` | Exact profile from `sparkring models`; asked in a terminal when omitted |
| `--plan` | Print and save the setup, checkpoint and model plan; change nothing. Before the first setup, use `sudo sparkring setup --plan` |
| `--yes` | Approve setup, the checkpoint plan, ConnectX restarts on an idle ring and the model switch; unknown SSH host keys still need confirmation |
| `--json` | One JSON result on stdout; progress on stderr |
| `--checkpoint NAME` | Another checkpoint the profile lists, by Hugging Face branch ([names](install-reference.md#checkpoints)); default: the profile's own |
| `--model-path [N=]PATH` | A checkpoint copy to reuse, for every Spark or for Node N; repeatable; never written |
| `--ignore-local-copies` | Use only SparkRing's own checkpoint directories and named copies |
| `--cache-path PATH` | Another writable compile cache on each Spark |
| `--download-limit RATE` | Cap the checkpoint download from Hugging Face: `850Mbit`, `2Gbit` or `none` ([details](install-reference.md#limit-the-download-rate)) |
| `--events FILE` | Also write progress to FILE, one JSON object per line ([fields](install-reference.md#event-stream)) |
| `--env FILE` | Preferences file: setup keys on first installation, the download limit on every run ([keys](install-reference.md#optional-preferences)) |
| `--stop-workloads` | Stop (never remove) GPU containers that are not SparkRing's |
| `--image-lock FILE` | Development image lock that replaces the shared installer image |
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
without a model. `sparkring install` runs it on first use.

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

## models

`sparkring models [--json]` lists every profile (exact model, version,
quantization and topology) and marks those `sparkring install` supports.

## status

`sudo sparkring status [flags]` prints Node A's state, one line per Spark with
the next action for any Spark that needs attention, and the saved model:

```text
Saved model operation: PROFILE | up complete
Checkpoint: NAME (REPOSITORY @ REVISION) | Image: RELEASE
```

A profile with one checkpoint shows only `REPOSITORY @ REVISION`.

| Flag | Meaning |
|---|---|
| `--refresh` | Contact every Spark and inspect the model containers |
| `--json` | Print the full observation as JSON |

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
every Spark, the deployments that use them and what a release frees.

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
- the releases it proposes.

Items are checkpoint directories, compile caches
(`/srv/sparkring/<cluster>/cache/<family>-<image>-<revision>`), deployment
workspaces (`/srv/sparkring/<cluster>/<profile>-i<identity>`), the remainder of
an interrupted release, and other entries. Sizes count hard-linked files once.
A size ending in `+` was still being measured when the Spark's 60-second limit
ran out.

| Class | Meaning |
|---|---|
| `installed` | Used by the active deployment, the rollback target or an unfinished model switch |
| `profile` | An installer profile of the installed package references it: a checkpoint the profile lists, the image the installer selects for it, or their compile cache. Kept for that profile's next installation |
| `unreferenced` | Neither. Proposed for release unless a running container uses it or it holds model files. Other retained deployments that use it are named; they need `sudo sparkring install` again after a release |
| `unmanaged` | Not created by SparkRing's installer, such as a directory you made in `/srv/sparkring`; never removed |

| Flag | Meaning |
|---|---|
| `--release PATH` | Remove this unreferenced cache directory or deployment workspace from every Spark that holds it |
| `--yes` | Approve the release without asking |
| `--json` | Print one JSON result |

A release asks first. Each Spark then checks again that no installed deployment
or running container uses the path. A release never removes:

- a checkpoint directory; `sudo sparkring checkpoints --release PATH` does;
- a workspace that holds model files;
- a Docker image; images are only listed, and `docker image rm ID` on that
  Spark removes one by hand;
- an `unmanaged` item.

`sparkring setup storage PROFILE` is a separate, local check of one profile's
space allowances before an installation.

## up and down

`sudo sparkring up [PROFILE] [flags]` starts the named profile's deployment, or
the active one. `sudo sparkring down [flags]` stops the active deployment. Both
print their steps and ask; `--execute` skips the question. `sparkring install`
is the usual way to start or switch a model.

| Flag | Meaning |
|---|---|
| `--plan` | Print the steps; change nothing |
| `--execute` | Apply the printed steps without asking |
| `--json` | Print the result as JSON |
| `--model-path PATH` | `up PROFILE` only: serve this complete copy read-only on every Spark |
| `--instance NAME` | `up PROFILE` only: a separate deployment beside the main one, for a rehearsal |
| `--fresh-mesh` | `up PROFILE` only: plan replacement of an existing four-Spark mesh |
| `--image-lock FILE` | `up PROFILE` only: another image lock, for a rehearsal |
| `--deployment DIR` | Use a deployment saved by `sparkring init` instead ([lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing)) |

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
- with optional `--variant`, `--image-lock`, `--name`, `--workspace` and
  `--output`.

`sparkring export --output FILE` writes the deployment as a zip;
`--deployment DIR` selects another.

- `--share` writes a portable template without private inputs.
- `--format compose` writes one standalone Compose file for the deployment's
  profile, or for the one that `--profile` and `--variant` name.

Both accept `--json`. See
[lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing).

## compose and validate-compose

- `sparkring compose render|check|start|stop` generates and coordinates
  profile-owned Compose deployments; see [Compose deployments](compose.md).
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
