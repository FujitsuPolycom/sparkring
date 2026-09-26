# SparkRing commands

The Debian package installs `sparkring` on Node A, the Spark connected to your
network. `sparkring COMMAND --help` prints the same flags. Setup and
installation: [Install SparkRing](install.md).

## Common examples

```bash
sudo sparkring install --profile qwen38-flash-next-tp2        # install on two Sparks
sudo sparkring install --profile qwen38-flash-next-qad-tp4    # install on four Sparks
sudo sparkring install --profile qwen38-flash-next-tp2 --plan # print the plan, change nothing
sudo sparkring install --profile qwen38-flash-next-tp2 --model-path /data/models/qwen  # reuse a copy
sudo sparkring install --profile qwen38-flash-next-tp2 --checkpoint qad-step-4000  # another listed checkpoint
sudo sparkring logs --follow                                  # follow progress
```

## All commands

| Command | Runs on | sudo | Does |
|---|---|---|---|
| [`install`](#install) | Node A | yes | Set up the Sparks and start one model profile |
| [`setup`](#setup) | Node A | yes | Set up the Sparks without installing a model |
| [`models`](#models) | any | no | List model profiles; marks those `install` supports |
| [`status`](#status) | Node A | yes | Show each Spark's state and the saved model |
| [`logs`](#logs) | Node A | yes | Show or follow the installation log |
| [`hairpin`](#hairpin) | Node A | yes | Apply the ConnectX setting that four-Spark rings need |
| [`checkpoints`](#checkpoints) | Node A | yes | List or release SparkRing's checkpoint directories |
| [`up`, `down`](#up-and-down) | Node A | yes | Start or stop a model deployment |
| [`node`](#node) | each Spark | yes, except the status reports | Per-Spark services; Node A calls most of them |
| [`init`, `export`](#init-and-export) | any | no | Save a deployment from hosts or a site file; export it as files |
| [`compose`](#compose-and-validate-compose) | a checkout | no | Render, check, start or stop profile Compose deployments |
| [`validate-compose`](#compose-and-validate-compose) | any | no | Check Compose files offline |
| [`deploy`](#deploy) | a checkout | no | Standalone discovery, network plans and staged model operations |
| [`cluster`, `doctor`, `host`](#cluster-doctor-and-host) | rank 0 or one Spark | no | Manual ring bootstrap and diagnosis |

`install`, `hairpin` and `checkpoints` print one JSON document with `--json`
and exit with 0 on success or plan, 3 when input is needed and 2 on failure.

## install

`sudo sparkring install [flags]` sets up the Sparks on first use, prepares the
image and checkpoint, then starts or switches the model.

| Flag | Meaning |
|---|---|
| `--profile PROFILE` | Exact profile from `sparkring models`; asked in a terminal when omitted |
| `--plan` | Print and save the setup, checkpoint and model plan; change nothing. On a Spark with no cluster yet, run `sudo sparkring setup --plan` instead |
| `--yes` | Approve setup, the checkpoint plan, ConnectX restarts on an idle ring and the model switch; unknown SSH host keys still need confirmation |
| `--json` | Print one JSON result on stdout; progress goes to stderr |
| `--checkpoint NAME` | Another checkpoint the profile lists, by Hugging Face branch; default: the profile's own |
| `--model-path [N=]PATH` | A checkpoint copy to reuse, for every Spark or for Node N; repeatable; never written |
| `--ignore-local-copies` | Use only SparkRing's own checkpoint directories and named copies |
| `--cache-path PATH` | Another writable compile cache on each Spark |
| `--env FILE` | Setup preferences file, read on first installation ([keys](install-reference.md#optional-preferences)) |
| `--stop-workloads` | Stop (never remove) GPU containers that are not SparkRing's |
| `--image-lock FILE` | Development image lock that replaces the shared installer image |
| `--allow-driver-reload` | Accepted and not needed; the approval covers ConnectX restarts |

## setup

`sudo sparkring setup [flags]` discovers the cabled Sparks and configures them,
without a model. `sparkring install` runs it on first use.

| Flag | Meaning |
|---|---|
| `--plan` | Discover and review over existing SSH access; configure nothing |
| `--yes` | Accept the listed changes; unknown SSH host keys still need confirmation |
| `--env FILE` | Literal preferences file; its keys set the defaults below |
| `--name NAME` | Cluster name: lowercase letters, digits and `-`, at most 35 characters (default `sparkring`) |
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

Two setup actions run offline, without sudo, and accept `--variant`:
`sparkring setup show PROFILE [--format text|json|shell]` prints the profile's
image and checkpoint, and `sparkring setup storage PROFILE --model-path P
--cache-path P --docker-path P [--reuse-model] [--reuse-image] [--json]`
checks free space on those filesystems without writing.

## models

`sparkring models [--json]` lists every exact model, version, quantization and
topology profile, and marks the ones `sparkring install` supports.

## status

`sudo sparkring status [flags]` prints Node A's state, one line per Spark with
the next action for a Spark that needs attention, and the saved model.

| Flag | Meaning |
|---|---|
| `--refresh` | Contact every Spark and inspect the model containers |
| `--json` | Print the full observation as JSON |

## logs

`sudo sparkring logs [flags]` prints the end of `/var/log/sparkring/install.log`.

| Flag | Meaning |
|---|---|
| `--follow` | Keep printing new lines until Ctrl-C |
| `--details` | Read `install-details.log`, the full command output, instead |
| `--lines N` | Lines to show first, 1 to 10000 (default 40) |
| `--plain` | No colors or spinner |

## hairpin

`sudo sparkring hairpin [flags]` applies the ConnectX hairpin setting on every
Spark of a four-Spark ring and at every boot. It first updates any Spark that
runs a different SparkRing revision than Node A. See
[Four-Spark rings](install-reference.md#four-spark-rings).

| Flag | Meaning |
|---|---|
| `--plan` | Show what each Spark needs; change nothing |
| `--yes` | Approve the listed driver restarts or records on an idle ring |
| `--json` | Print one `sparkring-hairpin-result/v1` document |
| `--revoke` | Stop applying the setting at boot; values in effect stay until reboot |

## checkpoints

`sudo sparkring checkpoints [flags]` lists SparkRing's checkpoint directories on
every Spark, which deployments use them and what a release frees.

| Flag | Meaning |
|---|---|
| `--release PATH` | Remove that directory from every Spark; copies it links to are never touched |
| `--yes` | Approve the release without asking |
| `--json` | Print one JSON result |

## up and down

`sudo sparkring up [PROFILE] [flags]` starts a deployment: the named profile's,
or the active one. `sudo sparkring down [flags]` stops the active deployment.
Both print their steps and ask; `--execute` skips the question.
`sparkring install` is the usual way to start or switch a model.

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
| `sparkring node hairpin status [--busy]` | Each ConnectX function's hairpin setting; `--busy`, which needs sudo, adds what blocks a restart |
| `sudo sparkring node hairpin apply --dry-run --boot` | The restarts the next boot performs |
| `sudo sparkring node assets --profile PROFILE` | Where this Spark holds copies of the profile's checkpoint (read-only) |

## init and export

`sparkring init` saves a locked deployment in `.sparkring/deployment` from SSH
discovery (`--host`, in rank order) or a site file (`--site`), with `--model
glm53|mimo26|qwen38` or `--profile`, and `--variant`, `--image-lock`, `--name`,
`--workspace`, `--output`. `sparkring export --output FILE` writes the
deployment as a zip (`--deployment DIR` selects another); `--share` writes a
portable template without private inputs, and `--format compose` writes one
standalone Compose file for the deployment's profile or the one `--profile`
and `--variant` name. Both accept `--json`. See
[lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing).

## compose and validate-compose

`sparkring compose render|check|start|stop` generates and coordinates
profile-owned Compose deployments; see [Compose](compose.md).
`sparkring validate-compose FILE` (or `--all`, `--json`, `--output FILE`)
validates Compose files without Docker or GPUs.

## deploy

`sparkring deploy discover|plan|network-plan|network-check|stage|runtime-plan|apply-plan`
is a standalone workflow for one four-Spark GLM-5.3 profile, run from a Linux
checkout; see
[Prepare and operate a four-Spark deployment](deployment-suite.md).

## cluster, doctor and host

`sparkring cluster init|show|configure` and `sparkring doctor` bootstrap and
diagnose a ring by hand; see [Bootstrap a blank SparkRing cluster](bootstrap.md).
`sparkring host check [--json] [--require-telemetry-disabled]` runs read-only
checks on one Spark.
