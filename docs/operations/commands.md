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
sudo sparkring install --profile PROFILE --on 4-7            # a model on positions 4 to 7 of a larger ring
sudo sparkring logs --follow                                 # follow progress
```

## All commands

| Command | Runs on | sudo | Does |
|---|---|---|---|
| [`install`](#install) | Node A | yes | Set up the Sparks and start one model |
| [`setup`](#setup) | Node A | yes | Set up the Sparks without a model |
| [`cabling`](#cabling) | Node A | yes | Show how the Sparks are cabled, the port map and what to move, or [measure each cable's speed](#cable-speed); changes nothing |
| [`fabric`](#fabric) | Node A | `verify` only | Show or verify the recorded fabric: ports, cables, relay table, boot units |
| [`models`](#models) | any | no | List profiles and mark those `install` supports |
| [`images`](#images) | any | no | List the installer images `install --image` can select |
| [`status`](#status) | Node A | yes | Show each Spark's state and the saved model |
| [`check`](#check) | Node A | yes | Send functional requests to the running model, check which transport carried its collectives, and write the tester report |
| [`logs`](#logs) | Node A | yes | Show or follow the installation log |
| [`hairpin`](#hairpin) | Node A | yes | Apply the ConnectX setting that relayed forwarding needs |
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
| `--on ARC` | Run the profile on these consecutive Sparks: `0,1`, `0-3`, `4-7`, or `6-1` across the cable to Node A ([models on part of the fabric](install-reference.md#models-on-part-of-the-fabric)); default: every Spark, or for a profile of fewer Sparks the one group that divides the fabric from Node A and serves no model. The published images' prepared transport runs only a four-Spark ring's halves, `0,1` and `2,3` |
| `--plan` | Print and save the setup, checkpoint and model plan; change nothing. Before the first setup, use `sudo sparkring setup --plan` |
| `--yes` | Approve setup, the checkpoint plan, ConnectX restarts on an idle ring and the model switch; unknown SSH host keys still need confirmation, and stopping another program's GPU containers still asks unless you add `--stop-workloads` |
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
| `--transport sircl` or `--transport prepared` | The collective transport ([transport and receipts](install-reference.md#transport-and-receipts)); default: `sircl` where the image and the recorded fabric carry it, else `prepared` |
| `--nccl never` or `--nccl auto` | NCCL on a SIRCL deployment: `never` (default) keeps it off; `auto` lets it carry what the cabling allows; `topology` is another name for `auto` |
| `--max-images N`, `--max-videos N`, `--context-length N`, `--max-concurrency N`, `--kv-cache-gib N`, `--save-cpu` | Replace one of the profile's serving values for this deployment ([serving settings](install-reference.md#serving-settings)) |
| `--reasoning-effort LEVEL`, `--thinking off` | How hard the model thinks, or that it doesn't, when a request doesn't say; requests can still choose ([thinking](install-reference.md#thinking)) |
| `--api-port N` | The port of the model's API, 1024 to 65535; a serving setting ([API endpoint](install-reference.md#api-endpoint)); default: the profile's |
| `--api-bind ADDRESS` | Let the API listen only on this IPv4 address of the Spark that serves it; a serving setting; default: every address |
| `--allow-loopback-bind` | Accept a loopback `--api-bind`, such as 127.0.0.1, which only programs on Node A can reach |
| `--api-address ADDRESS` | The name or address shown for the model, such as `llm.example.net`; shown only, not part of the deployment |
| `--allow-driver-reload` | Accepted and not needed; the approval covers ConnectX restarts |

In a terminal, without `--yes` and without the API options, `install` asks
which address the model's API listens on and which port it uses; Enter keeps
both ([API endpoint](install-reference.md#api-endpoint)).

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
| `--yes` | Answer the package question and pass `--yes` to `sparkring install`, with the same limits; required without a terminal |
| `--plan` | Install nothing; plan with the built version (below) |
| `--package-only` | Ask the package question, install or keep the package and stop before `sparkring install`; not with `--plan` |
| `--json` | One `sparkring-install-result/v1` document on stdout (below); progress, questions and `apt` output on stderr |
| `--ref BRANCH_TAG_OR_COMMIT`, `--repository URL` | Build another ref or repository ([Pin a commit](install-reference.md#pin-a-commit)) |
| `--relay-marker-binary PATH` | Ship this copy of the published relay marker instead of downloading it ([Build from a full clone](install-reference.md#build-from-a-full-clone)) |

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

`sudo sparkring setup [flags]` discovers the cabled Sparks (a pair, or a line
or ring of up to eight) and configures them, without a model. `sparkring
install` runs it on first use. Sparks that belonged to other SparkRing
clusters are
[re-formed](install-reference.md#re-form-sparks-into-another-pair-or-ring)
into the layout now cabled. Its plan shows the layout, the port map, the
[relay table](install-reference.md#the-relay-table) and the transports the
fabric can carry. Setup measures each cable's [speed](#cable-speed) when no
model serves (a degraded cable is a warning with its repair steps, not a
failure), then verifies the fabric and records the
[fabric document](install-reference.md#the-fabric-document) on every Spark.

| Flag | Meaning |
|---|---|
| `--plan` | Discover and review over existing SSH access; configure nothing |
| `--yes` | Accept the listed changes; unknown SSH host keys still need confirmation, and stopping GPU containers still asks unless you add `--stop-workloads` |
| `--env FILE` | Literal preferences file; its keys set the defaults below ([keys](install-reference.md#optional-preferences)) |
| `--name NAME` | Cluster name: a lowercase letter, then lowercase letters, digits or `-`; at most 35 characters (default `sparkring`) |
| `--ssh-user USER` | Worker account (default: the account that ran `sudo`; `root` with `--env` or port 2222) |
| `--ssh-port 22\|2222` | Worker SSH port; 2222 reaches workers prepared with `--worker-bundle` |
| `--control-cidr CIDR` | Administration network, an IPv4 `/29` or `/28` (default `10.253.255.0/29`; the `/28` containing it for seven or eight Sparks) |
| `--fabric-cidr CIDR` | Fabric addresses, an IPv4 `/16` to `/21` with room for two `/24` per cable (default `198.18.0.0/21` up to four cables, `198.18.0.0/20` above) |
| `--re-form` | Set the cabled Sparks up again as a new cluster, as after recabling them into another layout ([change the layout](install-reference.md#change-the-layout)) |
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
cables are cabled: the layout (`pair`, `path-N` or `cycle-N`, up to eight
Sparks), the port-to-Spark map and free ports, or what to move. It changes
nothing on any Spark. Setup stops with the same advice when the cables do not
fit ([cabling rules](install-reference.md#cabling)).

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

A layout cabled as SparkRing needs shows its port map:

```text
Cables:
  spark-a port 0 ↔ spark-b port 1
  spark-b port 0 ↔ spark-c port 1
Three-Spark line (path-3) from Node A, cabled as SparkRing needs.
Layout: path-3
Ports:
  position 0 spark-a: port 0 → position 1 spark-b port 1 (cable 0); port 1 free
  position 1 spark-b: port 0 → position 2 spark-c port 1 (cable 1); port 1 → position 0 spark-a port 0 (cable 0)
  position 2 spark-c: port 0 free; port 1 → position 1 spark-b port 0 (cable 1)
Line order: spark-a → spark-b → spark-c
Note: spark-c port 0 and spark-a port 1 are free; a cable from the first to the second makes a cycle-3.
```

It reads this Spark, the Sparks of its recorded cluster over the admin
network, and the other Sparks on the cables. It signs in to those over the
LAN or the cables as the account that ran `sudo`, with Node A's setup key
where they accept it; SSH asks for a password elsewhere.

| Flag | Meaning |
|---|---|
| `--json` | One `sparkring-cabling/v2` document (the `v1` keys, plus `shape`, `layout_size`, `layout_name`, `positions`, `free` and each cable's index); with `--bandwidth`, one `sparkring-fabric-bandwidth/v1` document |
| `--ssh-user USER` | Account for signing in to the other Sparks (default: the account that ran `sudo`) |
| `--no-sign-in` | Read only this Spark and its recorded cluster's Sparks |
| `--bandwidth` | Measure each cable's speed instead ([cable speed](#cable-speed)) |
| `--while-serving` | With `--bandwidth`, also measure cables a serving model uses |

It exits with 0 when the cables form a pair, line or ring as SparkRing needs,
1 when a cable needs to move or not every cable could be seen, and 2 on
failure. It reads up to nine Sparks, up to seven cables away, so a Spark
cabled beyond the eight that setup supports is named.

### Cable speed

`sudo sparkring cabling --bandwidth` measures each cable of the fabric that
setup recorded and saves the result for `sparkring status`. Setup runs it
before it records the fabric document. It takes about 30 seconds per cable.

```text
Fabric bandwidth, both directions at once (190 Gb/s or more per link is healthy):
spark-a port 0 ↔ spark-b port 1: healthy
  enp1s0f0np0 ↔ enp1s0f1np1      213.05 Gb/s  healthy
  enP2p1s0f0np0 ↔ enP2p1s0f1np1  212.87 Gb/s  healthy
spark-b port 0 ↔ spark-c port 1: degraded
  enp1s0f0np0 ↔ enp1s0f1np1      118.66 Gb/s  degraded
  enP2p1s0f0np0 ↔ enP2p1s0f1np1  119.02 Gb/s  degraded
  To repair: reboot both spark-b and spark-c, then run sudo sparkring cabling --bandwidth again.
  Restarting the link or the network driver does not clear this.
  If the cable is still degraded after the reboot, reseat it at both ends.
```

Each cable carries two links, and each is measured on its own. A degraded
cable shows no errors and models still run, but prompt processing over it is
slower. Reboot both Sparks on that cable to clear it.

The test slows a model that uses the cable, so cables of a serving model's
Sparks are listed as not measured; `--while-serving` measures them anyway.
A link the test cannot measure shows the reason, for example a RoCE GID
entry that does not hold the link's address.

It exits with 0 when every cable is healthy, 1 when a cable is degraded,
could not be measured or was skipped, and 2 when the check could not run,
for example while a model serves on every cable.
[Cable speed](install-reference.md#cable-speed) explains the test.

## fabric

`sparkring fabric show [--json]` prints the recorded
[fabric document](install-reference.md#the-fabric-document): the layout, each
Spark's ports and cables, cable health, the hairpin requirement, the relay
table, the transports and the last verification. It reads only Node A.

`sudo sparkring fabric verify [flags]` checks every Spark's links, addresses,
routes, hairpin setting, relay table, markers and boot units, and the
reachability of every other Spark's addresses
([what it checks](install-reference.md#verify-the-fabric)). It changes
nothing.

```text
Fabric verified: 8 cables on 8 Sparks (cycle-8), relays 96 rules and 96 routes, reboot-persistent.
Report: /var/lib/sparkring/controller/fabric-reports/fabric-verify-20261008T100500Z.json
```

| Flag | Meaning |
|---|---|
| `--traffic none\|light` | `light` adds one bidirectional RDMA write test per relayed lane (about 10 seconds each); default `none` |
| `--while-serving` | With `--traffic light`, test while a model serves |
| `--json` | Print the `sparkring-fabric-verify/v1` report |

It exits with 0 when every check passes, 1 when one fails, and 2 when it
could not run.

## models

`sparkring models [--json]` lists every profile (exact model, version,
quantization and topology) and marks those `sparkring install` supports. For
each installer profile it also shows what the model does with thinking when a
request doesn't say, such as `on · xhigh`, and the effort levels it accepts
([thinking](install-reference.md#thinking)).

## images

`sparkring images [--profile PROFILE] [--json]` lists the installer images
this package records, the default first, with the GitHub release that
published each, its download size, the transports it carries (`prepared`,
`sircl`) and the profiles it runs; an archived image is marked `archived`.
`--profile` lists only the images that run that profile. Any listed name, release tag or
part of a name that only one image has selects that image in
`sudo sparkring install --image NAME`.

## status

`sudo sparkring status [PROFILE [--instance NAME]] [flags]` prints Node A's
state, one line per Spark with the next action for any Spark that needs
attention, the saved model (the active deployment, or the one named) and
automatic recovery. On a fabric that serves
[several models](install-reference.md#models-on-part-of-the-fabric) it prints
each group's model under its Sparks, such as `Sparks 0-3:` and `Sparks 4-7:`,
with a `Group:` line naming its shape, positions and API Spark:

```text
Saved model operation: PROFILE | up complete
Checkpoint: NAME (REPOSITORY @ REVISION) | Image: RELEASE
Thinking: on · xhigh (model default)
Transport: sircl, NCCL: absent (checked 2026-10-08T10:00:00Z); sudo sparkring check repeats it
Automatic recovery: on
```

A profile with one checkpoint shows only `REPOSITORY @ REVISION`; a
[derived checkpoint](install-reference.md#derived-checkpoints) adds
`, derived from REPOSITORY @ REVISION` of its base. `Thinking` is what the
model does when a request doesn't say whether, or how hard, to think: the
model's default, or the deployment's own default beside it
([thinking](install-reference.md#thinking)). The model's API URL
follows, at the address `install --api-address` named when there is one,
with the URL SparkRing's own checks use in parentheses. The
recovery lines add the Spark it waits for, its last attempt and the next.
After `sparkring down` they read `on; idle until the next sudo sparkring up
--execute or sudo sparkring install`, and a deployment that recovery does not
restart, such as managed GLM, reads `not available`.

After the Spark lines, a `Fabric bandwidth:` line shows the last
[cable speed](#cable-speed) result and its age, such as `healthy on all 4
cables, measured 3 h ago`, or `never measured`. A degraded cable follows with
its repair steps. Status does not measure.

`Transport` names the deployment's transport and, on SIRCL ring sessions,
the last receipt verdict ([transport and receipts](install-reference.md#transport-and-receipts)):
`Transport check failed: ...` names the first rank and collective that
differs.

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
| `--on ARC` | Only the model on those Sparks |
| `--json` | Print the full observation as JSON; a fabric with models on part of it adds `slots`, one entry per group with its `placement` and `group`, and a recorded cluster adds `fabric_bandwidth` |

## check

`sudo sparkring check [flags]` checks each running model and changes no Spark.
It sends the functional requests of the acceptance harness to the model's API
(counting, arithmetic and code, and a tool call, an image and thinking where
the profile serves them) and, for a model on SIRCL ring sessions, reads every
rank's receipts and the NCCL lines of its log and judges them as the
installation does ([transport and receipts](install-reference.md#transport-and-receipts)).
It exits with 0 when every check passes, 1 when one fails and 2 when it cannot
run.

| Flag | Meaning |
|---|---|
| `--on ARC` | Only the model on those Sparks |
| `--json` | Print `sparkring-check/v1` |
| `--report DIR` | Also write `DIR/sparkring-report-<time>/` (`sparkring-test-report/v1`): the last installation's result, `fabric show` and the last `fabric verify` report, the status, this check, the receipts with the tuning table, the last 200 lines of the installation's details log and of each rank's model log, and Node A's package, image, DGX OS, driver, Docker, container toolkit and ConnectX firmware |

The report replaces management and LAN addresses, host names, MAC addresses
and account names with placeholders and keeps fabric addresses. A file that
still names a private item is left out and listed in its `report.json`.
Review the files, then attach the directory to a
[Test report](https://github.com/FujitsuPolycom/sparkring/issues/new?template=test_report.yml) issue.

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
Spark that relays between its cables (every Spark of a ring of four or more,
every Spark but the ends of a line of three or more) and at every boot. It
first updates any Spark that runs a different SparkRing revision than Node A.
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

On a fabric that serves [several models](install-reference.md#models-on-part-of-the-fabric),
each group of Sparks has its own active deployment: `sudo sparkring down --on
4-7 --execute` stops the model on Sparks 4 to 7. Without a profile or `--on`,
`up` and `down` act on the one recorded model and ask for `--on` when there
are several. `up` refuses to start a model while a model on one of its Sparks
runs, and names the `down` command that frees them.

| Flag | Meaning |
|---|---|
| `--plan` | Print the steps; change nothing |
| `--execute` | Apply the printed steps without asking |
| `--json` | Print the result as JSON |
| `--model-path PATH` | `up PROFILE` only: serve this complete copy read-only on every Spark |
| `--instance NAME` | With PROFILE: a deployment beside the main one, for example a rehearsal |
| `--on ARC` | Without PROFILE: the model on those Sparks. With `up PROFILE`: the profile on those Sparks, as instance `on-` and the positions (`on-2-3`, `on-4-5-6-7`) unless `--instance` names another |
| `--fresh-mesh` | `up PROFILE` only: plan replacement of an existing four-Spark mesh |
| `--max-images N`, `--reasoning-effort LEVEL` and the other [serving settings](install-reference.md#serving-settings) | `up PROFILE` only: replace one of the profile's serving values, or the model's thinking default, for a new deployment; an existing deployment keeps its own |
| `--allow-loopback-bind` | `up PROFILE` only: accept a loopback `--api-bind` for a new deployment |
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
| `sudo sparkring node relay-markers` | Run this Spark's relay markers until stopped; `sparkring-relay-marker.service` runs it |
| `sparkring node relay-marker-check` | Check the installed package's relay marker against the SHA-256 the package records; package installation runs it |

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
  flags of `sparkring install`, `--api-port` and `--api-bind` among them.
- `python scripts/generate_compose_builder.py --output DIR [--verify]` writes
  the [Install Builder](compose-builder.md) page for the checkout.
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
