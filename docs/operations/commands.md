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
sudo sparkring status                                        # each Spark and the model
sudo sparkring down --execute                                # stop the model
sudo sparkring up --execute                                  # start it again
```

## All commands

| Command | Runs on | sudo | Does |
|---|---|---|---|
| [`install`](#install) | Node A | yes | Set up the Sparks and start one model |
| [`name`](#name) | Node A | to set or clear | Read or change the ring display name |
| [`setup`](#setup) | Node A | yes | Set up the Sparks without a model |
| [`cabling`](#cabling) | Node A | yes | Show the cabling and what to move, or [measure cable speed](#cable-speed) |
| [`fabric`](#fabric) | Node A | `verify`, `tune` and `spread-check` | Show, verify or tune the recorded fabric |
| [`models`](#models) | any | no | List profiles and mark those `install` supports |
| [`images`](#images) | any | no | List the installer images `install --image` can select |
| [`status`](#status) | Node A | yes, or the `sparkring` group | Show each Spark's state and the saved model |
| [`check`](#check) | Node A | yes | Test the running model and its transport; write a tester report |
| [`dashboard`](#dashboard) | Node A | `export`, `quiet`, `enable`, `disable`, `external` | Web page and terminal view of the ring (Experimental) |
| [`host-load`](#host-load) | Node A or each Spark | yes, except `show` | Set where and how often host jobs run |
| [`logs`](#logs) | Node A | yes, or the `sparkring` group | Show or follow the installation log |
| [`hairpin`](#hairpin) | Node A | yes | Apply the ConnectX setting that relayed forwarding needs |
| [`checkpoints`](#checkpoints) | Node A | yes | List or release SparkRing's checkpoint directories |
| [`storage`](#storage) | Node A | yes | Report disk use; release unused caches |
| [`up`, `down`, `restart`](#up-down-and-restart) | Node A | yes | Start, stop or restart a model deployment |
| [`recover`](#recover) | Node A | yes, or the `sparkring` group to read | Show or switch the automatic restart |
| [`hold`](#hold) | Node A | yes, or the `sparkring` group to read | Reserve the cluster for one operator's longer task |
| [`node`](#node) | each Spark | yes, except status reports | Per-Spark services; Node A calls most of them |
| [`init`, `export`](#init-and-export) | any | no | Save a deployment from hosts or a site file; export it as files |
| [`compose`](#compose-and-validate-compose) | a checkout | no | Render, check, start or stop profile Compose deployments |
| [`validate-compose`](#compose-and-validate-compose) | any | no | Check Compose files offline |
| [`deploy`](#deploy) | a checkout | no | Standalone discovery, network plans and staged model operations |
| [`cluster`, `doctor`, `host`](#cluster-doctor-and-host) | rank 0 or one Spark | no | Manual ring bootstrap and diagnosis |

With `--json`, `install`, `hairpin`, `checkpoints` and `storage` print one JSON
document. They exit with 0 on success or plan, 3 when input is needed and 2 on
failure, and also on a run that `install --cancel` stopped (state
`cancelled`).

### Without sudo

Members of the `sparkring` group (`sudo usermod -aG sparkring USER`, then sign
in again) read `status`, `recover`, `logs` and `name` without sudo. `status` and
`recover` then print the state that the last sudo command recorded, with site
addresses removed; `status --refresh`, `recover on|off` and changing the name
still need sudo.

An optional root helper runs `up --execute`, `down --execute` (with or without
`PROFILE`) and `recover --auto` for group members. Turn it on with
`sudo systemctl enable --now sparkring-helper.socket` and set
`SPARKRING_HELPER_SOCKET=/run/sparkring/helper.sock`; it accepts no other
command. Status: implemented; untested on a Spark
([least privilege](../development/least-privilege.md#4-one-privileged-path-for-mutations)).

### Command highlighting

Commands that a message suggests, such as `sudo sparkring fabric verify` in a
`status` line, print in bold cyan on a terminal. Pipes, saved logs and `--json`
output stay plain, as do `NO_COLOR` (any value) and `TERM=dumb`. `install`,
`setup`, `up`, `down`, `restart`, `status`, `recover`, `check`, `models`,
`fabric`, `logs` and `export` accept `--color=auto|always|never`, which
overrides the `SPARKRING_COLOR` variable; `always` keeps the colour on a
non-terminal, for `watch -c sudo sparkring status --color=always`.

## install

`sudo sparkring install [flags]` sets up the Sparks on first use, prepares the
image and checkpoint, then starts or switches the model. Once the model
serves, it releases what older deployments hold on the Sparks
([automatic release](#automatic-release)).

| Flag | Meaning |
|---|---|
| `--profile PROFILE` | Exact profile from `sparkring models`; asked in a terminal when omitted |
| `--on ARC` | Run the profile on these consecutive Sparks: `0,1`, `0-3`, `4-7`, or `6-1` across the cable to Node A ([models on part of the fabric](install-reference.md#models-on-part-of-the-fabric)). Default: every Spark, or for a profile of fewer Sparks the one group that divides the fabric from Node A and serves no model. The `prepared` transport runs two-Spark models only on a four-Spark ring's halves, `0,1` and `2,3` |
| `--plan` | Print and save the setup, checkpoint and model plan; change nothing. A later `--yes` run stays within the saved plan. Before the first setup, use `sudo sparkring setup --plan` |
| `--cancel` | [Stop the running installation](#cancel-an-installation) at its next step boundary, or settle a model switch an interrupted installation left; takes no other installation option |
| `--yes` | Approve setup, the checkpoint plan, ConnectX restarts on an idle ring and the model switch. Unknown SSH host keys still need confirmation, and stopping another program's GPU containers still asks unless you add `--stop-workloads` |
| `--json` | One JSON result on stdout; progress on stderr |
| `--override-hold` | Proceed although another operator [holds the cluster](#hold); the override is recorded |
| `--display-name TEXT` | Set the [ring display name](#name) |
| `--checkpoint NAME` | Another checkpoint the profile lists ([names](install-reference.md#another-checkpoint-of-a-profile)). Default: the profile's preferred checkpoint on an image whose vLLM reads it, else the profile's own |
| `--model-path [N=]PATH` | A checkpoint copy to reuse, for every Spark or for Node N; repeatable; never written |
| `--ignore-local-copies` | Use only SparkRing's own checkpoint directories and named copies |
| `--cache-path PATH` | Another writable compile cache on each Spark |
| `--download-limit RATE` | Cap the checkpoint download from Hugging Face: `850Mbit`, `2Gbit` or `none` ([details](install-reference.md#limit-the-download-rate)) |
| `--events FILE` | Also write progress to FILE, one JSON object per line ([fields](install-reference.md#event-stream)) |
| `--env FILE` | Preferences file: setup keys on first installation, the download limit and the [retained deployments](#automatic-release) on every run ([keys](install-reference.md#optional-preferences)) |
| `--stop-workloads` | Stop (never remove) GPU containers that are not SparkRing's |
| `--no-auto-recover` | Do not restart this model by itself when a Spark stops serving ([automatic recovery](install-reference.md#automatic-recovery)) |
| `--image NAME` | Another installer image: a name or release tag from [`sparkring images`](#images) ([details](install-reference.md#another-image)). Default: the installer's own image |
| `--image-lock FILE` | Development image lock that replaces the installer image; not with `--image` |
| `--transport sircl\|prepared\|nccl\|libsircl` | The collective transport ([transport and receipts](install-reference.md#transport-and-receipts)). Default: `sircl` where the image and the recorded fabric carry it, else `prepared`. `nccl` (vLLM's PyNccl alone, on a cabled pair or a whole cycle) and `libsircl` (research-only) only by name |
| `--nccl never\|auto` | NCCL on a SIRCL deployment: `never` (default) keeps it off; `auto` lets it carry what the cabling allows; `topology` is another name for `auto` |
| `--image REF --transport libsircl --libsircl-library PATH --model-path DIR --plan -- VLLM_ARGUMENTS` | Plan a stock vLLM image (a registry reference or image ID) with libsircl as its NCCL: checks every Spark and writes each rank's Compose file. Research-only, plan only ([stock image](../architecture/libsircl.md#stock-image-option)) |
| `--max-images N`, `--max-videos N`, `--context-length N`, `--max-concurrency N`, `--kv-cache-gib N`, `--save-cpu` | Replace one of the profile's serving values for this deployment ([serving settings](install-reference.md#serving-settings)) |
| `--sparkcache-size SIZE` | Per-Spark SparkCache disk budget (default 32G) |
| `--sparkcache-integrity fast\|sha256\|none` | SparkCache payload check (default fast) |
| `--sparkcache-debug` | Detailed SparkCache logs for this deployment |
| `--reasoning-effort LEVEL`, `--thinking off` | How hard the model thinks, or that it doesn't, when a request doesn't say; requests can still choose ([thinking](install-reference.md#thinking)) |
| `--api-port N` | The port of the model's API, 1024 to 65535; a serving setting ([API endpoint](install-reference.md#api-endpoint)). Default: the profile's |
| `--api-bind ADDRESS` | Let the API listen only on this IPv4 address of the Spark that serves it; a serving setting. Default: every address |
| `--allow-loopback-bind` | Accept a loopback `--api-bind`, such as 127.0.0.1, which only programs on Node A can reach |
| `--api-address ADDRESS` | The name or address shown for the model, such as `llm.example.net`; shown only, not part of the deployment |
| `--allow-driver-reload` | Accepted and not needed; the approval covers ConnectX restarts |

In a terminal, without `--yes` and without the API options, `install` asks
which address the model's API listens on and which port it uses; Enter keeps
both ([API endpoint](install-reference.md#api-endpoint)).

### Cancel an installation

`sudo sparkring install --cancel` stops the installation that is running now;
killing the process leaves the candidate's containers and a switch record
behind instead. It records the request in
`/var/lib/sparkring/controller/install-cancel.json` and waits up to two
minutes. The installation reads the request before each step of its model
switch, stops there, stops the candidate deployment through that deployment's
own stop, starts the model the switch replaced again and records the switch as
`cancelled` with the time and the operator. The active pointer never leaves
the model that served. The interrupted run prints the reason and exits 2 with
state `cancelled`, or `failed` when the restore did not complete. `--cancel`
itself exits 0 when it asks, stops or settles a switch, and 2 when another
command holds the cluster or a switch it tried to settle still needs attention.

```bash
sudo sparkring install --cancel
```

- A step that cannot be abandoned, such as a checkpoint download, runs to its
  end; the installation stops at the next boundary. The request outlives the
  command, and `sparkring status` reports the switch until it settles.
- With no installation running, the same command settles a model switch an
  earlier installation left unfinished, for example after `SIGTERM`, `Ctrl-C`
  or a Spark restart. It stops that switch's candidate on every Spark and
  starts the model the switch replaced.
- `--cancel` takes no other installation option and no installation lock: the
  operation it ends already holds that lock. A hold does not refuse it. While
  another command changes the cluster (`up`, `down` or `restart`), it names
  that command and changes nothing: repeat it when the operation finishes.

## install.sh

[`install.sh`](../../install.sh) builds the SparkRing package from a full Git
clone, installs it with `apt` and runs `sudo sparkring install` with every
other option you pass it. Run it on Node A as a user with sudo; the one-line
command is in [Get the package](install-reference.md#get-the-package). The
copy in this repository builds `main`; the copy the Install Builder site
publishes builds the release that site describes.

- It builds in a temporary directory under `/var/tmp` and removes it before
  `sparkring install` starts.
- When piped, it reads its questions and the installer's from the terminal.
- It skips `apt` when the built version is already installed.
- Before replacing an installed package, it checks that the installed source
  revision is an ancestor of the requested one. A divergent or unidentified
  source needs terminal approval, even with `--yes`.

| Flag | Meaning |
|---|---|
| `--yes` | Approve package installation and pass `--yes` to `sparkring install`; replacing a divergent or unidentified source still needs terminal approval |
| `--keep-package` | Skip fetching, building and installing; run the installed package, also with `--plan`. Needs an installed package; not with `--package-only` |
| `--plan` | Install nothing; plan with the built version (below) |
| `--package-only` | Install or keep the package and stop before `sparkring install`; not with `--plan` |
| `--json` | One `sparkring-install-result/v1` document on stdout (below); progress, questions and `apt` output on stderr |
| `--ref BRANCH_TAG_OR_COMMIT`, `--repository URL` | Build another ref or repository ([Pin a commit](install-reference.md#pin-a-commit)) |
| `--relay-marker-binary PATH` | Ship this copy of the published relay marker instead of downloading it ([Build from a full clone](install-reference.md#build-from-a-full-clone)) |

`--plan` runs `sparkring install --plan` from the installed package when this
Spark has the built version, otherwise from the built package extracted into
the temporary directory. With no SparkRing package on this Spark it stops
with `needs_input` (field `package`). The plan inspects every Spark through
its installed package and compares their revisions with the built one. Like
any `sparkring install --plan`, it is saved on Node A and bounds a later
`--yes` run with the same source revision and options.

After `--package-only`, `sudo sparkring install --profile PROFILE --plan`
reviews the rest. With `--json`, `--package-only` reports `state`
`package-installed`, `version`, the `previous` version, whether it `changed`
and the `source_revision`.

The `--json` document is the installer's result, or the script's own when it
stops first or a plan from the extracted package ends without a result:
`state` `failed` with the failed `stage` (exit status 2), or `needs_input`
with the needed `field` (exit status 3).

## name

```bash
sudo sparkring name "Draupnir"   # set the display name
sparkring name                   # print the effective name
sudo sparkring name --clear      # return to the cluster name
sudo sparkring setup --display-name "Draupnir"
bash install.sh --display-name "Draupnir" --profile PROFILE
```

The display name is a label only: 1 to 40 printable Unicode characters after
trimming surrounding whitespace. Without one, the cluster name (`setup
--name`) is shown. Setting or clearing it changes no container, fabric or
deployment identity and needs no restart. Node A stores it in
`/var/lib/sparkring/controller/display-name.json` (mode 0600). Setup and
install plans show the effective name without saving it; omitting
`--display-name` keeps an existing one.

The name appears in the dashboard header, ring panel and page title, the
terminal view's header, `status`, and the Model ready card's
`Ring dashboard` line.

## setup

`sudo sparkring setup [flags]` discovers the cabled Sparks (a pair, or a line
or ring of up to eight) and configures them, without a model. `sparkring
install` runs it on first use. Sparks that belonged to other SparkRing
clusters are
[re-formed](install-reference.md#re-form-sparks-into-another-pair-or-ring)
into the cabled layout. The plan shows the layout, the port map, the
[relay table](install-reference.md#the-relay-table) and the transports the
fabric can carry. When no model serves, setup measures each cable's
[speed](#cable-speed); a degraded cable is a warning with its repair steps,
not a failure. It then verifies the fabric and records the
[fabric document](install-reference.md#the-fabric-document) on every Spark.

| Flag | Meaning |
|---|---|
| `--plan` | Discover and review over existing SSH access; configure nothing |
| `--yes` | Accept the listed changes; unknown SSH host keys still need confirmation, and stopping GPU containers still asks unless you add `--stop-workloads` |
| `--override-hold` | Proceed although another operator [holds the cluster](#hold); the override is recorded |
| `--env FILE` | Literal preferences file; its keys set the defaults below ([keys](install-reference.md#optional-preferences)) |
| `--name NAME` | Cluster name: a lowercase letter, then lowercase letters, digits or `-`; at most 35 characters (default `sparkring`) |
| `--display-name TEXT` | The [ring display name](#name) |
| `--ssh-user USER` | Worker account (default: the account that ran `sudo`; `root` with `--env` or port 2222) |
| `--ssh-port 22\|2222` | Worker SSH port; 2222 reaches workers prepared with `--worker-bundle` |
| `--control-cidr CIDR` | Administration network, an IPv4 `/29` or `/28` (default `10.253.255.0/29`; the `/28` containing it for seven or eight Sparks) |
| `--fabric-cidr CIDR` | Fabric addresses, an IPv4 `/16` to `/21` with room for two `/24` per cable (default `198.18.0.0/21` up to four cables, `198.18.0.0/20` above) |
| `--re-form` | Set the cabled Sparks up again as a new cluster, as after recabling them into another layout; what they keep from the earlier setup is moved aside with a receipt ([change the layout](install-reference.md#change-the-layout)) |
| `--no-share-internet` | Do not route workers' downloads and DNS through Node A |
| `--reset-links` | Replace incompatible fabric IPv4 settings, also without a terminal |
| `--stop-workloads` | Stop (never remove) GPU containers that block fabric preparation |
| `--worker-bundle` | Build a USB bundle for [workers without SSH](install-reference.md#if-a-worker-has-no-ssh) |
| `--admin-fallback` | On a set-up cluster, add [fallback paths](install-reference.md#admin-tunnel) (the other cables, the LAN) to the admin tunnel and change nothing else; with `--plan`, list them |
| `--allow-driver-reload` | Accepted and not needed |

With `--node` or `--inventory`, setup uses listed targets instead of discovery
over the cables. It then accepts `--name`, `--display-name`, `--fabric-cidr`,
`--plan`, `--yes`, `--allow-driver-reload` and these flags; a change without a
terminal needs `--apply --yes`:

| Flag | Meaning |
|---|---|
| `--node USER@IP` | A management target, Node A included; repeat for each Spark, each with the same package |
| `--apply` | Apply the reviewed plan; not with `--plan` |
| `--adopt` | Verify and record existing networking without changing links, addresses, NetworkManager connections or routes and without a driver restart. It adds only the relay table's missing objects, enables `sparkring-fabric.service` and `sparkring-relay-marker.service`, and records the fabric document. A Spark's record of another setup moves aside to `/var/lib/sparkring/retired/` with a receipt |
| `--skip-enroll` | SSH keys and host trust are already configured |
| `--inventory FILE` | Offline node records; planning only, with `--plan` |
| `--head-id ID` | Node A's identity for `--inventory` |
| `--output DIR` | Directory for the setup receipt |

Two setup actions run offline without sudo and accept
`--variant nvfp4-spark|nvfp4-qad`:

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
| `--json` | One `sparkring-cabling/v2` document; with `--bandwidth`, one `sparkring-fabric-bandwidth/v1` document |
| `--ssh-user USER` | Account for signing in to the other Sparks (default: the account that ran `sudo`) |
| `--no-sign-in` | Read only this Spark and its recorded cluster's Sparks; ask for no password |
| `--bandwidth` | Measure each cable's speed instead ([cable speed](#cable-speed)) |
| `--while-serving` | With `--bandwidth`, also measure cables a serving model uses |

It exits with 0 when the cables form a pair, line or ring as SparkRing needs,
1 when a cable needs to move or not every cable could be seen, and 2 on
failure. It reads up to nine Sparks, up to seven cables away, so a Spark
cabled beyond the eight that setup supports is named.

### Cable speed

`sudo sparkring cabling --bandwidth` measures each cable of the fabric that
setup recorded, one link at a time in both directions, and saves the result
for `sparkring status`. Setup runs it before it records the fabric document.
It takes about 30 seconds per cable.

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

Each cable carries two links, measured separately. A degraded cable shows no
errors and models still run, but prompt processing over it is slower.
Rebooting both Sparks on that cable clears it.

The test slows a model that uses the cable, so cables of a serving model's
Sparks are listed as not measured; `--while-serving` measures them anyway. A
link the test cannot measure shows the reason, for example a RoCE GID entry
that does not hold the link's address.

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
Fabric verified: 2 cables on 3 Sparks (path-3), relays 4 rules and 4 routes, reboot-persistent.
Report: /var/lib/sparkring/controller/fabric-reports/fabric-verify-20261008T100500Z.json
```

| Flag | Meaning |
|---|---|
| `--traffic none\|light` | `light` adds one bidirectional RDMA write test per relayed lane (about 10 seconds each); default `none` |
| `--while-serving` | With `--traffic light`, test while a model serves |
| `--json` | Print the `sparkring-fabric-verify/v1` report |

It exits with 0 when every check passes, 1 when one fails, and 2 when it
could not run.

`sudo sparkring fabric spread-check [flags]` copies test files from Node A to
every Spark along the cables, as an install spreads the serving image and the
checkpoint ([Spreading along the cables](install-reference.md#spreading-along-the-cables)).
Run it before an installation needs the spread or after recabling; it works
on any recorded layout.

- Each file is placed on a Spark only after its SHA-256 matches, and all are
  removed once every Spark holds them.
- It prints one line per pass and direction with each Spark's position, when
  it finished after the start and how long after the Spark before it. The
  report is saved in Node A's controller directory as
  `fabric-reports/spread-check-<time>.json` (`sparkring-spread-check/v1`).
- A Spark that stops answering stops it with the message an install prints.
  Repeating the command resumes: placed files are reused, and a partly
  written file is sent again from its first byte.
- It writes below `/var/lib/sparkring/spread/check` on every Spark and sends
  traffic on every cable.

| Flag | Meaning |
|---|---|
| `--files N` | Number of test files; default 16 |
| `--size MIB` | Size of each test file in MiB; default 1024 |
| `--while-serving` | Run while a model serves |

It exits with 0 when every Spark holds every file, 1 when the spread stopped
or is incomplete, and 2 when it could not run.

### fabric tune

`sudo sparkring fabric tune [flags]` plans the measurement of SIRCL's tuning
table on this fabric; `--execute` measures. Later installations use the
measured table instead of the release's default while the fabric, the image
and the Sparks' drivers stay the same
([measure the tuning table](install-reference.md#measure-the-tuning-table)).

```bash
sudo sparkring fabric tune                     # the plan; contacts nothing
sudo sparkring fabric tune --execute           # measure every group shape this fabric serves
sudo sparkring fabric tune --execute --quick   # every fourth size, about a quarter of the time
```

```text
  pair: measured, table HASH
  path-3: measured, table HASH
  cycle-4: measured, table HASH
Measured tuning table: /var/lib/sparkring/controller/sircl-tuning.json (sha256 DIGEST, measured DATE; rows cycle-4, pair, path-3).
```

| Flag | Meaning |
|---|---|
| `--execute` | Measure; without it the plan is printed and nothing is contacted |
| `--layouts NAMES` | Group shapes to measure, comma-separated, such as `pair,cycle-8`; default every shape the harness can measure here |
| `--quick` | Every fourth size, 4 KiB to 64 MiB |
| `--image NAME` | Measure with this installer image; default the one `sparkring install` uses |
| `--stop-serving` | Stop the serving models first; without it the command refuses while a model serves |
| `--max-hours H` | Start no group after H hours; the rest stay pending for the next run |
| `--layout-timeout SECONDS` | The longest measurement of one group; default 1800 |
| `--lan-interface NAME` | The interface of the measurement's control exchange; default the management interface |
| `--control-port PORT` | Its TCP port on the first Spark of each group; default 29650 |
| `--fresh` | Measure every named group again instead of keeping earlier results |
| `--distribute` | Copy the recorded measured tables to every Spark again; measures nothing |
| `--json` | Print the plan or the result as JSON |

A repeated command keeps the groups already measured and measures the rest.
It exits with 0 when every named group was measured (or the plan printed), 1
when a group failed or stayed pending, and 2 when it could not run.

## models

`sparkring models [--json]` lists every profile (exact model, version,
quantization and topology) and marks those `sparkring install` supports. For
each installer profile it also shows:

- what the model does with thinking when a request doesn't say, such as
  `on · xhigh`, and the effort levels it accepts
  ([thinking](install-reference.md#thinking));
- the checkpoint it installs and the others `--checkpoint` takes;
- any other name that selects it, such as `glm53-flash-tp4`.

## images

`sparkring images [--profile PROFILE] [--json]` lists the installer images
this package records, the default first. Each line names the image, its
release tag, its download size, the transports it carries (`prepared`,
`sircl`, `libsircl`) and the profiles it runs; an archived image is marked
`archived`. `--profile` lists only the images that run that profile. Any
listed name, release tag, or part of a name that only one image has selects
that image in `sudo sparkring install --image NAME`.

## status

`sudo sparkring status [PROFILE [--instance NAME]] [flags]` prints Node A's
state, one line per Spark with the next action for any Spark that needs
attention, the saved model (the active deployment, or the one named) and
automatic recovery. While an operator [holds the cluster](#hold), the first
line names the holder, the reason and the times, such as `Held by alice since
2026-11-02 09:00 UTC until 2026-11-02 18:00 UTC: acceptance run | release: sudo
sparkring hold --release`. A model switch that is in progress or stopped before
it finished adds its own line, `Model switch: ... | next: ...`, naming the step
that [settles](#cancel-an-installation) or resumes it. On a fabric that serves
[several models](install-reference.md#models-on-part-of-the-fabric) it prints
each group's model under its Sparks, such as `Sparks 0-3:` and `Sparks 4-7:`,
with a `Group:` line naming its shape, positions and API Spark:

```text
Saved model operation: PROFILE | up complete
Checkpoint: NAME (REPOSITORY @ REVISION) | Image: RELEASE
Thinking: on · xhigh (levels: low, medium or xhigh)
Transport: sircl, NCCL: absent (checked 2026-10-08T10:00:00Z); sudo sparkring check repeats it
Automatic recovery: on
```

- `Checkpoint`: a profile with one checkpoint shows only
  `REPOSITORY @ REVISION`; a
  [derived checkpoint](install-reference.md#derived-checkpoints) adds
  `, derived from REPOSITORY @ REVISION` of its base.
- `Thinking`: what the model does when a request doesn't say whether, or how
  hard, to think: the model's default, or the deployment's own default beside
  it ([thinking](install-reference.md#thinking)).
- The model's API URL follows, at the address `install --api-address` named
  when there is one, with the URL SparkRing's own checks use in parentheses.
- `Transport`: the deployment's transport and, on SIRCL ring sessions, the
  last receipt verdict ([transport and receipts](install-reference.md#transport-and-receipts)).
  `Transport check failed: ...` names the first rank and collective that
  differs.
- `Automatic recovery`: adds the Spark it waits for, its last attempt and the
  next. After `sparkring down` it reads `on; idle until the next sudo
  sparkring up --execute or sudo sparkring install`. A deployment that does
  not run as Compose containers reads `not available`.
- `Fabric bandwidth:` follows the Spark lines with the last
  [cable speed](#cable-speed) result and its age, such as `healthy on all 4
  cables, measured 3 h ago`, or `never measured`. A degraded cable follows
  with its repair steps. Status does not measure.
- `SIRCL tuning:` names the tuning table installations use: `the default
  table`, `measured on this fabric DATE for ROWS with image IMAGE`, or why a
  measured table does not apply, such as a GPU driver other than the one it
  was measured with
  ([measure the tuning table](install-reference.md#measure-the-tuning-table)).

With `--refresh`, a line per model container follows the saved model. When
the model does not serve, the first line says why and gives the command that
fixes it ([every case](install-reference.md#when-a-model-stops-serving)):

```text
The model runs on rank 0 (spark-a) but stopped on rank 1 (spark-b) | next: sudo sparkring down --execute, then sudo sparkring up --execute
  rank 1 (spark-b): stopped, exit code 255 at 2026-09-30 17:02:11 UTC
```

| Flag | Meaning |
|---|---|
| `--refresh` | Contact every Spark, inspect the model containers and ask rank 0's API `/health`; needs sudo |
| `--on ARC` | Only the model on those Sparks |
| `--json` | Print the full observation as JSON; a fabric with models on part of it adds `slots`, one entry per group with its `placement` and `group`, and a recorded cluster adds `fabric_bandwidth`, `fabric` and `sircl_tuning`; `hold` is the standing [hold](#hold) or null |

## check

`sudo sparkring check [flags]` checks each running model and changes no Spark.
It sends the functional requests of the acceptance harness to the model's API
(counting, arithmetic and code, and a tool call, an image and thinking where
the profile serves them). For a model on SIRCL ring sessions it also reads
every rank's receipts and the NCCL lines of its log and judges them as the
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

## dashboard

`sparkring dashboard serve` serves a web page of the fabric's ring and every
recorded model: state, output rate, requests, KV cache use and budget, draft
acceptance, settings and the last automatic-recovery check. Status:
**Experimental**.

It runs on Node A as `sparkring-dashboard.service` at
`http://NODE_A_LAN_ADDRESS:8090/` ([where it listens](dashboard.md#every-model-on-one-page)).
Installing the package or running `sudo sparkring setup` turns it on.
`install`, `up`, `down`, `restart` and automatic recovery refresh its site
file when they finish and ask the service to re-read it; a failed refresh
prints one line and never stops them. The service also notices a site file
written any other way within a second.

It reads files on Node A and sends only GET requests to the models' APIs:

- `/metrics` every 2 s while a viewer (the page or the terminal view) renews
  its six-second lease, and no API reads otherwise;
- `/v1/models` every 60 s;
- the model's [status report](dashboard.md) once per model start.

It sends no inference request, opens no SSH session, starts no process and
runs at idle priority on the efficiency cores. Each card shows the command
that prints the model's log; the page itself reads no log.

```bash
sudo sparkring dashboard quiet on      # no request at all, for a benchmark
sudo sparkring dashboard quiet off
sudo sparkring dashboard disable       # off for good, across reboots, setups and upgrades
sudo sparkring dashboard enable
sudo sparkring dashboard external add flare-spark-6 --host 192.0.2.60 --port 8000 --label "Flare · Qwen3.8-Flash-Next (Spark 6)"
sudo sparkring dashboard external add flare-spark-7 --host 192.0.2.61 --port 8000 --label "Flare (Spark 7)" --position 7   # on the ring figure
sudo sparkring dashboard external list   # endpoints served outside the installer, shown as read-only cards
sudo sparkring dashboard external remove flare-spark-6
```

A measurement holds quiet mode with a holder of its own, so overlapping
measurements and an operator's `quiet on` never end each other's pause:
`quiet hold --label NAME` prints the holder id, and `quiet release ID` ends
that hold. A hold lapses when its process (`--pid`) exits or after 12 hours
(`--ttl SECONDS`). `quiet status` lists every hold. The repository's benchmark
harnesses take a hold themselves ([profile validation](profile-validation.md)).

`sparkring dashboard tui` shows the same status in a terminal on Node A, over
SSH if need be:

- The header gives the models serving, Sparks busy, requests running, total
  output rate, the data's age and the read period.
- The ring shows each model's arc in its colour.
- Each model has a card with its state, TP size and arc, transport, uptime,
  output rate, running requests, KV use and budget, draft acceptance length
  and first-token time.
- `d` expands the selected card with rates, graphs, latency, settings, the
  log command and, per Spark, its rank, LAN address, memory and free disk.
  Free disk below 2 % of the root filesystem is red, below 5 % yellow.

The terminal view reads only the service's `/api/status` on this Spark, at
`http://127.0.0.1:8090` or, when nothing answers there, at the service's LAN
address. It sends no request to any model, opens no SSH session, and refuses
an address of another host before a connection opens. While open it counts
as a viewer, so the service reads `/metrics` every 2 s; `--passive` does not
count. The view needs Python's `curses` module, which Linux has; `--once`
prints one frame as text anywhere.

| Command or flag | Meaning |
|---|---|
| `export [--output FILE]` | Write `/var/lib/sparkring/dashboard/site.json` (`sparkring-dashboard-site/v1`) from the deployment records: each model's Sparks, API address and port and container names. `install`, `up`, `down` and automatic recovery run it themselves. Written to its default path, it then asks the service to re-read it |
| `enable`, `disable` | Turn the service on, or off for good (`/etc/sparkring/dashboard.disabled` records the choice) |
| `serve --bind ADDRESS` | Listen address; default Node A's LAN address from the fabric document, `127.0.0.1` on another host. The service also listens on `127.0.0.1` unless ADDRESS is a loopback or wildcard address ([why](dashboard.md#why-loopback-as-well)) |
| `serve --port N` | Port (default 8090) |
| `serve --fabric FILE --site FILE` | Another fabric's document and site file, once per fabric, for one page of several rings on another host |
| `serve --name NAME` | A host name browsers may use for the page; requests naming another host are refused |
| `serve --password-file FILE` | Require HTTP Basic authentication, user `sparkring`, with the password in FILE |
| `serve --manage --manage-token-file FILE` | Accept authenticated actions, sent with the bearer token in FILE (at least 32 characters), through the [root helper](#without-sudo) |
| `serve --cpus LIST` | CPUs to run on instead of the efficiency cores |
| `serve --no-node-status` | Do not read this host's `sparkring-agent` status |
| `quiet on`, `quiet off` | Pause or resume every request |
| `quiet hold --label NAME [--pid PID] [--ttl SECONDS]`, `quiet release ID`, `quiet status` | A measurement's own pause, its end, and every pause held |
| `tui` | The status in this terminal. Keys: up and down (or `j`, `k`) select a model, `d` or space expands its card, `e` shows events, `r` reads at once, `-` and `=` lengthen or shorten the read period, `q` quits |
| `tui --once [--details] [--events] [--width N]` | Print one frame as text and exit; `--details` expands every card, `--events` adds recent events |
| `tui --passive` | Do not count as a viewer; without another viewer the service stops API reads |
| `tui --url URL`, `tui --password-file FILE` | The service's address on this Spark (default `http://127.0.0.1:8090`, then the service's LAN address), and its password when it has one |

Like the model API, the page has no key unless `--password-file` gives one. It
shows host names, addresses and model settings, so keep Node A on a trusted
network. The site file and the fabric document hold the site's addresses; keep
copies of them out of Git.

## host-load

`sparkring host-load show|set [KEY=VALUE ...] [--all]` shows or changes where
and how often SparkRing's own periodic work runs on a Spark that serves a
model. Status: **Experimental**; without settings every job keeps its default
placement and interval.

```bash
sudo sparkring host-load show --all
sudo sparkring host-load set placement=efficiency agent_serving_interval=120 --all
sudo sparkring host-load set recovery_ssh_reuse=on        # on Node A
```

| Setting | Values | Meaning |
|---|---|---|
| `placement` | `default`, `efficiency` | `efficiency` runs the host agent, the recovery timer and its SSH master connections on the efficiency cores at idle priority, through systemd drop-ins |
| `agent_serving_interval` | 30 to 600 seconds (default 30) | Seconds between the host agent's observations while a model container runs on the Spark |
| `recovery_ssh_reuse` | `on`, `off` (default `off`) | The recovery timer's SSH commands share one connection per Spark, kept by `sparkring-ssh-mux.service`, instead of signing in each time |
| `control_refresh_interval` | 20 to 900 seconds (default 20) | Seconds between the administration tunnel's periodic checks. Link changes and ConnectX driver restarts start a check at once, and a check that leaves the tunnel moving between paths is repeated 20 s later |

`--all` applies the change to every Spark of the recorded fabric from Node A;
without it, to this Spark only (`sparkring node host-load` is the per-Spark
form). Settings live in `/etc/sparkring/host-load.json`. A change takes effect
without a reinstall or a model restart: placement restarts the host agent,
which serves no model, and an interval change restarts only its timer. The
recovery timer reads Node A's own status and container without SSH.

## logs

`sudo sparkring logs [flags]` prints the end of `/var/log/sparkring/install.log`,
the progress log that `install`, `setup`, `up`, `down`, `restart`, `hairpin`
and automatic recovery write.

| Flag | Meaning |
|---|---|
| `--follow` | Keep printing lines as they arrive, until Ctrl-C |
| `--details` | Read `install-details.log`, the full command output, instead |
| `--lines N` | Lines to show first, 1 to 10000 (default 40) |
| `--on ARC` | The latest operation on those Sparks only |
| `--operation ID` | One operation's log, by the identifier its start line prints; not with `--on` |
| `--plain` | No colors or spinner |
| `--color WHEN` | `auto` (default), `always`, or `never`; the log lines stay plain either way |

## hairpin

`sudo sparkring hairpin [flags]` applies the ConnectX hairpin setting, now and
at every boot, on every Spark that relays between its cables: every Spark of a
ring of four or more, and every Spark but the ends of a line of three or more.
It first updates any Spark that runs a different SparkRing revision than
Node A. [More about the setting](install-reference.md#four-spark-rings).

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
| `--override-hold` | With `--release`: proceed although another operator [holds the cluster](#hold); the override is recorded |
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
| `--override-hold` | With `--release`: proceed although another operator [holds the cluster](#hold); the override is recorded |
| `--json` | Print one JSON result |
| `--retain-deployments N` or `off` | Keep the N most recent deployments of each profile at each [automatic release](#automatic-release) (default 2), or turn it off |

A release asks first. Each Spark then checks again that no installed
deployment, no installed mesh (running or not) and no running container uses
the path ([Finding and freeing space](install-reference.md#finding-and-freeing-space)).
A release never removes:

- a checkpoint directory; `sudo sparkring checkpoints --release PATH` does;
- a workspace that holds model files;
- a Docker image; images are only listed, and `docker image rm ID` on that
  Spark removes one by hand. An image's size counts the layers it shares with
  other images, so removing it frees only the layers no other image uses
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

For the other deployments it removes, on each Spark, their stopped model
containers and workspaces, and compile caches that no kept deployment and no
installer profile uses, with the same checks as `--release`. It prints one
line, for example `Released 7 older deployments' containers, workspaces and
caches: 9.8 GiB`. Checkpoint directories, images and the deployments' records
on Node A stay, so `sudo sparkring up PROFILE --instance i<hash>` can start a
released deployment again once the running model is stopped; it recreates the
container and recompiles kernels whose cache was removed.

`sudo sparkring storage --retain-deployments off` turns it off;
`sudo sparkring storage` shows the setting. The preference
[`SPARKRING_RETAIN_DEPLOYMENTS`](install-reference.md#optional-preferences)
sets it too.

## up, down and restart

`sparkring install` is the usual way to start or switch a model. These
commands act on deployments it, or `up PROFILE`, created:

- `sudo sparkring up [PROFILE] [flags]` starts the named profile's deployment,
  or the active one, and makes it the active one.
- `sudo sparkring down [PROFILE] [flags]` stops the named deployment, or the
  active one; the active one stays active.
- `sudo sparkring restart [--on ARC] [flags]` stops and starts the active
  deployment; it takes no profile.
- `sparkring status [PROFILE]` reports one ([status](#status)).

Each prints its steps and asks; `--execute` skips the question. Without a
terminal and without `--execute`, they only print the steps.

A deployment keeps the SparkRing source that created it, so a deployment made
by an earlier package still starts and stops after the package changes.
Without PROFILE, the commands act on the deployment that was active when they
printed their steps; when a `sparkring install` makes another one active
before the steps start, they stop and ask for a new review.

Repeating `up` re-checks a running deployment and starts one whose containers
stopped on every Spark, for example after a restart
([when a model stops serving](install-reference.md#when-a-model-stops-serving)).
A completed `up` clears [automatic recovery's](#recover) failed attempts,
keeps its on or off choice, then runs [automatic release](#automatic-release).
After `down` the model stays stopped.

`sparkring install` names its deployments with instances `i<hash>`:
`sparkring down PROFILE --instance i<hash>` stops one of them. The deployment
directories are under `/var/lib/sparkring/controller/deployments/`.

On a fabric that serves [several models](install-reference.md#models-on-part-of-the-fabric),
each group of Sparks has its own active deployment: `sudo sparkring down --on
4-7 --execute` stops the model on Sparks 4 to 7. Without a profile or `--on`,
the commands act on the one recorded model and ask for `--on` when there are
several. `up` refuses to start a model while a model on one of its Sparks
runs, and names the `down` command that frees them.

| Flag | Meaning |
|---|---|
| `--plan` | Print the steps; change nothing |
| `--execute` | Apply the printed steps without asking |
| `--yes` | Proceed despite a warning that another operation on other Sparks may be affected; an operation that holds these Sparks still stops it |
| `--override-hold` | Proceed although another operator [holds the cluster](#hold); the override is recorded |
| `--json` | Print the result as JSON |
| `--instance NAME` | With PROFILE: a deployment beside the main one, for example a rehearsal |
| `--on ARC` | Without PROFILE: the model on those Sparks. With `up PROFILE`: the profile on those Sparks, as instance `on-` and the positions (`on-2-3`, `on-4-5-6-7`) unless `--instance` names another |
| `--model-path PATH` | `up PROFILE` only: serve this complete copy read-only on every Spark |
| `--fresh-mesh` | `up PROFILE` only: plan replacement of an existing four-Spark mesh |
| `--image NAME` | `up PROFILE` only: another installer image from [`sparkring images`](#images) |
| `--image-lock FILE` | `up PROFILE` only: another image lock, for a rehearsal; not with `--image`. An existing deployment keeps the image it recorded, and naming another lock for it is refused |
| `--transport NAME`, `--nccl MODE` | `up PROFILE` only: the transport of a new deployment, as for [`install`](#install) |
| `--max-images N`, `--reasoning-effort LEVEL` and the other [serving settings](install-reference.md#serving-settings) | `up PROFILE` only: replace one of the profile's serving values, or the model's thinking default, for a new deployment; an existing deployment keeps its own |
| `--allow-loopback-bind` | `up PROFILE` only: accept a loopback `--api-bind` for a new deployment |
| `--deployment DIR` | Use a deployment saved by `sparkring init` instead ([lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing)) |

## recover

`sudo sparkring recover [status|on|off] [flags]` shows or sets the automatic
restart of the active models when a Spark stops serving
([how it works](install-reference.md#automatic-recovery)). `on` and `off`
apply to every active deployment, the whole ring's and each group's, unless
`--on` names one.

| Command or flag | Does |
|---|---|
| `sudo sparkring recover` | Show whether it is on, the Spark it waits for, the last attempt and the next |
| `sudo sparkring recover off` | Stop restarting the active models by themselves |
| `sudo sparkring recover on` | Restart them by themselves again; clears failed attempts and the restart count |
| `--on ARC` | Only the active deployment on those Sparks |
| `--plan` | With `on` or `off`: print what changes; change nothing |
| `--yes` | With `on` or `off`: proceed despite a warning about another operation |
| `--override-hold` | With `on` or `off`: proceed although another operator [holds the cluster](#hold); the override is recorded |
| `--json` | Print one JSON document |

Each `sparkring install` sets the choice for the model it installs: on, or
off with `--no-auto-recover`, also after `recover off`. `sparkring up` keeps
the deployment's choice; a deployment without one starts with recovery on.
To stop the model and keep it stopped, use `sudo sparkring down --execute`.
`sparkring-recover.timer` runs `sparkring recover --auto` once a minute; you
do not run it by hand.

## hold

`sudo sparkring hold --reason TEXT [--until TIME]` reserves the cluster for
one operator's longer task, such as a checkpoint experiment or an acceptance
run, between the single operations that `install.lock` already serializes.
The hold names the holder, the reason, its start and an optional end; it is
kept with the controller records on Node A and survives a reboot.

While another operator's hold stands, `install`, `setup`, `up`, `down`,
`restart`, `recover on|off`, `storage --release` and `checkpoints --release`
refuse before they change anything and name the holder and the reason; the
JSON result carries the hold. `--yes` does not pass a hold. `--override-hold`
passes the hold and appends the override (time, operator, command, Sparks and
the hold) to `/var/lib/sparkring/controller/hold-overrides.jsonl`. The entry
records that the command passed the hold; the command's other checks and
prompts still apply, so one that stops at a later check also leaves an entry. Automatic
recovery never overrides a hold: it keeps checking, reports `held` when
it would restart a model, and restarts it only after the hold ends. Dashboard actions cannot override
a hold either.

The holder is the operator's login, as `sudo` records it. Sessions under the
same login, such as an operator and an agent session on one account, are one
operator, so the hold does not refuse them. A hold whose `--until` has passed
refuses nothing. A `hold.json` that is not a valid hold record refuses every
operator, and keeps automatic recovery waiting, until
`sudo sparkring hold --release --override-hold` clears it; `sparkring status`
names that command.
The acceptance harness (`performance/harnesses/acceptance/accept_profile.py`)
holds the cluster for its whole run when no hold stands, under its `--node-a`
login (`sudo -n`), so its own installation is not refused, and releases that
hold when the run ends; every result record carries the hold's state. A hold is separate from the dashboard's
[quiet holds](#dashboard), which pause the page around one measurement and
only warn.

| Command or flag | Does |
|---|---|
| `sparkring hold` | Show the standing hold; a member of the `sparkring` group reads it from the operator view |
| `--reason TEXT` | Take the hold, or replace your own, with this reason (at most 200 characters) |
| `--until TIME` | With `--reason`: the hold ends by itself, after a duration such as `90m`, `4h` or `2d`, or at a time such as `2026-11-02T18:00` (Node A's local time unless the time names a zone) |
| `--release` | Clear the hold |
| `--override-hold` | With `--reason` or `--release`: replace or clear another operator's hold; the override is recorded |
| `--json` | Print one JSON document (`sparkring-hold/v1`) |

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
| `sudo sparkring node tuning-facts [--interface NAME]` | This Spark's GPU driver and kernel, which a measured tuning table binds, and with `--interface` that interface's IPv4 address (read-only) |

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

`export` accepts `--json`. See
[lower-level commands](install-reference.md#lower-level-commands-and-compose-sharing).

## compose and validate-compose

- `sparkring compose render|check|start|stop` generates and coordinates
  profile-owned Compose deployments; see [Compose deployments](compose.md).
  `render` takes `--image NAME`, `--checkpoint NAME` and the serving-setting
  flags of `sparkring install`, `--api-port` and `--api-bind` among them.
- `python scripts/generate_compose_builder.py --output DIR [--verify]` writes
  the [Install Builder](compose-builder.md) site for the checkout.
- `sparkring validate-compose FILE` (or `--all`, `--json`, `--output FILE`)
  checks Compose files without Docker or GPUs.

## deploy

`sparkring deploy discover|plan|network-plan|network-check|stage|runtime-plan|apply-plan`
is a standalone workflow, run from a Linux checkout, for one four-Spark GLM-5.3
profile. See [Prepare and operate a four-Spark deployment](deployment-suite.md).

## cluster, doctor and host

- `sparkring cluster init|show|configure` bootstraps a blank ring of four or
  six Sparks by hand (`init --size 4|6`), and `sparkring doctor [--cluster
  FILE]` diagnoses it; see [Bootstrap a blank SparkRing cluster](bootstrap.md).
- `sparkring host check [--json] [--require-telemetry-disabled]` runs read-only
  checks on one Spark.
