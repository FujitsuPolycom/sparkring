# Install SparkRing

`sparkring install` sets up two or four cabled DGX Sparks and starts one model
on them. Run it on Node A, the Spark connected to your network. It sets up the
other Sparks through the cables, downloads the serving image and the model (or
reuses a copy already on the Sparks), and prints the API address when the
model is ready.

All commands and flags: [SparkRing commands](commands.md). Details:
[Install SparkRing: reference](install-reference.md).

## Requirements

- Two or four DGX Sparks with Ubuntu 24.04 ARM64 (DGX OS), working NVIDIA
  drivers, Docker, NVIDIA Container Toolkit, NetworkManager and SSH, all on
  the same OS version. Setup does not install or replace drivers or firmware.
- A root login, or a login with sudo, on every Spark.
- Cables: a pair uses one cable from port p0 to port p0; a four-Spark ring
  connects each Spark's p0 to the next Spark's p1, in any order.
- Node A's 10GbE port on your network, with outbound HTTPS to `ghcr.io`,
  `huggingface.co`, your Ubuntu mirror and, for four Sparks, `github.com`.
  Workers need no network cable of their own.
- Free disk on a Spark that holds neither the image nor the model: about
  203 GiB for Qwen, 278 GiB for MiMo or 280 GiB for GLM, plus 14.2 GiB on
  Node A ([details](install-reference.md#downloads-storage-and-outbound-hosts)).

Not supported: six-Spark rings, other port layouts, and Docker's containerd
image store ([check which store a Spark uses](install-reference.md#image-distribution-and-caches)).

## Install

Two Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

Four Sparks:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

Run it on Node A as a user with sudo. It builds the SparkRing Debian package,
installs it, then runs `sudo sparkring install` with your options.

To use a published package instead, download `sparkring_*_arm64.deb` and its
`.sha256` file from a [prerelease](https://github.com/FujitsuPolycom/sparkring/releases), then:

```bash
sha256sum --check sparkring_*_arm64.deb.sha256 && sudo apt install ./sparkring_*_arm64.deb
sudo sparkring install --profile qwen38-flash-next-tp2
```

To build the package yourself, see [Get the package](install-reference.md#get-the-package).

| Profile | Sparks | API port |
|---|---|---|
| `qwen38-flash-next-tp2` | 2 | 8000 |
| `qwen38-flash-next-qad-tp4` | 4 | 8015 |
| `glm53-flash-nvfp4-spark-tp2` | 2 | 8000 |
| `glm53-flash-nvfp4-spark-tp4` | 4 | 8015 |
| `mimo-v26-flash-rl-tp2` | 2 | 8020 |
| `mimo-v26-flash-rl-tp4` | 4 | 8020 |

Without `--profile`, the command lists the profiles for the cabled Sparks and
asks for one.

## What it asks and changes

- On first use it lists every change and asks `Proceed? [Y/n]` once. Enter
  approves.
- It asks `Proceed with this checkpoint plan? [y/N]` when it would download
  more than 1 GiB of model files or a search for existing copies did not
  finish. Answer `y`.
- SSH asks for a worker's password when no key login exists.
- It asks before stopping a GPU container that is not SparkRing's.
- It installs SparkRing on the workers, sets addresses on the cabled ports,
  creates an administration network, and shares Node A's Internet connection
  with the workers.
- It prepares the image and model before it stops the running model. If the
  selected model fails to start, it restores the model that was running.
- It ends with `Model ready:` and the API URL. If it stops before the model
  switch, run the same command again; it re-checks finished work and continues
  an unfinished download.

## Four-Spark rings

Four-Spark profiles need a ConnectX driver setting, the hairpin setting, on
every Spark. SparkRing applies it at the first installation, one Spark at a
time, and again at every boot, which adds about 30 seconds to each boot. The
approval question lists it; no flag is needed. Pairs do not use it.

A model does not start by itself after a reboot. Start it again with the
install command, for example:

```bash
sudo sparkring install --profile qwen38-flash-next-qad-tp4
```

If `sudo sparkring status` reports a Spark as `needs-attention` for a ConnectX
function, run `sudo sparkring hairpin` on Node A.
[More about the setting](install-reference.md#four-spark-rings).

## Reuse a model already on disk

Before it plans, the installer searches every Spark for the profile's
checkpoint: SparkRing's own directories, Hugging Face caches, Docker model
mounts, and folders under `/var/tmp/models`, `/models`, home directories,
`/data`, `/srv`, `/mnt`, `/opt` and other local disks. It never searches
network storage. It checks every file's SHA-256, then links or copies it into
`/srv/sparkring/<cluster>/checkpoints`. It never writes, moves or deletes your
files.

- `--model-path PATH` names a copy for every Spark; `--model-path N=PATH`
  names one for Node N.
- `--ignore-local-copies` uses only SparkRing's own directories and named
  copies.
- A file named `.sparkring-ignore` keeps SparkRing out of its folder and
  everything below it.
- `sudo sparkring checkpoints` lists SparkRing's checkpoint directories, and
  `sudo sparkring checkpoints --release PATH` removes one.

[How copies are found and used](install-reference.md#checkpoints).

## Logs and status

```bash
sudo sparkring logs --follow             # installation progress
sudo sparkring logs --details --follow   # full command output
sudo sparkring status --refresh          # every Spark and the model
```

The [status dashboard](dashboard.md) is at
`http://NODE_A:PORT/v1/sparkring/status/view`, where `PORT` is the profile's API
port. Logs are in `/var/log/sparkring/`.

## Security

- The model API has no key and listens on every interface of Node A. Keep
  Node A on a trusted network or restrict the port with a firewall.
- Setup adds a WireGuard administration network (`sr-control`) and an SSH
  service on port 2222 that admits only Node A's key. Node A shares its
  Internet connection with the workers.
- If a Spark lacks passwordless sudo, the installer prints a command that adds
  it (`/etc/sudoers.d/USER`); review it before you run it.

[Everything the installer exposes](install-reference.md#security-and-host-exposure).

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `runs only from the installed ARM64 Debian package` | Install the package on Node A (see [Install](#install)) |
| `Run sudo sparkring install on Node A.` | Run the command with `sudo` on Node A |
| `Some Sparks need noninteractive SSH and sudo` | Run the printed command once on each listed Spark, then repeat |
| `Approve this installation with ... --yes` | Run the same command in a terminal, or add `--yes` |
| A Spark lacks free space | Free space on the filesystem the message names, then repeat |
| `The checkpoint plan differs from the plan reviewed with --plan` | Run with `--plan`, review, then repeat with `--yes` |
| The plan does not use your copy | Name it with `--model-path N=PATH` |
| `in use. No ConnectX driver was restarted` | Stop what the message lists, then repeat |
| `needs-attention` for a ConnectX function | `sudo sparkring hairpin` on Node A |
| The model is stopped after a reboot | `sudo sparkring install --profile PROFILE` |
| Deleting a copy freed no space | `sudo sparkring checkpoints`, then `--release PATH` |

## Uninstall

```bash
sudo sparkring down --execute    # on Node A: stop the model
sudo apt remove sparkring        # on each Spark
```

Removal stops and disables SparkRing's services. It keeps configuration,
models, caches, network settings and the sudo file, and leaves `avahi-daemon`
and `lldpd` enabled; no command reverts setup's changes.
[What removal keeps](install-reference.md#what-is-installed).
