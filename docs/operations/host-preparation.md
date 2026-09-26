# Prepare the Spark hosts

Run this once on every Spark before the manual [pair](pair-network.md) or
four-node ring procedure in [Set up SparkRing manually](setup.md). The normal
path, `sudo sparkring install`, prepares the hosts itself; its requirements are
in [Install SparkRing](install.md).

Commands use Bash on Linux. Rank 0 is the controller. On an existing
installation, check each step's result and skip what is already done.

## 1. Finish first boot on every Spark

Label ranks from 0. Complete the vendor's first-boot wizard and supported OS
and component updates, rebooting when asked. Keep a separate management
Ethernet or Wi-Fi connection with a reserved IP, and record each rank's username
and management address privately. Use the vendor-supported GPU driver stack.

On **every rank**:

```bash
cat /etc/os-release
uname -r
nvidia-smi
ip -br address
ip route show default
```

**Pass:** the GPU is listed and management stays reachable.

## 2. Check host tools and operator access

On **every rank**, install missing Ubuntu utilities and check the tools:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client openssh-server \
  python3 python3-venv python3-yaml rsync iproute2 ethtool rdma-core ibverbs-utils pciutils
sudo systemctl enable --now ssh
python3 --version
docker version
nvidia-ctk --version
command -v nmcli devlink ibdev2netdev ibv_devinfo
systemctl is-active NetworkManager
```

Python must be 3.12 or later. Docker and NVIDIA Container Toolkit come with the
DGX software stack; if either is missing, repair it with the vendor's procedure.
No host CUDA toolkit or host vLLM is needed. Keep the existing network manager;
do not enable NetworkManager on interfaces another backend owns.

If the operator cannot use Docker, run on that rank:

```bash
sudo usermod -aG docker "$(id -un)"
```

Log out and back in; `docker info` must then work without sudo. Docker-group
membership is equivalent to root. Ring repair and managed mesh operations also
need noninteractive sudo on the affected ranks; set that up before running
bootstrap repairs, as the [mesh authorization procedure](../GLM53_SPARK_MESH_HOST_SETUP.md#4-configure-management-ssh-and-root-authorization)
describes. Pair networking uses interactive sudo.

## 3. Install and record one checkout

On **rank 0**, resolve the branch to one commit and download the installer
script from that commit. `BRANCH=main` selects the default branch; set another
branch name to install that branch.

```bash
REPOSITORY=https://github.com/FujitsuPolycom/sparkring.git
BRANCH=main
REF=$(git ls-remote "$REPOSITORY" "refs/heads/$BRANCH" | awk '{print $1}')
test "${#REF}" -eq 40
echo "$REF"
curl --fail --location \
  "https://raw.githubusercontent.com/FujitsuPolycom/sparkring/$REF/bootstrap.sh" \
  --output /tmp/sparkring-bootstrap.sh
less /tmp/sparkring-bootstrap.sh
bash /tmp/sparkring-bootstrap.sh --ref "$BRANCH"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/.local/share/sparkring"
test "$(git rev-parse HEAD)" = "$REF"
```

The script clones the branch into `~/.local/share/sparkring` and installs the
`sparkring` command in `~/.local/bin`; the last `test` confirms the checkout is
the resolved commit. It installs only on the machine where it runs and refuses
to update a checkout with local changes.

Save the printed 40-character revision. On **each other rank**, run the same
block with `REF` set to that revision instead of the `git ls-remote` line. If
the final `test` fails because the branch has moved, select the recorded commit:

```bash
git fetch origin "$REF"
git checkout --detach "$REF"
test "$(git rev-parse HEAD)" = "$REF"
```

Run all later repository commands from the checkout:

```bash
cd "$HOME/.local/share/sparkring"
python3 scripts/sparkring.py host check
```

For a checkout in another directory, `cd` there instead. All ranks must have
identical committed source. `host check` requires only 20 GiB free on the root
filesystem; step 5 checks model storage.

## 4. Select the profile once per shell

On **every rank**, from the checkout, pick the same profile ID from
[setup](setup.md#1-choose-your-deployment). This example is GLM on two Sparks:

```bash
PROFILE_ID=glm53-flash-spark-tp2-dcp1-sparkcache
mkdir -p .sparkring
python3 scripts/sparkring.py setup show "$PROFILE_ID"
python3 scripts/sparkring.py setup show "$PROFILE_ID" --format shell \
  > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
```

`setup show` reads the profile and its image publication offline and prints
quoted assignments: `PROFILE_ID`, `PROFILE_CONFIG`, `RELEASE`, `IMAGE_REF`,
`EXPECTED_IMAGE_ID`, `MODEL_REPO`, `MODEL_REV`, `NODE_COUNT`,
`SPARKCACHE_ENABLED` and `TARGET_MODEL_VARIANT`. For GLM's alternate QAD
checkpoint, add `--variant nvfp4-qad` to both `show` commands. Source only a
file you just generated and reviewed; regenerate it when you change profiles.

## 5. Check storage on every rank

Mount the intended volumes first, then choose dedicated absolute model and
cache paths. These examples are fine if the disks have room:

```bash
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
CACHE_DIR="/srv/cache/${PROFILE_ID}/${RELEASE}"
DOCKER_DIR=$(docker info --format '{{.DockerRootDir}}')
python3 scripts/sparkring.py setup storage "$PROFILE_ID" \
  --model-path "$MODEL_DIR" --cache-path "$CACHE_DIR" --docker-path "$DOCKER_DIR"
```

For GLM QAD, add `--variant nvfp4-qad`. **Pass:** every listed filesystem
prints `PASS`. The check adds up allocations that share a filesystem and uses
the nearest existing directory for paths not yet created. It writes nothing.

| Allowance per rank | GLM | Qwen |
|---|---:|---:|
| Checkpoint | 200 GiB | 120 GiB |
| Image | 68 GiB | 68 GiB |
| Cache and JIT | 32 GiB | 32 GiB |
| All on one filesystem | 300 GiB | 220 GiB |

These are conservative planning budgets from
[storage-planning.json](../../profiles/storage-planning.json), not measured
minimums. `--reuse-model` (an existing, nonempty model directory) and
`--reuse-image` drop those allowances; they do not verify the assets. Also allow
for a separate containerd content store, quotas, image archives and concurrent
downloads.

After a pass, create operator-owned model and cache directories on each rank:

```bash
sudo install -d -m 0755 -o "$(id -un)" -g "$(id -gn)" "$MODEL_DIR" "$CACHE_DIR"
```

Do not point these at another deployment's directories. If the serving guide
uses different paths, run the storage check again.

## 6. Prepare model-download tooling

On every host that downloads weights itself:

```bash
python3 -m venv "$HOME/.venvs/sparkring-download"
"$HOME/.venvs/sparkring-download/bin/python" -m pip install huggingface_hub
source "$HOME/.venvs/sparkring-download/bin/activate"
hf --help
```

Operator scripts use the system `python3-yaml`, so deactivate this environment
or open a new shell before running them. Call
`"$HOME/.venvs/sparkring-download/bin/hf"` directly for downloads. Log in
interactively if the publisher requires it, and keep tokens out of site files.

Next: [configure the data network](setup.md#3-configure-and-verify-the-data-network).
