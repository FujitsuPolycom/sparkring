# Prepare the Spark hosts

Use this once before the [pair](pair-network.md) or four-node ring procedure.
An existing installation can skip completed actions after checking their results.
Commands use Bash on Linux. Rank 0 is the controller; run each block where stated.

## 1. Finish first boot on every Spark

Label ranks starting at 0. Complete the vendor's first-boot wizard and supported
OS/component updates; reboot when requested. Keep a separate management Ethernet
or Wi-Fi connection and reserve stable management IPs. Record each rank's username
and management address privately. Use the vendor-supported GPU driver stack.

On **every rank**:

```bash
cat /etc/os-release
uname -r
nvidia-smi
ip -br address
ip route show default
```

**Pass:** the GPU is visible and management remains reachable. These commands do
not check inter-node RDMA. Resolve host failures before network configuration.

## 2. Check host tools and operator access

On **every rank**, install missing Ubuntu utilities:

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

Use Python 3.12 or later for these shared-image paths. Docker and NVIDIA Container
Toolkit normally arrive with the supported DGX software stack. If missing, repair
that installation using the vendor's procedure. A host CUDA toolkit or host vLLM
installation is unnecessary. Preserve the existing network manager; do not enable
NetworkManager over interfaces owned by another backend.

If the trusted operator cannot use Docker, on that rank run:

```bash
sudo usermod -aG docker "$(id -un)"
```

Log out and back in, then require `docker info` to work without sudo. Docker-group
membership grants effective root access. Ring repair and managed mesh operations
also require noninteractive sudo on the affected ranks. Arrange the policy before
running bootstrap repairs; the [mesh authorization procedure](../GLM53_SPARK_MESH_HOST_SETUP.md#4-configure-management-ssh-and-root-authorization)
explains that scope. Pair networking below uses interactive sudo during preparation.

## 3. Install and record one checkout

On **rank 0**, resolve the installation ref once and download the installer from
that exact commit. `main` below selects a maintained checkout, not a floating image.
For a supervised branch rehearsal, use the supplied committed checkout instead
and skip this download/install block on every host.

```bash
REPOSITORY=https://github.com/FujitsuPolycom/sparkring.git
REF=$(git ls-remote "$REPOSITORY" refs/heads/main | awk '{print $1}')
test "${#REF}" -eq 40
curl --fail --location \
  "https://raw.githubusercontent.com/FujitsuPolycom/sparkring/$REF/bootstrap.sh" \
  --output /tmp/sparkring-bootstrap.sh
less /tmp/sparkring-bootstrap.sh
bash /tmp/sparkring-bootstrap.sh --ref "$REF"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/.local/share/sparkring"
git rev-parse HEAD
```

Save the printed 40-character revision. On **each other rank**, repeat this block
with `REF` set to that recorded revision instead of the `git ls-remote` assignment.
Do not independently resolve `main` again. The installer refuses a dirty managed
checkout. It installs only on the machine where it runs.

For all subsequent repository commands, use this directory in each new shell:

```bash
cd "$HOME/.local/share/sparkring"
python3 scripts/sparkring.py host check
```

For an explicitly supplied test checkout, `cd` to its actual path instead. All
ranks must have identical committed source. The generic host check's 20-GiB root
space threshold is only a basic host check; perform the storage check below.

## 4. Select the profile once per shell

On **every rank**, from the checkout, choose the same ID from [setup](setup.md#1-choose-your-deployment).
This example is GLM on two nodes. Change `PROFILE_ID` for your selected row:

```bash
PROFILE_ID=glm53-flash-spark-tp2-dcp1-sparkcache
mkdir -p .sparkring
python3 scripts/sparkring.py setup show "$PROFILE_ID"
python3 scripts/sparkring.py setup show "$PROFILE_ID" --format shell \
  > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
```

The generated file contains quoted assignments from the tracked profile and
publication. It performs no remote actions. `--variant nvfp4-qad` explicitly selects
GLM's alternate pinned checkpoint; pass it to both `show` commands when using that
variant. Source only the file you just generated and reviewed. Keep this selection
with the deployment; regenerate it deliberately when changing profiles.

## 5. Check storage on every rank

Choose dedicated absolute model/cache paths; mount intended storage volumes first.
The following are examples you may keep if those disks have enough space:

```bash
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
CACHE_DIR="/srv/cache/${PROFILE_ID}/${RELEASE}"
DOCKER_DIR=$(docker info --format '{{.DockerRootDir}}')
python3 scripts/sparkring.py setup storage "$PROFILE_ID" \
  --model-path "$MODEL_DIR" --cache-path "$CACHE_DIR" --docker-path "$DOCKER_DIR"
```

For GLM QAD, add `--variant nvfp4-qad`. **Pass:** every reported filesystem has
enough additional free space. The check groups allocations sharing a filesystem
and uses the nearest existing directory for paths not yet created. If Docker
uses a separate containerd content store, check its filesystem too; the Docker
root alone may not cover it. Quotas and simultaneous downloads need separate
allowance.

Planning allowances are 300 GiB for GLM or 220 GiB for Qwen when all three paths
share one filesystem: checkpoint 200/120 GiB, image 68 GiB, cache/JIT 32 GiB.
These are conservative budgets, not measured minima. [The policy](../../profiles/storage-planning.json)
owns them. Image archives require additional space. A verified existing checkpoint
can use `--reuse-model`; a verified installed image can use `--reuse-image`.
These flags change the budget only and do not verify the assets.

After a pass, create model/cache directories owned by the operator on each rank:

```bash
sudo install -d -m 0755 -o "$(id -un)" -g "$(id -gn)" "$MODEL_DIR" "$CACHE_DIR"
```

Do not point these commands at directories owned by another deployment. Follow
the serving guide for its final paths; repeat storage planning if they change.

## 6. Prepare model-download tooling

On every host that will download weights directly:

```bash
python3 -m venv "$HOME/.venvs/sparkring-download"
"$HOME/.venvs/sparkring-download/bin/python" -m pip install huggingface_hub
source "$HOME/.venvs/sparkring-download/bin/activate"
hf --help
```

Open a fresh shell or deactivate this environment before running operator scripts
that use the system's `python3-yaml`. Activate it again for `hf download` only, or
call `"$HOME/.venvs/sparkring-download/bin/hf"` directly. Authenticate interactively
if the publisher requires access. Keep tokens out of site files and receipts.

Continue to [data-network setup](setup.md#3-configure-and-verify-the-data-network).
