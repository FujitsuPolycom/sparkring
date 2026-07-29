# Four-Spark quickstart

This is the shortest path to four stock DGX Sparks running GLM-5.2, switchless: 

```text
pull the pinned public ARM64 base
        ↓
apply the hash-checked SparkRing Python layer
        ↓
build patched NCCL + the small native SparkRing transport
        ↓
download one pinned GLM-5.2 checkpoint and copy it around the ring
        ↓
preflight → eager first launch → graph tuning
```

The faststart path avoids rebuilding vLLM, Torch, FlashInfer, SparkInfer,
DeepGEMM, and the GB10 kernels. The exact public base is:

```text
aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7
```

The mutable `production-hybrid-1.3` tag is recorded for humans, but none of
the commands below use it.

> Status: the source, patch contracts, launcher, and GPU-free tests are public
> and validated. A native ARM64 build and four-rank bring-up have passed through
> identical image distribution, 116/116 preflight checks, full model/MTP load,
> B12X prewarm, and KV allocation. The corrected image still awaits API/request
> acceptance. A patch preimage mismatch is therefore a useful
> result: report it; do not bypass the check.

## 1. Prerequisites

You need:

- four DGX Sparks with Docker and NVIDIA Container Toolkit;
- four direct 200 GbE DAC cables in a closed ring;
- SSH key access to all four nodes over Wi-Fi, USB Ethernet, LAN, or
  Tailscale;
- roughly 450 GB free per node for the checkpoint;
- one node with additional temporary room for the built image archive;
- Python 3.12 on the computer that will run the launcher.

Use management networking for SSH and bootstrap. Never put management traffic
on the four point-to-point RoCE subnets.

A 10GbE diagonal may be used temporarily during initial setup. Once Wi-Fi 7,
USB Ethernet, wired LAN, or Tailscale management has been proven across a
reboot, move SSH/bootstrap there and reserve the isolated 10GbE diagonals for
future cache or inference-sideband experiments. They are not part of the
public serving topology.

The physical ring is:

```text
                 link 0-1
             ┌─────────────┐
             │             │
           rank 0         rank 1
             │             │
 link 3-0    │             │    link 1-2
             │             │
           rank 3         rank 2
             │             │
             └─────────────┘
                 link 2-3
```

Each cable gets its own `/24`. For example:

| Link | First endpoint | Second endpoint |
|---|---|---|
| 0-1 | rank 0 `10.10.1.10/24` | rank 1 `10.10.1.11/24` |
| 1-2 | rank 1 `10.10.2.11/24` | rank 2 `10.10.2.12/24` |
| 2-3 | rank 2 `10.10.3.12/24` | rank 3 `10.10.3.13/24` |
| 3-0 | rank 3 `10.10.4.13/24` | rank 0 `10.10.4.10/24` |

Configure MTU 9000 and confirm every port negotiates at 200000 Mb/s. The
complete netplan, RoCEv2 GID, and cable-qualification procedure is
[Stages 1-3 of the bring-up guide](SETUP.md#stage-1--hardware-cabling). Do not
start a model until all four edges pass.

## 2. Clone SparkRing on rank 0

Run this on one Spark, preferably rank 0:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
git status --short
```

The last command should print nothing. Record the source revision:

```bash
git rev-parse HEAD
```

## 3. Build the thin SparkRing image

Still on rank 0:

```bash
chmod +x runtime/build-faststart.sh scripts/download-glm52.sh
OUTPUT_IMAGE=sparkring/glm52-faststart:trial \
  ./runtime/build-faststart.sh
```

What this does:

1. pulls the pinned public ARM64 base;
2. verifies that its installed vLLM files match every recovered patch
   preimage;
3. applies 59 modified-file patches and 12 additions, then the two SparkCache
   compatibility patches;
4. builds patched NCCL 2.30.7 and `libspark_transport_capi.so` for `sm_121`;
5. bakes the SparkRing integration layer and a complete runtime hash manifest
   into a thin derived image.

This should take tens of minutes rather than hours. The exact time is bounded
mainly by the base-image pull and the two native builds. It does **not** build
vLLM or the kernel stack.

If it stops with `preimage mismatch`, stop there. Include the expected and
actual hashes plus this output in a GitHub issue:

```bash
docker image inspect \
  aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7
git rev-parse HEAD
```

Do not remove the SHA checks or patch with fuzz.

Record the resulting immutable local image ID:

```bash
IMAGE=sparkring/glm52-faststart:trial
docker image inspect "$IMAGE" --format '{{.Id}}'
```

## 4. Put that exact image on ranks 1-3

Before creating the archive, verify the control-host connections and every
direct-cable SSH direction used by the fanout:

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope fanout
```

The report names the exact failed direction and distinguishes an unknown host
key, missing public-key authorization, and an unreachable link. To repair
direct-neighbour trust and authorization through the already-working
management connections, run:

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope all-adjacent \
  --fix
```

`--fix` may create an Ed25519 key on a source rank, but its private key never
leaves that rank. It transfers only the public key. It obtains the
destination's host public key through the authenticated management connection
instead of blindly trusting `ssh-keyscan`. It cannot and will not repair a
broken management connection; establish that trust manually first.

Do not begin the archive transfer until the fanout report passes. Passwordless
access from the control host is a separate relationship from passwordless
rank-to-rank access over a direct cable.

The simplest no-registry method is one image archive. On rank 0:

```bash
docker save sparkring/glm52-faststart:trial \
  -o /tmp/sparkring-glm52-faststart.tar
sha256sum /tmp/sparkring-glm52-faststart.tar
```

Copy it to rank 1 and rank 3 over their direct fabric addresses:

```bash
scp /tmp/sparkring-glm52-faststart.tar <user1>@10.10.1.11:/tmp/
scp /tmp/sparkring-glm52-faststart.tar <user3>@10.10.4.13:/tmp/
```

From rank 1, copy it across link 1-2:

```bash
scp /tmp/sparkring-glm52-faststart.tar <user2>@10.10.2.12:/tmp/
```

On ranks 1, 2, and 3:

```bash
docker load -i /tmp/sparkring-glm52-faststart.tar
docker image inspect sparkring/glm52-faststart:trial --format '{{.Id}}'
```

All four image IDs must be identical. You may delete the tar files after all
four `docker load` operations and ID comparisons succeed.

If you have a registry, pushing once and pulling the resulting digest on all
four ranks is equally valid and usually more convenient.

## 5. Download the pinned model once

On rank 0:

```bash
sudo mkdir -p /srv/models/GLM-5.2-MXFP4-Experts-GPTQ
sudo chown -R "$USER":"$USER" /srv/models

./scripts/download-glm52.sh \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ
```

The script uses the base image's `huggingface_hub`, resumes partial downloads,
pins Hugging Face commit
`46537e0e16fcd156627800139b41b9c497fc7ee2`, and refuses a `config.json`
whose SHA-256 is not
`ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69`.
It also creates a zero-copy root `model-mtp.safetensors` symlink to the
checkpoint's `mtp-draft/model-mtp.safetensors`. The pinned root weight index
references that filename, but the Hugging Face repository stores it in the
draft subdirectory; the public model preflight intentionally rejects the
dangling layout.

If Hugging Face requires authentication in your environment, export
`HF_TOKEN` before running it. The script passes the token through the
environment and does not write it to the repository.

Record the safetensors index hash:

```bash
sha256sum \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/model.safetensors.index.json
```

## 6. Copy the model around the direct ring

First create the destination on ranks 1-3, using each rank's management SSH
address:

```bash
ssh -t <user1>@<rank1-management-ip> \
  'sudo mkdir -p /srv/models/GLM-5.2-MXFP4-Experts-GPTQ &&
   sudo chown -R "$USER":"$USER" /srv/models'
ssh -t <user2>@<rank2-management-ip> \
  'sudo mkdir -p /srv/models/GLM-5.2-MXFP4-Experts-GPTQ &&
   sudo chown -R "$USER":"$USER" /srv/models'
ssh -t <user3>@<rank3-management-ip> \
  'sudo mkdir -p /srv/models/GLM-5.2-MXFP4-Experts-GPTQ &&
   sudo chown -R "$USER":"$USER" /srv/models'
```

Use the same two-hop fanout. From rank 0:

```bash
rsync -aH --info=progress2 \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/ \
  <user1>@10.10.1.11:/srv/models/GLM-5.2-MXFP4-Experts-GPTQ/

rsync -aH --info=progress2 \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/ \
  <user3>@10.10.4.13:/srv/models/GLM-5.2-MXFP4-Experts-GPTQ/
```

From rank 1:

```bash
rsync -aH --info=progress2 \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/ \
  <user2>@10.10.2.12:/srv/models/GLM-5.2-MXFP4-Experts-GPTQ/
```

On every rank:

```bash
sha256sum \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/config.json \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ/model.safetensors.index.json
```

Both hashes must agree across all four nodes.

## 7. Create cache directories

On every rank:

```bash
sudo mkdir -p \
  /var/lib/sparkring/jit-cache \
  /var/lib/sparkring/context-cache
sudo chown -R "$USER":"$USER" /var/lib/sparkring
```

SparkCache is not required merely to reach the first API. The directories are
mounted now so later cache experiments do not require changing the container
contract.

## 8. Fill the two launch files

On the computer from which you will control the cluster:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt

cp scripts/config/site.example.yaml scripts/config/site.yaml
cp scripts/config/launch.example.json scripts/config/launch.json
```

Edit `scripts/config/site.yaml`:

1. replace the four `ssh_target`, management interface/address, ring
   interface/address, RDMA device, and GID entries;
2. replace the four topology subnets;
3. set `runtime.container_image` to
   `sparkring/glm52-faststart:trial`;
4. set `runtime.container_image_digest` to the `sha256:...` image ID recorded
   in Step 3;
5. set `runtime.model_path` to `/models/glm52`;
6. retain the pinned model repository and revision;
7. set `runtime.checkpoint_sha256` to the safetensors index hash from Step 5;
8. replace the entire loose artifact list with `artifacts: []`;
9. reduce `paths.min_free_bytes.context_cache` if you do not yet intend to use
   disk-backed SparkCache.

Edit `scripts/config/launch.json`:

```json
"model_host_path": "/srv/models/GLM-5.2-MXFP4-Experts-GPTQ"
```

Leave `--enforce-eager` in place for the first launch. This deliberately
separates basic four-rank correctness from CUDA-graph capture.

Validate locally without contacting a Spark:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml \
  --strict-placeholders

python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  plan

python scripts/preflight.py \
  --site scripts/config/site.yaml \
  --print-plan
```

Read the generated four `docker run` commands. In particular, confirm each
rank's two peers are its physical ring neighbors.

## 9. Run the read-only four-node preflight

```bash
python scripts/preflight.py \
  --site scripts/config/site.yaml
```

Do not continue until it passes SSH, management-address ownership, 200 Gb/s
link state, MTU, RoCEv2 GIDs, peer reachability, image identity, model/cache
paths, free space, and required-port checks on every rank.

## 10. Launch the eager correctness candidate

This is the first command in this guide that starts serving containers:

```bash
python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  --execute start
```

Follow rank 0:

```bash
ssh <rank0-management-ssh-target> \
  'docker logs --follow --tail 200 glm52-sparkring-public-r0 2>&1'
```

The first load can take roughly ten minutes for weights, followed by kernel
prewarm and KV allocation. The API is ready only after this
returns a model:

```bash
curl http://<rank0-management-ip>:8000/v1/models
```

If logs stop at `Autotuning process starts` during a distributed launch, stop
the run instead of waiting for the NCCL watchdog. The current launch template
must include `--no-enable-flashinfer-autotune`; SparkRing retains the bounded
B12X prewarm but disables the unsafe generic full-model FlashInfer tuner.

The public template reserves `4600000000` bytes per rank for KV cache. On the
validated checkpoint this reported 4.28 GiB and 465,663 logical tokens, enough
for `max_model_len: 458752`. Do not reduce it to 4,000,000,000 bytes while
keeping that maximum context length.

Then run one short request:

```bash
curl http://<rank0-management-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Write a Python hello-world."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 11. Stop, inspect, and report

Stop only SparkRing-managed containers:

```bash
python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  --execute stop
```

If the faststart lane fails, include:

- SparkRing commit;
- image ID from every rank;
- the first failing build/preflight/entrypoint message;
- `docker logs` from all four ranks;
- sanitized `site.yaml`;
- whether eager startup reached `/v1/models`.

Do not include passwords, tokens, SSIDs, private IPs you do not want public, or
model-provider credentials.

## 12. After eager works

Only after the eager launch is correct:

1. qualify the standalone native TP4, vocabulary, and DCP probes;
2. remove `--enforce-eager`;
3. add the captured decode/prefill bucket plan;
4. run the acceptance matrix;
5. publish measured performance with the exact image ID, source commit, model
   revision, DCP/MTP settings, context, and concurrency.

The full source-reproducible image remains available through
`runtime/build-runtime.sh`. That lane rebuilds the whole pinned stack and is
appropriate for provenance work; it is not required for a first trial.
