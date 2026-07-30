# Four-Spark NF3 quickstart

SparkRing has one current deployment target:

```text
madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
TP4 / DCP4 / adaptive MTP2-4
four directly cabled DGX Sparks
```

The former Aiden MXFP4/GPTQ lane is historical. Its configuration, results,
and independent bring-up findings are preserved in
[history/AIDEN_MXFP4_GPTQ.md](history/AIDEN_MXFP4_GPTQ.md).

## Current publication status

The NF3 model, SparkRing transport, and upstream kernel inputs are public. The
validated Spark runtime is ARM64/SM121-specific and currently exists as an
attested local image on the four-node development cluster. Its remaining NF3
integration delta still needs to be consolidated into the public source tree.

The remaining no-build blockers are publishing that source delta and the
complete derived image at an immutable OCI digest. The public `madeby561`
serving images are AMD64/SM120 images and cannot be substituted on DGX Spark.

Check the machine-readable recipe:

```bash
python scripts/sparkring_recipe.py \
  --recipe glm52-nf3-hybrid \
  plan
```

Until `runtime.final_image` contains a published ARM64 digest, the command
prints `BLOCKED` and exits non-zero. This is intentional. It prevents a bot
from silently compiling a different runtime or pulling an incompatible
AMD64 image.

## 1. Hardware

You need:

- four DGX Sparks;
- four 200 Gb/s DAC cables in a closed cycle;
- Docker and NVIDIA Container Toolkit on every rank;
- key-based management SSH over Wi-Fi, USB Ethernet, LAN, or Tailscale;
- at least 400 GB free per rank for the replicated checkpoint;
- additional space for the runtime image and JIT cache.

```text
            200 Gb/s
       S0 --------- S1
       |             |
200 Gb/s             200 Gb/s
       |             |
       S3 --------- S2
            200 Gb/s
```

Use management networking for SSH and bootstrap. Keep the four direct
200 Gb/s subnets dedicated to the inference fabric.

Follow [SETUP.md](SETUP.md#stage-1--hardware-cabling) for netplan, MTU 9000,
RoCEv2 GIDs, and cable qualification.

## 2. Clone and configure the site

On the controller:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring

cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
```

Fill only operator-owned facts:

- management SSH targets;
- management interfaces and addresses;
- the two ring interfaces, addresses, RDMA devices, and GID indices per rank;
- model/cache paths and free-space policy.

Then validate offline:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
```

## 3. Verify SSH and the four cables

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope all-adjacent
```

If direct-neighbour key authorization is missing:

```bash
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope all-adjacent \
  --fix
```

Qualify every edge with
[the cable guide](../spark_transport/CABLE_QUALIFICATION.md). Do not start a
model while any link, MTU, GID, or jumbo-frame check fails.

The native transport peer schedule is fixed:

```text
rank 0 -> rank 1, rank 3
rank 1 -> rank 0, rank 2
rank 2 -> rank 3, rank 1
rank 3 -> rank 2, rank 0
```

This is `[rank ^ 1, rank ^ 3]`, not numeric peer sorting.

## 4. Download the model

Run one command on a rank with enough free space:

```bash
./scripts/download-glm52.sh
```

It downloads and verifies:

- the pinned NF3 target; and
- only the small MTP draft from the historical Aiden repository.

It does **not** download the old 382 GiB Aiden target. If your model root is
not `/srv/models`, pass both destinations explicitly:

```bash
./scripts/download-glm52.sh \
  /your/model/root/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  /your/model/root/GLM-5.2-NF3-MTP-Draft
```

Pinned identity:

| Field | Value |
|---|---|
| Revision | `66f3623dd8fefb5ca8046706912d5d31c8d196af` |
| `config.json` SHA-256 | `254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef` |
| weight-index SHA-256 | `6eb773222d932418dd0530c63aca498f86ef424da2a4526ccba76b59726da234` |
| checkpoint shards | 184 |
| approximate checkpoint size | 341 GiB |

Do not publish a final model directory until the script prints `PASS`. Copy
both verified directories to the same paths on the other three ranks using
resumable `rsync`.

## 5. Pull the SparkRing NF3 image

This step becomes:

```bash
docker pull <published-arm64-nf3-image>@sha256:<manifest-digest>
```

The exact reference will live in
[`recipes/glm52-nf3-hybrid.json`](../recipes/glm52-nf3-hybrid.json). Do not
replace it with:

- `madeby561/vllm-glm52-nvfp4-nf3-hybrid:v3`;
- the AMD64 `voipmonitor/vllm` image;
- a mutable tag;
- the Aiden base-image digest; or
- a locally rebuilt image with an unrecorded identity.

All four ranks must report the same repository digest.

## 6. Preflight and launch

Copy the one launch profile:

```bash
cp scripts/config/launch.example.json scripts/config/launch.json
```

Change only the two host model paths if you passed custom paths to the
download helper. Keep the checked-in NF3 flags unchanged.

The validated NF3 serving contract is:

| Setting | Value |
|---|---:|
| TP / DCP | 4 / 4 |
| MTP | adaptive 2/4, 32-round window |
| max model length | 458,752 |
| KV dtype | FP8 |
| KV bytes per rank | 7,000,000,000 |
| reported KV tokens | 511,488 |
| max batched tokens | 4,096 |
| max sequences | 8 |
| transport query capacity | Q40 |
| startup profile | Q2 while retaining Q4096 serving |
| first launch | eager; graph semantic gate remains separate |

The startup profile and workspace reserve are correctness controls, not
optional tuning values.

Before execution:

```bash
python scripts/preflight.py --site scripts/config/site.yaml

python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  plan
```

Review the four generated commands, then launch:

```bash
python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  start --execute
```

The first public launch intentionally includes `--enforce-eager`. Graph mode
must pass the semantic canary before it becomes the default.

Smoke-test rank 0:

```bash
curl -fsS http://<rank-0-management-address>:8000/v1/models
```

## 7. Acceptance

Require:

1. every rank builds 76 NF3 plans;
2. all 184 shards load;
3. every rank reports the same 511,488-token KV pool;
4. tool calling returns structured `tool_calls`;
5. ordinary, code, JSON, and long-context prompts are coherent;
6. no rank drops or transport fallback surprises occur;
7. the eager/graph semantic canary passes before graph results are published.

CUDA capture completion and `/health` alone are insufficient. See
[CUDAGRAPH_CORRECTNESS_GATE.md](CUDAGRAPH_CORRECTNESS_GATE.md).

## Source-build recovery

The source-build lane remains for maintainers and provenance work. It is not
the normal onboarding path. The public quickstart will not ask every user to
recompile NCCL, SIRCL, vLLM, or the NF3 kernels.

Until the final ARM64 image is published, use this document to prepare
hardware, networking, SSH, and the pinned checkpoint. Do not improvise the
runtime layer.
