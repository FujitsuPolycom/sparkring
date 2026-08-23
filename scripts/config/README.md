# Serving configuration templates

This directory contains sanitized inputs for the supported GLM-5.2 EXL3
3.5-bpw, DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6 serving
configurations. Templates describe contracts; they are not deployment receipts
or evidence of a healthy cluster.

## GLM-5.2 EXL3 3.5-bpw R7

The R7 configuration is split by responsibility:

| File | Role |
|---|---|
| `exl3-r7-site.example.yaml` | Four-rank site topology, host paths, and serving parameters |
| `exl3-r7-candidate.example.json` | Candidate image identity, model hashes, transport selection, and enabled R7 options |
| `exl3-r7-pins.json` | R7 model and runtime identity pins |

Copy the site and candidate templates to local, untracked paths. Replace every
placeholder with values for one appliance, then retain the resolved inputs and
image identity with the resulting evidence. Do not commit host addresses,
credentials, model files, local paths, or a locally produced image ID.

The site template is a declarative input for four directly connected ranks.
The candidate template binds the selected image and model hashes to the
transport and runtime options. Treat a mismatch between candidate, pins, and
the built image as a failed configuration, not a value to normalize manually.

Run the focused offline tests after changing an R7 configuration contract:

```bash
python -m pytest \
  scripts/test_glm35_profile.py \
  runtime/exl3-r7/test_exl3_r7_verify_runtime.py -q
```

## DeepSeek-V4-Flash-0731

`deepseek-v4-flash-0731.env.example` is the per-rank Docker environment file
for DeepSeek-V4-Flash-0731. Copy it once per rank and replace:

- `<NCCL_SOCKET_IFNAME>` with the rank's configured fabric interface;
- `<RANK_FABRIC_IP>` with that interface's address.

All other values are the common runtime contract. Keep each rank's resolved
environment file local. The template configures the loader, patched NCCL,
RoCE transport, B12X kernels, and local cache paths required by the pinned
serving image. It does not supply a model path, launch command, registry
credential, or serving acceptance result.

The template's immutable image digest is also represented in
[`runtime/faststart-lock.json`](../../runtime/faststart-lock.json). If those
inputs differ, stop and resolve the identity drift before launching.

## Qwen3.8-27B EXL3 K5/K6

`qwen38-27b-exl3-k5k6.env.example` is the per-rank environment for the
four-Spark Qwen candidate. Copy it once per rank and replace:

- `<RENDEZVOUS_IFNAME>` with the rank's management interface;
- `<RANK_RENDEZVOUS_IP>` with that interface's address;
- `<NCCL_IB_HCA>` with the two cycle-facing RoCE devices in local ring order;
- `<NCCL_IB_GID_INDEX>` with the rank-global index of both devices' RoCEv2
  entries for their fabric IPv4 addresses.

The template combines the Qwen EXL3 graph/prefill settings with the
four-Spark patched-NCCL cycle. Management carries rendezvous and random worker
TCP ports; the RoCE devices carry collectives. The template assumes the
source-built runtime is mounted at `/ws`; it does not supply that runtime, the
model, a site address, or live evidence. Follow
[`docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md`](../../docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md).

## Safety

Copying or validating a template is **OFFLINE**. Contacting configured ranks
without changing them is **READ-ONLY REMOTE**. Pulling an image, creating a
container, changing host files, or stopping a serving stack is **MUTATES HOST**
or **STOPS SERVING** and requires explicit authorization for the named hosts
and action.
