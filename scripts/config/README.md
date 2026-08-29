# Serving configuration templates

This directory contains sanitized inputs for the supported GLM-5.2 EXL3
3.5-bpw, GLM-5.3 Flash, DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6 serving
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

## GLM-5.3 Flash with BF16 DFlash2

The GLM-5.3 Flash deployment uses three sanitized inputs:

| File | Role |
|---|---|
| `glm53-flash-tp4-site.example.yaml` | Four-rank direct-cycle topology and the shared TP4/DCP1 serving geometry |
| `glm53-flash-dflash2-bf16-tp4-dcp1.example.json` | Cache-disabled runtime profile |
| `glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json` | SparkCache-enabled runtime profile |

Copy the selected files outside version control and replace the documentation
addresses, SSH targets, interfaces, devices, immutable image identity, target
model host path, draft model host path, and writable cache path. Both runtime
profiles keep asynchronous scheduling, native prefix caching, and chunked
prefill enabled. The cache-enabled profile adds only the external key-value
connector to the serving arguments.

The generic launcher rejects the zero image ID in each template. A resolved
profile also fails before launch if the image labels, BF16 DFlash hashes,
patched NCCL binary, vLLM configuration postimage, or vLLM lease contract
differs. Follow the [SparkCache-enabled quickstart](../../docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md)
or [cache-disabled quickstart](../../docs/GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md).

Run the focused CPU-only contract suite after changing these inputs:

```bash
python -m pytest scripts/test_glm53_flash_profile.py -q
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

`qwen38-27b-exl3-k5k6-pair.env.example` is the per-rank environment for the
two-Spark Qwen research profile. It names one direct-link interface, one RoCE
device and one site-specific RoCEv2 GID. The pair leaves NCCL algorithm and
channel selection at library defaults and sets the long-context override used
by the 1M static-YaRN launch. Follow
[`docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md`](../../docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md).

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
Qwen image built by `runtime/qwen38/build-image.sh` supplies the immutable
runtime under `/ws`; it does not supply the model, rank environment, a site
address, or live evidence. Follow
[`docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md`](../../docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md).

## DeepSeek SIRCL research overlay

`deepseek-v4-flash-0731-sircl-research.env.example` is a second per-rank
environment file for the research-only width-4096 SIRCL graph candidate. It
does not replace the canonical environment. Apply it after the canonical file
only for an authorized matched A/B, and resolve both direct peer addresses
against the device named beside each address. The offline plan and evidence
requirements are documented in
[`docs/DEEPSEEK_V4_FLASH_SIRCL_AB.md`](../../docs/DEEPSEEK_V4_FLASH_SIRCL_AB.md).
The environment overlay is not sufficient by itself: both generated A/B
commands require the same six read-only, SHA-bound runtime mounts for the
vLLM capture overlays and SIRCL implementation. The control leaves their
activation variables unset.

## Safety

Copying or validating a template is **OFFLINE**. Contacting configured ranks
without changing them is **READ-ONLY REMOTE**. Pulling an image, creating a
container, changing host files, or stopping a serving stack is **MUTATES HOST**
or **STOPS SERVING** and requires explicit authorization for the named hosts
and action.
