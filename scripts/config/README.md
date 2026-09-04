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
| `exl3-r7-candidate.example.json` | Image identity, model hashes, transport selection, and enabled R7 options |
| `exl3-r7-pins.json` | R7 model and runtime identity pins |

Copy the site and image-identity templates to local, untracked paths. Replace every
placeholder with values for one appliance, then record the resolved inputs and
image identity with the resulting evidence. Do not commit host addresses,
credentials, model files, local paths, or a locally produced image ID.

The GLM configuration provides the operator-facing equivalent of the
env-driven profiles:

| Operator setting | Config field | Container use |
|---|---|---|
| Model host path | `model_host_path` and `runtime.model_path` | Read-only model mount |
| JIT cache parent | `jit_cache_host_path` and `paths.jit_cache_dir` | `/cache/jit` |
| Online EXL3 weight cache | `online_weight_cache_host_path` | `/cache/exl3-online` |
| API/master ports | `serving.api_port`, `serving.master_port` | Rank-0 API and rendezvous |
| MTP depth, context, sequences | `serving.mtp_tokens`, `serving.max_model_len`, `serving.max_num_seqs` | vLLM serving arguments |

The profile generator owns the qualified scheduler-token budget; it is not a
site override. SparkCache recipes inherit the model, cache, ports, and serving
values from the generated GLM base profile.

The site template is a declarative input for four directly connected ranks.
The image-identity template, `exl3-r7-candidate.example.json`, binds the
selected image and model hashes to the transport and runtime options. Treat a
mismatch between that template, pins, and
the built image as a failed configuration, not a value to normalize manually.

Run the focused offline tests after changing an R7 configuration contract:

```bash
python -m pytest \
  scripts/test_glm35_profile.py \
  runtime/exl3-r7/test_exl3_r7_verify_runtime.py -q
```

## GLM-5.3 Flash with BF16 DFlash2

The operator template is
[`runtime/glm53-flash-jj-r8-gb10/runtime.env.example`](../../runtime/glm53-flash-jj-r8-gb10/runtime.env.example).
It configures one published Linux/ARM64 image for TP4 with DCP1, DCP2, or
DCP4. Its defaults are TP4/DCP4, a 1,048,576-token request limit, 16 sequences,
an 8,192-token batched-token budget, scheduler interval 2, and a 24 GiB FP8 KV
allocation per rank.

The implemented SIRCL performance-testing composition appends
[`sircl-fused.env.example`](../../runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example)
to a rank-local copy of the operator template. The overlay contains every
non-site transport setting and leaves the peer addresses and secondary device
names as explicit placeholders. The image identified by the operator template
contains the SIRCL bundle; `SIRCL_BUNDLE_HOST_ROOT` is an optional read-only
developer override.

`SPARKCACHE_ENABLED=1` enables persistent SparkCache plus vLLM prefix caching.
`SPARKCACHE_ENABLED=0` omits the persistent connector and retains vLLM prefix
caching. Both modes use the same image and
[`GLM-5.3 quickstart`](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md).

`glm53-flash-tp4-site.example.yaml` is the companion topology and preflight
input. Its image identity and DCP4 memory settings match the operator image.
Its `preflight.memory` block requires at least 96 GiB of available RAM and 200
equivalent free blocks of at least 32 MiB before model loading. Larger buddy
blocks count as multiple 32 MiB blocks. These thresholds describe a rank before
model loading; they do not describe memory headroom while a model is serving.

The `glm53-flash-dflash2-bf16-tp4-dcp1-site.example.yaml` and
`glm53-flash-dflash2-bf16-tp4-dcp1*.example.json` files describe separate
source-bound TP4/DCP1 profiles. They are not launch inputs for the DCP4
operator profile.

Run the focused CPU-only contract suites after changing GLM-5.3 inputs:

```bash
python -m pytest \
  runtime/glm53-flash-jj-r8-gb10 \
  scripts/test_glm53_flash_profile.py \
  scripts/test_sparkring_generic_launcher.py -q
```

## GLM-5.3 adaptive MTP with live-tensor B12X KDA

Status: **implemented, not qualified**. The source-built runtime composition
uses these sanitized inputs:

| File | Role |
|---|---|
| `glm53-flash-b12x-kda-adaptive-mtp-tp4-site.example.yaml` | Four-rank TP4/DCP1 site and 20 GiB FP8 KV reservation per rank |
| `glm53-flash-b12x-kda-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json` | Adaptive MTP 3→5, 32-step window, fastsafetensors queue one, native SparkCache restore |

Resolve both templates with
`scripts/prepare_glm53_b12x_kda_adaptive_mtp_profile.py`. The resolver rejects
source, contract, image, loader, adaptive-policy, and cache-identity drift.
Follow the [executable quickstart](../../docs/GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md).

## DeepSeek-V4-Flash-0731

**Status: implemented.**

`deepseek-v4-flash-0731-pair.env.example` is both the host launch contract and
the container environment for the two-Spark profile. It is consumed by
`scripts/deepseek_v4_pair_serve.sh`, which validates model/cache host paths,
rank/rendezvous inputs, API and master ports, speculative depth, context,
sequence count, scheduler budget, and pair transport before rendering or
running Docker. Keep one resolved copy per rank outside version control.

`deepseek-v4-flash-0731.env.example` is the corresponding host launch contract
and container environment for the four-Spark cycle. It is consumed by
`scripts/deepseek_v4_cycle_serve.sh`. Copy it once per rank and resolve:

- `<NCCL_SOCKET_IFNAME>` with the rank's configured fabric interface;
- `<RANK_FABRIC_IP>` with that interface's address.

The file also records model/cache host paths, API and master ports,
speculative depth, context, sequence count, and scheduler budget. Keep each
rank's resolved environment file local. Both DeepSeek env files map
`CACHE_HOST_PATH` to `/cache`; their `/cache/jit` values are container paths
and must remain unchanged. DeepSeek SparkCache recipes inherit these
base-profile inputs.

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

The pair env also records host model/cache/log paths, API and master ports,
speculative depth, context, sequence count, and scheduler budget.
`CACHE_HOST_PATH` is mounted at `/ws/cache`; the `/ws/cache/jit` values are
container paths and must remain unchanged. `scripts/qwen38_dgx2_serve.sh`
validates and consumes the serving values inside the prepared runtime
container.

`qwen38-27b-exl3-k5k6.env.example` is the per-rank environment for the
implemented four-Spark Qwen profile. Copy it once per rank and replace:

- `<RENDEZVOUS_IFNAME>` with the rank's management interface;
- `<RANK_RENDEZVOUS_IP>` with that interface's address;
- `<NCCL_IB_HCA>` with the two cycle-facing RoCE devices in local ring order;
- `<NCCL_IB_GID_INDEX>` with the rank-global index of both devices' RoCEv2
  entries for their fabric IPv4 addresses.

The template combines the Qwen EXL3 graph/prefill settings with the
four-Spark patched-NCCL cycle. Management carries rendezvous and random worker
TCP ports; the RoCE devices carry collectives. The template assumes the
Qwen image built by `runtime/qwen38/build-image.sh` supplies the immutable
runtime under `/ws`. Host paths in the env select separately mounted model,
cache, and log directories; they do not attest model bytes or provide live
evidence. Follow
[`docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md`](../../docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md).

The cycle env uses the same host/cache/serving contract as the pair, with
four-rank defaults from `recipes/qwen38-27b-exl3-k5k6.json`.

## DeepSeek SIRCL research overlay

`deepseek-v4-flash-0731-sircl-research.env.example` is a second per-rank
environment file for the research-only width-4096 SIRCL graph overlay. It
does not replace the canonical environment. Apply it after the canonical file
only for an authorized matched A/B, and resolve both direct peer addresses
against the device named beside each address. The offline plan and evidence
requirements are documented in
[`docs/DEEPSEEK_V4_FLASH_SIRCL_AB.md`](../../docs/DEEPSEEK_V4_FLASH_SIRCL_AB.md).
The environment overlay is not sufficient by itself: both generated A/B
commands require the same six read-only, SHA-bound runtime mounts for the
vLLM capture overlays and SIRCL implementation. The control leaves their
activation variables unset.

## Direct-fabric image archive distribution

The rank and edge data in a validated four-rank site file can drive
`scripts/fanout_image_archive.py`. The utility downloads one checksum-bound
archive on a seed rank and forwards it through adjacent direct links. See
[`docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md`](../../docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md)
for planning, verification, create-only, and image-import behavior.

## Safety

Copying or validating a template is **OFFLINE**. Contacting configured ranks
without changing them is **READ-ONLY REMOTE**. Pulling an image, creating a
container, changing host files, or stopping a serving stack is **MUTATES HOST**
or **STOPS SERVING** and requires explicit authorization for the named hosts
and action.
