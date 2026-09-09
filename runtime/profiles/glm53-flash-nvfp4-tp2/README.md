# Original GLM-5.3 Flash NVFP4 on two Sparks

Status: **implemented**. This shared-image profile uses the model
settings of a bounded TP2/DCP1 text qualification. It requires an explicit
image identity and a matching verification receipt before creating or starting
a container. The measured source deployment does not qualify the shared
image's common sources, toolchain, or NCCL 2.30.7 binary.

Select this profile for `local-inference-lab/GLM-5.3-Flash-NVFP4`, recorded
revision prefix `520de24`. The separate
[`glm53-flash-spark-tp2`](../glm53-flash-spark-tp2/profile.json) profile selects
the NVFP4-Spark checkpoint at its own revision. Check the checkpoint identity
before launch; the `config.json` existence check does not authenticate shard
contents, and the recorded prefix is not a full checkpoint-content digest.

| Setting | Profile value |
|---|---|
| Nodes / parallelism | Two GB10 nodes, one GPU each; TP2/DCP1 |
| Transport | Both Socket Direct PCI functions of physical cage p0 |
| HCA inventory order | `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1` |
| RoCEnante peer map | Rank 0: `1=0/2`; rank 1: `0=0/2` |
| NCCL | Shared candidate: 2.30.7 with dual-domain routing, eight channels, `=rocep1s0f0,roceP2p1s0f0` |
| RoCEnante dispatch | Two paths; all-reduce limit 16 MiB; all-gather input-shard limit 16 MiB |
| Loader / speculation | Managed B12X; static MTP3; Humming draft MoE and B12X draft attention |
| Prefill | mHC sharding and recurrent-checkpoint coalescing enabled |
| KV / context | FP8, 6.75 GiB per node; maximum context 262144 |
| Scheduler | Eight sequences, 8192 batched tokens, prefill interval 8 |
| Graphs | `FULL_AND_PIECEWISE`, maximum capture 32, compilation mode 0 |
| Multimodal limits | Four images, zero videos; accuracy remains unqualified |
| SparkCache | Disabled; enabling it requires a separate composition profile |
| Host protection | 2 GiB memory-guard floor; explicit manual start; Docker restart `no` |

`profile.json` owns model arguments and feature switches. `rank.env.example`
contains only the host IP and two socket-interface settings. Resolve it for
each rank. Transport maps, model switches, and SparkCache cannot be overridden
through this site file. Confirm that the named HCA functions map to p0 on the
actual hardware before using this profile.

The measured reference deployment used NCCL 2.30.4 and a loader that
hardcoded managed allocation. The shared candidate selects
`/opt/sparkring/nccl-pci/libnccl.so.2.30.7` and passes
`--model-loader-extra-config '{"allocation":"managed"}'`. The shared loader
retains its `pinned_wc` default for profiles that omit that option. These
source and binary differences require new GPU qualification.
The TP2 profile sets `VLLM_GLM53_KDA_GATE_SIDE_STREAM=0` to preserve sequential
KDA projection scheduling; the common TP4 source keeps its default overlap.

## Prepare and inspect

The shared image must contain the selected
[transport bundle and startup hook](../../transport_profiles/README.md),
profile assets, and common source verifier. The
[source-image recipe](../../sparkring/source_image/README.md) prepares those
inputs from public source bases and repository files. The
[dependency matrix](dependencies.json) lists the common-source requirements.

For a local source-built image, first produce a CPU verification receipt from
the exact prepared context:

The [published shared image](../../sparkring/source_image/README.md#download-the-published-image)
can also be pulled and verified using this path. Prepare its matching source
context, pull the declared parent for verification, and set
`SPARKRING_LOCAL_IMAGE_ID` to the result of
`docker image inspect --format '{{.Id}}' "$SPARKRING_IMAGE"`.

```bash
python3 runtime/sparkring/source_image/verify_image.py \
  --image "$SPARKRING_LOCAL_IMAGE_ID" \
  --context /tmp/sparkring-glm-image-context \
  --profile glm53-flash-nvfp4-tp2-mtp3 \
  --output /srv/config/glm53-tp2-source-receipt.json
```

`SPARKRING_LOCAL_IMAGE_ID` is an exact `sha256:...` image config ID. This
`sparkring-source-image-receipt/v1` receipt is checked against the repository's
exact common source-lock bytes, complete installed-package/native witnesses,
the locked TP2 profile hashes, and the installed transport file-map witness.
It permits an explicit research launch; it is not a GPU qualification or a
registry publication record.

A published image may instead use an immutable
`ghcr.io/fujitsupolycom/sparkring@sha256:...` registry manifest digest with its
registry profile receipt. That receipt must bind `registry_digest` and the
`profiles.glm53-flash-nvfp4-tp2-mtp3` entry's `profile_sha256`,
`transport_manifest_sha256`, and recorded `source_compatibility: "passed"`.
The launcher does not construct passing compatibility evidence from launch
flags. A local-image receipt cannot stand in for a registry receipt.

Provide the original checkpoint directory and a separate writable compilation
cache directory. The launcher binds the cache to `/cache/jit`, then separates
artifacts by profile hash and rank. No external prompt/KV cache is enabled.

From the repository root, print a plan on each rank using resolved local paths:

```bash
python3 runtime/profiles/glm53-flash-nvfp4-tp2/launch.py plan \
  --rank 1 --master rank0.example \
  --model-dir /srv/models/GLM-5.3-Flash-NVFP4/rev-520de24 \
  --cache-dir /srv/cache/glm53-nvfp4 \
  --env-file /srv/config/glm53-rank1.env \
  --image "$SPARKRING_LOCAL_IMAGE_ID"
```

Use the same exact local image ID that the verifier recorded, or the
registry manifest digest from a matching publication receipt. The launcher
has no image default. The older
published Spark-checkpoint image is not a source-compatible selection for
this profile merely because it shares the repository name.

The existing host memory-guard service must be installed. The supplied
`memory-guard.conf` is its profile-specific systemd drop-in. The launcher
requires an active service whose effective command specifies the 2 GiB floor.
Installing the file and restarting systemd services are host administration
actions; this local profile preparation does neither.

## Create and start manually

Use `create` instead of `plan`, with the same arguments and
`--runtime-receipt /srv/config/glm53-tp2-source-receipt.json`, to create a
stopped container. The command fails on an existing name. It neither replaces
that container nor starts it.

Use `start` with the same arguments and receipt to start an existing container.
Start rank 1, then rank 0. Before starting, the launcher verifies the stopped
container's image reference, command, environment, labels, model/cache mounts,
and disabled restart policy. Both lifecycle actions require the memory guard
and refuse to proceed while a GPU container is running on that host. Stop a
serving pair explicitly before switching profiles.

The container enters through
`python3 -S -B /opt/sparkcache-jj-runtime/verify_sources.py --serve ...`.
After verifying common source and profile assets, the verifier dispatches the
selected TP2 profile to its transport entrypoint with normal Python startup.
The profile clears inherited `PYTHONPATH`; the installed `.pth` hook adds the
TP2 transport selector without activating the parent image's TP4 mesh hooks.
The transport entrypoint verifies its active source selection before importing
vLLM, and the `.pth` hook applies in spawned workers.

The TP2 launcher passes `--no-healthcheck` because the parent image's Docker
healthcheck requires a TP4 readiness marker that this entrypoint does not
create. TP2 has no TP4 startup-admission gate. Check API health and run a
semantic smoke request manually before using the server.

Rank 0 serves `GLM-5.3-Flash-NVFP4` on port 8000. This profile binds all
interfaces and supplies no API authentication; expose it through a trusted
network or authenticated gateway. There is no automatic model startup after
boot and no automatic restart after a guard stop.

## Evidence and limits

**Conditions.** The measured configuration used NCCL 2.30.4, two GB10 nodes, the original checkpoint,
TP2/DCP1, and the exact source archives in `dependencies.json`. The two rank
image IDs differed while their checked source hashes matched. Both physical
cables remained connected; measured serving traffic used only both p0
functions, with zero p1 traffic.

**Measurement.** Two passes measured C1/C8 decode at context 0/8192 with
512-token output limits and 15-second windows after warmup, plus 8K/32K/64K
prefill scouts. Prefill is client prompt tokens divided by TTFT; decode is
continuous-usage output tokens divided by the measured window. The
[evidence record](../../../performance/records/glm53-flash/tp2-single-dac-source-20260908.md)
links the per-pass observations and states the harness revision and gates.

**Result.** Warm single-DAC prefill
was 2311/2176/2283 tok/s, compared with 2300/2175/2272 for the two-DAC control.
Two-pass mean decode differed by at most 2.8%. Requests completed without
benchmark errors or detected loops; the semantic smoke returned `FINAL=42`
and both guards stayed active.

**Conclusion.** These results support short-run parity for the measured text workloads.

**Limitations.** They do not establish statistical equivalence, long-duration stability,
multimodal accuracy, physical cable-removal behavior, or a maximum usable
context/concurrency capacity. A shared image with different common vLLM/B12X
sources needs its own source checks and serving qualification.
