# Original GLM-5.3 Flash NVFP4 on two Sparks

Status: **implemented**. This profile reproduces the settings of a bounded
TP2/DCP1 text qualification. It requires an explicit shared-image digest and
passing source compatibility checks for that image before creating or starting
a container. The measured source deployment does not qualify an assembled
shared image.

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
| NCCL | 2.30.4, eight channels, `=rocep1s0f0,roceP2p1s0f0` |
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

## Prepare and inspect

The shared image must contain the selected
[transport bundle and startup hook](../../transport_profiles/README.md).
Its runtime receipt must identify the image with `registry_digest` and include
a `profiles.glm53-flash-nvfp4-tp2-mtp3` entry with the SHA-256 of `profile.json`,
`transport_manifest_sha256` from that profile, and
`source_compatibility: "passed"`. Those fields record source compatibility;
they do not claim a serving performance result. The
[dependency matrix](dependencies.json) lists the required common-source checks.

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
  --image "$SPARKRING_IMAGE_DIGEST"
```

`SPARKRING_IMAGE_DIGEST` must be an immutable
`ghcr.io/fujitsupolycom/sparkring@sha256:...` reference for the compatible
shared image. The launcher deliberately has no image default. The older
published Spark-checkpoint image is not a source-compatible selection for
this profile merely because it shares the repository name.

The existing host memory-guard service must be installed. The supplied
`memory-guard.conf` is its profile-specific systemd drop-in. The launcher
requires an active service whose effective command specifies the 2 GiB floor.
Installing the file and restarting systemd services are host administration
actions; this local profile preparation does neither.

## Create and start manually

Use `create` instead of `plan`, with the same arguments and
`--runtime-receipt /srv/config/sparkring-runtime-receipt.json`, to create a
stopped container. The command fails on an existing name. It neither replaces
that container nor starts it.

Use `start` with the same arguments and receipt to start an existing container.
Start rank 1, then rank 0. Before starting, the launcher verifies the stopped
container's image reference, command, environment, labels, model/cache mounts,
and disabled restart policy. Both lifecycle actions require the memory guard
and refuse to proceed while a GPU container is running on that host. Stop a
serving pair explicitly before switching profiles.

Rank 0 serves `GLM-5.3-Flash-NVFP4` on port 8000. This profile binds all
interfaces and supplies no API authentication; expose it through a trusted
network or authenticated gateway. There is no automatic model startup after
boot and no automatic restart after a guard stop.

## Evidence and limits

**Conditions.** The measured configuration used two GB10 nodes, the original checkpoint,
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
