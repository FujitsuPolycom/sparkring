# GLM-5.3 Flash NVFP4-Spark on a switched fabric

Status: **Experimental**. Switched deployments are provided as-is. This
profile has not been validated on switched hardware.

This profile uses the common SparkRing source image with ordinary NCCL
collectives. It keeps mHC prefill sharding and recurrent-checkpoint coalescing
enabled. Custom SIRCL/RoCEnante collectives, switchless routing, compact index
ownership, mesh top-k ownership, and SparkCache are disabled.

| Setting | Value |
|---|---|
| Model | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision `df116c4fb16b1d37ae43d2cfd624de26ffbc832e` |
| Parallelism | Four nodes, one GPU each; TP4/DCP1 |
| Speculation / loader | Static MTP3; fastsafetensors target loader |
| Context / KV | 1048576 configured tokens; 24 GiB FP8 KV per node |
| Scheduler | 16 sequences, 8192 batched tokens, prefill interval 2 |
| Graphs | FULL_AND_PIECEWISE; maximum capture 64 |
| Communication | NCCL 2.30.7; exact HCA/port selection supplied per rank |
| API / liveness | Port 8000 / port 8001 on rank 0 |
| Lifecycle | Manual create/start, no automatic restart, minimum 4 GiB host-memory guard |
| Startup | Generic warmup and admission wrapper; one shape at C1 plus six sampler request recipes |

The source audit in [dependencies.json](dependencies.json) identifies the
common vLLM/B12X revisions and relevant file hashes. The CUDA communicator
constructs PyNccl for the TP group; with custom and symmetric-memory paths
disabled, ordinary all-reduce/all-gather/reduce-scatter use NCCL. mHC uses that
same PyNccl communicator directly. The coalescing validator admits TP4/DCP1,
the V2 runner, B12X KDA, aligned prefix caching, and static MTP3 without a
physical ring requirement.

The dependency field `nccl.audited_lock_sha256` contains the patched NCCL
library's SHA-256, matching `runtime.nccl_sha256` in the
[source lock](../../sparkring/source_image/glm53-tp4-lock.json).
The image receipt's `source_lock_sha256` identifies the complete lock.
Image verification checks the build lock, profile hash and installed file
witnesses; switched-hardware qualification remains required.

## Supply the switch-connected interfaces

Copy `rank.env.example` into a private file for each rank. Fill its five
settings from that host's real switch-port and RoCE configuration. The HCA
selector must begin with `=` and contain exact names, optionally with `:port`.
The launcher preserves order, case, and the number of selected functions.

Do not copy a four-node ring map into this file. A count of HCA functions is
not a count of physical uplinks; one or multiple uplinks are operator choices.
The launcher neither infers those connections nor changes interfaces, GIDs,
MTU, routes, or switch configuration. Its software `NCCL_ALGO=Ring` setting
does not enable physical switchless routing.

The extended-IPv4 and PCI-domain capability flags are enabled, while
`NCCL_SWITCHLESS_RING_ONLY` and `NCCL_IB_SUBNET_AWARE_ROUTING` remain zero.
In the [pinned NCCL patch](../../../spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch), extended-GID advertisement and PCI-root preference
run only inside the subnet-aware path. That path stays inactive here, leaving
ordinary L2 topology-selected connections unchanged. These flags neither
choose the operator's HCA list nor imply a number of physical uplinks.

Only the five site fields can be supplied through this file. Model and
transport feature switches remain in `profile.json`. Checkpoint and cache
directories are explicit CLI inputs. The cache directory's numeric owner
becomes the container UID/GID; create it as the intended serving user.

## Build and launch the common image

Use the [common source-image recipe](../../sparkring/source_image/README.md).
Its lock must include `glm53-flash-spark-tp4-switched-mtp3`, the exact profile
SHA-256, and this profile's installed assets. The same assembled image can
contain ring, mesh, direct-pair, and switched profiles; selection happens when
the container is created. Existing profiles keep their own settings.

The [switched quickstart](../../../docs/GLM53_SWITCHED_TP4_QUICKSTART.md)
shows source preparation, local receipt verification, and manual startup.
`plan` only prints Docker argv. `create` requires an exact image receipt and
active memory guard, creates a stopped container, and fails on an existing
name. `start` checks the stopped container against that plan. Start ranks
1–3, then rank 0. Stop another GPU serving stack explicitly before switching.

Local `sha256:...` image config IDs use the common
`sparkring-source-image-receipt/v1` verifier. Immutable registry manifest
digests use their matching registry profile receipt. The launcher checks the
exact common source-lock bytes, installed-package/native witnesses, profile
hash, and switched NCCL declaration. It does not manufacture a passing receipt
from feature flags.

## Readiness and health

The launcher sets `SOURCE_IMAGE_PROFILE` and enters the common source verifier.
The verifier then uses its existing `warmup_argv`. Initially empty
`PYTHONPATH` and an empty transport selector prevent inherited mesh hooks;
the generic wrapper adds only its startup-admission directory for the child.

Rank 0 waits for the API, runs the bounded shape warmup, and completes all six
existing sampler recipes: unfiltered, temperature, top-k, top-p, combined
top-k/top-p, and seeded combined sampling. The wrapper owns
`/tmp/sparkring-engine-ready`, and the explicit Docker healthcheck observes
that marker. The wrapper removes stale markers before startup and removes the
marker on exit or failure. Other ranks use the
same wrapper with the correct rank environment and `--headless`.

Wait for rank-zero readiness and check `/health` and `/liveness` before a
benchmark. Warmup establishes completion of those request recipes, not proof
that every future request shape has finished JIT compilation. The API binds
all interfaces; this profile supplies no API key, so use the site's access
controls or authenticated gateway.
