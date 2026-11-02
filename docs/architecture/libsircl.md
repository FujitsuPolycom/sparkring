# libsircl in the installer

libsircl is a C11 shared library that implements the public C interface of
NVIDIA NCCL 2.32 on SIRCL's ring-session wire protocol. Its ELF SONAME is
`libnccl.so.2`, so a program written against NCCL's C API can load it in
place of NCCL. It is an independent implementation: it is not NVIDIA NCCL and
contains no NCCL implementation source. The library's own
[README](../../spark_transport/libsircl/README.md),
[STATUS](../../spark_transport/libsircl/STATUS.md) and
[RUNBOOK](../../spark_transport/libsircl/RUNBOOK.md) describe it.

This page covers how SparkRing carries libsircl: its source, the image layer
that installs it, how vLLM loads it, the installer transport
`--transport libsircl`, and an option that plans an unmodified ARM64 vLLM
image with libsircl. Every part except the source is **research-only**: no
serving A/B on DGX Sparks has measured a deployment that uses libsircl, and
every plan that selects it says so.

| Part | Path | Status |
|---|---|---|
| Library source | [`spark_transport/libsircl/`](../../spark_transport/libsircl/README.md) | implemented |
| vLLM general plugin `libsircl` | [`integrations/vllm/libsircl/`](../../integrations/vllm/libsircl/README.md) | research-only |
| Image layer `installer-libsircl-layer` | [`runtime/images/libsircl_layer.py`](../../runtime/images/libsircl_layer.py) | research-only |
| Installer transport `libsircl` | [`runtime/common/libsircl.py`](../../runtime/common/libsircl.py) | research-only |
| Stock-image option | [`runtime/common/stock_image.py`](../../runtime/common/stock_image.py), [`runtime/host/stock_install.py`](../../runtime/host/stock_install.py) | research-only; plans and checks only |

## Source placement

`spark_transport/libsircl/` is libsircl's source of record. Images, locks and
receipts name the source a library was built from by the git tree id of that
directory (`source_tree`, as `git rev-parse HEAD:spark_transport/libsircl`
prints it). Before the directory became the source of record it held copies
of the library written by a separate development workspace. Each copy, a
*snapshot*, is named by the first eight digits of the SHA-256 of its file
list; STATUS.md names the snapshot each recorded run used.

The library, version 0.6.0 ([`VERSION`](../../spark_transport/libsircl/VERSION)), has:

- a fail-stop mode ([Fail-stop](#fail-stop));
- four kernel packs (transport, fold, link and point-to-point), compiled for
  `sm_120` and `sm_121` by its build;
- ring schedules from 8 MiB on a communicator whose ring closes over cables
  (the *cycle plan*);
- SIRCL's point-to-point channels between ranks of a communicator of three or
  more, off by default (`LIBSIRCL_P2P_CHANNELS=on`);
- a route planner, [`tools/site_routes.py`](../../spark_transport/libsircl/tools/site_routes.py),
  which prints each rank's routing settings.

The library carries two native sources of SIRCL byte for byte
(`src/transport/sircl_roce_proxy.c`, `src/transport/sircl_p2p_proxy.c`). Its
Makefile and CMake build refuse a copy whose SHA-256 differs from the one
recorded beside it, and `.gitattributes` keeps the directory's bytes
unconverted on every platform. CI's `libsircl` job builds the library and
runs its CPU checks, including the route planner's tests against this
repository's SIRCL package.

Evidence for the library:

- STATUS.md: GPU emulation on one RTX 5090 workstation, and snapshots on
  Sparks: cabled pairs at positions 0-1 and 6-7, the path of four at positions
  4-7 and the cycle of eight.
- The library of image `1a8c10354eb0` (source tree `dbf36074`), with the cycle
  plan, passed libsircl's gate on the cycle of eight
  ([record](../../performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md))
  and on two rings of four
  ([record](../../performance/records/transport/libsircl-ring4-image-1a8c10354eb0-20261010.md)).
  STATUS.md changes only with the library's source, so it does not list these
  runs.
- The library of tree `030419b8` passed the same gates in release 2026.10.2's
  requalification. It differs from tree `dbf36074` in one change: a
  communicator created on a thread with no current CUDA context uses the
  current device's primary context, which vLLM's PyNccl needs
  ([record](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010.md)).

libsircl is under the Apache License 2.0. Its third-party components (the
NCCL header copy, SIRCL's notice, rdma-core header code and the CUDA object
code of built kernel packs) are listed in section 18 of
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md); the source holds no
compiled pack.

## Image layer

`installer-libsircl-layer` ([libsircl_layer.py](../../runtime/images/libsircl_layer.py))
adds libsircl to an installer image that has a `sparkring-installer-image/v3`
lock, normally one with SIRCL's layer, and writes the derived image's v3 lock.
It is a separate layer so that libsircl rebuilds only its own files, a Spark
that holds the parent loads only the difference
([layer_delta.py](../../runtime/images/layer_delta.py)), and one image carries
SIRCL and libsircl over the same vLLM, CUDA and NCCL bytes.

Release 2026.10.2's image (`d52737a109e0`) has this layer, and the images of
release 2026.10.4 (`3cc9a81fd344`) and the installer's default release
2026.10.5 (`f625053225d4`) inherit it unchanged: libsircl 0.6.0 from source
tree `030419b8`, library SHA-256 `ffdba5b24a2b…`, with the fail-stop mode
([2026.10.5 lock](../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/installer-image.json)).

The layer adds:

| Image path | Content |
|---|---|
| `/opt/sparkring/libsircl/lib/libsircl.so.<version>` | the library, built from the committed source |
| `/opt/sparkring/libsircl/LICENSE`, `NOTICE`, `vendor/NCCL-LICENSE.txt`, `vendor/SIRCL-NOTICE`, `LICENSES/CUDA-NOTICE.txt`, `LICENSES/rdma-core-verbs.txt` | the notices every binary copy carries |
| `<site-packages>/sparkring_libsircl.py` and `sparkring_libsircl-<version>.dist-info/` | the vLLM general plugin `libsircl` |
| `/opt/sparkring/receipts/libsircl-layer.json` | the layer receipt, `sparkring-libsircl-layer/v1` |

No `libnccl.so.2` link is created; vLLM loads the library by its path.

Actions; none pushes or publishes an image:

| Action | What it does |
|---|---|
| `natives --parent-lock LOCK --output DIR` | Builds the committed source (it refuses uncommitted changes under `spark_transport/libsircl`) with `make` and `make check` in a network-less container of the parent, at the fixed path `/tmp/libsircl` so the bytes depend only on the compiler and source. Writes the library, kernel packs, check log and `natives.json` with every SHA-256, the compiler and the nvcc. |
| `prepare --parent-lock LOCK --natives DIR --output CONTEXT` | Records every added file in the parent's external-base receipt, so the image's `verify` checks it, and writes the Docker context. |
| `build --context CONTEXT --tag TAG --name NAME --output LOCK` | Builds the context, then runs `record`: probes the image (plugin registered, library loads, reports NCCL API level 22705, identifies itself and exports every function PyNccl binds), checks the layer as installation does, admits every profile and writes the v3 lock. |
| `host-library --builder-image ID --output DIR` | The same build in another local image, for the [stock-image option](#stock-image-option). |

Installer images carry CUDA 13.4, so their kernel packs can differ in bytes
from the CUDA 13.3 packs of STATUS.md's emulation evidence; `natives.json`
names the packs each image embeds.

The v3 lock adds `libsircl` to `transports` and a `libsircl` block
([image_lock.py](../../runtime/common/image_lock.py)): version, `source_tree`,
library path and SHA-256, NCCL API level, `fail_stop`, plugin path and
SHA-256, and layer receipt path and SHA-256. Locks of layers built from a
snapshot name it as `snapshot` in place of `source_tree` (for example image
`27e9f75c0d09`,
[record](../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009.md));
admission compares whichever field the lock names. A v3 lock without
libsircl has no `libsircl` field.

## How vLLM loads libsircl

vLLM's PyNccl loads its NCCL with `ctypes.CDLL(path)`, where `path` is
`VLLM_NCCL_SO_PATH` when set, else `libnccl.so.2`. If that load or binding
fails, PyNccl disables itself and vLLM falls back to torch's NCCL.

The installer image's entrypoint
([toolchain_runtime.py](../../runtime/images/toolchain_runtime.py),
`configure_environment`) sets `VLLM_NCCL_SO_PATH` to the image's NVIDIA NCCL
2.32.3 and puts that NCCL first in `LD_PRELOAD`, so a container variable
cannot select libsircl. The layer's vLLM general plugin does
([sparkring_libsircl.py](../../integrations/vllm/libsircl/sparkring_libsircl.py)).
With `libsircl` in `VLLM_PLUGINS`, vLLM calls its `register` in every process
before that process creates a PyNccl communicator. `register`:

1. checks the file `SPARKRING_LIBSIRCL_LIBRARY` against
   `SPARKRING_LIBSIRCL_SHA256`;
2. loads it as PyNccl will and requires it to identify itself as libsircl
   (`sirclGetInfo`) and to export every function PyNccl binds;
3. sets `VLLM_NCCL_SO_PATH` to it.

Any failure ends the process, so PyNccl cannot silently fall back to torch's
NCCL.

libsircl links with `-Bsymbolic-functions` and exports only `nccl*`,
`pnccl*` and three `sircl*` functions
([exports.map](../../spark_transport/libsircl/exports.map)), so loaded by path
beside the preloaded NVIDIA NCCL it keeps its own calls inside itself.

### PyTorch's ProcessGroupNCCL

- **In the installer image**, only vLLM's PyNccl uses libsircl. The
  collectives vLLM sends through PyNccl (all-reduce, all-gather,
  reduce-scatter, broadcast and point-to-point of its device communicators)
  run on libsircl; a `torch.distributed` collective on a vLLM device group
  runs on NVIDIA NCCL. Pointing torch at libsircl would mean changing the
  sealed toolchain entrypoint. NVIDIA NCCL can connect the ranks of a pair or
  a whole cycle, but not ranks of a path that share no cable; the plan says
  so. The transport refuses the settings known to issue such collectives
  ([Refusals](#refusals)), and the containers log NCCL's initialization, so a
  communicator that NVIDIA NCCL creates is visible in the model log.
- **In a stock image** whose PyTorch needs `libnccl.so.2` and whose entrypoint
  does not rewrite loader settings, `LD_PRELOAD` of libsircl redirects torch's
  ProcessGroupNCCL too, because the dynamic linker resolves that dependency to
  the preloaded object with the same SONAME. The evidence is GPU emulation
  with torch 2.10.0 (STATUS.md, "PyTorch ProcessGroupNCCL, unmodified, through
  LD_PRELOAD"), not Sparks. The stock-image [preflight](#preflight) checks
  which library each caller bound.

## Installer transport

`sudo sparkring install --profile PROFILE --transport libsircl` (or
`sparkring up PROFILE --transport libsircl`) deploys an installer profile with
vLLM's PyNccl on libsircl. The transport is never the default; it is chosen by
name. [libsircl.py](../../runtime/common/libsircl.py) implements it.

It runs only where all of these hold, and otherwise refuses with the reason:

- the image lock lists `libsircl` among its transports, and its library has
  the fail-stop mode;
- `sudo sparkring setup` recorded a fabric document that lists `sircl` (the
  relay table is installed where the fabric has relays) and whose Sparks name
  their fabric devices alike;
- the group is one SIRCL's group placement accepts: two to eight consecutive
  Sparks, or the whole cycle, with no lane through more than three relays;
- the profile's decode-context parallelism is 1;
- `--nccl` is not given (it applies to SIRCL deployments).

Evidence on Sparks: `qwen38-flash-next-tp2 --transport libsircl` installed on
a pair of each ring of four with release 2026.10.2's image, answered three
known-answer requests and passed `sparkring check`'s functional checks
([requalification](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010.md)).
No larger group has run through this transport.

Each rank's routing settings come from libsircl's
[`tools/site_routes.py`](../../spark_transport/libsircl/tools/site_routes.py),
which runs SIRCL's route planner on the group's explicit layout with
`SIRCL_FABRIC_DOCUMENT` naming a copy of the fabric document, so the route
maps use the device names setup discovered. Rank `r` of the group is libsircl
position `r`. The transport carries the eight routing variables the planner
prints (`ROUTE_VARIABLES`) and refuses a planner row with any other. A pair's
rows hold four of them (no forward or point-to-point windows); a group of four
consecutive Sparks holds all eight, with point-to-point windows toward peers
two or three positions away; the whole cycle of eight holds six, without
point-to-point windows.

### Fail-stop

vLLM's PyNccl checks only that each call was queued. A libsircl wait that
times out poisons its communicator, and the step whose output it spoiled can
still complete; the error surfaces at a later call or, in a replayed CUDA
graph, at none. The transport therefore sets `LIBSIRCL_FAIL_STOP=1`, with
which the library ends the process on a recorded asynchronous error, and
refuses a library without the mode. The layer records `fail_stop` in the lock
when the library's bytes name `LIBSIRCL_FAIL_STOP`; the stock-image preflight
reads the same mark from the host library.

### Container settings

Each rank's container gets the profile's settings with these changes
([`libsircl.adapt`](../../runtime/common/libsircl.py)):

| Settings | Values |
|---|---|
| `VLLM_PLUGINS` | the image's plugins and `libsircl`, never `sircl` |
| `SPARKRING_LIBSIRCL_LIBRARY`, `SPARKRING_LIBSIRCL_SHA256` | the lock's library path and SHA-256, which the plugin checks |
| `LIBSIRCL_FAIL_STOP` | `1` |
| `LIBSIRCL_NCCL_API_VERSION` | `22705`, libsircl's default: it passes vLLM's and torch's feature gates without opening newer ones |
| `LIBSIRCL_TRANSPORT` | `verbs` |
| `LIBSIRCL_POSITION`, `SIRCL_PEER_ROUTES`, `LIBSIRCL_CHAIN_ORDER`, `LIBSIRCL_FORWARD_WINDOWS`, `SIRCL_FORWARD_CHUNK_BYTES`, `LIBSIRCL_RING_WINDOW`, `LIBSIRCL_P2P_WINDOWS`, `SIRCL_P2P_CHUNK_BYTES` | the rank's routing settings, recorded per rank in the deployment lock |
| `SIRCL_BOOTSTRAP_ADDR` | the rank's `VLLM_HOST_IP` |
| `SIRCL_GID_INDEX` | the profile's `NCCL_IB_GID_INDEX`, else 3 |
| `LIBSIRCL_RECEIPT` | `/run/sparkring/sircl/receipts/libsircl`: one receipt per communicator in the deployment's receipt directory on each Spark |
| `VLLM_DISABLE_PYNCCL`, `VLLM_ALLREDUCE_USE_SYMM_MEM`, `VLLM_USE_NCCL_SYMM_MEM`, `VLLM_ALLREDUCE_USE_FLASHINFER`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, `VLLM_ENABLE_PCIE_ALLREDUCE` | `0`, replacing the profile's: PyNccl on; torch and NCCL symmetric memory (and NCCL's optional `ncclMemAlloc` allocator), FlashInfer all-reduce and B12X PCIe all-reduce off |
| `VLLM_ENABLE_ROCE_ALLREDUCE`, `SPARKRING_TRANSPORT_PROFILE`, `SPARKRING_TRANSPORT_MANIFEST_SHA256`, `SPARK_TP4_ENABLED`, `VLLM_SPARK_TP4_MODE`, `VLLM_SPARK_TP4_VOCAB_MODE` | SIRCL's `DISABLED_TRANSPORTS`: the prepared RoCE transport and SIRCL's four-rank adapter off |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | `INFO`, `INIT` |
| vLLM arguments | `--disable-custom-all-reduce` added |

### Refusals

A profile is refused, naming each reason, when:

- its collectives would bypass PyNccl and reach torch's NCCL, or libsircl's
  communicators would lose their single issuing order: SIRCL's
  `nccl_free_problems` (`--load-format instanttensor`, `--enable-eplb`,
  `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`), SIRCL's `relay_conflicts`
  (GEMM-communication and all-reduce-RMSNorm fusion passes,
  `--enable-batch-sharded-sampling`, all-to-all backends that connect every
  pair of ranks themselves) and vLLM micro-batching;
- it needs what libsircl does not carry (`libsircl.capability_problems`):
  `--enable-sleep-mode` (libsircl exports `ncclCommSuspend` and
  `ncclCommResume` only as refusals); pipeline parallelism above 2
  (point-to-point runs only between the two ranks of a two-rank communicator,
  because the transport leaves libsircl's point-to-point channels off); data
  parallelism; expert parallelism with an all-to-all backend other than
  `allgather_reducescatter`; and the sequence-parallelism pass
  (`pass_config.enable_sp`);
- it sets decode-context parallelism, a variable the adapter owns, or any
  `LIBSIRCL_*` variable.

### Lock, admission and receipts

The deployment lock's `transport` section (`sparkring-transport/v1`,
`backend: libsircl`) records the status, the image, the fabric document's
identity, the group, each rank's RDMA devices and routing settings, the route
planner's inputs and the image's `libsircl` block. The lock's identity covers
the section, so another fabric, group or library is another deployment.

Installation admits the image's libsircl layer on every Spark: the image's
external-base receipt must record the layer receipt, the library and the
plugin with the SHA-256 the lock names, and each Spark's fabric document must
have the deployment's identity. The installer does not judge libsircl's
receipts: the result's transport verdict is `unknown`, and
`sudo sparkring check` reports the same.

The plan states what runs and that it is research-only. For the default
image on a path of four:

```text
Transport: libsircl (research-only): vLLM's PyNccl carries its collectives on libsircl 0.6.0; SIRCL's adapter and RoCEnante are off
  libsircl group: path-4 at positions 4, 5, 6, 7; 2 lanes per peer, at most 2 relays on a lane
  Library: /opt/sparkring/libsircl/lib/libsircl.so.0.6.0, SHA-256 ffdba5b24a2b, source tree 030419b8a2b6; fail-stop on (LIBSIRCL_FAIL_STOP=1)
  Off: vLLM's custom all-reduce, torch and NCCL symmetric memory, FlashInfer all-reduce and B12X PCIe all-reduce, so PyNccl carries the device collectives
  Research-only: no serving A/B has measured libsircl; torch.distributed's own collectives stay on the image's NCCL
  Note: torch.distributed's NVIDIA NCCL cannot connect this group's ranks that share no cable; a collective vLLM sends through torch instead of PyNccl would wait at NCCL's connection setup
```

## Stock-image option

```bash
sudo sparkring install --image REF --transport libsircl --libsircl-library LIBRARY \
  --model-path PATH [--on POSITIONS] --plan -- VLLM_ARGUMENTS
```

This plans a deployment of an unmodified ARM64 vLLM image with libsircl as the
NCCL of both vLLM's PyNccl and torch's ProcessGroupNCCL. `REF` is a registry
reference or local image ID (it contains `/` or `:`), such as
`eugr/spark-vllm-b12x:nightly-20261001`. It is not an installer release:
it carries no SparkRing receipts, and SparkRing does not pin or pull it.

Status: **research-only**; no stock-image deployment has run on Sparks. The
option only plans and checks. It writes each rank's Compose file and prints
the commands that copy, start and stop them; the installer itself never
starts, stops or switches a stock deployment, and `sparkring status` does not
list it. Without `--plan` it refuses, and it takes no `--profile`.

Every Spark of the group needs:

- the image, loaded locally (the plan uses `--pull never`);
- the host build of libsircl at `LIBRARY`, the same bytes on every Spark.
  `libsircl_layer.py host-library --builder-image ID --output DIR` builds it
  from the committed source in a network-less container of a builder image
  with nvcc (CUDA 13.3 or later), gcc, make, Python 3 and the rdma-core
  headers; an installer image has them. It names the install path
  `/var/lib/sparkring/libsircl/<sha256>/libsircl.so.<version>`. The builder's
  glibc must be no newer than the stock image's;
- the model files at `PATH`, mounted read-only at `/model`.

Group placement, routing settings and shape refusals are those of the
[installer transport](#installer-transport): pairs, paths and whole cycles of
two to eight Sparks.

### Preflight

On every Spark, over SSH and read-only,
[stock_image.py](../../runtime/common/stock_image.py) inspects the image,
hashes the library and runs a probe
([stock_image_probe.py](../../runtime/common/stock_image_probe.py)) in a
network-less, read-only container of the image with the GPUs and the library
mounted. It refuses, naming each reason and Spark, when any check fails:

| Check | Rule |
|---|---|
| Architecture | the image is `arm64` and its interpreter runs on `aarch64` |
| Library | it loads in the image, its SONAME is `libnccl.so.2`, `ncclGetVersion` reports 22705, `sirclGetInfo` names libsircl, and it has the [fail-stop](#fail-stop) mode |
| glibc | the image's glibc is at least the newest `GLIBC_x.y` version the library needs |
| CUDA driver API | 13000 or later: the kernel packs hold `sm_120` and `sm_121` code from a CUDA 13 nvcc and no PTX |
| GPU architecture | the container sees a GPU, and every visible GPU's compute capability is one the packs carry (GB10 is 12.1) |
| `VLLM_NCCL_SO_PATH` | the image's vLLM defines it and `find_nccl_library` reads it (read from the sources, not by importing vLLM) |
| PyNccl's functions | every function vLLM's `NCCLLibrary` binds resolves in the library, and none is one libsircl exports only as a refusal ([tests/api_manifest.json](../../spark_transport/libsircl/tests/api_manifest.json)), except `ncclCommSuspend` and `ncclCommResume`, which only sleep mode calls |
| Multi-node launch | the CLI accepts `--nnodes`, `--node-rank`, `--master-addr`, `--master-port` and `--headless`, and the image has a `vllm` command |
| torch's NCCL | torch's `libtorch_cuda.so` needs `libnccl.so.2`; a static or absent NCCL is refused, because `LD_PRELOAD` cannot redirect it |
| Arguments | `VLLM_ARGUMENTS` set none of the options the plan owns and pass the installer transport's [refusals](#refusals) |
| Same bytes | every Spark holds the same image ID and library SHA-256 |

When those pass, a second probe runs on every Spark with the planned
`VLLM_NCCL_SO_PATH` and `LD_PRELOAD`. It refuses unless PyNccl's load and
the process's global scope both resolve `ncclGetVersion` in libsircl,
`torch.cuda.nccl.version()` is 2.27.5 (libsircl's API level 22705), and
libsircl is the only mapped file with the SONAME `libnccl.so.2`. This verifies
the binding, not a collective.

### Each rank's container

- **Command:** the image's own entrypoint with `vllm serve /model`,
  `VLLM_ARGUMENTS`, the multi-node arguments (tensor-parallel size and
  `--nnodes` equal to the group size, `--master-port 29511`, `--headless`
  above rank 0, port 8000 on rank 0), `--disable-custom-all-reduce` and,
  unless `VLLM_ARGUMENTS` give their own, a `--compilation-config` that turns
  off the communication fusions the image's vLLM has.
- **Library:** mounted read-only at
  `/opt/sparkring/libsircl/lib/libsircl.so.<version>`, named by
  `VLLM_NCCL_SO_PATH`, and first in `LD_PRELOAD` with the image's other
  `libnccl.so` entries removed.
- **Settings:** the `LIBSIRCL_*`, routing, bootstrap, receipt and
  `NCCL_DEBUG` values of the installer transport's
  [container settings](#container-settings), with `SIRCL_GID_INDEX=3`; `0`
  for each bypassing collective switch the image's vLLM defines (others are
  not set); and `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME`
  for the rank's fabric address and interface.
- **Access:** `/dev/infiniband`, all GPUs, host networking and IPC, unlimited
  memlock and `IPC_LOCK`; SparkRing's loader seccomp policy only when the
  image's loader needs io_uring (`--load-format fastsafetensors` or
  `instanttensor`, or B12X's `b12x_loader` plugin).

The files (`compose.rank<r>.yaml`, `plan.json` and, where needed,
`loader-seccomp.json`) land in `/var/lib/sparkring/controller/stock/<name>/`
on Node A. The printed commands copy them to `/var/lib/sparkring/stock/<name>/`
on each Spark. `<name>` is `sr-stock-` and a digest of the image ID, library
digest, positions, model path, arguments and fabric identity.

## Build and load commands

These commands build a libsircl layer on one Spark (`BUILD`) as an account in
the `docker` group, from a checkout of this repository. They need a local
parent image with SIRCL's layer and its v3 lock at `$W/parent-lock.json` (for
example the [SIRCL image recipe](../../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md)).
`NAME` is the release name the lock records and `TAG` the local image tag.
They push nothing.

```bash
W=~/sparkring-image; cd $W/sparkring
PARENT_LOCK=$W/parent-lock.json
PARENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' $PARENT_LOCK)

# 1. The source the layer builds: its git tree id, and no uncommitted changes.
git rev-parse HEAD:spark_transport/libsircl && git status --porcelain --untracked-files=no spark_transport/libsircl

# 2. Build libsircl in the parent image. Expect libsircl.so.0.6.0 and make check passed.
python3 runtime/images/libsircl_layer.py natives --parent-lock $PARENT_LOCK --output $W/libsircl-natives

# 3. Context: the library, six notices, the plugin's six files and the layer receipt.
python3 runtime/images/libsircl_layer.py prepare --parent-lock $PARENT_LOCK \
  --natives $W/libsircl-natives --output $W/libsircl-context

# 4. Build, probe and admit; writes the v3 lock. Expect "nccl_api_version": 22705 and "fail_stop": true.
python3 runtime/images/libsircl_layer.py build --context $W/libsircl-context \
  --tag TAG --name NAME --output $W/libsircl-lock.json

# 5. Delta archive: the image without the parent's layers.
python3 runtime/images/layer_delta.py --image TAG --parent $PARENT --output $W/libsircl-delta.tar
```

Load it on each other Spark that holds the parent image, from a machine that
reaches them:

```bash
ssh BUILD 'cat ~/sparkring-image/libsircl-delta.tar' | ssh HOST "sudo -n docker image inspect $PARENT >/dev/null \
  && sudo -n docker load && sudo -n docker image inspect --format '{{.Id}}' TAG"
```

Plan a deployment with the lock: `sudo sparkring install --profile PROFILE
--on 0,1 --image-lock $W/libsircl-lock.json --transport libsircl --plan`.
