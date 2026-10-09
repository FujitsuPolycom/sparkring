# libsircl in the installer

libsircl is a C11 shared library that implements the public C interface of
NVIDIA NCCL 2.32 on SIRCL's ring-session wire protocol. Its ELF SONAME is
`libnccl.so.2`, so a program written against NCCL's C API loads it in place
of NCCL. It is an independent implementation: it is not NVIDIA NCCL and
contains no NCCL implementation source. Its own documents describe the
library: [README](../../spark_transport/libsircl/README.md),
[STATUS](../../spark_transport/libsircl/STATUS.md) and
[RUNBOOK](../../spark_transport/libsircl/RUNBOOK.md).

This page describes how SparkRing carries libsircl: where its source lives,
the image layer that installs it, how vLLM loads it, the installer transport
`--transport libsircl`, and a stock-image option that runs an unmodified
ARM64 vLLM image with libsircl. Status of every part: **research-only**. No
serving A/B on DGX Sparks has measured a deployment that uses libsircl; the
installer prints that status in every plan that selects it.

| Part | Path | Status |
|---|---|---|
| Source of record | [`spark_transport/libsircl/`](../../spark_transport/libsircl/README.md) | implemented |
| vLLM general plugin `libsircl` | [`integrations/vllm/libsircl/`](../../integrations/vllm/libsircl/README.md) | research-only |
| Image layer `installer-libsircl-layer` | [`runtime/images/libsircl_layer.py`](../../runtime/images/libsircl_layer.py) | research-only; no image built |
| Installer transport `libsircl` | [`runtime/common/libsircl.py`](../../runtime/common/libsircl.py) | research-only |
| Stock-image option | [`runtime/common/stock_image.py`](../../runtime/common/stock_image.py), [`runtime/host/stock_install.py`](../../runtime/host/stock_install.py) | research-only; plans and checks only |

## Source placement

`spark_transport/libsircl/` is libsircl's source of record: the library is
developed here, and git commits are its provenance. An image, a lock or a
receipt names the source a library was built from by the git tree id of
`spark_transport/libsircl` at that commit (`source_tree`, as
`git rev-parse HEAD:spark_transport/libsircl` prints it). The library
(version 0.6.0) has the fail-stop mode ([Fail-stop](#fail-stop)), the ring
schedules from 8 MiB on a communicator whose ring closes over cables (the
cycle plan), SIRCL 0.3.1's native sources, four kernel packs and SIRCL's
point-to-point channels between two ranks of a larger communicator
(`LIBSIRCL_P2P_CHANNELS=on`, off by default). Its route planner
(`tools/site_routes.py`) also prints each rank's point-to-point windows
(`LIBSIRCL_P2P_WINDOWS`, `SIRCL_P2P_CHUNK_BYTES`).

The directory's history here starts from libsircl snapshot `e31abc5c`, a
directory of the library's files that libsircl's development workspace
wrote, identified by its tree digest (the SHA-256 of its `FILES.sha256` list,
`e31abc5ca510f592cd0e2d895d2134a70f26625fe23fe75cf2dffcde3c447cf1`): the
commit "Vendor libsircl snapshot e31abc5c" holds its files except its run
logs (`verification/`) and its change requests to SIRCL's package
(`requests/`), whose texts name that workspace's directories. STATUS.md
names the snapshot each recorded run used by the first eight digits of its
tree digest; the run logs it cites stay with the workstation that made them.
Evidence: GPU emulation on one RTX 5090 workstation and runs of earlier
snapshots on Sparks (cabled pairs, a path of four and the cycle of eight);
the cycle plan has not run on Sparks. The installer's libsircl transport
(`runtime/common/libsircl.py`) carries six routing variables per rank and
refuses a planner row with others. For a group of four consecutive Sparks (a
path of four, or four positions of the ring of eight) this snapshot's planner
adds the two point-to-point variables, so the transport refuses that group;
a pair and the whole ring of eight plan as before.

`.gitattributes` keeps the directory's bytes unconverted on every platform.
The library carries two native sources of SIRCL byte for byte
(`src/transport/sircl_roce_proxy.c`, `src/transport/sircl_p2p_proxy.c`); its
Makefile and CMake build refuse a copy whose SHA-256 differs from the one
recorded beside it. CI's `libsircl` job builds the library and runs its CPU
checks (`make check`, the route planner's tests against this repository's
SIRCL package included). The release-safety scan
(`scripts/check_release_safety.py`) reports a local Windows user path in any
tracked file.

The repository's [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
(section 18) and [NOTICE](../../NOTICE) name libsircl's components: its
Apache-2.0 `LICENSE` and `NOTICE`, the NCCL 2.32.3 header copy and NCCL's
licence in `vendor/`, SIRCL's notice in `vendor/SIRCL-NOTICE`, the rdma-core
header code under the OpenIB.org BSD licence and the prebuilt kernel packs,
whose object code is under NVIDIA's CUDA Toolkit End User License Agreement
(`LICENSES/`).

## Image layer

`installer-libsircl-layer` ([libsircl_layer.py](../../runtime/images/libsircl_layer.py))
adds libsircl to a kraken-line image with a `sparkring-installer-image/v3`
lock, normally the SIRCL image
`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034`, and writes the
derived image's v3 lock. It stacks on that image as a separate layer instead
of extending the SIRCL layer:

- libsircl has its own version and release cadence. A separate layer
  rebuilds only its own files; the SIRCL wheel, native libraries, receipts and
  the SIRCL layer builder's v3 lock stay byte for byte those of the parent.
- One image then carries the prepared transport, SIRCL ring sessions and
  libsircl over the same vLLM, B12X, CUDA and NCCL bytes, so a serving A/B
  between them differs only in the transport.
- A Spark that holds the SIRCL image loads only the libsircl layer
  ([layer_delta.py](../../runtime/images/layer_delta.py)).
- libsircl compiles SIRCL's native proxy into itself; it shares no file with
  the SIRCL layer.

The layer adds:

| Image path | Content |
|---|---|
| `/opt/sparkring/libsircl/lib/libsircl.so.<version>` | the library, built from the committed source |
| `/opt/sparkring/libsircl/LICENSE`, `NOTICE`, `vendor/NCCL-LICENSE.txt`, `vendor/SIRCL-NOTICE`, `LICENSES/CUDA-NOTICE.txt`, `LICENSES/rdma-core-verbs.txt` | the notices every binary copy carries |
| `<site-packages>/sparkring_libsircl.py` and `sparkring_libsircl-<version>.dist-info/` | the vLLM general plugin `libsircl` |
| `/opt/sparkring/receipts/libsircl-layer.json` | the layer receipt, `sparkring-libsircl-layer/v1` |

`<site-packages>` is the serving interpreter's directory where the parent's
external-base receipt records `vllm/__init__.py`, as for the SIRCL layer. No
`libnccl.so.2` link is created: vLLM loads the library by its path.

Actions, none of which pushes or publishes an image:

1. `natives --parent-lock LOCK --output DIR` reads every file git tracks
   under `spark_transport/libsircl` at `HEAD` from git's objects (it refuses
   to start while tracked files there have changes not committed), copies
   them into a network-less container of the parent
   image at the fixed path `/tmp/libsircl`, runs `make -j BUILD=build` and
   `make check BUILD=build` there with `LD_PRELOAD` unset, and saves the
   library, the check log and `natives.json`
   (`sparkring-libsircl-natives/v1`: parent image, source tree id, version,
   library SHA-256, compiler, kernel pack SHA-256 values). The build uses the
   prebuilt kernel packs, so it needs the image's gcc, make, Python 3 and
   rdma-core headers, and no nvcc. The library's debug information names the
   build directory, so the fixed path makes its bytes depend only on the
   image's compiler and the source: two x86_64 builds of snapshot `ba5a337b` at one path
   gave identical bytes, and a build at another path did not.
2. `prepare --parent-lock LOCK --natives DIR --output CONTEXT` reads the
   parent's external-base and toolchain receipts (copies, or the local parent
   image), refuses a parent that already records an added path, records every
   added file in the external-base receipt (so the image's `verify` checks
   its bytes) with `capabilities.libsircl` naming the layer receipt,
   re-records the toolchain receipt, and writes the Docker context and
   `plan.json`.
3. `build` tags the parent, builds the context and runs `record`; `record`
   confirms that the parent has none of the added paths, probes the built
   image (the plugin's entry point is registered, the library loads, reports
   NCCL API level 22705 and identifies itself through `sirclGetInfo`, the
   plugin selects it, and every function the image's PyNccl binds resolves
   in it), checks the layer receipt against the lock the way installation
   does, runs the installer's admission for every profile, and writes the v3
   lock. The probe ran against an x86_64 build of `ba5a337b` and a stand-in
   site-packages on a workstation: all 20 functions of PyNccl's list in Local
   Inference Lab's `integration/karmic-kraken-beta` at `57a80980bb`
   resolved.

The lock is the parent's v3 lock with the built image's identity, the two
re-recorded receipt digests, `libsircl` added to `transports` and a
`libsircl` block ([image_lock.py](../../runtime/common/image_lock.py)):
version, the source's git tree id (`source_tree`), library path and SHA-256, NCCL API level,
whether the library has the fail-stop mode (its bytes name
`LIBSIRCL_FAIL_STOP`), plugin module path and SHA-256, and layer receipt
path and SHA-256. A lock of a layer built while this repository vendored
libsircl snapshots names the snapshot's tree digest (`snapshot`) in place of
`source_tree`; image `27e9f75c0d09`'s locks
([record](../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009.md))
name snapshot `a3477af2` that way, and admission compares the field the lock
names with the layer receipt's. A v3 lock without libsircl has no `libsircl`
field, so every existing lock validates unchanged.

## How vLLM loads libsircl

vLLM's PyNccl (`vllm/distributed/device_communicators/pynccl_wrapper.py`)
loads its NCCL with `ctypes.CDLL(path)` where `path` is
`vllm.utils.nccl.find_nccl_library()`: `VLLM_NCCL_SO_PATH` when set, else
`libnccl.so.2`. It binds every function of its list and fails on a missing
one, except functions it marks optional; when that load or binding fails,
PyNccl disables itself and vLLM's device communicator falls back to torch's
own NCCL. `vllm.envs` reads variables when they are used, until
`enable_envs_cache()` after a worker's `init_device`. Verified in the source
of Local Inference Lab's `integration/karmic-kraken-beta` at `57a80980bb`, the
branch the image's vLLM (`d51b4181`, `bc9ea774`) merges; the image's own files
were not read, and the CSF merge's
[source manifest](../../runtime/images/compositions/kraken-csf-sources-20261007/sources.json)
does not change `pynccl_wrapper.py` or `vllm/utils/nccl.py`.

The installer image's verified entrypoint,
`/opt/sparkring/toolchain/toolchain.py serve`
([toolchain_runtime.py](../../runtime/images/toolchain_runtime.py),
`configure_environment`), sets `VLLM_NCCL_SO_PATH` to the image's NVIDIA
NCCL 2.32.3, `/opt/sparkring/toolchain/nccl/lib/libnccl.so.2`, removes every
`LD_PRELOAD` entry whose file name contains `libnccl.so` and puts that NCCL
first in `LD_PRELOAD` before vLLM starts. A container variable therefore
cannot select libsircl in that image. The layer's vLLM general plugin does
([sparkring_libsircl.py](../../integrations/vllm/libsircl/sparkring_libsircl.py)):
with `libsircl` in `VLLM_PLUGINS`, vLLM calls its `register` in every
process before that process creates a PyNccl communicator (workers load
general plugins in `init_worker`, before `init_device`). `register` checks
the file `SPARKRING_LIBSIRCL_LIBRARY` against `SPARKRING_LIBSIRCL_SHA256`,
loads it as PyNccl will and requires it to identify itself as libsircl
(`sirclGetInfo`) and to export every function PyNccl binds, then sets
`VLLM_NCCL_SO_PATH` to it. Any failure ends the process, so PyNccl cannot
silently fall back to torch's NCCL for a load or binding reason.

libsircl links with `-Bsymbolic-functions` and exports only `nccl*`,
`pnccl*` and its three `sircl*` functions, so loading it by path beside the
preloaded NVIDIA NCCL keeps its own calls inside it; libsircl's hardware runs
on a cabled pair loaded it that way in this image's parent
(STATUS.md, hardware evidence).

### PyTorch's ProcessGroupNCCL

- Inferred, not verified: the installer image's PyTorch binds NCCL
  dynamically as `libnccl.so.2`. The toolchain layer selects one NCCL "for
  both PyTorch and vLLM" with `LD_PRELOAD` and an `nvidia/nccl/lib` alias in
  its Python search namespace, and its GPU smoke requires exactly one mapped
  `libnccl`; neither would be needed for a statically linked NCCL. The
  `NEEDED` entries of the image's `libtorch_cuda.so` were not read. The layer
  builder's probe records `torch.cuda.nccl.version()` in the built image
  (`torch_nccl_version` of `record`'s summary): 2.32.3 means torch bound the
  image's NVIDIA NCCL. The stock-image probe reads the `NEEDED` entries
  (below).
- Consequence in the installer image: torch's collectives cannot be pointed
  at libsircl without changing the sealed toolchain entrypoint, which forces
  NVIDIA NCCL first in `LD_PRELOAD`. Only vLLM's PyNccl uses libsircl. The
  collectives vLLM sends through PyNccl (all-reduce, all-gather,
  reduce-scatter, broadcast and point-to-point of its device communicators)
  run on libsircl; a `torch.distributed` collective on a vLLM device group,
  such as `GroupCoordinator.broadcast` of a GPU tensor, runs on NVIDIA NCCL.
  On a pair or a whole cycle NVIDIA NCCL can connect the ranks; on a path it
  cannot connect ranks that share no cable, and the plan says so. The
  transport refuses the settings known to issue such collectives (below),
  and the containers log NCCL's initialization (`NCCL_DEBUG=INFO`,
  `NCCL_DEBUG_SUBSYS=INIT`), so a communicator NVIDIA NCCL creates is visible
  in the model log.
- In a stock image whose entrypoint does not rewrite loader settings and whose
  PyTorch needs `libnccl.so.2`, `LD_PRELOAD` of libsircl redirects torch's
  ProcessGroupNCCL too: the dynamic linker resolves the `libnccl.so.2`
  dependency to the preloaded object with that SONAME. libsircl's evidence for
  this is GPU emulation with torch 2.10.0+cu128 (STATUS.md, "PyTorch
  ProcessGroupNCCL, unmodified, through LD_PRELOAD"), not a Spark. The
  stock-image preflight checks which library each caller bound (below).

## Installer transport

`sudo sparkring install --profile PROFILE --transport libsircl` (and
`sparkring up PROFILE --transport libsircl`) deploys an installer profile
with vLLM's PyNccl on libsircl. The transport is never the default; it is
chosen by name. [libsircl.py](../../runtime/common/libsircl.py) owns it;
[transport.py](../../runtime/common/transport.py) passes `libsircl` sections
to it.

It runs where all of these hold, and refuses with the reason otherwise:

- the image lock lists `libsircl` among its transports, and its library has
  the fail-stop mode (below);
- `sudo sparkring setup` recorded a fabric document that lists `sircl` among
  its transports (the relay table is installed where the fabric has relays)
  and whose Sparks name their fabric devices alike;
- the group is one SIRCL's group placement accepts: two to eight consecutive
  Sparks of the fabric, or the whole cycle, with no lane through more than
  three relays (SIRCL's qualified limit);
- the profile's decode-context parallelism is 1;
- `--nccl` is not given (it applies to SIRCL deployments).

Supported shapes, from libsircl's STATUS: a communicator of 1 to 8 ranks;
pairs; paths, whose ends reach each other through relays, with the routing
settings of `tools/site_routes.py`; cycles of up to eight, the whole ring
included. On Sparks only a cabled pair has run (positions 0-1); paths and
cycles ran in GPU emulation and SIRCL's emulation harness.

### Fail-stop

vLLM's PyNccl checks only that each call was queued. A libsircl wait that
times out poisons its communicator and can let the step whose output it
spoiled complete; the error surfaces at a later call, or, in a replayed CUDA
graph, at none. The transport therefore sets `LIBSIRCL_FAIL_STOP=1`, with
which the library ends the process on a recorded asynchronous error, and
requires a library that has the mode: the layer records `fail_stop` in the
lock's `libsircl` block when the library's bytes name `LIBSIRCL_FAIL_STOP`,
and the stock-image preflight reads the same mark from the host library. The
library reads the variable (`src/engine.c`), so a layer or host library built
from this repository's source passes the gate; a library built from a source
without the mode is refused, naming the reason.

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
| `LIBSIRCL_POSITION`, `SIRCL_PEER_ROUTES`, `LIBSIRCL_CHAIN_ORDER`, `LIBSIRCL_FORWARD_WINDOWS`, `SIRCL_FORWARD_CHUNK_BYTES`, `LIBSIRCL_RING_WINDOW` | the rank's routing settings, recorded per rank in the deployment lock |
| `SIRCL_BOOTSTRAP_ADDR` | the rank's `VLLM_HOST_IP`, the address it publishes in the unique id of a communicator it roots |
| `SIRCL_GID_INDEX` | the profile's `NCCL_IB_GID_INDEX`, else 3, as for SIRCL deployments |
| `LIBSIRCL_RECEIPT` | `/run/sparkring/sircl/receipts/libsircl`: one receipt file per communicator in the deployment's receipt directory on each Spark |
| `VLLM_DISABLE_PYNCCL`, `VLLM_ALLREDUCE_USE_SYMM_MEM`, `VLLM_USE_NCCL_SYMM_MEM`, `VLLM_ALLREDUCE_USE_FLASHINFER`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, `VLLM_ENABLE_PCIE_ALLREDUCE` | `0`, replacing the profile's: PyNccl on; torch and NCCL symmetric memory off, and with NCCL's the optional `ncclMemAlloc` allocator, whose callers ignore its errors; FlashInfer all-reduce and B12X PCIe all-reduce off |
| `VLLM_ENABLE_ROCE_ALLREDUCE`, `SPARKRING_TRANSPORT_PROFILE`, `SPARKRING_TRANSPORT_MANIFEST_SHA256`, `SPARK_TP4_ENABLED`, `VLLM_SPARK_TP4_MODE`, `VLLM_SPARK_TP4_VOCAB_MODE` | SIRCL's `DISABLED_TRANSPORTS`: RoCEnante and SIRCL's four-rank adapter off |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | `INFO`, `INIT` |
| vLLM arguments | `--disable-custom-all-reduce` added: vLLM's custom all-reduce, which covers NVLink and PCIe peer copies, stays off |

The routing settings come from libsircl's own tool,
[`tools/site_routes.py`](../../spark_transport/libsircl/tools/site_routes.py),
which runs SIRCL's route planner (`sparkring_sircl.routes`). The installer
places the group with its existing route code (`transport.group_topology`,
SIRCL's `describe_group`), passes the group's explicit layout
(`cables=...;positions=...`) and lane count to the tool with
`SIRCL_FABRIC_DOCUMENT` naming a copy of the fabric document, so the route
maps use the device names setup discovered, and records the tool's lines in
the lock's `transport.routes`. Rank `r` of the group is libsircl position `r`.

### Refusals

A profile is refused, naming each reason, when:

- its collectives would bypass PyNccl and reach torch's NCCL, or libsircl's
  communicators would lose their one issuing order: SIRCL's
  `nccl_free_problems` (`--load-format instanttensor`, `--enable-eplb`,
  `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`), SIRCL's `relay_conflicts`
  (GEMM-communication and all-reduce-RMSNorm fusion passes,
  `--enable-batch-sharded-sampling`, all-to-all backends that connect every
  pair of ranks themselves) and vLLM micro-batching;
- it needs what libsircl does not carry (`libsircl.capability_problems`):
  `--enable-sleep-mode` (libsircl exports `ncclCommSuspend` and
  `ncclCommResume` only as refusals); pipeline parallelism above 2
  (the transport leaves libsircl's point-to-point channels off, so
  point-to-point runs only between the two ranks of a two-rank
  communicator); data parallelism; expert parallelism with an all-to-all
  backend other than `allgather_reducescatter`; and the sequence-parallelism
  pass (`pass_config.enable_sp`);
- it sets a variable the adapter owns or any `LIBSIRCL_*` variable.

### Lock, admission and receipts

The deployment lock's `transport` section (`sparkring-transport/v1`,
`backend: libsircl`) records the status, the image, the fabric document's
identity, the group, each rank's RDMA devices, each rank's routing settings,
the tool's inputs and the image's `libsircl` block; the lock's identity
covers it, so another fabric, group or library is another deployment. The
installation admits the image's libsircl layer on every Spark: the image's
external-base receipt must record the layer receipt, the library and the
plugin with the SHA-256 the lock names, and each Spark's fabric document must
have the deployment's identity. After the model answers, the installer does
not judge libsircl's receipts; the result's transport verdict is `unknown`
with that reason, and `sudo sparkring check` reports the same.

The plan says what runs and that it is research-only, for example:

```text
Transport: libsircl (research-only): vLLM's PyNccl carries its collectives on libsircl 0.6.0; SIRCL's adapter and RoCEnante are off
  libsircl group: path-4 at positions 4, 5, 6, 7; 2 lanes per peer, at most 2 relays on a lane
  Library: /opt/sparkring/libsircl/lib/libsircl.so.0.6.0, SHA-256 0123456789ab, source tree 3e51d70dd67f; fail-stop on (LIBSIRCL_FAIL_STOP=1)
  Off: vLLM's custom all-reduce, torch and NCCL symmetric memory, FlashInfer all-reduce and B12X PCIe all-reduce, so PyNccl carries the device collectives
  Research-only: no serving A/B has measured libsircl; torch.distributed's own collectives stay on the image's NCCL
  Note: torch.distributed's NVIDIA NCCL cannot connect this group's ranks that share no cable; a collective vLLM sends through torch instead of PyNccl would wait at NCCL's connection setup
```

## Stock-image option

```bash
sudo sparkring install --image REF --transport libsircl --libsircl-library LIBRARY \
  --model-path PATH [--on POSITIONS] --plan -- VLLM_ARGUMENTS
```

plans a deployment of an unmodified ARM64 vLLM image, such as
`ghcr.io/spark-arena/dgx-vllm-eugr-nightly` or
`eugr/spark-vllm-b12x:nightly-20261001`, with libsircl as the NCCL of both
vLLM's PyNccl and torch's ProcessGroupNCCL. `REF` is a registry reference or
a local image ID (it contains `/` or `:`); it is not an installer image
release, carries no SparkRing receipts and is not pinned by SparkRing.
Status: **research-only**. The option plans and checks; it never pulls the
image, and the installer does not start, stop or switch a stock deployment:
it writes each rank's Compose file and prints the commands that copy, start
and stop them ([stock_install.py](../../runtime/host/stock_install.py)).
Without `--plan` it refuses and says so; it takes no `--profile`.

The plan needs on every Spark of the group:

- the image, loaded locally (`--pull never` everywhere);
- the host build of libsircl at `LIBRARY`, the same bytes on every Spark:
  `libsircl_layer.py host-library --builder-image ID --output DIR` builds it
  from the committed source in a network-less container of a builder image
  that has gcc, make, Python 3 and the rdma-core headers (an installer image
  has them) and names its content-addressed path,
  `/var/lib/sparkring/libsircl/<sha256>/libsircl.so.<version>`; the stock
  image needs no build tools. The builder's glibc bounds the images the
  library runs in: an x86_64 build of snapshot `ba5a337b` with GCC 13.3 on
  Ubuntu 24.04 (glibc 2.39) needs `GLIBC_2.38`, so an image with an older
  glibc needs a library built with that glibc or an older one;
- the model files at `PATH`, mounted read-only at `/model`.

The group, its routing settings and its refusals of shapes are those of the
[installer transport](#installer-transport): pairs, paths and whole cycles of
two to eight Sparks, so a pair and the eight-Spark cycle (TP8 through relays)
are planned the same way.

### Preflight

On every Spark, over SSH and read-only,
[stock_image.py](../../runtime/common/stock_image.py) inspects the image,
hashes the library and runs a standard-library probe
([stock_image_probe.py](../../runtime/common/stock_image_probe.py)) in a
network-less, read-only container of the image with the GPUs and the library
mounted as serving mounts them. It refuses, naming each reason and Spark,
when any check fails:

| Check | Rule |
|---|---|
| Architecture | the image is `arm64` and its interpreter runs on `aarch64` |
| Library | it loads in the image, its SONAME is `libnccl.so.2`, `ncclGetVersion` reports 22705, `sirclGetInfo` names libsircl, and its bytes name `LIBSIRCL_FAIL_STOP` (the [fail-stop](#fail-stop) mode) |
| glibc | the image's glibc (`os.confstr("CS_GNU_LIBC_VERSION")`) is at least the newest `GLIBC_x.y` symbol version the library needs (its ELF `.gnu.version_r`) |
| CUDA driver API | `cuDriverGetVersion` through the container's `libcuda.so.1` is 13000 or later: the kernel packs hold `sm_120` and `sm_121` code built by nvcc 13.3 and no PTX, which a CUDA 13 driver loads (CUDA's minor-version compatibility; inferred) |
| GPU architecture | the container sees a GPU, and every visible GPU's compute capability is one the kernel packs carry (`sm_120`, `sm_121`; GB10 is 12.1) |
| `VLLM_NCCL_SO_PATH` | `vllm/envs.py` defines it and `find_nccl_library` reads it (read from the installed vLLM's sources, not by importing vLLM) |
| PyNccl's functions | every function vLLM's `NCCLLibrary` binds resolves in the library; none is one libsircl exports only as a refusal ([tests/api_manifest.json](../../spark_transport/libsircl/tests/api_manifest.json)), except `ncclCommSuspend` and `ncclCommResume`, which only sleep mode calls and which `--enable-sleep-mode`'s refusal covers |
| Multi-node launch | the CLI accepts `--nnodes`, `--node-rank`, `--master-addr`, `--master-port` and `--headless`, and the image has a `vllm` command |
| torch's NCCL | torch's `libtorch_cuda.so` needs `libnccl.so.2` (`dynamic`). A `static` or absent NCCL is refused: `LD_PRELOAD` cannot redirect it, and torch's own collectives would reach another NCCL |
| Arguments | `VLLM_ARGUMENTS` set none of the options the plan owns and pass the installer transport's [refusals](#refusals) |
| Same bytes | every Spark holds the same image ID and library SHA-256 |

When those pass, a second probe on every Spark runs with the planned
`VLLM_NCCL_SO_PATH` and `LD_PRELOAD`, imports torch and reports which
library each caller bound; it refuses unless PyNccl's `ctypes` load of
`VLLM_NCCL_SO_PATH` and the process's global scope both resolve
`ncclGetVersion` in libsircl (`dladdr`), `torch.cuda.nccl.version()` is
2.27.5 (libsircl's API level 22705), and libsircl is the only mapped file
with the SONAME `libnccl.so.2`. This verifies the binding, not a collective:
a qualification on Sparks still runs one PyNccl and one ProcessGroupNCCL
collective on every rank and requires each to add a libsircl receipt.

### Each rank's container

- the image's own entrypoint, followed by `vllm serve /model`
  `VLLM_ARGUMENTS`, the group's multi-node arguments
  (`--tensor-parallel-size` and `--nnodes` equal to the group size,
  `--node-rank`, `--master-addr` rank 0's address, `--master-port 29511`,
  `--headless` above rank 0, `--host 0.0.0.0 --port 8000` on rank 0),
  `--disable-custom-all-reduce` and, unless `VLLM_ARGUMENTS` give their own,
  `--compilation-config` setting the communication fusions the image's vLLM
  has (`fuse_allreduce_rms`, `fuse_gemm_comms`, `enable_sp`) to false;
- the library mounted read-only at
  `/opt/sparkring/libsircl/lib/libsircl.so.<version>`, `VLLM_NCCL_SO_PATH`
  naming it, and `LD_PRELOAD` with libsircl first and no other `libnccl.so`
  entry of the image's own `LD_PRELOAD`;
- `LIBSIRCL_FAIL_STOP=1`, `LIBSIRCL_NCCL_API_VERSION=22705`,
  `LIBSIRCL_TRANSPORT=verbs`, the rank's routing settings,
  `SIRCL_BOOTSTRAP_ADDR` and `VLLM_HOST_IP` the rank's address,
  `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` its interface,
  `SIRCL_GID_INDEX=3`, the receipt prefix in `receipts/` of the plan's
  directory, and `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT`;
- `0` for each switch of a bypassing collective path that the image's
  `vllm/envs.py` defines (`VLLM_DISABLE_PYNCCL`, symmetric memory, FlashInfer
  all-reduce, PCIe all-reduce, `VLLM_ENABLE_ROCE_ALLREDUCE`); switches the
  image does not define are not set;
- `/dev/infiniband`, all GPUs, host networking and IPC, unlimited memlock and
  `IPC_LOCK`, and no other added capability;
- SparkRing's loader seccomp policy (io_uring admitted) only when the image's
  loader needs it: `--load-format fastsafetensors` or `instanttensor`, or the
  image registers B12X's `b12x_loader` plugin.

The files land in `/var/lib/sparkring/controller/stock/<name>/` on Node A
(`compose.rank<r>.yaml`, `plan.json` and, where needed, `loader-seccomp.json`),
with the commands that copy them to `/var/lib/sparkring/stock/<name>/` on each
Spark and start and stop the containers. `<name>` is derived from the image
ID, library digest, positions, model path, arguments and fabric identity.

## Build and load commands

The commands run on one Spark (`BUILD`) as an account in the `docker` group,
from a copy of the repository at the revision that holds this page, with the
SIRCL image `dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034`
loaded and its v3 lock at `$W/sircl-lock.json`
([SIRCL release recipe](../../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md)).
They push nothing.

```bash
W=~/sparkring-image; cd $W/sparkring
PARENT_LOCK=$W/sircl-lock.json
PARENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' $PARENT_LOCK)
RELEASE=dev-20261008-kraken-csf-sircl-libsircl-cuda1342-nccl2323-status034

# 1. The libsircl source the layer builds: the git tree id it records.
git rev-parse HEAD:spark_transport/libsircl && git status --porcelain --untracked-files=no spark_transport/libsircl

# 2. Build libsircl in the parent image. Expect libsircl.so.0.6.0, make check passed and the image's gcc line.
python3 runtime/images/libsircl_layer.py natives --parent-lock $PARENT_LOCK --output $W/libsircl-natives

# 3. Context. Expect the library, six notices, the plugin's six files and the layer receipt.
python3 runtime/images/libsircl_layer.py prepare --parent-lock $PARENT_LOCK \
  --natives $W/libsircl-natives --output $W/libsircl-context

# 4. Build, probe and admit; writes the v3 lock. Expect "libsircl": "0.6.0", "nccl_api_version": 22705,
#    "fail_stop": true.
python3 runtime/images/libsircl_layer.py build --context $W/libsircl-context \
  --tag sparkring-dev/kraken:csf-sircl-libsircl-20261008 --name $RELEASE --output $W/libsircl-lock.json

# 5. Delta archive: the image without the parent's layers.
python3 runtime/images/layer_delta.py --image sparkring-dev/kraken:csf-sircl-libsircl-20261008 \
  --parent $PARENT --output $W/libsircl-delta.tar
```

Load on each other Spark that holds the parent image, from a machine that
reaches them:

```bash
ssh BUILD 'cat ~/sparkring-image/libsircl-delta.tar' | ssh HOST "sudo -n docker image inspect $PARENT >/dev/null \
  && sudo -n docker load && sudo -n docker image inspect --format '{{.Id}}' sparkring-dev/kraken:csf-sircl-libsircl-20261008"
```

Plan a deployment with the lock: `sudo sparkring install --profile PROFILE
--on 0,1 --image-lock $W/libsircl-lock.json --transport libsircl --plan`.
