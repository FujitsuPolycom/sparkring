# Installer image builders

The `sparkring install` profiles run on development images. Each installer
image release under [`runtime/releases/`](../releases/README.md) is produced by
a builder in [builders.json](builders.json); the entry's `releases` field names
it. The [repository layout check](../../scripts/check_repository_layout.py)
requires a builder for every release that has an `installer-image.json` lock.

The default image, `dev-20261004-kraken-cuda1342-nccl2323-status034`, is built
in two layers over `eugr/spark-vllm-b12x` nightly-20261001: a software layer
that [external_context.py](external_context.py) prepares from SparkRing's merges
of Local Inference Lab's Karmic Kraken beta vLLM and B12X branches and from the
integration assets of `dev-20261001-statusrows-cuda1342-nccl2323-status034`,
then the CUDA 13.4.2 and NCCL 2.32.3 layer of
[toolchain_assembly.py](toolchain_assembly.py). Its
[composition record](compositions/external-kraken-20261004/README.md) lists the
source commits and merge decisions and refers to the rebuild steps of
`dev-20261001-kraken-cuda1342-nccl2323-status034`
([composition record](compositions/external-kraken-20261001/README.md)), the
rollback image, which is built the same way from other vLLM and B12X commits.
Neither image's `publication.json` has a `derivation`, so the install plan
counts each one's whole download.

The images of the chain below build on one another; each adds one layer to its
parent.

| Image | Parent | Layer | Builder |
|---|---|---|---|
| `dev-20260924-cuda1342-nccl2323-status031` | Software layer `sha256:01869ac593003b144416828b000b33febb626179dc901d14907ce6a202150373` | CUDA 13.4.2 and NCCL 2.32.3 | `cuda134-nccl232-assembly`, [toolchain_assembly.py](toolchain_assembly.py) |
| Intermediate, image ID `sha256:372ed77b81f724f4eeacb8ddb909f039567de4ce8b0d151e0a4897fe65fd345d`, no release record | `dev-20260924-cuda1342-nccl2323-status031` | Prepared RoCEnante proxy paces hardware-forwarded stripes within a send window | `installer-transport-window`, [derive_transport_window.py](derive_transport_window.py) |
| `dev-20260925-cuda1342-nccl2323-status031` | the intermediate image | Qwen HC token-row ownership on TP2 as well as TP4 | `installer-tp2-hc`, [derive_tp2_hc.py](derive_tp2_hc.py) |
| `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | `dev-20260925-cuda1342-nccl2323-status031` | Qwen4Exp decode GEMM plans and MXFP8 hyper-connection projections | `installer-qwen-decode`, [build_qwen_decode.py](../../performance/records/qwen38-flash-next/installer-tuning-20260925/programs/build_qwen_decode.py) |
| `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` | `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | vLLM host-to-device copies staged through fresh pinned memory | `installer-staging-fix`, [derive_staging_fix.py](derive_staging_fix.py) |
| `dev-20260927-b12xcache-cuda1342-nccl2323-status032` | `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` | [B12X reconciled selection-cache correction](../../integrations/b12x/selection_cache/README.md) and runtime-status 0.3.2 | `installer-derived-layer`, [derived_layer.py](derived_layer.py) with descriptor [installer-b12xcache-status032](compositions/installer-b12xcache-status032/descriptor.json) |
| `dev-20260927-mimovision-cuda1342-nccl2323-status032` | `dev-20260927-b12xcache-cuda1342-nccl2323-status032` | MiMo vision encoder attention sinks in the softmax denominator | `installer-mimo-vision`, [derive_mimo_vision.py](derive_mimo_vision.py) |
| `dev-20260928-peerwait-cuda1342-nccl2323-status032` | `dev-20260927-mimovision-cuda1342-nccl2323-status032` | Supervised RoCEnante peer waits: a late peer is waited for up to `B12X_ROCE_PEER_TIMEOUT_S`, and each stall is logged | `installer-transport-peer-wait`, [derive_transport_peer_wait.py](derive_transport_peer_wait.py) |
| `dev-20260928-toolchoice-cuda1342-nccl2323-status032` | `dev-20260928-peerwait-cuda1342-nccl2323-status032` | Named and required Chat Completions `tool_choice` requests without a complete call fail with HTTP 400 when the token limit ended generation and HTTP 500 otherwise ([tool-result contract](../../integrations/vllm/tool_choice_contract/README.md#installer-images)) | `installer-tool-choice-contract`, [derive_tool_choice_contract.py](derive_tool_choice_contract.py) |
| `dev-20260928-plainstatus-cuda1342-nccl2323-status033` | `dev-20260928-toolchoice-cuda1342-nccl2323-status032` | Runtime-status 0.3.3 | `installer-derived-layer`, [derived_layer.py](derived_layer.py) with descriptor [installer-plainstatus-status033](compositions/installer-plainstatus-status033/descriptor.json) |
| `dev-20260930-spinwait-cuda1342-nccl2323-status033` | `dev-20260928-plainstatus-cuda1342-nccl2323-status033` | vLLM's shared-memory readers poll for `SPARKRING_SHM_BUSY_LOOP_S` seconds after a read when it is set, one second otherwise | `installer-spin-wait`, [derive_spin_wait.py](derive_spin_wait.py) |
| `dev-20261001-portgid-cuda1342-nccl2323-status033` | `dev-20260930-spinwait-cuda1342-nccl2323-status033` | Each HCA of a RoCEnante runtime uses the RoCE GID index of its fabric address's RoCE v2 GID, read at startup ([GID index per port](../../integrations/vllm/rocenante_prepared/README.md#gid-index-per-port)); proxy ABI 6, transport manifest `d5e790c5173c` | `installer-transport-port-gid`, [derive_transport_port_gid.py](derive_transport_port_gid.py) |
| `dev-20261001-statusrows-cuda1342-nccl2323-status034` | `dev-20261001-portgid-cuda1342-nccl2323-status033` | Runtime-status 0.3.4, whose settings tables add the reasoning parser, tool-call parser, default chat template arguments and shared-memory reader window | `installer-derived-layer`, [derived_layer.py](derived_layer.py) with descriptor [installer-statusrows-status034](compositions/installer-statusrows-status034/descriptor.json) |

Each release's `publication.json` names its parent and describes its layer.
The install plan follows `derivation.parent_release` and
`derivation.parent_image_id` from the selected lock (`lineage` in
[install_space.py](../host/install_space.py)): a Spark holding any image of
that chain lacks only the layers added after it, and the plan counts them from
the difference of the two locks' `image_bytes` and `download_bytes`
([storage](../../docs/operations/install-reference.md#downloads-storage-and-outbound-hosts)).
The plan relies on each derived image keeping its parent's layers unchanged
and adding its own on top, as `docker build` on the parent image produces
them.
[cuda134-nccl232.md](cuda134-nccl232.md) documents the toolchain layer and the
inputs recorded for `dev-20260924`.

Rebuilding the software layer below `dev-20260924` from this repository is
**unsupported**. That layer composes the `eugr/spark-vllm-b12x:nightly-20260924`
base with pinned vLLM and B12X source archives, SparkRing integration assets
exported from the `shared-2026.09.4-rc.4` image, deployment add-ons and a
runtime-status 0.3.1 wheel built from
[integrations/vllm/runtime_status](../../integrations/vllm/runtime_status/README.md).
It was prepared by an earlier revision of
[external_context.py](external_context.py), and the Qwen prefill controller
version it packages is not in this repository. The layer's installed receipt has
SHA-256 `1354e3d0270b297dafafd39e5341210f4d0e7bd908b15b0026531c4e0350da62`
and composition SHA-256
`46ecfc99659799934e7429fb0f8c80310b93aee485337114139bb5fe1df096d4`, as the
release's installer image lock records.

## Capabilities

Some serving settings need what a layer adds. `--save-cpu` sets
`SPARKRING_SHM_BUSY_LOOP_S`, which only an image with the shared-memory reader
window reads.
[installer-capabilities.json](../releases/installer-capabilities.json)
(`sparkring-installer-capabilities/v1`) names, for each capability, the
releases whose own layer adds it and the file that describes that layer:
`dev-20260930-spinwait-cuda1342-nccl2323-status033` through
[derive_spin_wait.py](derive_spin_wait.py), and
`dev-20261001-kraken-cuda1342-nccl2323-status034` and
`dev-20261004-kraken-cuda1342-nccl2323-status034`, whose vLLM branches carry
the same edit, through their composition records.
`installer_image.capabilities` gives an image what its own layer adds and what
every image it derives from has, following `derivation.parent_release` in each
`publication.json`. `runtime/common/serving.py` `NEEDS` names the capability
each setting needs; the installer, `sparkring compose render` and the [Install
Builder](../../docs/operations/compose-builder.md) refuse the setting, or do
not offer it, on an image without it. A release whose own layer adds a
capability, such as an image built from new sources, is listed in the file.
So is an unpublished derived release, which has no `publication.json` to name
its parent: `dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034`
keeps the shared-memory reader of `dev-20261004-kraken-cuda1342-nccl2323-status034`,
as its [CSF-sources record](compositions/kraken-csf-sources-20261007/README.md)
states.

## Derived layers

Every layer above `dev-20260924` except the Qwen decode layer is built by
[derived_layer.py](derived_layer.py). The installer image lock
(`sparkring-installer-image/v2`) lists the profiles it admits and pins the
image configuration, the registry manifest, the composition, the prepared
transport manifest, the runtime-status version and the SHA-256 of two receipts
inside the image: the external-base receipt
(`/opt/sparkring/receipts/external-base-installed.json`), whose file map the
image's `verify` checks, and the toolchain receipt
(`/opt/sparkring/toolchain/installed.json`), which records the external-base
receipt. A derived layer copies its files over the parent, records each file in
the external-base receipt, re-records the toolchain receipt and writes a
provenance receipt under `/opt/sparkring/receipts/` that lists every path with
its inherited and resulting SHA-256 (`null` for an added file).

A layer is defined in one of two ways.

A `sparkring-derived-layer-descriptor/v1` **descriptor** names the parent
installer lock, each image path with its repository source, pinned SHA-256 and
inherited SHA-256 (`null` for an addition), the provenance receipt path and the
layer's purpose. Descriptor files are Python sources in the serving
interpreter's site-packages. The builder refuses native libraries, the startup
hooks that select the transport, feature and status packages, and the B12X
sources that the prepared RoCE transport verifies at startup; changing those
needs a new transport manifest or composition instead. The provenance receipt
of a descriptor layer also records the descriptor `id`, the parent release and
each file's repository source.

A **code layer** is a `Layer` in a `derive_*.py` module. It computes each
replaced file from the parent's own bytes, with exact substitutions that must
each match once, or from a repository bundle, and may pin each replaced file's
inherited and resulting SHA-256 and edit other receipt fields:
[derive_transport_window.py](derive_transport_window.py) installs
[integrations/vllm/rocenante_prepared](../../integrations/vllm/rocenante_prepared/README.md)
and records the new transport manifest in the receipt and the derived lock;
[derive_transport_peer_wait.py](derive_transport_peer_wait.py) installs the
same bundle's supervised peer wait over
`dev-20260927-mimovision-cuda1342-nccl2323-status032`, replacing exactly six
bundle files pinned to their parent and resulting SHA-256;
[derive_transport_port_gid.py](derive_transport_port_gid.py) installs the
bundle's RoCE GID index per HCA over
`dev-20260930-spinwait-cuda1342-nccl2323-status033` the same way, with four
pinned files;
[derive_tp2_hc.py](derive_tp2_hc.py) lists the TP2 row-sharding HC mode;
[derive_staging_fix.py](derive_staging_fix.py) pins `vllm/v1/utils.py`. Every
replaced path must already be recorded by the parent receipt, and its bytes in
the parent must match that record. A code layer may also add a site-packages
Python file under the descriptor rules: `pins` names the added path with
inherited SHA-256 `None`, and the parent receipt must not record it.
[derive_tool_choice_contract.py](derive_tool_choice_contract.py) adds the
tool-result policy as a vLLM module and pins the `serving.py` that installs it.
[derive_glm53_plugins.py](derive_glm53_plugins.py) adds the GLM-5.3 vLLM
general plugins `glm_dsa_indexer_split` and `glm53full_speedups` 1.1.0 and
`glm_dcp_decode_comm` 2.0.0 with their dist-info directories and names all
three in `Layer.plugins`.

The derived lock has its parent lock's schema. A parent with a
`sparkring-installer-image/v3` lock, such as the SIRCL or libsircl image, gives
a v3 lock that keeps the parent's image line, transports, default tuning table
digest and `sircl` and `libsircl` blocks unchanged, is not archived, and lists
the layer's plugins with the parent's in `vllm_plugins`. Its `record` also
requires that the built image registers each listed plugin at its version in
`vllm.general_plugins` and that the built image's external-base receipt still
covers the kept SIRCL and libsircl layers (`transport.check_layer`,
`libsircl.check_layer`), and runs admission through the lock's v2 fields. A v1
or v2 parent gives the same context and lock as before, without plugins:
`sparkring install` refuses a profile whose `VLLM_PLUGINS` names an added
plugin on an image whose lock does not list it
([image_lock.py](../common/image_lock.py), `plugin_problem`).

The peer-wait layer's build takes the parent lock and reads the parent's
installed bundle from the local parent image:

```bash
python3 runtime/images/derive_transport_peer_wait.py prepare \
  --parent-lock runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/installer-image.json \
  --output CONTEXT
```

`record` then writes a lock whose `transport_manifest_sha256` names the
installed manifest, `9f2c0ae62e1e` for the published layer; installer
containers export it as `SPARKRING_TRANSPORT_MANIFEST_SHA256`. Ranks of this image and of its parent refuse to
connect to each other (proxy ABI 5 and 4), so every Spark of a deployment must
run the same image. A transport layer reads its replacement files from the
repository bundle, so it prepares only from a revision whose bundle differs
from its parent in exactly its pinned files: the peer-wait layer from
`ce396adef06d5ff17a621465ba75d1c83830d7b0`, the port-GID layer from a revision
that holds its four resulting files.

The port-GID layer produces
`dev-20261001-portgid-cuda1342-nccl2323-status033`, in which each HCA of a
RoCEnante runtime uses the RoCE GID index of its fabric address's RoCE v2 GID,
read at startup, and the configured index only when its GID table does not
identify that GID
([GID index per port](../../integrations/vllm/rocenante_prepared/README.md#gid-index-per-port)).
Its build takes the parent lock:

```bash
python3 runtime/images/derive_transport_port_gid.py prepare \
  --parent-lock runtime/releases/dev-20260930-spinwait-cuda1342-nccl2323-status033/installer-image.json \
  --output CONTEXT
python3 runtime/images/derive_transport_port_gid.py build --context CONTEXT \
  --tag sparkring:portgid --name dev-20261001-portgid-cuda1342-nccl2323-status033 --output LOCK
```

`prepare` computes the installed manifest from the parent's, and `record`
writes its SHA-256 as the lock's `transport_manifest_sha256`, `d5e790c5173c`
for the published layer. Ranks of this image and of its parent refuse to
connect to each other (proxy ABI 6 and 5), so every Spark of a deployment must
run the same image. The `installer-transport-port-gid` entry of
[builders.json](builders.json) lists the release.

### Replacing the runtime-status package

A descriptor may also replace the runtime-status package with a
`runtime_status` entry that pins a pure wheel and its source archive by file
name and SHA-256; `prepare --status-artifacts DIRECTORY` reads both. The image
installs that package from its wheel into `/opt/sparkring/python`. The image's
`verify` requires the complete inventory of that directory to equal the
receipt's `python_roots` and the receipt's `files` entries under it, and
requires every path in `removed_files` to be absent. The installer's admission
requires the receipt's `capabilities.runtime_status.version` to equal the lock's
`status_version`, which must match `0.3.x`. The builder therefore admits the
wheel with the image preparer's rules (package modules equal to the source
archive's, official entry points, `fastapi>=0.115` as the only dependency,
consistent `METADATA`, `WHEEL` and `RECORD`), requires a `0.3.x` version other
than the parent's, and rewrites those receipt fields together: the parent
version's files leave `files` and `python_roots` and enter `removed_files`, and
`capabilities.runtime_status` records the wheel's version and the wheel and
source-archive SHA-256. The layer's Dockerfile removes the parent version's
`dist-info` directory before copying the wheel's files, and `record` writes the
wheel's version into the lock's `status_version`. `composition_sha256` keeps
naming the parent's composition; the provenance receipt records the replaced
status version and files. The `installer-b12xcache-status032` descriptor pins
the version 0.3.2 wheel and source archive, built from Git tree
`74407675db01502e57ad6131103a8bbdb3db3bd8` of
[integrations/vllm/runtime_status](../../integrations/vllm/runtime_status/README.md#building-the-image-artifacts),
whose README gives the build commands. The `installer-plainstatus-status033`
descriptor pins the version 0.3.3 wheel and source archive, built the same way
from Git tree `b82d56e0a8a5a04470fc679be9c7a665a7ab7fef`. The
[installer-statusrows-status034](compositions/installer-statusrows-status034/descriptor.json)
descriptor pins the version 0.3.4 wheel and source archive, built from Git tree
`899362a503a4ac2a85addf3aa9abc68d2e3f89bd` with archive time `1790816225`, over
`dev-20261001-portgid-cuda1342-nccl2323-status033`; the image built from it is
`dev-20261001-statusrows-cuda1342-nccl2323-status034`, which the
`installer-derived-layer` entry of [builders.json](builders.json) lists. Its
build follows the descriptor commands below, with `--status-artifacts` naming a
directory that holds both files.

### Commands

`prepare` checks the parent's two receipts against the parent lock and writes a
Docker context and `plan.json`; it never builds. A descriptor layer needs only
the receipts, copied from the parent image:

```bash
docker run --rm --pull never --network none --entrypoint cat PARENT_IMAGE_ID \
  /opt/sparkring/receipts/external-base-installed.json > base.json
docker run --rm --pull never --network none --entrypoint cat PARENT_IMAGE_ID \
  /opt/sparkring/toolchain/installed.json > toolchain.json
python3 runtime/images/derived_layer.py prepare \
  --descriptor runtime/images/compositions/DESCRIPTOR_DIRECTORY/descriptor.json \
  --base-receipt base.json --toolchain-receipt toolchain.json \
  [--status-artifacts STATUS_DIRECTORY] --output CONTEXT
```

A code layer takes the parent lock and reads the parent files it edits, through
a network-less container of the local parent image or, with `--parent-root`,
from an exported root filesystem (for example a `docker export` of a created
container, unpacked):

```bash
python3 runtime/images/derive_staging_fix.py prepare \
  --parent-lock runtime/releases/PARENT_RELEASE/installer-image.json --output CONTEXT
```

`build` tags the parent, builds the context and records the result; `record`
does the same for an image built separately with
`docker build --build-arg PARENT_IMAGE=LOCAL_PARENT_TAG -t TAG CONTEXT`
(BuildKit resolves a bare image ID as a registry name, so the parent needs a
local tag). Recording confirms that the parent has none of the added paths,
runs the installer's admission for every profile of the lock, including the
image's isolated `verify`, and writes the lock:

```bash
python3 runtime/images/derive_staging_fix.py build --context CONTEXT \
  --tag sparkring:TAG --name RELEASE --output LOCK
python3 runtime/images/derived_layer.py record --context CONTEXT \
  --image BUILT_IMAGE_ID --name RELEASE --output LOCK
```

`--profiles` replaces the parent lock's profile list; the
`dev-20260925-cuda1342-nccl2323-status031` lock lists the six profiles
`glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`,
`mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4`, `qwen38-flash-next-qad-tp4`
and `qwen38-flash-next-tp2`. The written lock is a development lock: its
`image_reference` is the local configuration ID and its `download_bytes` an
upper bound. Publication replaces both from the registry and adds the release's
`publication.json` and `release.json`
([release procedure](../../docs/development/releases.md)). No step selects the
image for a profile. `python3 scripts/build_image.py BUILDER -- ACTION ...`
prints the same commands from the catalog.

The Qwen decode builder is a program of the
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md#decode-image),
which describes its inputs; it builds from the files that `qwen_decode_patch.py`
writes and takes positional arguments.

## SIRCL layer

[sircl_layer.py](sircl_layer.py) (`installer-sircl-layer`) adds SIRCL ring
sessions to a kraken-line image with a v2 lock and writes a
`sparkring-installer-image/v3` lock
([release procedure](../../docs/development/releases.md)). Its layer holds:

- the package, installed in the serving interpreter's site-packages (the
  `site-packages` or `dist-packages` directory where the parent receipt
  records `vllm/__init__.py`; B12X's `b12x/integration/vllm` subpackage is
  not one) from a
  reproducible wheel `sparkring_sircl-<version>-py3-none-any.whl`: every
  Python, C, header and JSON file of `spark_transport/sircl/sparkring_sircl`,
  SparkRing's RoCE GID resolver as the top-level module `spark_roce_gid`, and a
  dist-info directory whose entry points register the `sircl` platform and
  general plugins. vLLM loads them only when a container's `VLLM_PLUGINS` names
  `sircl`, so deployments on the prepared transport are unchanged;
- `roce_proxy-<digest>.so` and `p2p_proxy-<digest>.so` in
  `/opt/sparkring/sircl/lib`, where `<digest>` is the first 16 hexadecimal
  digits of the SHA-256 of the C source. SIRCL's own build code compiles them
  in a network-less container of the parent image, against that image's glibc
  and `libibverbs.so.1`; no Spark compiles them;
- `/opt/sparkring/receipts/sircl-layer.json` (`sparkring-sircl-layer/v1`): the
  version, ABI, wheel, libraries, tuning key, compiler and every installed
  file with its SHA-256.

Every added file is recorded in the external-base receipt, so the image's
`verify` checks its bytes, and `sparkring install` admits the layer by
requiring that receipt to record the layer receipt and both libraries with the
SHA-256 the lock names. The receipt's `capabilities.sircl` names the layer
receipt.

```bash
python3 runtime/images/sircl_layer.py wheel --output WHEELS
python3 runtime/images/sircl_layer.py natives --parent-lock PARENT_LOCK \
  --wheel WHEELS/sparkring_sircl-0.3.2-py3-none-any.whl --output NATIVES
python3 runtime/images/sircl_layer.py prepare --parent-lock PARENT_LOCK \
  --wheel WHEELS/sparkring_sircl-0.3.2-py3-none-any.whl --natives NATIVES \
  [--base-receipt base.json --toolchain-receipt toolchain.json] --output CONTEXT
python3 runtime/images/sircl_layer.py build --context CONTEXT --tag sparkring:sircl \
  --name RELEASE [--profiles PROFILE,PROFILE,...] --output LOCK
```

`wheel` and `prepare` are offline and the wheel's bytes depend only on the
checkout; `natives` needs the parent image on the build host. `record` (run by
`build`) confirms that the parent has none of the added paths, runs SIRCL's
probe in the built image (package and entry points found in site-packages,
both libraries present), runs the installer's admission for every profile of
the lock and writes the v3 lock with the pinned vLLM builds the probe
reports. The lock lists the parent lock's profiles, or those of `--profiles`,
which may add the profiles that run only on SIRCL ring sessions and that a v2
parent lock cannot list. The lock's `image_reference` is the local
configuration ID until publication replaces it. The deployment's SIRCL
sessions read the layer's libraries through `SIRCL_NATIVE_LIBRARY` and
`SIRCL_P2P_NATIVE_LIBRARY` ([transport.py](../common/transport.py)).

The SIRCL layer adds no model sources. Runtime-status 0.3.5, which adds
SIRCL's facts to the dashboard's Transport table, enters an image through a
[status descriptor layer](#replacing-the-runtime-status-package) below the
SIRCL layer.

## libsircl layer

[libsircl_layer.py](libsircl_layer.py) (`installer-libsircl-layer`) adds
libsircl, SIRCL's NCCL-compatible C library
([design](../../docs/architecture/libsircl.md)), to a kraken-line image with a
v3 lock, normally the SIRCL image, and writes the derived image's v3 lock with
`libsircl` among its `transports` and a `libsircl` block. Status:
**research-only**; no release lists it. The layer holds:

- `/opt/sparkring/libsircl/lib/libsircl.so.<version>`, built by libsircl's
  own Makefile from the vendored [snapshot](../../spark_transport/libsircl/README.md)
  (`make -j BUILD=build`, then `make check`) in a network-less container of
  the parent image, at the fixed path `/tmp/libsircl` and with `LD_PRELOAD`
  unset. The build embeds the prebuilt kernel packs and needs no nvcc;
- libsircl's notices under `/opt/sparkring/libsircl`: `LICENSE`, `NOTICE`,
  `vendor/NCCL-LICENSE.txt`, `vendor/SIRCL-NOTICE` and `LICENSES/`;
- the vLLM general plugin `libsircl`
  ([sparkring_libsircl.py](../../integrations/vllm/libsircl/README.md)) in
  the serving interpreter's site-packages with its dist-info directory. vLLM
  loads it only when `VLLM_PLUGINS` names `libsircl`, so the image's other
  deployments are unchanged;
- `/opt/sparkring/receipts/libsircl-layer.json` (`sparkring-libsircl-layer/v1`):
  version, snapshot, library (path, SHA-256, SONAME `libnccl.so.2`), NCCL API
  level, whether the library has the fail-stop mode (its bytes name
  `LIBSIRCL_FAIL_STOP`; the transport requires it), kernel packs and
  architectures, plugin, compiler, build commands and every installed file
  with its SHA-256.

Every added file is recorded in the external-base receipt, whose
`capabilities.libsircl` names the layer receipt, so the image's `verify`
checks their bytes.

```bash
python3 runtime/images/libsircl_layer.py natives --parent-lock PARENT_LOCK --output NATIVES
python3 runtime/images/libsircl_layer.py prepare --parent-lock PARENT_LOCK --natives NATIVES   [--base-receipt base.json --toolchain-receipt toolchain.json] --output CONTEXT
python3 runtime/images/libsircl_layer.py build --context CONTEXT --tag sparkring:libsircl   --name RELEASE [--profiles PROFILE,PROFILE,...] --output LOCK
```

`natives` first checks the vendored tree against its manifest
(`scripts/sync_libsircl.py check`) and needs the parent image on the build
host; `prepare` is offline given copied receipts. `record` (run by `build`)
confirms that the parent has none of the added paths, probes the built image
(the plugin's entry point, `ncclGetVersion` 22705, `sirclGetInfo` naming
libsircl and its version, and the plugin selecting the library), checks the
layer as installation does (`libsircl.check_layer`), runs the installer's
admission for every profile and writes the v3 lock. `host-library
--builder-image ID --output DIR` builds the same library in another local
image for the stock-image option, which mounts it from the host. The
[design](../../docs/architecture/libsircl.md#build-and-load-commands) gives
the commands that build the layer on one Spark and load it on the others.

## CSF sources of the kraken line

A parent that serves the GLM-5.3-Flash CSF checkpoint needs vLLM and B12X
sources that carry its `nvfp4_csf` quantization and loader.
[derive_kraken_csf_sources.py](derive_kraken_csf_sources.py)
(`installer-kraken-csf-sources`) is a code layer over
`dev-20261004-kraken-cuda1342-nccl2323-status034` that replaces 47 and adds 2
Python files of `vllm/` and `b12x/` in site-packages with those of SparkRing's
merges vLLM `bc9ea774` and B12X `cc36aa6f`. Its
[source manifest](compositions/kraken-csf-sources-20261007/README.md) pins
each file in the parent and in the merges, and `prepare` takes the files from
a directory that holds them at their site-packages paths (`--sources`, such as
the CSF source overlay) or from the payload archive (`--payload`). The layer
writes no compiled file and none of the B12X sources the prepared transport
verifies; its provenance receipt is
`/opt/sparkring/receipts/derived-kraken-csf-sources.json`. `record` and
`build` are those of [derived_layer.py](derived_layer.py).

The SIRCL layer over this layer makes
[`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034`](../releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md),
whose recipe gives the build, load and check commands. Only that image's v3
lock records the pinned vLLM build in `sircl.vllm_pins`, which the installer
requires for the CSF checkpoint (`image_lock.CHECKPOINT_BUILDS`). An
installation with the v2 lock of this layer alone therefore keeps each
profile's default checkpoint and refuses `--checkpoint csf`; with the v3 lock,
the GLM-5.3-Flash profiles of two and four Sparks install the CSF checkpoint
by default.
[layer_delta.py](layer_delta.py) writes a `docker load` archive of a derived
image without the layers its parent provides, so a Spark that holds the parent
loads only the added layers.

## Evidence

Conditions: offline replay of `prepare` with copies of each parent's installed
receipts and of the parent files each code layer reads, starting from the
`dev-20260924` receipts; the Qwen decode step used the files of the published
`dev-20260925-qwendecode` layer, and the status 0.3.2 step the pinned wheel and
source archive. No image was built.

Result: each replayed layer produced the external-base and toolchain receipt
SHA-256 values that the installer locks record:

| Layer result | External-base receipt | Toolchain receipt |
|---|---|---|
| Transport window (intermediate) | `f8d66127cff95a7e…` | `79f84c03b9129c6c…` |
| `dev-20260925` | `7411678301f5fadb…` | `7345dc10a0a6b94a…` |
| `dev-20260925-qwendecode` | `4a7aa7ec1d84f586…` | `6b3104327f0dc5d3…` |
| `dev-20260927-h2dstaging` | `9861346f291e7808…` | `fa551a7581a24a7b…` |
| `dev-20260927-b12xcache` | `7824d15be44953b7…` | `62ec5a5d47bbe9d6…` |

The transport-window layer produced the installed manifest SHA-256
`2eef276d54030a71…`, and the TP2 HC layer a `derived-tp2-hc.json` identical to
the published image's. Conclusion: these builders reproduce the recorded file
and receipt identities of the published chain. Docker layer metadata and image
IDs of a rebuild differ from the published images; a rebuilt image is not a
published release and carries no serving qualification.

The peer-wait, tool-choice and runtime-status 0.3.3 layers were built on one
Spark from their parent images; each build ran the installer's admission,
including the image's `verify`, for all nine profiles. Building the peer-wait
layer twice on that Spark from the same parent produced the same image ID,
`sha256:e6ea1f241b16…`.
