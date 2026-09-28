# Installer image builders

The `sparkring install` profiles run on development images that are built as a
chain. Each image adds one layer to its parent, and each installer image release
under [`runtime/releases/`](../releases/README.md) is produced by a builder in
[builders.json](builders.json); the entry's `releases` field names it. The
[repository layout check](../../scripts/check_repository_layout.py) requires a
builder for every release that has an `installer-image.json` lock.

| Image | Parent | Layer | Builder |
|---|---|---|---|
| `dev-20260924-cuda1342-nccl2323-status031` | Software layer `sha256:01869ac593003b144416828b000b33febb626179dc901d14907ce6a202150373` | CUDA 13.4.2 and NCCL 2.32.3 | `cuda134-nccl232-assembly`, [toolchain_assembly.py](toolchain_assembly.py) |
| Intermediate, image ID `sha256:372ed77b81f724f4eeacb8ddb909f039567de4ce8b0d151e0a4897fe65fd345d`, no release record | `dev-20260924-cuda1342-nccl2323-status031` | Prepared RoCEnante proxy paces hardware-forwarded stripes within a send window | `installer-transport-window`, [derive_transport_window.py](derive_transport_window.py) |
| `dev-20260925-cuda1342-nccl2323-status031` | the intermediate image | Qwen HC token-row ownership on TP2 as well as TP4 | `installer-tp2-hc`, [derive_tp2_hc.py](derive_tp2_hc.py) |
| `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | `dev-20260925-cuda1342-nccl2323-status031` | Qwen4Exp decode GEMM plans and MXFP8 hyper-connection projections | `installer-qwen-decode`, [build_qwen_decode.py](../../performance/records/qwen38-flash-next/installer-tuning-20260925/programs/build_qwen_decode.py) |
| `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` | `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | vLLM host-to-device copies staged through fresh pinned memory | `installer-staging-fix`, [derive_staging_fix.py](derive_staging_fix.py) |
| `dev-20260927-b12xcache-cuda1342-nccl2323-status032` | `dev-20260927-h2dstaging-cuda1342-nccl2323-status031` | [B12X reconciled selection-cache correction](../../integrations/b12x/selection_cache/README.md) and runtime-status 0.3.2 | `installer-derived-layer`, [derived_layer.py](derived_layer.py) with descriptor [installer-b12xcache-status032](compositions/installer-b12xcache-status032/descriptor.json) |
| `dev-20260927-mimovision-cuda1342-nccl2323-status032` | `dev-20260927-b12xcache-cuda1342-nccl2323-status032` | MiMo vision encoder attention sinks in the softmax denominator | `installer-mimo-vision`, [derive_mimo_vision.py](derive_mimo_vision.py) |

The `installer-tool-choice-contract` builder,
[derive_tool_choice_contract.py](derive_tool_choice_contract.py), derives a
layer from `dev-20260927-mimovision-cuda1342-nccl2323-status032` in which named
and required Chat Completions `tool_choice` requests without a complete call
fail with `ToolChoiceContractError`
([tool-result contract](../../integrations/vllm/tool_choice_contract/README.md#installer-images)).
No release records an image built by it.

Each release's `publication.json` names its parent and describes its layer.
[cuda134-nccl232.md](cuda134-nccl232.md) documents the toolchain layer and the
inputs recorded for `dev-20260924`.

Rebuilding the software layer below `dev-20260924` from this repository is
**unsupported**. That layer composes the `eugr/spark-vllm-b12x:nightly-20260924`
base with pinned vLLM and B12X source archives, SparkRing integration assets
exported from the `shared-2026.09.4-rc.4` image, deployment add-ons and a
runtime-status 0.3.1 wheel built from
[integrations/vllm/runtime_status](../../integrations/vllm/runtime_status/README.md).
Its external-image composition preparer and the Qwen prefill controller version
it packages are not in this repository. The layer's installed receipt has
SHA-256 `1354e3d0270b297dafafd39e5341210f4d0e7bd908b15b0026531c4e0350da62`
and composition SHA-256
`46ecfc99659799934e7429fb0f8c80310b93aee485337114139bb5fe1df096d4`, as the
release's installer image lock records.

## Derived layers

Every layer above `dev-20260924` except the Qwen decode layer is built by
[derived_layer.py](derived_layer.py). The installer image lock pins the SHA-256
of two receipts inside the image: the external-base receipt
(`/opt/sparkring/receipts/external-base-installed.json`), whose file map the
image's `verify` checks, and the toolchain receipt
(`/opt/sparkring/toolchain/installed.json`), which records the external-base
receipt. A derived layer copies its files over the parent, records each file in
the external-base receipt, re-records the toolchain receipt and writes a
provenance receipt that lists every path with its inherited and resulting
SHA-256 (`null` for an added file).

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
[derive_tp2_hc.py](derive_tp2_hc.py) lists the TP2 row-sharding HC mode;
[derive_staging_fix.py](derive_staging_fix.py) pins `vllm/v1/utils.py`. Every
replaced path must already be recorded by the parent receipt, and its bytes in
the parent must match that record. A code layer may also add a site-packages
Python file under the descriptor rules: `pins` names the added path with
inherited SHA-256 `None`, and the parent receipt must not record it.
[derive_tool_choice_contract.py](derive_tool_choice_contract.py) adds the
tool-result policy as a vLLM module and pins the `serving.py` that installs it.

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
status version and files. The `installer-b12xcache-status032` wheel and source
archive were built from the source now in
[integrations/vllm/runtime_status](../../integrations/vllm/runtime_status/README.md)
(version 0.3.2).

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

The tool-choice layer's `prepare`, replayed offline with copies of the
`dev-20260927-mimovision-cuda1342-nccl2323-status032` receipts, which match the
SHA-256 values its lock records, and of its `serving.py`, accepted the pinned
inherited `serving.py` and wrote a context that adds one file and replaces one.
No image was built from it.
