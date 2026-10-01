# External ARM64 image composition

[external_context.py](external_context.py) prepares a Docker context that adds
SparkRing source changes and integration assets to a digest-pinned external
vLLM/B12X image. [external_base.py](external_base.py) installs and verifies that
composition inside the image. The supported layout is Linux ARM64 with both
packages under `/usr/local/lib/python3.12/dist-packages`.

The composition includes the Qwen collective policy, TP4 HC prefill helper,
adaptive prepared RoCEnante bundle, patched NCCL and SparkCache assets. It binds
each source-dependent hook to the composed package bytes. Qwen profiles select
one supported HC ownership scheme explicitly. Deployment profiles retain
responsibility for model, rank, cache and network settings.

The external image owns its compiled framework binaries and generated/vendor
files. Candidate archives supply source files under `vllm/` and `b12x/`.
Only files present in the pinned baseline Git archive and absent from the
candidate archive may be removed. Source archives cannot supply native binaries,
symlinks or bytecode. A native-code change requires a separately rebuilt base;
this preparer does not establish native compatibility.

## Prepare from local artifacts

Create a `sparkring-external-inputs/v1` JSON manifest. Paths resolve relative to
the manifest directory; absolute input paths also work. SHA-256 values identify
the actual artifact bytes, including archive line endings. Git revisions are
full 40-character commit hashes. The following example uses placeholders for
the caller's hashes and image reference:

```json
{
  "schema": "sparkring-external-inputs/v1",
  "platform": "linux/arm64",
  "base": {
    "reference": "eugr/spark-vllm-b12x@sha256:<manifest digest>",
    "config_id": "sha256:<image config digest>"
  },
  "base_inventory": {"path": "base-inventory.json", "sha256": "<sha256>"},
  "installer": {"path": "controller/external_base.py", "sha256": "<sha256>"},
  "parent_cache_contract": {"path": "parent-cache-contract.json", "sha256": "<sha256>"},
  "sources": {
    "vllm": {
      "commit": "<integrated source commit>",
      "upstream": "<upstream comparison commit>",
      "baseline_commit": "<base image source commit>",
      "archive": {"path": "vllm-sources.tar", "sha256": "<sha256>"},
      "baseline_archive": {"path": "vllm-base-sources.tar", "sha256": "<sha256>"}
    },
    "b12x": {
      "commit": "<integrated source commit>",
      "upstream": "<upstream comparison commit>",
      "baseline_commit": "<base image source commit>",
      "archive": {"path": "b12x-sources.tar", "sha256": "<sha256>"},
      "baseline_archive": {"path": "b12x-base-sources.tar", "sha256": "<sha256>"}
    }
  },
  "assets": {
    "root": "assets",
    "export_manifest": {"path": "assets/export.json", "sha256": "<sha256>"},
    "parent_receipt": {"path": "assets/parent-receipt.json", "sha256": "<sha256>"}
  },
  "prefill_controller": {
    "root": "controller/qwen4_prefill",
    "files": {
      "package_prefill.py": "<sha256>",
      "qwen4_prefill_bootstrap.py": "<sha256>",
      "qwen4_hc_fusion.py": "<sha256>",
      "qwen4_mtp_gemm.py": "<sha256>",
      "qwen4_fused_gate_kernel.py": "<sha256>"
    }
  }
}
```

The installed base inventory records `architecture: "aarch64"`, active
distribution `versions`, and `packages.vllm` / `packages.b12x`. Each package
contains its absolute `root` and a `files` map from package-relative path to
`{"sha256": "..."}`. Inventory every installed regular file except `.pyc`,
including native and generated files. Use the active distribution version
resolved by Python, rather than a shadowed duplicate package.

The asset root contains `sparkcache`, `sparkcache-overrides`, `sparkcache-native`,
`licenses`, `transports`, `features` and `nccl-lib`. Its export manifest records
an immutable `parent_image` config ID, `parent_receipt_sha256`, and a `files` map
from export-relative name to `{"origin": "/original/image/path", "sha256": "..."}`.
Every exported regular file except `.pyc` must be listed. Non-NCCL assets must
also match the parent's installed receipt. NCCL is independently pinned by the
export manifest and consists of one versioned `libnccl.so.MAJOR.MINOR.PATCH` plus
the `libnccl.so.2` and `libnccl.so` aliases resolving to the same bytes. Only
these NCCL sibling symlinks are supported in the asset export.

Copy the maintained prefill controller files from
[`integrations/vllm/qwen4_prefill`](../../integrations/vllm/qwen4_prefill/README.md)
and pin every listed file. Preparation runs their packaging entry point for the
`external-base` image and requires the source bindings it declares to equal the
composed bytes of `vllm/models/qwen4_exp/nvidia/hyperconnection.py` and
`b12x/sequence/mtp_feedback/_kernels.py`. A change to either source file
therefore requires a matching controller update. It changes a collective-policy source
binding only when CRLF normalization matches its already admitted LF digest;
semantic changes to that target require a reviewed integration update.

The optional [runtime status plugin](../../integrations/vllm/runtime_status/README.md)
uses a pinned pure Python wheel and matching source bundle. Add this object to
the input manifest only when the plugin should be present in the image:

```json
{
  "runtime_status": {
    "wheel": {
      "path": "artifacts/sparkring_runtime_status-0.1.0-py3-none-any.whl",
      "sha256": "<sha256>"
    },
    "source_archive": {
      "path": "artifacts/runtime-status-source.tar.gz",
      "sha256": "<sha256>"
    }
  }
}
```

The source tar is rooted at `runtime_status/` or at the component directory and contains `pyproject.toml`,
the `sparkring_runtime_status` Python package, and optionally `README.md`,
`schema-v1.json` and the component tests. Version 0.2 also includes the source-bound
`sparkring_runtime_status/dashboard.html` asset. Exclude build output, bytecode and
egg-info directories. Build the wheel from an isolated copy of these same source
bytes, using the package's declared setuptools backend. The preparer requires
`py3-none-any`, checks every RECORD hash and size, matches package bytes to the
source archive, and verifies both official `sparkring_status` entry points:
`vllm.endpoint_plugins` and `vllm.general_plugins`.

Status files and distribution metadata are installed under
`/opt/sparkring/python`; one owned `.pth` adds that path. No dependency installer
runs inside the image: the pinned base must already satisfy FastAPI's declared
minimum. Source/wheel hashes and source-file provenance are recorded in the
composition. Installation rejects an existing status distribution or plugin
name. Startup verifies the complete Python-root inventory, including rejection
of extra files and standalone bytecode. The image's
`PYTHONDONTWRITEBYTECODE=1` prevents generated bytecode from changing that root.

```bash
python runtime/images/external_context.py \
  --manifest /path/to/inputs.json --output /path/to/context
```

The output directory must not exist. Input failure leaves no partial output.
The result includes `composition.json`, its payload, the installed base
inventory, installer, prefill bundle and generated Dockerfile. The builder
catalog exposes the same preparer as `external-arm64`. Preparation uses local
files only and does not invoke Docker, download dependencies or install a host
service. Build the context separately on a compatible ARM64 Docker host:

```bash
docker build --tag sparkring:external-candidate /path/to/context
```

The descriptor records candidate and baseline archive hashes, source commits,
installer and preparer hashes, prefill-controller hashes, parent cache contract,
and the complete asset export provenance. Keep the input manifest and its
referenced artifacts with the release inputs so another host can reconstruct the
context. Identical inputs and preparer produce identical context file bytes;
Docker layer metadata and resulting image IDs are a separate build concern.

Installation verifies the inherited inventory and package versions, applies
declared source/assets, preserves framework native files and updates package
RECORD entries. The installed entry point verifies the complete resulting package
inventory before serving. A successful preparation, build or source-binding check
does not establish CUDA correctness, RDMA behavior, cache restoration or model
serving qualification. Publish those results separately with the image digest,
source pins and exact deployment scope described in the
[release procedure](../../docs/development/releases.md).

## Serve the composition through an installer image

The built context is the software layer of an installer image. The
[CUDA 13.4.2 and NCCL 2.32.3 toolchain assembly](cuda134-nccl232.md#installer-image-assembly)
takes it as its parent, using `/opt/sparkring/receipts/external-base-installed.json`
from the built image as the parent receipt. Derived layers and the installer
image lock are described in [installer images](installer-images.md); installer
profiles admit the resulting image through
[`installer_image.py`](../common/installer_image.py). Model, rank, cache and
network settings remain in each profile, and `VLLM_PLUGINS` must name
`sparkring_status` when the runtime status artifact is present.

## Carry deployment addons into a new composition

The optional `deployment_addons` input pins a manifest and tar archive. Its
`sparkring-deployment-addons/v1` manifest lists every archive member's SHA-256,
the `av`, `soundfile` and `sparkring-startup-identity` versions, and artifact
provenance. Archive paths are rooted at `python/`, `prefill/` or `sources/`.
Python payloads belong to those three distributions; framework packages and
path hooks cannot be supplied through this interface.

The image installs this snapshot under `/opt/sparkring/addons`. Its verifier
checks the complete inventory and distribution versions, and adds the addon
Python directory before serving even when a profile overrides `PYTHONPATH`.
The startup identity plugin still requires an explicit, receipt-bound deployment
manifest and selection in `VLLM_PLUGINS`. Merely packaging the prefill payload
does not activate it. Native compatibility and served audio require separate
qualification.

For assets exported from a previous external composition, `integration_sources`
can pin its `vllm` and `b12x` source archives. The collective adapter permits a
line-ending-only rebinding when the inherited file matches the adapter's
recorded digest and both source texts normalize identically. Semantic changes
continue to require an adapter update. Both original virtualenv and external
distribution paths are accepted for the recorded transport source bindings.
