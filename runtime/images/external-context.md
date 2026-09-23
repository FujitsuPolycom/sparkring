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
and pin every listed file. Preparation executes their packaging entry point
with exact composed source bindings. It changes a collective-policy source
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

The source tar is rooted at `runtime_status/` and contains `pyproject.toml`,
the `sparkring_runtime_status` Python package, and optionally `README.md`,
`schema-v1.json` and `test_runtime_status.py`. Exclude build output, bytecode and
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

## Admit a published Qwen profile

The [external host admission helper](../common/external_candidate.py) uses the
shared publication identity contract with `runtime_layout: "external-base/v1"`.
In addition to the registry/config digest, anonymous pull evidence and installed
receipt SHA-256, its publication requires:

- `composition_sha256`, `base` and `sources` matching the installed receipt.
  Each source entry includes candidate/baseline commits and archive SHA-256s.
- `features` containing `qwen-collectives` and `qwen4-prefill`, plus the prepared
  `transport.profile` and `transport.manifest_sha256`.
- `hc_supported_modes`, copied from the composition capabilities. It describes
  supported alternatives rather than asserting which one a running profile uses.
- `sparkcache_contract` containing its installed `path` and `sha256`.
- `profiles`, a map of repository-relative serving configuration JSON paths to
  their raw SHA-256 digests. These explicitly opt into this release.
- Optional `runtime_status`, copied from the composition capability, including
  distribution/version, both entry points, Python root and source/wheel hashes.

Each registered configuration uses `image_extension: "external-base"`, its
`image_release`, and this explicit loader envelope:

```json
{
  "container_envelope": {
    "cap_add": ["IPC_LOCK"],
    "security_opt": ["seccomp=unconfined"]
  }
}
```

The supported Qwen configuration is TP2 or TP4, one rank per node, PP1/DCP1,
context 262144, 16 sequences, 8192 batched tokens, 24 GiB KV cache, MTP3 and
CUDA graph capture ceiling 64. Its `--model-loader-extra-config` is
`{"read_mode":"bounce","io_threads":8}`. TP4 selects
`SPARKRING_FEATURES=qwen-collectives,qwen4-prefill`; TP2 leaves that setting empty.
These are explicit bounds for this adapter, not performance recommendations for
other configurations.

The supported HC settings are mutually exclusive:

| TP | `VLLM_QWEN3_8_FLASH_NEXT_HC_TP` | `VLLM_QWEN3_8_HC_PREFILL_MODE` |
| --- | --- | --- |
| 2 | `1` | `off` |
| 4 | `1` | `off` |
| 4 | `0` | `shard` |

The first scheme shards HC projections. The TP4 `0/shard` alternative keeps
replicated HC parameters and shards prefill token rows. Combining `1/shard` is
rejected; the launcher never silently substitutes a scheme. This is one
process-wide profile choice, not automatic switching between decode and prefill.

The existing [Qwen launch adapter](../common/qwen_flash_next.py) accepts these
registered configurations and retains its `plan`, `check` and `create` actions.
It selects `python3` and the external verifier, sets working directory `/`,
binds the release's transport and cache contract, selects patched NCCL, and
preserves the profile's explicit HC scheme. Persistent cache paths include both
the composition identity and a canonical hash of the complete serving profile.
Different HC, model, argument or environment settings therefore use separate
cache namespaces; rank-local mount and network inputs are applied afterward.
Other model, serving
capacity and rank-network settings remain in the selected profile. Host
admission checks the exact image/entrypoint, installed receipt, source pins,
transport file and cache contract before launch. Existing native and legacy
profiles keep their previous entrypoints and container defaults.

When the status artifact is selected, add `sparkring_status` to the profile's
existing `VLLM_PLUGINS` list. The shared container specification carries that
same list to the API rank and every worker. Admission rejects activation without
the artifact, or omission of activation from a profile selecting that artifact.
After startup, query `GET /v1/sparkring/status` through the normal authenticated
inference API; the plugin's guide defines its passive reporting limits.

The Compose adapter accepts `qwen38-flash-next-qad-tp2-eugr` and
`qwen38-flash-next-qad-tp4-eugr` after their profile metadata and publication are
registered. Their release selection uses `published-immutable-reference`, names
the matching release ID and registry image, and pins its `publication.json` as
the first input. Staged controller inventories include both configurations named
by that publication, since host admission verifies every profile hash.

Adding these opt-in configurations does not change default profile selection.
Checked-in public Compose examples have a separate inventory and remain generated
from their registered site examples. The image builder and host admission helpers
never install a cluster daemon.
