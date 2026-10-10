# SparkCache connector-job lease layer over image `sha256:d52737a109e0`

Status: **research-only**. The layer is written and reviewed; no Spark has
built or run it. GPU capture, restore and serving qualification remain
required.

The [descriptor](descriptor.json) derives an installer image from release
2026.10.2's image (`sha256:d52737a109e0…`, lock
`runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json`)
by adding one file and restating three, all Python or JSON under
`/usr/local/lib/python3.12/dist-packages/sparkcache/`:

| Image path | Change | Source |
|---|---|---|
| `…/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json` | added | `integrations/sparkcache/gab86b70/runtime_patches/vllm-connector-jobs-gab86b70.json` |
| `…/sparkcache/runtime_patches/verify_lease_contract.py` | restated | `integrations/sparkcache/gab86b70/runtime_patches/verify_lease_contract.py` |
| `…/sparkcache/runtime_patches/generic_connector_contract.py` | restated | `integrations/sparkcache/gab86b70/runtime_patches/generic_connector_contract.py` |
| `…/sparkcache/streaming/manager_page_factory.py` | restated | `integrations/sparkcache/gab86b70/streaming/manager_page_factory.py` |

The added contract pins the SHA-256 of eleven vLLM source files of the parent
image's vLLM (`gab86b7073`) and requires the `jj-block-state-read-leases/v1`
ownership semantics; its review notes are in the contract. The three restated
files are the SparkCache code that reads the connector settings and verifies
that contract (`streaming/manager_page_factory.py` loads the contract path from
the connector's extra config; `runtime_patches/generic_connector_contract.py`
gates connector-job mode; `runtime_patches/verify_lease_contract.py` hashes and
symbol-checks each file). They carry the parent image's installed bytes
unchanged, so the restated rows pin the parent's bytes and the layer changes
only which contract SparkCache verifies against.

The parent image keeps its own lease contract,
`/opt/sparkring/contracts/vllm-connector-jobs-eugr-<16 hex>.json`, whose
`vllm/envs.py` fingerprint the image's CSF layer replaced; deployments that
point `spark_cache_async_page_capture_lease_contract` at it still refuse. The
profiles of the [SparkCache variants](../../../profiles/glm53-flash-nvfp4-sparkcache-tp4/README.md)
point at this layer's contract instead.

## Build

On a Spark with the parent image local (the derived layer's `prepare` needs only
the parent's two receipts and never contacts a registry):

```bash
docker run --rm --pull never --network none --entrypoint cat \
  sha256:d52737a109e0… /opt/sparkring/receipts/external-base-installed.json > base.json
docker run --rm --pull never --network none --entrypoint cat \
  sha256:d52737a109e0… /opt/sparkring/toolchain/installed.json > toolchain.json
python3 runtime/images/derived_layer.py prepare \
  --descriptor runtime/images/compositions/installer-sparkcache-gab86b70/descriptor.json \
  --base-receipt base.json --toolchain-receipt toolchain.json --output CONTEXT
python3 runtime/images/derived_layer.py build --context CONTEXT \
  --tag sparkring:sparkcache-gab86b70 \
  --name dev-20261010-sparkcache-gab86b70-plugins-status036 \
  --output runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/installer-image.json \
  --profiles deepseek-v41-flash-tp4,deepseek-v41-flash-tp8,glm53-flash-csf-tp8,glm53-flash-nvfp4-spark-tp2,glm53-flash-nvfp4-spark-tp4,glm53-flash-nvfp4-sparkcache-tp2,glm53-flash-nvfp4-sparkcache-tp4,glm53-nvfp4-tp8,glm53-nvfp4-tp8-dcp1,mimo-v26-flash-mopd-tp2,mimo-v26-flash-mopd-tp4,qwen38-flash-next-qad-tp4,qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp4
```

`build` tags the parent, builds the context (one `COPY files/ /` layer over the
parent) and records: it confirms the parent lacks the added path, runs the
installer's admission for every profile of the written lock, keeps the parent's
v3 fields (image line, transports, SIRCL block with its
`sparkring-kraken-beta-20261007-bc9ea774` vLLM pin, default tuning digest) and
lists the two SparkCache variants beside the parent's profiles. The written
lock is a development lock (`image_reference` is the local configuration ID)
until publication. No step here pushes or selects an image for a profile.

## Expected layer payload

`plan.json` in the prepared context records the four files, their inherited and
resulting SHA-256 (the added contract's inherited value is `null`) and the two
receipt digests. `runtime/images/test_sparkcache_gab86b70_layer.py` pins the
same file list and hashes from the repository side, so a descriptor or source
edit that changes the layer fails the CPU suite before any build.

## Qualification state

The contract's source review and CPU conformance are the only evidence today.
The image-level GPU gates of `docs/jj-connector-read-leases.md` (capture
ownership on CUDA streams, restore placement, and the profile qualification of
`glm53-flash-hybrid` on this vLLM line) remain open; the hardware test plan in
`performance/records/images/` for the first run of these profiles records them.
