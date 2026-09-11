# R33 GLM-5.3 Spark profile integration

Status: **TP2 and TP4 SparkCache profiles bounded-qualified**. These inputs
define the settings and evidence gates for generic R33 ARM64 image
`ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef`.
They do not render a private site or authorize a host action. Other packaged
profiles retain their own qualification status.

`image.env.example` is the single image-selection boundary. Fill its three
image fields from [the public runtime receipt](../public-image-receipt.json).
The registry manifest digest and Docker config ID have distinct roles and must
come from the same receipt. For a separate local build, `image_reference` may
equal its local config ID until that exact image is published.
Use the same receipt on every TP2 and TP4 rank. Keep the rollback image receipt
and archive named by `SPARKRING_ROLLBACK_RECEIPT` until qualification completes.

The generic R33 image entrypoint consumes this contract in
`SPARKRING_PROFILE_MODE=custom`; it does not contain a second profile catalog.
The external renderer or launcher must provide `SOURCE_IMAGE_PROFILE` and the
complete rank-local environment before `serve`. TP4 additionally requires the
managed renderer marker and concrete library, peer, device, GID, control-port,
HCA, rank, and management values. Placeholder site values are rejected.

`tp2-dcp1.env.example` preserves the existing one-DAC layout: both functions
of physical cage p0 are addressed through the primary and secondary host PCIe
domains. It uses the byte-pinned `tp2-rocenante-adaptive` transport, TP2/DCP1,
MTP3, native vLLM `instanttensor` loading, a 1,048,576-token request limit, and the existing exact
TP2 graph sizes. It clears `PYTHONPATH` so the TP4 mesh overlay cannot leak into
the pair. This InstantTensor profile has not been qualified with SparkCache.
Use `tp2-dcp1-sparkcache` for the tested managed-B12X cache configuration.

`tp4-dcp1.env.example` is an overlay for the existing managed mesh renderer and
lifecycle in `runtime/glm53-spark-mtp3-mesh`. That renderer remains responsible
for private addresses, four-HCA `NCCL_IB_HCA` values, SIRCL endpoints, marker
ownership, memory gates, coordinated stop, startup, and recovery. The template
selects TP4/DCP1, MTP3, the 1,048,576-token request limit, dual-domain NCCL, and
the exact graph sizes recorded by the mesh profile. Apply
`tp4-dcp1-sparkcache.env.example` only for the separate bounded cache run.

The cache-disabled TP2 and both TP4 profiles use native vLLM `instanttensor`
loading. All profiles enable mHC prefill sharding and DCP1. For an 8,192-row prefill, TP4 divides mHC ownership into
2,048 rows per rank; this is not a fixed owner-row count for every topology.
The TP4 template requests continuation-prefill coalescing and up to four
diagnostic records per rank. The SparkCache overlay inherits those settings.
The cache-disabled TP2 profile disables coalescing. The separate
`tp2-dcp1-sparkcache` profile requires managed B12X loading, TP2 coalescing,
7.5 GiB KV per rank and source-bound capability evidence. The 6.75 GiB pin is
retained as reference metadata and an explicit override, not the default for
the R33 one-million-token target. The public R33 image carries the required
source implementation and packaged evidence. A different image cannot gain
those capabilities by selecting the profile name.
See [the TP2 cache plan and evidence gates](../../../profiles/glm53-flash-spark-tp2/R33_SPARKCACHE.md).
The locked vLLM composition implements sparse checkpoint scheduling and
the multi-checkpoint B12X execution path. The pinned
InstantTensor revision selects its I/O backend automatically; these profiles do
not require an `INSTANTTENSOR_*` environment override. Configuration is
only admission evidence. The [activation evidence contract](ACTIVATION_EVIDENCE.md)
defines the immutable logs, process maps, status snapshots, and request receipts
required to prove that every rank used the same image and NCCL 2.31.2, both host
domains carried NCCL, the exact graph set was captured, and InstantTensor, MTP3,
mHC, and the selected custom transport executed. Package verification of the
prefix-hit metadata fix in source tree `547f7091841728f21ab419012a766fd1df70a569`
does not by itself qualify model execution. The exact public image's TP2 and
TP4 SparkCache model results are in the linked qualification records. The
cache-disabled TP2 profile still requires disabled-coalescing evidence for its
own qualification. TP4 requires two completed
32,768-token-or-longer prompts with no `sample_tokens` timeout or fatal engine
error. The cache profile also requires successful capture, restore, payload
comparison, and recovery after an injected fault.

Run the offline contract checks with:

```bash
python -m pytest runtime/sparkring/jovian-r33/profiles -q
python runtime/sparkring/jovian-r33/profiles/verify_profile.py template --profile tp2-dcp1 --asset-root runtime
python runtime/sparkring/jovian-r33/profiles/verify_profile.py template --profile tp4-dcp1 --asset-root runtime
```

After a hardware run, validate its independently collected receipt with:

```bash
python runtime/sparkring/jovian-r33/profiles/verify_profile.py activation \
  --receipt /srv/sparkring/private/r33-tp4-activation.json
```

The public runtime receipt supplies the generic image config ID, registry
manifest digest, and source/package evidence. Operators still supply a rollback
receipt and SparkCache namespace. Rank, management, HCA, and peer addresses
remain private site inputs. The managed lifecycle validates the R33 receipt and
profile contract before it creates containers; do not replace published image
fields in generated files.
