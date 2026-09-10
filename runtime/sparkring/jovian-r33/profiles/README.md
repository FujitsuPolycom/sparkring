# R33 GLM-5.3 Spark profile integration

Status: **research-only**. These inputs define the settings and evidence gates
for one generic R33 ARM64 image. They do not select an image, alter the
published source-image lock, render a private site, or authorize a host action.

`image.env.example` is the single image-selection boundary. Replace its three
image fields from one construction receipt only after the image has an immutable
Docker config ID and matching source-lock evidence. Before publication,
`image_reference` may equal that local config ID. After publication it must be
the registry manifest digest from a receipt that also binds the same config ID.
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
MTP3, managed loading, a 1,048,576-token request limit, and the existing exact
TP2 graph sizes. It clears `PYTHONPATH` so the TP4 mesh overlay cannot leak into
the pair. SparkCache remains disabled because the pair has no matching recovery
qualification.

`tp4-dcp1.env.example` is an overlay for the existing managed mesh renderer and
lifecycle in `runtime/glm53-spark-mtp3-mesh`. That renderer remains responsible
for private addresses, four-HCA `NCCL_IB_HCA` values, SIRCL endpoints, marker
ownership, memory gates, coordinated stop, startup, and recovery. The template
selects TP4/DCP1, MTP3, the 1,048,576-token request limit, dual-domain NCCL, and
the exact graph sizes recorded by the mesh profile. Apply
`tp4-dcp1-sparkcache.env.example` only for the separate bounded cache run.

All profiles enable continuation-prefill coalescing, 2,048-row mHC prefill
sharding, InstantTensor through the generic image, and DCP1. Configuration is
only admission evidence. A profile qualifies only when a structured activation
receipt proves that every rank used the same image and NCCL 2.31.2, both host
domains carried NCCL, the exact graph set was captured, and nonzero runtime
counters show InstantTensor, MTP3, continuation coalescing, mHC, and the selected
custom transport were exercised. TP4 additionally requires two completed
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

The remaining unresolved inputs are the generic image config ID, registry
manifest digest, image receipt path, rollback receipt path, and SparkCache
namespace. Rank, management, HCA, and peer addresses remain private site inputs.
The existing managed lifecycle must reject the R33 receipt until its image
adapter validates this contract; do not bypass that gate by replacing the
published image fields in generated files.
