# Rebuilt GLM-5.2 3.5-bpw image: clean-checkout build and four-rank bring-up

## Scope

**Status: implemented; not qualified.** This record covers one image built
from a clean checkout by `runtime/exl3-r7/build-image.sh`, placed on four
directly cabled NVIDIA DGX Spark systems, and brought up under the generated
exact-Q40 operator profile until the API served requests.

It establishes that the documented GLM-5.2 EXL3 3.5-bpw path executes from a
clean checkout to a serving endpoint, and that the deployed bytes match the
pinned identities. It establishes no throughput, latency, or output-quality
result, and it does not promote the profile: the acceptance workload named by
[the promotion checklist](../../../docs/GLM52_35BPW_PROMOTION_CHECKLIST.md) is
outstanding. The qualified status of any other image ID does not transfer to
this one.

Conditions: four directly cabled DGX Spark systems, GB10, TP4 with DCP4, one
ring of four 200 Gb/s ConnectX-7 links, no switch in the inference fabric.

## Immutable identities

| Item | Value |
|---|---|
| Repository revision built from | `c1a520027b502faa15ad29cd41f59dacfefea2a4` |
| Parent image manifest digest | `sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Parent image ID | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Built image ID | `sha256:5569c4778c9561a8595ac283c7adf31e22be1d35517aa208569af5224244a2da` |
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` |
| Model revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Model config SHA-256 | `fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126` |
| Model index SHA-256 | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| KV representation | dynamic NVFP4 latent with FP8 RoPE |
| Speculation | fixed MTP4 |

## Checkpoint verification

Method: `scripts/download_exl3_r7.py verify` against the pinned revision,
which hashes every indexed shard against the LFS SHA-256 metadata published at
that revision. The command requires network access to the model host and a
Python environment providing `huggingface_hub`; the serving image supplies
both when the Hugging Face offline flag is cleared for the run.

Result: `status: pass`, 157 runtime shards, 346,218,639,128 runtime weight
bytes, matching the pinned index total.

Conclusion: the deployed checkpoint bytes are the pinned revision, established
independently of any runtime observation.

## Generated profile artifacts

Every artifact was produced from the tracked scripts against the built image
ID, and each generated profile and site names that image ID.

| Artifact | SHA-256 |
|---|---|
| Complete pre-exact-Q40 profile | `4f64a798307737446553619562f9a70b87e31bbb684df2ead7d58f2e23cb92bf` |
| Exact-Q40 serving profile | `735e2ccad4ef17e9dc9a9d21cbcc888e9d4d0cb5b80227e5ff8a1d1fce919eca` |
| Exact-Q40 manifest | `7e1f4d553c10f154c7e3748a45637469216e4e9cd1e01eefe7a049309c6490e4` |
| Exact-Q40 `exl3.py` overlay | `8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2` |
| Exact-Q40 `model_runner.py` overlay | `2c2b79326e00fb1ae7494a4921b13300a75f4dcf6aa7e9943db2e9e5b63f7a23` |

The attestation overlay binds the serving profile to the built image ID and to
the pinned model revision, so neither can be substituted without regenerating
it.

## Preflight

Method: `scripts/preflight.py` against the generated site, covering SSH
reachability, ring link state, MTU, link speed, addressing, RDMA port and GID
state, jumbo-frame reachability on each ring edge, transport control-channel
reachability, port availability, cache-path presence and free space, and image
identity on each rank.

Result: 116 of 116 checks passed on four ranks.

## Runtime attestation

The launcher verifies a 17-entry SHA-256 manifest inside the container on
every rank before serving starts. It covers the image entrypoint, the patched
`weight_utils`, `cudagraph_utils`, and `parallel_state` modules, the two QuACK
annotation edits, the collective audit module, the SIRCL native library and
its four adapter modules, both exact-Q40 overlays, two B12X kernel modules,
and the model config and index.

Result: 17 of 17 entries matched on all four ranks.

## Bring-up observations

| Observation | Value |
|---|---|
| Weight load, each of four workers | 87.54 GiB |
| Load duration range | 217.9 s to 238.0 s |
| Mixed-Trellis tier structure | varies per layer, three- through five-bit tiers |
| SIRCL sessions | eager sessions ready on all four ranks across the captured payload sizes |
| CUDA graph capture | 40 sizes, piecewise and full |
| Exact-Q40 receipt | written once per rank |
| API state | `Application startup complete` |

The per-layer varying tier structure and the 87.54 GiB per-worker weight load
together distinguish the pinned checkpoint from a requantized rendering of the
same model, which loads a smaller resident size and reports a uniform tier
structure on every routed layer.

## Serving observations

| Check | Result |
|---|---|
| `/health` | HTTP 200 |
| `/v1/models` identity | `glm-5.2-exl3-r7-3.5bpw` |
| `/v1/models` maximum length | 262,144 tokens |
| Speculative depth | 4 |
| Deterministic arithmetic prompt | correct |

A bring-up chat completion reported a mean acceptance length of 4.14 at
speculative depth 4, with per-position acceptance rates 0.966, 0.897, 0.690,
and 0.586. This is a single short bring-up request, not a measurement: it
establishes that fixed-MTP4 speculation is active and accepting, and supports
no throughput or acceptance-rate claim.

## Limitations

- **Not qualified.** The promotion checklist's fixed-seed, bounded C1, C2, and
  C8 acceptance workload has not been run against this image ID, no post-run
  rank and transport health confirmation exists, and no acceptance receipts
  have been preserved.
- **The acceptance workload has no procedure in this repository.** The
  checklist item names a workload that no tracked document defines. The item
  cannot be satisfied until the procedure is specified.
- No throughput, latency, or output-quality figure in this record is a
  measurement. Existing performance figures for this profile belong to other
  image IDs and do not transfer.
- Native transport tests passed on one GB10 host: 18 of 18 ctest targets from
  the same checkout that produced the image.
- One bring-up on one appliance. Nothing here is a distribution, a repeat, or a
  portable hardware claim.
