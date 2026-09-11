# GLM-5.3 Flash TP4 Ring

Status: **TP4/DCP1 ring with SparkCache bounded-qualified**.
The exact evidence and limits are in the
[image020 TP4 record](../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md).

This guide downloads or builds a source-pinned image and selects it through SparkRing's
existing managed mesh deployment. It uses four NVIDIA Sparks, native MTP depth
three, continuation-prefill coalescing, token-sharded mHC, and NCCL across both
host PCIe domains. DCP1 is the baked-in qualified profile. A TP4/DCP4
profile (SparkCache and cache-disabled variants) is available through the
profile-contract overlay (`R33_PROFILE_CONTRACT_HOST_ROOT`; see the
Reproduction section of the record) and is bounded-qualified in
[r33-image020-tp4-dcp4-sparkcache-20260911](../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md).

| Setting | DCP1 profile |
|---|---|
| Model | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` |
| Model revision | `df116c4fb16b1d37ae43d2cfd624de26ffbc832e` |
| Tensor/context parallelism | TP4 / DCP1 |
| Maximum context | 1,048,576 tokens |
| Scheduler token budget | 8,192 tokens per pass |
| KV allocation | 24 GiB per rank |
| Coalescing and mHC prefill sharding | Enabled |
| TP4 mesh and dual-domain NCCL | Enabled |
| SparkCache connector / compact index cache | Enabled / disabled |
| DCP top-k owner exchange | DCP1: disabled (implementation requires DCP4). DCP4 overlay: active (`full-CKV gather` prefill path) |

The maximum context is a request-length limit. It is distinct from the total
KV capacity reported at startup and does not establish a tested concurrency
at that length.

The bounded-qualified profile is `tp4-dcp1-sparkcache`. The generic image also
packages a cache-disabled TP4/DCP1 profile, which does not inherit this
profile's model qualification.

## Prepare the hosts and image

Use the R33 deployment tooling from the integration branch in a separate checkout:

```bash
git clone --branch feat/jj-r33-integration https://github.com/FujitsuPolycom/sparkring.git sparkring-r33
cd sparkring-r33
```

Complete the [managed mesh host setup](GLM53_SPARK_MESH_HOST_SETUP.md) and read
the [deployment suite guide](DEPLOYMENT_SUITE.md). Preserve its networking,
authenticated ownership, stop, memory and readiness gates. Use an independent
management connection and private deployment inputs.

The image uses the source-locked R33 ARM64 composition documented under
[`runtime/sparkring/jovian-r33`](../runtime/sparkring/jovian-r33/README.md).
Pull the immutable digest and validate its tracked image receipt:

```bash
SPARKRING_IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef'
SPARKRING_RECEIPT='runtime/sparkring/jovian-r33/public-image-receipt.json'
docker pull "$SPARKRING_IMAGE"
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt "$SPARKRING_RECEIPT"
```

The receipt preserves the exact Docker config ID, source lock, component
receipts and installed-file verification, while its `image_reference` selects
the public registry manifest. The separate publication record is distribution
evidence and is not accepted as a runtime receipt.

To build the same composition locally, follow the
[R33 image recipe](../runtime/sparkring/jovian-r33/image/README.md). A local
build requires a fresh construction receipt and separate GPU qualification;
do not relabel the public receipt for different image bytes.

## Use the managed deployment suite

R33 keeps the forwarding marker outside the serving image. Staging downloads
the small [host tool release](https://github.com/FujitsuPolycom/sparkring/releases/tag/r33-host-tools-c8646b0)
and verifies its SHA-256 against the repository's
[host contract](../runtime/sparkring/jovian-r33/mesh-host-contract.json).
The mesh bundle is extracted from the verified serving image. The explicit
`--runtime-profile` below selects the SparkCache-enabled ring configuration.

Follow the discovery step in [the deployment guide](DEPLOYMENT_SUITE.md#discover-and-review).
Pass the image receipt when creating the private preparation document:

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.private/glm-tp4-deployment"

sr plan --inventory "$STATE/inventory.json" --name glm-tp4 \
  --workspace /srv/sparkring/glm-tp4 --fabric-range 198.18.0.0/21 \
  --image-receipt "$SPARKRING_RECEIPT" \
  --runtime-profile tp4-dcp1-sparkcache \
  --output "$STATE/preparation.json"
```

Use a fabric range that does not overlap management or VPN routes. Review and
verify networking through the deployment guide, then stage the verified
preparation. Staging requires all added implementation files to be tracked by
Git; untracked files are intentionally excluded from the source archive.

The selected receipt is copied to the controller and all four hosts. The same
identity is checked during staging, rendering, stopped-container verification,
installation and native qualification. Local images must exist on the staging
seed; they are saved and loaded by config ID rather than pulled as registry
digests. Omitting `--image-receipt` selects the existing canonical public
profile, not this source image.

Continue with the deployment guide's container creation, managed installation,
native tests and coordinated startup. Do not bypass those steps with direct
container starts or replace networking underneath active queue pairs.

## Qualify the deployment

Wait for all four containers and the API readiness gate before sending test
traffic. Verify the effective command contains `--max-model-len 1048576`,
`--tensor-parallel-size 4` and the selected DCP size. Verify actual coalescing
and 2,048-row mHC owner execution during an eligible 8,192-token prefill;
environment variables alone are not activation evidence.

Check cold, repeated and extended prompts, including a cache-hit continuation
that computes more than one additional 8K pass. Measure cold 8K/16K/32K prefill
with warmup excluded and preserve each raw sample. Report decode separately
with MTP acceptance and normalized steps/s. Record image, source, launch,
topology and benchmark identities with the results.

The [R33 source recipe](../runtime/sparkring/jovian-r33/README.md) describes the
source and verification contracts. A 1M context setting does not imply that
1M accuracy, mixed-request scheduling or sustained high concurrency has been
qualified by short prefill and decode tests.
