# GLM-5.3 Flash TP4 Ring

Status: **research-only**. The source recipe and managed setup integration are
implemented and CPU-tested. The image built from this recipe still requires
the serving qualification described below. Measurements from a different image
do not qualify a rebuild.

This guide downloads or builds a source-pinned image and selects it through SparkRing's
existing managed mesh deployment. It uses four NVIDIA Sparks, native MTP depth
three, continuation-prefill coalescing, token-sharded mHC, and NCCL across both
host PCIe domains. DCP1 is the selected profile; DCP4 is an explicit alternative.

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
| SparkCache connector and compact index cache | Disabled |
| DCP top-k owner exchange | Disabled; its implementation requires DCP4 |

The maximum context is a request-length limit. It is distinct from the total
KV capacity reported at startup and does not establish a tested concurrency
at that length.

For asynchronous persistent caching, select `tp4-dcp1-mtp3-sparkcache` in both
the source-image verifier and private site. That profile retains the same
prefill and network settings and enables bounded connector-job capture and
verified restore. Its source configuration is implemented; the shared image
still requires the cache checks described in the
[source recipe](../runtime/sparkring/source_image/README.md).

## Prepare the hosts and source

Complete the [managed mesh host setup](GLM53_SPARK_MESH_HOST_SETUP.md) and read
the [deployment suite guide](DEPLOYMENT_SUITE.md). Preserve its networking,
authenticated ownership, stop, memory and readiness gates. Use an independent
management connection and private deployment inputs.

Run the build and verification commands on an ARM64 Docker host with Python
3.12 or later and Git. The staging seed is rank 0: build there, or transfer the
resulting image to rank 0 with Docker save/load before staging. Allow disk space
for the source, compiler outputs, image archive, loaded image and model.

From the SparkRing checkout, choose initially absent writable directories:

```bash
curl --fail --location \
  https://github.com/FujitsuPolycom/sparkring/releases/download/native-runtime-sm121-aa8fa11831af/native-runtime-files-20260908.tar \
  --output /tmp/native-runtime-files-20260908.tar
python runtime/sparkring/source_image/prepare_image.py \
  --output "$PWD/.private/glm-tp4-context" \
  --source-cache "$PWD/.private/glm-tp4-sources" \
  --native-files /tmp/native-runtime-files-20260908.tar

SPARKRING_IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:86516f319b505e94686e0ba59200f91dda4b65599b08b3508c931c1e4e42b2a6
docker pull "$SPARKRING_IMAGE"
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746
```

Preparation fetches exact public source commits, applies the packaged patches,
and verifies complete source trees. The downloaded image retains the pinned
parent's framework dependencies and exact hash-verified native libraries.
The parent pull supplies the local parent identity required by verification;
Docker reuses shared layers. The [publication record](../runtime/sparkring/source_image/publication.json)
binds the image digest and source lock.

To build the image yourself instead of pulling it:

```bash
docker build --platform linux/arm64 --network none \
  -t sparkring-glm53-tp4-source "$PWD/.private/glm-tp4-context"
SPARKRING_IMAGE=sparkring-glm53-tp4-source
```

The optional compiler-rebuild path is described in the common recipe and
requires separate native-output qualification.

## Verify the image and select a profile

Keep the prepared context: verification uses it to check the image's verifier,
source lock and manifest before executing the CPU verification program.
Select the cache-disabled DCP1 profile:

```bash
python runtime/sparkring/source_image/verify_image.py \
  --image "$SPARKRING_IMAGE" \
  --context "$PWD/.private/glm-tp4-context" \
  --profile tp4-dcp1-mtp3-prefill \
  --output "$PWD/.private/glm-tp4-dcp1-image-receipt.json"
SPARKRING_RECEIPT="$PWD/.private/glm-tp4-dcp1-image-receipt.json"
```

This receipt identifies a local Docker config ID, not a published registry
manifest. CPU verification checks files, source identities and dependency
metadata; it does not test GPU kernels or model outputs.

To select DCP4, generate a separate receipt with
`--profile tp4-dcp4-mtp3-prefill`. That profile enables the DCP4 owner exchange
and fused endpoints. SparkCache and compact index cache remain disabled.
Do not change individual flags inside a verified receipt.

To enable bounded SparkCache capture and restore, use this profile instead:

```bash
python runtime/sparkring/source_image/verify_image.py \
  --image "$SPARKRING_IMAGE" \
  --context "$PWD/.private/glm-tp4-context" \
  --profile tp4-dcp1-mtp3-sparkcache \
  --output "$PWD/.private/glm-tp4-dcp1-sparkcache-receipt.json"
SPARKRING_RECEIPT="$PWD/.private/glm-tp4-dcp1-sparkcache-receipt.json"
```

The receipt selects the deployment profile. For DCP4, set `SPARKRING_RECEIPT`
to the output path from its separate verification command. Pass that selected
receipt to the deployment plan below.

## Use the managed deployment suite

Follow the discovery step in [the deployment guide](DEPLOYMENT_SUITE.md#discover-and-review).
Pass the image receipt when creating the private preparation document:

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.private/glm-tp4-deployment"

sr plan --inventory "$STATE/inventory.json" --name glm-tp4 \
  --workspace /srv/sparkring/glm-tp4 --fabric-range 198.18.0.0/21 \
  --image-receipt "$SPARKRING_RECEIPT" \
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

The [source recipe](../runtime/sparkring/source_image/README.md) describes the
source and verification contracts. A 1M context setting does not imply that
1M accuracy, mixed-request scheduling or sustained high concurrency has been
qualified by short prefill and decode tests.
