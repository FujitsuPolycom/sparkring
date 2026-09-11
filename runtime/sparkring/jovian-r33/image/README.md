# Generic Jovian R33 ARM64 image

This directory assembles and verifies the model-neutral ARM64 image from exact
R33 artifacts. Image
`sha256:3c7779ad71dd0d5d6fae4c98e04b94c377429306158c2259fc44635892b8b8e4`
has bounded TP2 and TP4 ring qualification; the
[TP2](../../../../performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md)
and
[TP4](../../../../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md)
records state the exact scope. The public ARM64 manifest is
`ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef`.
Use the tracked [runtime receipt](../public-image-receipt.json) for launch
admission and [publication receipt](../publication.json) for registry evidence.
The original [construction receipt](image020-construction-receipt.json) retains
its pre-publication bytes. Building and verification do not publish, deploy,
or start a model. The default command prints help.

```bash
SPARKRING_IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef'
docker pull "$SPARKRING_IMAGE"
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt runtime/sparkring/jovian-r33/public-image-receipt.json
```

The final image uses content-addressed media runtime `sha256:a1a72e18ad49d99f6194a2585bfdc5f32d79180cdf2cd015c0d0d451479d6a42`. That layer is verified as a child of `local/sparkring:r33-arm64-foundation` and already contains the qualified Ubuntu FFmpeg 6.1.1 runtime and exact media wheels. It inherits CUDA 13.3 and PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`. The candidate replaces the canonical NCCL byte with SparkRing routing SHA `84a4b8d8...` at the same single authoritative `/opt/local-inference/nccl/lib` path. SIRCL SHA `bea00f2b...`, the LMCache cuMem interposer, canonical external profile contract, and lease contract are installed separately and content-checked.

`artifact-lock.json` rejects changed stable inputs and explicitly excludes the CUDA 13.0 TorchAudio wheel. The source-built CUDA 13.3 TorchAudio wheel is required. Context preparation requires rebuilt vLLM and B12X wheels plus the terminal FlashInfer Python and ARM64 JIT-cache wheel pair. Each artifact must match its completed SHA256SUMS receipt.

The vLLM source composition is reproduced from
`patches/vllm-r33-sparkring.patch`; its manifest binds the base commit, the
continuation port, every changed file, the resulting tree and the required
B12X checkpoint capability. The package step also materializes the exact
commit-bound FlashAttention Python helpers before checking the wheel RECORD.

The goal-complete image requires the ARM64/SM121 SparkCache placement and snapshot libraries at `artifacts/sparkcache-native/`, with schema `sparkring-r33-sparkcache-native-build/v1`. The artifact lock binds placement SHA `d89c9fda...`, snapshot SHA `7da9e72f...`, source tree `86ef46de...`, and source archive SHA `d61bb093...`. Context preparation validates these bytes and their native build receipt before admitting the cache profile.

Run the local contract tests from the repository root:

```bash
python -m unittest runtime/sparkring/jovian-r33/image/test_r33_image_contract.py -v
```

On the ARM64 build host, place this directory at a stable path and make the repository checkout available. After all component builds finish, generate the three terminal source-identity receipts in one fail-closed step:

```bash
python3 runtime/sparkring/jovian-r33/image/finalize_component_receipts.py \
  --build-root /var/tmp/sparkring-r33-20260910
```

The command runs `capture_post_build_sources.py`, `validate_flashinfer_resume.py`, and `capture_flashkda_identity.py` in that order. It generates all results in a temporary directory before copying them to their documented artifact locations, and refuses existing targets. To prove existing receipts reproduce exactly without changing them, run:

```bash
python3 runtime/sparkring/jovian-r33/image/finalize_component_receipts.py \
  --build-root /var/tmp/sparkring-r33-20260910 \
  --verify-existing
```

Then prepare a new context path:

```bash
python3 runtime/sparkring/jovian-r33/image/prepare_context.py \
  --build-root /var/tmp/sparkring-r33-20260910 \
  --repository-root /path/to/sparkring-checkout \
  --context /var/tmp/sparkring-r33-20260910/work/image-context-001
```

Resolve the complete ARM64 binary-only Python dependency closure and write its immutable source lock:

```bash
bash runtime/sparkring/jovian-r33/image/resolve_closure.sh \
  /var/tmp/sparkring-r33-20260910/work/image-context-001
```

The resolver uses the exact content-addressed media runtime on the ARM64 host, asks pip for a binary-only full closure, records every selected URL and SHA-256, downloads those exact wheels, installs the entire closure in an isolated probe `/opt/venv`, and records every resulting payload byte. The isolated environment includes the exact local Torch and media wheels instead of accepting unrelated packages from the parent interpreter. It then creates `source-lock.json`. The Docker build verifies the complete finalized context before consuming any wheel or runtime file, installs Python packages offline, and compares the final installed payload with the probe manifest.

After reviewing `source-lock.json`, the candidate-only build command is:

```bash
docker build --pull=false \
  --file /var/tmp/sparkring-r33-20260910/work/image-context-001/Dockerfile.candidate \
  --tag local/sparkring:r33-arm64-candidate \
  /var/tmp/sparkring-r33-20260910/work/image-context-001
```

That build is intentionally not launched by the preparation scripts. Coordinate before starting it because the 25.8 GB media runtime plus the full wheel closure makes it a large local build.

After the local build, create an immutable local-only image receipt and validate it with the canonical profile contract:

```bash
python3 runtime/sparkring/jovian-r33/image/qualify_image.py \
  --context /var/tmp/sparkring-r33-20260910/work/image-context-001 \
  --image local/sparkring:r33-arm64-candidate \
  --output /var/tmp/sparkring-r33-20260910/artifacts/r33-candidate-image-receipt.json
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt /var/tmp/sparkring-r33-20260910/artifacts/r33-candidate-image-receipt.json
```

The entrypoint contract is:

```text
sparkring-r33 --help
sparkring-r33 verify
sparkring-r33 serve MODEL [vLLM options...]
```

Serving requires `SPARKRING_PROFILE_MODE=custom` and an externally rendered
profile from `runtime/sparkring/jovian-r33/profiles`. The entrypoint validates
common source settings and concrete site inputs before invoking vLLM; it does
not render profiles. TP4 also requires the managed-mesh marker and resolved
native-library, peer, device, GID, control-port, HCA, rank, and management values.
SparkCache is admitted by `tp2-dcp1-sparkcache` and `tp4-dcp1-sparkcache`.
Their separate qualification records above define the tested scope. Installed
SparkCache or LMCache packages do not enable caching for other profiles.
