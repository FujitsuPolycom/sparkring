# DeepSeek V4.1 Flash on SGLang

Status: **Development**. This source-pinned adapter and launcher support a
four-Spark direct cycle. The [controlled prefill comparison](../../performance/records/deepseek-v41-flash/sglang-decoder-replay-20260911.md)
passed its bounded quality checks. A separate [six-hour streaming record](../../performance/records/deepseek-v41-flash/sglang-soak-20260912.md)
covers a 430080-context site configuration; the recipe retains its 262144-context
default. Rebuilt images and unattended recovery require separate validation.
No public image is published.

| Input | Contract |
|---|---|
| Hardware | Four GB10 Sparks, Linux ARM64, local NVMe checkpoint and packed Engram files on every rank |
| Runtime | External Mia adapter and SGLang base image pinned in [pins.json](pins.json) |
| Measured settings | TP4/EP4, context 262144, chunk 4096, eight requests, memory fraction 0.90, DSpark block five |
| Transport | SparkRing patched NCCL 2.30.7, both RoCE devices, four ring channels |
| Authentication | Required file containing one distinct bearer key per nonempty line |
| Deployment | Operator-managed rank startup; start workers 3, 2, 1, then rank 0 |
| Fallback | [vLLM runtime](../deepseek-v41-gb10/README.md) |

## Build and prepare

On an ARM64 build host, choose a build directory that does not already exist:

```bash
WORK=/absolute/new-build-directory IMAGE=local/sparkring-deepseek-v41:sglang \
  runtime/deepseek-v41-sglang/build-image.sh
```

The builder clones Mia's repository at the recorded commit and replaces the
floating SGLang base tag with the recorded ARM64 digest. It builds the external
Dockerfile and prints the resulting image ID. Distribute that image to all four
ranks and record its ID in each private environment file. Source and base pins
do not guarantee byte-identical compiler output on independently built hosts.

Copy the [environment template](../../scripts/config/deepseek-v41-flash-sglang-cycle.env.example)
once per rank outside the checkout. Resolve all placeholders. Values are literal
`KEY=VALUE` entries: shell expansion, inline credentials, and unknown settings
are rejected. Keep node-local paths and credentials out of version control.

Obtain SparkRing's patched `libnccl.so.2` as described in the
[vLLM library setup](../../profiles/deepseek-v41-flash-cycle/README.md#patches-and-nccl), resolve any symlink,
and record `sha256sum` of that library as `NCCL_SO_SHA256`. The launcher mounts
it **over the image's pip NCCL library**. Do not add `LD_PRELOAD`: loading a second
NCCL runtime triggers SGLang's runtime check. This substitution assumes the
required NCCL 2.x ABI; the host checksum identifies the actual library, while
Torch's compiled header version may still report 2.29.7.

Render a launch plan without Docker or GPU access:

```bash
python3 scripts/deepseek_v41_sglang_cycle_serve.py --check /private/rank-0.env
```

On each rank, prepare the CPU-only authentication patch, then pack that rank's
Engram files into an empty directory:

```bash
python3 scripts/deepseek_v41_sglang_cycle_serve.py --prepare /private/rank-0.env
python3 scripts/deepseek_v41_sglang_cycle_serve.py --pack /private/rank-0.env
```

Packing uses local checkpoint shards and approximately 48 GiB per rank for two
files named `engram-l1-r0of4.bin` and `engram-l14-r0of4.bin` on rank 0. Other ranks
use their own rank index. These are Mia's packed layout, incompatible with the
vLLM packed files. A directory containing existing files is refused.

The authentication patch checks the pinned upstream source shape before writing
a private generated module under the state directory. Its
`operator/auth-receipt.json` binds the generated source to the
selected image, patcher and source pins. Rerun `--prepare` after changing the
image or patch inputs, or when using state prepared without this receipt.
Startup rejects missing or mismatched authentication receipts. Each key is
accepted. SGLang's internal caller can use the complete joined value. Separately
configured admin credentials retain exact matching; ordinary keys cannot unlock
endpoints that require an admin key. Health and metrics keep SGLang's existing
unauthenticated behavior. The entrypoint reads the mounted key file, so keys
never appear in Docker arguments or the launch receipt. Restrict access to the
state directory, which also holds the external launcher's internal key file.

## Serve and validate

Remove the backend from public routing and stop its existing model owner. The
launcher refuses an occupied GPU, an existing `sgl_dsv41` container, less than
100 GiB available host memory, missing shards, or an image/NCCL identity mismatch.
Reboot idle ranks if necessary to recover unified-memory headroom.

Start ranks 3, 2, 1, then 0 with their respective files:

```bash
python3 scripts/deepseek_v41_sglang_cycle_serve.py --run /private/rank-0.env
```

Allow approximately 9–11 minutes for weights, the draft model, and graph capture.
Verify rank 0 `/health`, each configured key against `/v1/models` and an actual
generation, and rejection of missing/invalid credentials. The served name is
`deepseek-v4.1-flash`; the external launcher sets the `deepseekv41` tool parser
and `deepseek-v41` reasoning parser. Clients can request non-thinking responses
with `chat_template_kwargs: {"thinking": false}`.

The external image enables decoder bounded replay: layers 0–20 process the
whole chunk, then layers 21–39 process the final 128 extend tokens per request.
Global KV memory is retained, but local late-layer SWA visibility changes.
This is not a demonstrated mathematical equivalence to full prefill. Prompt
logprobs and full prompt hidden-state capture are not supported by that shortcut.
See the [controlled replay evidence](../../performance/records/deepseek-v41-flash/sglang-decoder-replay-20260911.md).

Before adopting a site deployment, validate long-context retrieval, tools and
vision, mixed prefill/decode, tail/chunk boundaries, sampled responses, and a
20-minute followed by six-hour C8 soak with per-rank memory observations. Test
coordinated restart under the site's service owner. The launcher does not
install services, alter routing, or provide unattended recovery after a partial
rank failure.

Increasing `CONTEXT_LENGTH` to 430080 permits testing a roughly 400K prompt, but
is outside the recorded 262144-context A/B profile. Inspect the actual KV pool;
`MAX_TOTAL_TOKENS` alone does not establish that a request fits. Do not describe
an untested context setting as a measured result.

DSpark uses the verify-all schedule when the SPS table is absent. Table paths
refer to files inside the mounted state directory. Profile SPS/STS on the
running configuration before enabling them and compare decode throughput and
quality independently. vLLM's adaptive verification and Engram tuning do not
transfer automatically to this implementation.

## Source ownership

Mia's adapter, launcher, and packer remain in the external AGPL-3.0 repository
and the operator-built image. They are not vendored here. This directory holds
SparkRing's original wrapper, a source-checked modification to SGLang's auth
module, and source pins. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
