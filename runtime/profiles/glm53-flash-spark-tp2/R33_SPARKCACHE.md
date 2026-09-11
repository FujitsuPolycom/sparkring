# R33 TP2 SparkCache composition

Status: **research-only**. The local launcher implements a separate
`tp2-dcp1-sparkcache` plan. Its default KV pool is 7.5 GiB per rank, matching
the bounded R33 test configuration. It preserves the TP2 reference's
managed B12X loader, TP2/DCP1, MTP3 with Humming draft MoE,
eight sequences, 8,192-token scheduler budget, prefill interval eight,
three-image/one-video admission, exact graph list, single-DAC two-domain
transport, and active 2 GiB memory guard. It selects the R33 image's pinned
SparkCache native libraries and lease contract and uses a fresh image-specific
namespace. It does not reuse the reference deployment's disk cache.

The pinned R33 B12X loader uses managed allocation internally. The launcher
omits the legacy `allocation` extra configuration because R33's safetensors
superclass rejects it. Target and MTP both select B12X; the draft explicitly
uses an empty extra-config dictionary. The legacy profile retains its explicit
managed-allocation option for its different loader implementation.

The request limit is explicitly 1,048,576 tokens. The reference used 262,144;
neither that reference nor this plan proves one-million-token serving capacity.
The R33 startup capacity check rejected the 6.75 GiB reference pin at this
request limit: it estimated 7.27 GiB required, about 6.74 GiB available, and
a 962,560-token maximum. Those estimates are specific to that image/run.
The packaged profile and launcher now default to 7.5 GiB
(`8053063680` bytes). The contract records 6.75 GiB separately as the reference
pin, not the R33 one-million-token default. Explicit overrides remain available
for 6.75, 7.5 and 8.75 GiB; they do not inherit qualification from another pin.
The 7.5 GiB configuration is bounded-qualified on generic R33 image
`3c7779ad71dd…` for text correctness, three cold starts, managed loading,
prefill and decode, native 8,192-token cache capture and restart restoration,
and active memory guards. The
[qualification record](../../../performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md)
states the exact image, source identities, measurements and evidence hashes.
Continuation-prefill coalescing is implemented for TP2/DCP1 under the
[source and component evidence contract](../../sparkring/jovian-r33/profiles/evidence/tp2-continuation-prefill-coalescing.json).
During the bounded model run, the scheduler selected an 8,192-token coalesced
span with checkpoint targets at tokens 6,144 and 7,936. Both workers configured
B12X checkpoint export with 32 local heads and four checkpoint slots. The
worker logs do not contain a distinct export-completion event, so the live
evidence proves plan selection and compatible worker configuration rather than
independently observing the checkpoint writes.
The configured one-million-token limit was admitted by the KV pool but was not
exercised by a completed request. Multimodal correctness and sustained guarded
memory stability remain unqualified.

The cache-disabled `tp2-dcp1` profile continues to use InstantTensor with
coalescing disabled. The cache composition requires managed B12X loading and
TP2 continuation coalescing; it never silently disables either requirement.
An image with only the existing TP4 coalescing implementation cannot run it.
Adding these local files does not change an already built image. An image must
contain the extended entrypoint, matching source composition, and capability
evidence before serving this profile.

## Prepare and validate a plan

Use the existing launcher arguments and add `--r33-sparkcache`:

```bash
python3 runtime/profiles/glm53-flash-spark-tp2/launch.py plan \
  --rank 0 --master 198.18.200.1 \
  --model-dir /srv/models/nvfp4-spark-df116c4f \
  --cache-dir /srv/sparkring/r33-tp2-cache-r0 \
  --env-file /srv/sparkring/private/tp2-rank0.env \
  --image sha256:IMAGE_CONFIG_ID \
  --runtime-receipt /srv/sparkring/private/r33-image.json \
  --r33-sparkcache
```

Model/cache paths must exist. The site file supplies the host IP and NCCL/Gloo
socket interfaces. The plan prints `activation_blockers` when the image lacks
capabilities. `create` and `start` reject that receipt before inspecting guards
or contacting Docker. Their other guard and existing-GPU-container checks are
unchanged. Candidate names are `sparkring-r33-tp2-dcp1-sparkcache-r0/r1`.

## Final image packaging

The canonical profile records the 7.5 GiB default, implicit managed B12X
allocation, explicit B12X draft loader with empty extra configuration, and
request scheduling/multimodal limits. The launcher reads these fields rather
than inheriting the legacy profile's values for the cache composition.

An already built image retains its packaged profile bytes. Final packaging
requires a new context/source lock, a rebuilt generic image with this contract,
and a new verified image receipt. The pinned runtime sources and native
artifacts are unchanged and can be reused. Distribute the external launcher
from the same repository revision and bind its rendered plans to the final
image receipt. Repeat the required TP2 and TP4 qualification on that exact
image; neither these configuration changes nor a successful package build
promotes earlier bounded results to complete release qualification.

The TP2 and TP4 bounded checks have passed on the published image. Their
separate qualification records state the measured scope and limitations.

## Required capability evidence

The source build supplies `tp2-sparkcache-capabilities.json` in the canonical
R33 profile directory before context preparation. The file records implemented
source/component capabilities with live qualification pending. Its schema is
`sparkring-r33-runtime-capabilities/v1`,
its `profile` is `tp2-dcp1-sparkcache`, and its `sources` must exactly match the
contract's `vllm_integrated_tree`, `b12x_tree`, and `sparkcache_tree`.

Both `checks` and `evidence_sha256` must contain exactly these keys:

- `tp2_continuation_prefill_coalescing`
- `managed_b12x_loader`
- `tp2_sparkcache`

Each check must be `implemented` and backed by a retained source-bound CPU or
component-test artifact's SHA-256. Set `evidence_kind` to
`source-component-tests` and `live_qualification` to `pending`. Loader evidence
must establish registration and managed-allocation support; cache evidence
must establish the TP2 connector/native interface and lease-contract support.
These checks admit the first research model run. They do not require that run
to have passed already. A source path that is absent or whose compatibility
is not established cannot be marked implemented.
Serialize the file with `json.dumps(document, sort_keys=True,
separators=(",", ":")) + "\n"`. Context finalization binds its bytes; image
verification checks the source lock and validates the capability document.
Each evidence digest identifies the matching canonical receipt under
`runtime/sparkring/jovian-r33/profiles/evidence/`; those receipts state the
source revisions, test conditions, result, conclusion, and qualification limit.
The resulting image receipt carries `runtime_capabilities.document` and its
`sha256`, matching `verification.checked_files`. The launcher also requires
both pinned SparkCache native library hashes in that image verification.
These are source/package capability gates. Completed GPU/model/cache-recovery
results belong in the separate activation receipt and remain release gates.
Do not change admission status to `qualified` or use a live result as a
substitute for a missing source/component compatibility check.

Activation requires positive managed-B12X allocation, coalescing, mHC and
RoCEnante activity on both ranks. For an 8,192-row TP2 prefill, each owner must
execute 4,096 rows. Retain exact image/source identities, loaded NCCL and
dual-domain routes, graph capture, MTP acceptance, correctness, cache payload
comparison and fault recovery. Do not substitute InstantTensor counters or
TP4 results for the managed-B12X TP2 run.
