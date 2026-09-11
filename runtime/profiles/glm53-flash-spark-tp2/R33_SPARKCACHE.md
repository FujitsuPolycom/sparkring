# R33 TP2 SparkCache composition

Status: **research-only**. The local launcher implements a separate
`tp2-dcp1-sparkcache` plan. It preserves the bounded TP2 cache reference's
6.75 GiB KV pool, managed B12X loader, TP2/DCP1, MTP3 with Humming draft MoE,
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
For one-million-token qualification, explicitly select the 8.75 GiB candidate
with `--r33-cache-kv-memory-bytes 9395240960`. The default remains the 6.75 GiB
reference configuration and is not sufficient for that observed R33 startup.
The larger pin retains the same source-capability gates and active memory
guard; its startup, request capacity and memory stability must be measured.
Multimodal accuracy, guarded memory stability, native cache capture/restore,
recovery, and performance need new measurements on the selected image.

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
  --r33-sparkcache --r33-cache-kv-memory-bytes 9395240960
```

Model/cache paths must exist. The site file supplies the host IP and NCCL/Gloo
socket interfaces. The plan prints `activation_blockers` when the image lacks
capabilities. `create` and `start` reject that receipt before inspecting guards
or contacting Docker. Their other guard and existing-GPU-container checks are
unchanged. Candidate names are `sparkring-r33-tp2-dcp1-sparkcache-r0/r1`.

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
