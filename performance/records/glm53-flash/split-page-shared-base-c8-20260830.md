# GLM-5.3 split-page SparkCache shared-base C8 validation

Status: **qualified** for exact output, eight external restores, and one
authenticated shared-base read per rank by the local image named below.
Latency is **research-only**. Soak behavior and rebuilt images are
**unsupported** by this record.

## Conditions

The artifact was local Linux/ARM64 image
`sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818`.
All four NVIDIA DGX Spark ranks had that exact image ID. It combined vLLM pull
request head `ead9d8a4e21b3818b21ec6f4d4d94564dd60c3f8`, SparkCache's vLLM
composition `6da4865d440608a46eada50f27b2fff0e698c574`, SparkCache commit
`59ac0b04db6035a9a9d2a52e92405ceaf84daa40`, and B12X commit
`b1d541f9e71a35f030d45fae437630fff7507c2a`.

Serving used TP4/DCP1, FP8 KV, 20 GiB of KV memory per rank, external BF16
DFlash7, a 262,144-token model limit, 16 maximum sequences, and a 4,096-token
prefill batch. Target and recurrent pages were both 512 tokens. SparkCache used
eight restore lanes, eight pending restore slots, and a 256 MiB CUDA arena per
lane. The cache identity retained the `manager-pages-v1` geometry namespace.

After a clean restart, each worker discovered 20 persistent manifests. One
five-token prompt with a one-token completion then executed to deliver worker
inventory to the scheduler. The measured cohort contained eight distinct
16,384-token stored roots presented as 16,385 prompt tokens by the request
client. Each root shared one stored 8,192-token base.

## Measurement

The request client checked the exact expected output for every request. Worker
records reported the authenticated shared-base flight independently on each
rank. The machine-readable receipt preserves the artifact identities, serving
configuration, request digests, counts, byte size, and per-rank timing values:
[`validation.json`](../../receipts/glm53-flash/split-page-shared-base-c8-20260830/validation.json).

HTTP `/health` was already successful before the tiny inference. That health
response proved process readiness but not scheduler knowledge of the workers'
persistent manifests. A diagnostic C8 cohort run before any model execution
restored seven requests and safely recomputed one. The qualified cohort ran
after one real model step populated scheduler inventory.

## Result

- All eight requests returned HTTP 200 with their exact expected output.
- All eight requests restored external state, a 100% external-hit ratio.
- Each rank served the eight restores with one 100,868,258-byte physical base
  read and avoided seven duplicate base reads.
- Base-read time was 161.461 ms on rank 0, 172.076 ms on rank 1, 188.138 ms on
  rank 2, and 173.349 ms on rank 3.
- Worker restore service time ranged from 0.5160 to 0.9905 seconds across the
  per-rank observations.
- Client elapsed time ranged from 1.392026 to 2.337652 seconds.

## Conclusion

After one inference established scheduler inventory readiness, the exact image
correctly restored all eight different request roots that shared one stored
base. SparkCache authenticated and read that base once per rank, then kept each
request's tail and placement transaction independent.

## Limitations

This single cohort does not establish latency variance, throughput, or a
production service-level objective. No soak or fault-injection run was
included. HTTP health is not sufficient to assert persistent-hit readiness
after a restart; one real inference must complete first. The local image has no
published OCI digest, and a rebuild does not inherit its qualification.
