# MTP3 image deployment and persistent recall

Status: qualified for the bounded trial below. Fresh-host installation and
unattended managed lifecycle are not established by this test.

## Conditions

- Four NVIDIA DGX Sparks, TP4/DCP4, using an already-installed managed mesh.
- Target: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
  `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP depth 3.
- Image: `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:23f00af873ccc784cfb742b7be2a29c6d3c20ebec9741843c025320bb9c04685`.
  Local image ID: `sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47`.
- FP8 KV, 24 GiB per rank, 8,192 batched tokens, 16 sequences, 1,048,576-token
  configured limit; CUDA graph sizes 4 through 64 in steps of 4.
- Local lil image-command extension and SparkRing canonical argument exporter.
  Image, checkpoint, native library and managed-fabric preflight checks passed.
- The lookup request contained 12,848 tokens, used temperature zero and default
  reasoning settings, and requested the exact code `LIL-7391`.

## Results

| Check | Result |
| --- | --- |
| Cache disabled | Exact code; zero cached tokens; 8.203 seconds for the complete response |
| Cache enabled, initial request | Exact code; all four ranks committed 12,288 tokens with digest prefix `a2ac226092ae` |
| Full container stop and fresh launch | All ranks and API became healthy using the persisted cache |
| Identical request after restart | Exact code; 12,288 cached tokens; no new publication; complete response 2.141 seconds |
| Rank 0 / 1 / 2 / 3 restore | 109.1 / 131.5 / 126.3 / 119.0 ms |
| lil status, logs and stop | Commands completed successfully against the trial containers |
| Direct-fabric copy | A 1 MiB fixture crossed three rank-to-rank edges with matching SHA-256 at every destination |

Initial and replay request SHA-256:
`a12d997b749171a19ef13afd4419e5e2abf027256fa7515967645f0f36741362`.
Cache-enabled bundle SHA-256:
`94fb0c4a83e08c92f5588adde214628d216ab16c9ab6db6b22c280fe4068d933`.

The complete-response times include reasoning and generation, whose lengths
differed. They are not an isolated prefill-speed comparison. Restore timings are
worker-reported durations. The copy fixture validates routing and integrity, not
bulk transfer throughput.

The operator supervised trial containers while the existing mesh supervisors
retained network ownership. lil did not install the mesh or retarget its systemd
model units. Per-rank connection ordering must match the physical fabric; uniform
device ordering is not valid for every rank. Existing containers and caches were
retained throughout the trial.

Private raw evidence is kept under `integrations/lil/.private/`: `probe-mtp-off.json`,
`probe-mtp-prime.json`, `probe-mtp-replay.json`, `cache-mtp-prime.json`,
`cache-mtp-replay.json`, `fanout-receipt.json`, and timestamped lil lifecycle
receipts. These include site information and are excluded from version control.
This record does not claim larger-context, concurrency, corruption-injection,
multimodal, or unattended failure-recovery coverage.
