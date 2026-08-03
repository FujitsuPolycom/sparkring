# GLM-5.2 EXL3 LMCache campaign, 2026-08-03

Status: **CS512 performance-promoted and live; release/public acceptance blocked**

## Scope and evidence boundary

This document records an external operator campaign against the four-Spark
fixed-MTP2 EXL3 configuration documented in
[`EXL3_FIXED_MTP2_RECIPE_20260802.md`](EXL3_FIXED_MTP2_RECIPE_20260802.md).
The campaign is testing a patched LMCache integration for TP4/DCP4 without
changing the legacy 3.25-bpw checkpoint, MTP depth, graph ceiling, KV layout,
or serving envelope.

This is a **public-functional-lane, external-evidence campaign in progress**.
The full-envelope L0 arm reached four-rank initialization with one local
LMCache server per Spark, but CUDA allocation failed while registering the KV
cache and L0 was rejected before API readiness. Exact-label rollback succeeded.
A reduced-envelope C0 capacity-control arm subsequently reached API readiness,
registered all four rank-local cache contexts, passed a bounded deterministic
gate, stored symmetric objects, and reused the stored prefix after the vLLM
engine was destroyed and restarted with native prefix caching disabled. This
is **bounded live validation of the LMCache lifecycle**. Later C0-best
measurements established a fast combined native-prefix-cache plus LMCache warm
path and fair repeated C8 lanes, but did not promote performance. Two
fixed-seed 128-token reruns produced different hashes, so release correctness
is failed/open. This is not durable-backend persistence, public-bootstrap
acceptance, a clean-checkout result, or a reference-lane result.

The final chunk-size tuning arm, CS512, changes only LMCache `chunk_size` from
256 to 512 and otherwise retains the C0-best engine configuration. It is the
campaign's performance-promoted live arm. Release and public acceptance remain
blocked by the inherited token-124 nondeterministic correctness gate and by
non-identical per-host image IDs.

The raw artifacts remain outside the tracked tree because they contain site
identities and paths. Management addresses, SSH identities, hostnames, local
paths, filled launch definitions, and credentials are omitted here. Public
artifact identities, configuration values, sanitized measurements, and claim
boundaries are retained in the companion
[`docs/configurations/glm52-exl3-lmcache-campaign-20260803.json`](configurations/glm52-exl3-lmcache-campaign-20260803.json).

LMCache is not the repository's `sparkcache/` implementation. Evidence and
maturity claims for the two systems are separate.

## Campaign state

| Item | Current evidence |
|---|---|
| B0 control | live-validated with LMCache absent and vLLM native prefix caching enabled |
| patched LMCache package | ARM64 wheel built; focused topology tests passed |
| one-Spark prerequisites | CUDA interop, cross-process CUDA IPC, and one local MP server passed bounded smoke gates |
| L0 full envelope | rejected before readiness when KV-cache registration hit CUDA OOM |
| L0 server state | all four remained unregistered and empty |
| L0 rollback | exact-label B0 rollback succeeded |
| C0 capacity control | API ready at the reduced envelope |
| connector registration | one `world_size=1` context on each of four local servers |
| miss, put, and all-shard hit lifecycle | bounded live proof complete |
| engine-restart reuse | passed with native prefix caching disabled |
| LMCache performance | CS512 performance-promoted relative to C0-best; release/public acceptance remains false |
| latest repeated C8 fairness | passed the lane-ratio check and supports the bounded CS512 performance promotion; it is not a release/public acceptance result |
| 128-token repeat correctness | failed/open: fixed-seed output hashes differed |
| durable persistence or storage-tier offload | not demonstrated |
| C0-best promotion | not promoted; full matrix completed with capacity-limited cells |
| final live arm | CS512, performance-promoted relative to C0-best |
| release/public acceptance | blocked: correctness gate and identical-image requirement fail |

The C0 claim is limited to the observed live lifecycle. It does not establish a
default recipe, performance superiority, NVMe durability, server-restart
recovery, eviction safety, or prolonged mixed-traffic stability.

## L0 rejection and C0 transition

L0 preserved the full B0 serving envelope: a 1,048,576-token maximum model
length, 9,000,000,000 KV-cache bytes per rank, and an eagerly allocated 1-GiB
LMCache L1 per rank. All four workers initialized the LMCache C-operations and
pointer plumbing. During `REGISTER_KV_CACHE`, CUDA allocation failed. The local
servers had not registered KV state and remained empty, so L0 produced no
cache-hit or performance evidence. It was rejected before API readiness.

The operator restored the exact labeled B0 stack successfully, then launched
C0. C0 changes only the capacity envelope and LMCache allocation timing:

| Setting | L0 rejected arm | C0 bounded-live arm |
|---|---:|---:|
| maximum model length | 1,048,576 | 524,288 |
| KV-cache bytes/rank | 9,000,000,000 | 4,500,000,000 |
| LMCache L1/rank | eager 1 GiB | lazy 1 GiB, starting at zero |
| TP / DCP / MTP | TP4 / DCP4 / fixed MTP2 | unchanged |
| batch / sequences / graph | Q4096 / C8 / Q32 | unchanged |
| disposition | rejected before readiness | lifecycle live-validated; performance not promoted |

C0 is a capacity-control tradeoff, not an expected kernel speedup. Its live
reported KV capacity is 562,688 tokens.

## C0 bounded live evidence

### Artifact inventory

| Artifact | SHA-256 | Valid scope |
|---|---|---|
| initial prefix probe | `67c87379d55746d0e25d8b1eccefd64b9e143c532399e011d84ccb9849a94763` | first store and pre-restart control |
| NPC-off engine-restart replay | `4777bc367c72daf684005dbd95f3ab0036f71af6abf143409521c80f402fc446` | server-resident reuse after engine replacement |
| NPC-off fresh-salt probe | `269974780bf5516dc814a0265cd3873a4e01f38fd985eaf1e106ac731e45bddb` | LMCache attribution with changed-prefix control |
| NPC-off standard 16K matrix | `7fe6dba1bd7b98a3dc1bb80c994c9db3b6d296e2a85febb8cd9125826aac7003` | all cells valid; C8 fairness failed |
| NPC-off C8 repeats | `4c7d7a9dbd1898ef54131b33580dd84e34172cf590880a4c82ce16dfb33faaf9` | three valid aggregates; fairness failed |
| C0-best standard 16K matrix | `343cff7616ae0442ccab19bc259cb4a58a65eb30542401d497a0d0a507c519d1` | all cells valid; fully unique-context candidate evidence |
| C0-best C8 repeat 1 | `7ebe401e36b9129af40fcd1d17529fe2c50de0538f14c5fac6fe8e33dac3660e` | valid, fair lanes |
| C0-best C8 repeat 2 | `13989d93644ee0c915bf0e033dbe6718c054172a2bc65e307f93bd43c374dfe8` | valid, fair lanes |
| C0-best C8 repeat 3 | `9448601f5f62db6bd26753ceb8816312d630412f0c7f50c4a452ef655a413184` | valid, fair lanes |
| C0-best fresh prefix probe | `31f3f21ee2e00a6128cd2898ca685c6b4f7888b971427b04bed6177f368e83c0` | combined native-prefix-cache plus LMCache warm path |
| C0-best 128-token gate 1 | `0d213edf0d128406b8016e90fcdc653bc093f5d31a8e29cdb470f00419d20a71` | individual gate passed; repeat hash differed |
| C0-best 128-token gate 2 | `3a1112570af053481c9ad18cd657322a0f12de49dd6f5ba020fad71c8f6e441e` | individual gate passed; repeat hash differed |
| C0-best full matrix | `3b74e3b55bf349564fdb7c13fd39c3a23ba285b4ccb22674058d2893b6a3e510` | 13 valid decode cells, 2 invalid capacity-limited cells, 1 suppressed cell |
| CS512 prefill/C1 matrix | `c711fc60d34115e23cb32b7986494e18ec95c0a4943efc7fe94ef9fe7d74bb5a` | matched prefill and C1 evidence |
| CS512 quick 16K matrix | `2bd3f7300b9bf9ae936de5ffd6f69fe14739c4a441de2d8ec9fc370170c4fc8d` | all C1/C2/C4/C8 cells valid |
| CS512 C8 repeat 1 | `d57bf0e6349d4fdec904568ba29ed7d6a9193ff3fda3d7639755b243d8882184` | valid, 8/8 effective |
| CS512 C8 repeat 2 | `5dead9d3b1fc8b57468b69ee78f12c7b70fe0fcc8512720f555341133c77f509` | valid, 8/8 effective |
| CS512 C8 repeat 3 | `a495e09190769446569c331cdbbfb5447c1a0a22c3f1ce4bf522fc21f5b0296b` | valid, 8/8 effective |
| CS512 fresh prefix probe | `32d47f68c0ee4ed5144e952d9f03b0e3813cac84cd64f19e91f380be503bc07e` | combined native-prefix-cache plus LMCache warm path |

### Registration and deterministic gate

C0 reached the API with the expected 524,288-token model limit and explicit
4,500,000,000-byte KV allocation per rank. Every local server registered
exactly one independent `world_size=1` context with the same schema:

| Field | Value on each of four servers |
|---|---:|
| registered contexts | 1 |
| layers | 101 |
| blocks | 2,198 |
| bytes/token | 7,994 |
| reported engine KV capacity | 562,688 tokens |

A deterministic 32-token gate passed. This bounded gate does not supersede the
separate 128-token correctness boundary recorded for the underlying EXL3
configuration.

### Store symmetry and engine-restart reuse

The first 16K cold store produced exactly 152 objects and 311,296,000 bytes on
each server. The original engine was then destroyed. The replacement retained
the same explicit KV allocation, disabled native prefix caching, and set
`gpu_memory_utilization=0.80` only to satisfy admission checks. It did not
shrink the explicit 4,500,000,000-byte KV reservation.

The replacement engine retrieved the same stored prefix in 2.489 seconds,
versus its earlier 35.491-second cold request. Because the four LMCache servers
remained alive while the engine was replaced and native prefix caching was
disabled, this is LMCache server-resident reuse across an engine restart. It is
not proof of NVMe durability or recovery after LMCache-server restart.

### Fresh-salt NPC-off attribution

The fresh-salt negative-control probe kept native prefix caching disabled:

| Request | TTFT |
|---|---:|
| cold seed | 37.615 s |
| changed-first-byte control | 38.158 s |
| warm median | 1.543 s |
| warm/cold ratio | 24.37x |

After this proof, all four servers again matched exactly at 304 objects and
622,592,000 bytes each. The changed-prefix miss and four-server symmetry make
this bounded result attributable to LMCache rather than vLLM's native prefix
cache. It remains a repeated-prefix avoided-compute result, not unique cold
prefill throughput.

### NPC-off decode measurements

The standard v0.4.31 16K matrix completed every requested concurrency without
errors or readiness timeouts:

| Cell | Aggregate tok/s | Validity / fairness |
|---|---:|---|
| C1 | 16.7430 | valid |
| C2 | 28.2000 | valid |
| C4 | 44.1483 | valid |
| C8 | 66.6533 | valid aggregate; one lane was 74.7% of the lane median |

This run used fully shared prompts with native prefix caching disabled, so it
is useful candidate evidence but is not a cold unique-prompt match to B0. The
C8 lane result fails the campaign's fairness gate.

A separate custom C8 three-repeat probe produced three valid aggregate windows:

| Repeat | Aggregate tok/s | Minimum lane / median lane |
|---:|---:|---:|
| 0 | 66.24 | 100.0% |
| 1 | 73.60 | 78.1% |
| 2 | 70.44 | 33.3% |
| median | 70.44 | fairness gate failed |

All requests completed, but the last repeat contained severe lane collapse.
These data support live operation, not performance promotion. A later C0-best
arm corrected the observed lane imbalance but still did not satisfy the
campaign's complete promotion gates.

### C0-best with native prefix caching enabled

C0-best retained the C0 model/KV envelope and LMCache topology, re-enabled
vLLM native prefix caching, and used `gpu_memory_utilization=0.80` as an
admission guard only. The explicit 4,500,000,000-byte KV allocation remained
unchanged.

The standard 16K quick matrix completed all cells:

| Cell | Aggregate tok/s |
|---|---:|
| C1 | 17.1200 |
| C2 | 28.9045 |
| C4 | 42.2924 |
| C8 | 64.2823 |

Three additional C8 windows were valid and lane-fair:

| Repeat | Aggregate tok/s | Minimum lane / median lane |
|---:|---:|---:|
| 0 | 61.2744 | 94.2% |
| 1 | 61.4711 | 92.4% |
| 2 | 60.1809 | 95.1% |

The fresh 16K prefix probe recorded 37.964 seconds cold, 37.244 seconds for
the changed-prefix control, and a 0.5585-second warm median, or 67.97x versus
cold. Native prefix caching and LMCache were both enabled, so this is a
**combined NPC+LMCache warm-path benefit**. It must not be attributed to
LMCache alone or described as increased cold model prefill throughput.

Despite fair repeated C8 lanes, C0-best is not performance-promoted. The quick
artifacts use fully shared prompts, and the campaign lacks the required matched
cold/no-reuse bracket. The later unique-prompt full matrix completed but
contained capacity-limited invalid cells.

### C0-best full unique-prompt matrix

The full standard v0.4.31 matrix used temperature 0, fully unique prompts, a
562,688-token KV budget, and a 1,200-second readiness timeout. It is
**public-functional-lane, external live evidence**. Thirteen decode cells are
valid, two are invalid capacity-limited readiness results, and 128K/C8 was
suppressed after the 128K/C4 capacity failure. The invalid raw rates below are
diagnostics only and must not be quoted as throughput.

Prefill is client-side `prompt_tokens / TTFT`; Prometheus validation is shown
where available. Each row has one sample.

| Requested context | Observed prompt tokens | TTFT | Client tok/s | Prometheus tok/s |
|---:|---:|---:|---:|---:|
| 8K | 8,199 | 16.953 s | 484 | unavailable |
| 16K | 16,254 | 31.937 s | 509 | 510 |
| 32K | 32,347 | 63.359 s | 511 | 511 |
| 64K | 64,536 | 128.234 s | 503 | 504 |
| 128K | 128,911 | 261.297 s | 493 | 494 |

Valid sustained aggregate decode results:

| Context | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| 16K | 16.8040 | 27.6405 | 41.7152 | 58.9600 |
| 32K | 16.4013 | 26.9575 | 41.7200 | 57.3412 |
| 64K | 12.2783 | 27.1937 | 39.6676 | invalid |
| 128K | 17.2000 | 29.5723 | invalid | suppressed |

The 64K/C8 cell timed out after 1,200.172 seconds with only 4/8 requests
running, four errors, and effective concurrency 4. Its raw partial rate was
43.4031 tok/s. The 128K/C4 cell timed out after 1,200.437 seconds with 3/4
requests running, one error, and effective concurrency 2.7. Its raw partial
rate was 29.4784 tok/s. Both are invalid and capacity-limited. The harness then
suppressed 128K/C8 rather than producing a misleading value.

This matrix completes the planned execution but does not promote performance:
long-context admission is constrained by C0's deliberately reduced capacity,
release correctness remains failed/open, and no closing matched B0 bracket has
been collected.

### CS512 final chunk-size tuning arm

CS512 changes exactly one setting from C0-best: LMCache `chunk_size` increases
from 256 to 512. The C0-best model, TP4/DCP4 fixed-MTP2 topology, Q4096/C8/Q32
serving envelope, 4,500,000,000-byte KV allocation, native prefix caching, and
other engine settings are unchanged. LMCache retains a lazy 1-GiB L1.

The first cutover attempt automatically rolled back because the private helper
issued its HTTP readiness probe immediately and raced ordinary startup. No
LMCache runtime failure or benchmark result was established. The helper was
corrected to wait 30 seconds before probing. The second coordinated cutover
succeeded.

At final collection all four engines were running with restart count zero.
Each of the four healthy MP servers reported the same state:

| Server field | Per-server value |
|---|---:|
| `chunk_size` | 512 |
| tracked contexts | 1 |
| objects | 193 |
| bytes | 790,528,000 |
| pending/inflight store | 0 |
| pending/inflight prefetch | 0 |
| L1 | lazy 1 GiB |

Matched prefill improved at every measured context:

| Context | C0 tok/s | CS512 tok/s | Delta | C0 TTFT | CS512 TTFT | CS512 prompt tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 484 | 544 | +12.40% | 16.953 s | 15.063 s | 8,199 |
| 16K | 509 | 553 | +8.64% | 31.937 s | 29.391 s | 16,257 |
| 32K | 511 | 540 | +5.68% | 63.359 s | 59.859 s | 32,349 |
| 64K | 503 | 519 | +3.18% | 128.234 s | 124.359 s | 64,536 |
| 128K | 493 | 510 | +3.45% | 261.297 s | 252.859 s | 128,904 |

The matched CS512 C1 results were 18.9714, 17.8000, 18.7312, and
15.4493 tok/s at 16K, 32K, 64K, and 128K respectively.

The separate fully unique-context 16K quick comparison was:

| Arm | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| C0-best | 17.1200 | 28.9045 | 42.2924 | 64.2823 |
| CS512 | 18.4576 | 29.1496 | 43.3878 | 60.9055 |

Three CS512 C8 repeats completed at 60.8744, 60.6453, and 60.8400 tok/s;
all were 8/8 effective with zero errors, no timeout, and no capacity flag.
Their 60.8400 median is 0.71% below the 61.2744 C0-best repeat median. This
bounded decode cost is accepted for the consistent matched-prefill gain.

The CS512 fresh prefix probe recorded 41.8994 seconds cold, 43.1969 seconds
for the changed-prefix control, and a 0.931086-second warm median, or 45.0006x
versus cold. Native prefix caching and LMCache were both enabled, so this is a
combined warm-path result, not an LMCache-only attribution or cold-prefill
throughput claim.

CS512 is therefore **performance-promoted relative to C0-best** and remains
the live arm. This is public-functional-lane, external-live evidence. It is not
a reference-lane result, clean-checkout reproduction, release acceptance, or
public acceptance. Those broader gates remain blocked by the inherited
token-124 nondeterministic correctness failure and non-identical per-host image
IDs.

## Unchanged B0 serving configuration

| Setting | B0 value |
|---|---|
| hardware | four directly cabled DGX Sparks / GB10s |
| model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| EXL3 layout | legacy `per_expert_v1`, not `shared_h_v1` |
| parallelism | TP4 / PP1 / DP1 / DCP4 |
| MTP | fixed MTP2 |
| DCP / attention | `ag_rs` / `B12X_MLA_SPARSE` |
| model length | 1,048,576 tokens |
| KV | `nvfp4_ds_mla`, 9,000,000,000 bytes/rank, 1,125,632 tokens reported |
| batch / graph | 4,096 batched tokens, maximum 8 sequences, Q32 graph ceiling |
| native prefix cache | enabled |
| chunked prefill / async scheduling | enabled / disabled |
| SparkCache | disabled |
| LMCache | absent and disabled |

B0 used the same local ARM64 image identity and model attestation recorded by
the fixed-MTP2 external recipe. The campaign does not convert B0 itself into a
new recipe.

## B0 evidence

### Artifact inventory

| Artifact | SHA-256 | Valid scope |
|---|---|---|
| standard 16K quick matrix | `89c43fea02c7c42cede9e13108faa9ef5ebbfec42c4e452d19490ee247f2126c` | standard-harness C1/C2/C4/C8 control |
| custom cold-prefill/decode matrix | `71b496fda5658dc6275623f7c1b6da274c4c744b621cf5b76c8373ebb6df55c3` | separate custom harness |
| native prefix-cache probe | `1b6911a7ddccc55cd041103c7b75a7bea387df3330b4672ce22d8795d5f0b2b6` | in-process vLLM prefix reuse only |
| bounded 128-token live gate | `49d34f1391add9dc2eeb3882b30a98389c0ce395b73cacb40ed000b1ffbc57f5` | one passing rerun, not release correctness |

Several earlier overlapping benchmark processes and disposable TP1 smoke
attempts were invalidated and are excluded. They are not part of any table or
comparison in this record.

### Standard 16K decode control

The standard control used `llm_decode_bench.py` v0.4.31, SHA-256
`8de7c32c0abae3c664226fb9c1c197d0752c0a0f3f5a87b3357326f1407f9c07`,
with temperature 0, 25-second measurement cells, a 300-second readiness
timeout, unique prompts, and separate prefill disabled.

| Cell | Aggregate tok/s | Effective concurrency | Errors | Valid |
|---|---:|---:|---:|---|
| C1 | 18.1425 | 1 | 0 | yes |
| C2 | 28.0423 | 2 | 0 | yes |
| C4 | 43.9680 | 4 | 0 | yes |
| C8 | 61.3113 | 8 | 0 | yes |

### Custom cold-prefill control

These are cold, fresh-prompt samples from a separate custom harness. The
observed prompt-token counts exceed their nominal generator targets and are
reported explicitly.

| Nominal target | Observed prompt tokens | Repeats | Median TTFT | Median prompt tok/s |
|---:|---:|---:|---:|---:|
| 16K | 19,472-19,478 | 3 | 34.867 s | 558.63 |
| 32K | 38,933-38,935 | 3 | 71.147 s | 547.22 |
| 64K | 77,848 | 3 | 144.459 s | 538.89 |

The same custom artifact recorded C1/C2/C4/C8 decode windows of
23.16/37.44/54.52/74.60 aggregate tok/s. Different prompts, output limits,
ordering, and measurement mechanics make those numbers unsuitable for direct
comparison with the standard quick matrix.

### Native prefix-cache control

| Request | TTFT |
|---|---:|
| cold seed | 34.991 s |
| warm shared prefix 1 | 1.256 s |
| warm shared prefix 2 | 1.052 s |
| changed-first-byte control | 35.420 s |
| warm shared prefix 3 | 0.800 s |
| warm median | 1.052 s |

The apparent warm/cold ratio was 33.25x. This proves a warm hit in vLLM's
in-process native prefix cache. LMCache was not installed in B0, so this result
does not prove LMCache activation, KV offload, persistent storage, eviction
recovery, cross-process reuse, or restart persistence.

### Bounded correctness gate

One supervised B0 rerun produced two identical greedy fixed-seed 128-token
outputs and passed its short C1/C2/C8 thresholds with post-gate health intact.
Later C0-best repeats each passed their individual throughput gates but
returned different deterministic output hashes:
`de47ae1598231248ecccf12239920226d5000b3680f3ab052ae3e5766b2ce2ad`
and `a241f436e57991a8f242dd2bfa374f833622f44b6688cee22c992a7cfa24deb9`.
The within-gate fixed-seed comparison therefore failed. Together with the
earlier reproducible token-124 branch, this keeps release correctness
failed/open; the 32-token bounded gate remains only a startup/lifecycle check.

## LMCache candidate provenance

The upstream-derived integration is from
`local-inference-lab/LMCache`, based on
`release/v0.5.2-glm52-dcp-base`. The r20 OCI labels report base commit
`9cebd405d0caf4bebe01d694b5a8bf4e3e354314`, integration tree
`a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa`, package version
`0.5.2+glm52dcp.4`, patch SHA-256
`34a0b73b2603ce2fb8c6d9383551871e23981ff2c0b837c000961c38afe73337`,
and integration-lock SHA-256
`85e45dc2afcc2532995731a015504e6f943a22fed4b9b7b37bd0d3321cca6582`.
Those label identities describe the r20 source provenance; the AMD64/sm120 r20
image itself was not run on ARM64 Sparks.

The candidate adds a narrow four-local-server topology patch:

| Candidate item | Identity / evidence |
|---|---|
| topology patch | SHA-256 `4342ee03050e55a466a8016cfd23a5dee5f8ab2bebb152e3ec791575784a0813` |
| focused tests | 22 passed in isolation |
| patched ARM64 wheel | `lmcache-0.5.2+glm52dcp4.1-cp312-cp312-linux_aarch64.whl` |
| wheel SHA-256 | `9e032f6d50e2c44cd05a0d61b78f9d75423e62e2ecac9ddc8e4c738857824fd6` |
| CuPy | 13.6.0, SHA-256 `a3bb49fb023757bfaf0b82c5a1740739a2108ea46d944d699bcff92963c7b87f` |
| `fastrlock` | 0.8.3, SHA-256 `85a49a1f1e020097d087e1963e42cea6f307897d5ebe2cb6daf4af47ffdd3eed` |
| derived ARM64 image | `sha256:211efa7ca520b5b30b391d4ef433153dd297f3421c7c3a52202142c0612a26e3` |

One-Spark bounded tests passed CuPy stream interop, CuPy-to-Torch DLPack,
LMCache CUDA IPC export, cross-process CUDA IPC import, and local MP-server
startup. Those are prerequisite tests, not evidence that the four-rank engine
is ready or serving correctly. Candidate observability is also limited because
the Prometheus OpenTelemetry exporter was absent during the bounded smoke.

## Four-local-server DCP4 contract

Each Spark runs one LMCache MP server local to its GPU. Rank `N` routes its
shard to server `N`; each server-local object uses a local world size and worker
identity of one. The scheduler must query all four ordered servers and accept
only the minimum consensus hit length. A missing, shorter, corrupt,
schema-incompatible, or failed shard must force recomputation rather than a
partial hit.

L0 established that this patched geometry entered coordinated four-rank
initialization, but its full memory envelope failed before readiness. C0 proved
the same geometry at reduced KV/model capacity with lazy L1 allocation. Before
C0 can be promoted as a performance recipe, it still must prove:

1. one-shard-missing, corrupt, timeout, and schema-mismatch recomputation;
2. correct release, cancellation, eviction, and bounded storage accounting;
3. stable cold-path performance against an opening/closing B0 bracket;
4. prolonged mixed hit/miss stability without OOM or restart;
5. server-restart recovery before any durable-persistence claim.

## Cache terminology and attribution

- **Native prefix-cache warm hit** means KV reuse retained inside the running
  vLLM engine under `--enable-prefix-caching`.
- **LMCache hit** requires connector/server evidence for every required DCP
  shard. TTFT alone is insufficient attribution.
- **LMCache offload** requires measured movement of KV bytes from the
  engine-managed cache into a configured CPU-memory or NVMe tier.
- **LMCache persistence** requires a hit after the original engine/container
  state is gone, from a durable named backend with matching namespace and
  schema.
- **Cold prefill throughput** uses unique or fresh-salt prompts with no reusable
  prefix. A warm-hit TTFT reduction is avoided computation, not increased
  unique-prompt model throughput.

Client, filesystem, and operating-system page caches must also be controlled
or reported when interpreting a warm result.

## Arm register and promotion gates

| Arm | Intended single delta | Required proof | State |
|---|---|---|---|
| B0-close | repeat unchanged LMCache-disabled control | drift bracket, health, C1-C8 | pending |
| L0 | four local LMCache servers with patched DCP4 topology; full B0 memory envelope; eager 1-GiB L1 | readiness and registration | rejected: CUDA OOM during `REGISTER_KV_CACHE` |
| NPC-off | disable native prefix cache after a viable LMCache capacity arm | attribute reuse to LMCache | bounded attribution proof complete |
| L1 | bounded LMCache DRAM tuning | cold overhead, unified-memory bound, fair C8 | future |
| L2 | bounded NVMe backend | measured IO, eviction, capacity, restart restore | future |
| C0 | 524,288 model envelope, 4.5 GB KV/rank, lazy 1-GiB L1 starting at zero | lifecycle, performance, and fairness | lifecycle live-validated; performance not promoted |
| P32 | qualified mixed K3/K4 block-32 prefill route | paired vLLM/SparkInfer source chain and correctness | blocked/no-go by corrected offline audit |
| CKV0/CKV1 | CKV gather with prefetch depth 0 versus 1 | exact base preimages, resolved source chain, SM121 correctness | blocked/no-go by corrected offline audit |
| Q6144 | Q4096 to Q6144 after earlier arms stabilize | at least 5% bracketed cold-prefill/admission gain and at most 3% decode regression | deferred |
| BEST | C0 with native prefix caching and LMCache enabled | correctness, soak, rollback proof | full matrix complete; not promoted |
| CS512 | LMCache `chunk_size` 256 to 512; C0-best otherwise unchanged | matched prefill and bounded decode | performance-promoted; live |

Performance-arm promotion requires repeatable all-shard hits, bounded
resources, meaningful matched gains, and no C8 readiness failure or unfair
lane collapse. Release or public acceptance separately requires correctness,
reproduction, and identical image attestation on every host.

The legacy mixed K3/K4 checkpoint uses a fixed mixed-route block size of eight;
`VLLM_EXL3_PREFILL_BLOCK_M=64` affects the uniform path and is not a credible
arm here. Earlier Q8192 and wide prefill-graph experiments were flat or
negative, so they are not repeated in this campaign.

## Remaining disposition gates

L0 has been rejected and its exact-label rollback completed. C0 passed bounded
readiness and lifecycle gates. C0-best produced fair repeated C8 lanes, but is
not performance-promoted and failed repeat release correctness. Its full matrix
completed with 13 valid, two invalid capacity-limited, and one suppressed decode
cell. CS512 supersedes C0-best only as the live performance arm. A closing B0
bracket, additional resource and eviction accounting, failure-path controls,
and a mixed hit/miss soak remain useful before recipe publication. Bounded
engine-restart reuse is not durable persistence or full acceptance.

P32 and CKV remain future no-go items based only on a corrected offline source
audit. P32 requires a coordinated vLLM and SparkInfer upgrade plus unresolved
ABI/correctness work; it is not an isolated block-size override. CKV lacks the
exact inherited base-file preimages and resolved source chain needed for a safe
port, and SM121 correctness is unproven. The audit is not public-source or live
validation of either feature, and neither should be attempted from the
retracted first-pass design.

CS512 is the final live performance arm. Its promotion is relative to C0-best
and does not supersede the failed/open release-correctness result. Public or
release acceptance additionally requires one identical, attested image ID on
all four hosts; the final live deployment has non-identical per-host image IDs.
