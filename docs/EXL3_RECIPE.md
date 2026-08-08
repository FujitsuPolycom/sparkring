# GLM-5.2 EXL3 3.25-bpw recipe

Status: **default, main advertised, and currently running public-functional
configuration; clean-checkout four-Spark live-validated; not fully accepted**

This recipe defines the current public EXL3 serving contract for four directly
cabled DGX Sparks:

[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
at revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`.

The machine-readable source of truth is
[`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json).
It contains the model hashes, EXL3 source commits, complete static environment,
and vLLM argument vector. Site-specific addresses, SSH users, NIC names, GID
indices, and paths remain in the ignored site configuration.

## Current contract

| Setting | Value |
|---|---|
| Hardware | 4x DGX Spark, direct 200-Gb/s cycle |
| Parallelism | TP4 / DCP4, `ag_rs` |
| Quantization | EXL3/Trellis 3.25 bpw; K3x192 and K4x64 expert tiers |
| MTP | fixed MTP2 |
| Batching | 8 sequences (C8), 4,096 batched tokens, Q32 graph ceiling |
| Maximum model length | 524,288 tokens |
| KV allocation | 4,500,000,000 bytes per rank |
| Reported KV capacity | 562,688 tokens |
| KV representation | NVFP4 latent plus FP8 RoPE, per-token scale |
| Attention | `B12X_MLA_SPARSE`, CKV gather, exact global top-k |
| CUDA graphs | full and piecewise through Q32 |
| Prefix cache | enabled |
| SparkCache | disabled |
| LMCache | CS512; one local MP server per rank, 512-token chunks, lazy 1-GiB L1 |
| Served name | `glm-5.2-exl3-tr3-3.25bpw` |

The Q4096/C8/Q32 relationship is part of the fail-closed contract: the engine
accepts at most 4,096 batched tokens, eight sequences, and 32 query rows. The
LMCache launcher starts one host-local server for each rank, verifies server
health, and only then starts the four distributed vLLM engines.

The fixed-MTP2 engine profile and the later LMCache CS512 campaign were first
validated as external operator configurations. Their exact deltas, bounded
gates, and evidence limitations are recorded in the
[DCP4 fixed-MTP2 recipe](EXL3_FIXED_MTP2_RECIPE_20260802.md) and
[LMCache campaign](EXL3_LMCACHE_CAMPAIGN_20260803.md). Publishing the same
settings in the executable recipe does not relabel those earlier external runs.
A later clean-checkout deployment of the public bootstrap has its own bounded
receipt below. In that exact deployment, the repeated 128-token gate passed;
the earlier external token-124 divergence remains part of the external
campaign record.

## Inspect the recipe offline

```bash
python scripts/sparkring_recipe.py plan \
  --recipe glm52-exl3-tr3-3.25bpw
```

This verifies the immutable model metadata, source pins, topology assumptions,
Q4096/C8/Q32 relationship, fixed-MTP2 policy, packed-KV settings, LMCache CS512
contract, and required explicit unsets. It does not contact or alter any Spark.

## Build and launch from public sources

Complete [PREREQUISITES.md](PREREQUISITES.md) and the site/SSH/fabric sections
of [QUICKSTART.md](QUICKSTART.md) first. Use the same ignored `site.yaml` as
the NF3 bootstrap. Plan mode is entirely offline:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

The first execution is intentionally explicit and should normally use
`--no-launch` so the generated files can be reviewed before serving:

```bash
python scripts/bootstrap_exl3.py execute \
  --site scripts/config/site.yaml \
  --no-launch \
  --confirmation BOOTSTRAP-EXL3-ALL-FOUR
```

It performs these receipt-gated operations:

1. verifies management SSH and the exact three-hop direct-ring fanout tree;
2. verifies the 200 GbE/RDMA fabric before downloading or building;
3. builds or reuses the exact NF3 NVFP4/FP8-RoPE base layer;
4. adopts or resumes the pinned 81-shard model on rank 0, with a capacity gate
   and verification of every execution input against the pinned Hugging Face
   manifest;
5. copies model bytes to both neighbors in parallel, then relays to the opposite
   rank over direct 200 GbE addresses using resumable `rsync`;
6. reconstructs the exact public ExLlamaV3, SparkInfer, and LMCache Git trees
   from pinned base commits plus hash-checked patches;
7. builds and verifies one derived ARM64 image on rank 0, then fans that exact
   image ID over the same direct-ring tree; and
8. writes ignored resolved files under `.sparkring/bootstrap-exl3/` and runs
   the full public preflight.

Review the generated contract:

```bash
cat .sparkring/bootstrap-exl3/site.yaml
cat .sparkring/bootstrap-exl3/launch.json

python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  plan
```

Then start exactly that profile:

```bash
python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute \
  --confirmation START-EXL3-LMCACHE-CS512-ALL-FOUR \
  start
```

Both the model download and derived-image build are resumable: exact existing
payloads are verified and skipped. Partial Hugging Face and `rsync` transfers
resume in place. First-time adoption hashes all 88 execution inputs: 81 weight
shards and seven runtime metadata files. It can take several minutes without
consuming GPU time. Mutable source heads, incomplete or same-size-corrupted
shard sets, wrong metadata, unattested source trees, mismatched base-image IDs,
and dirty build contexts fail closed.

The pinned revision has one upstream publication inconsistency: the served
`README.md` bytes disagree with the README digest recorded in that revision's
`MANIFEST.sha256`. The README is not a model execution input. The downloader
therefore verifies the manifest identity, downloads and hashes all 88 runtime
inputs, and deliberately excludes nine manifested release-only sidecars from
content verification. This exception does not include weights, configuration,
tokenizer files, the chat template, generation settings, the tier bitmap, or
the safetensors index. Unmanifested non-cache files are still rejected.

## Clean-checkout four-Spark receipt

The public source composition, full vLLM EXL3 overlay, derived-image builder,
model verifier, 200 GbE fanout, and dry-run-first launcher were exercised from
a clean checkout on four directly cabled DGX Sparks. The image source was
commit `19523482c29860024c3a3cf51e793e8436e1c441`; launcher correction
`cc9cc1e` was then used for deployment.

| Gate | Observed result |
|---|---|
| Exact image fanout | `sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f` on all four ranks |
| Post-stop preflight | 116/116 passed |
| Processes | four engines + four LMCache CS512 servers; zero restarts |
| Model/API | 524,288 maximum model length; 562,688 reported KV tokens |
| Model load | 84.43 GiB/rank |
| CUDA graphs | 16/16 piecewise and 12/12 full captures |
| Repeated live gate | five consecutive C1/C2/C8 gate runs passed every floor |
| Fixed-seed output | ten identical 128-token completions; SHA-256 `a310b67d304b36f5dea88cbbcb18ba7be640001cc463590fe4e8cbb31042131c` |
| Standard sustained decode | unique 16K C1/C2/C4/C8: 18.33 / 27.61 / 45.11 / 59.40 aggregate tok/s; exact requested concurrency; zero errors |
| Offline suite | local: 2,046 passed, 13 skipped; clean host: 2,035 passed, 4 skipped, 113 subtests |

This makes EXL3+LMCache CS512 the default, main advertised, and currently running
public-functional configuration. The evidence is intentionally bounded: it
does not prove blanket model correctness, LMCache persistence, arbitrary-host
reproducibility, release promotion, or the complete public-functional
acceptance matrix. NF3 remains an accepted deterministic alternative.
SparkCache is a separate implementation
and is disabled in this profile.

The offline-validated candidate workflow for closing those blockers is the
[EXL3 + LMCache acceptance runbook](EXL3_ACCEPTANCE_RUNBOOK.md). Its published
plan does not itself close a blocker: broader correctness, restart-boundary
behavior, resources, rollback, and performance still require a reviewed live
bundle from the immutable profile.

Do not substitute mutable model tags, source heads, image tags, or arbitrary
EXL3 wheels and call the result this recipe.

## Reproduce the bounded live gate

After the launcher reports that all four LMCache servers and all four engines
are running and the API is ready, first re-attest the deployed bytes and check
both halves of the profile:

```bash
python scripts/sparkring_exl3_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute verify-image

python scripts/sparkring_exl3_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute verify-model

python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute \
  status
```

Then run the bounded API gate. The initial public-path regression floors are
deliberately below the short maintainer sanity measurements; they detect a
broken path rather than claim a final performance band:

```bash
python scripts/exl3_live_gate.py \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --max-tokens 128 \
  --timeout-seconds 1800 \
  --min-c1-tps 15 \
  --min-c2-tps 24 \
  --min-c8-tps 35 \
  | tee .sparkring/bootstrap-exl3/live-gate.json
```

This requires two byte-identical greedy fixed-seed outputs, measures C1/C2/C8,
enforces the stated floors, and checks `/health` both before and after the
matrix. A successful clean-checkout run must retain its generated site/profile,
image IDs, bootstrap output, live-gate JSON, and four-rank logs as the
publication receipt. These are candidate regression floors, not reference-lane
performance claims.

For a standard sustained-decode check, use `llm_decode_bench.py` v0.4.31 with
16K context, concurrency 1/2/4/8, 25-second cells, 2,048 maximum tokens,
temperature 0, 100% unique contexts, DCP4, KV budget 562688, three-second decode
warmup, and prefill skipped. Set the cell-warmup timeout to 300 seconds. The
default automatic 60-second readiness allowance was insufficient on the clean
deployment and suppressed C2/C4/C8 even though KV capacity was not exhausted.
Only quote cells whose effective concurrency equals the request and whose
error count is zero.
