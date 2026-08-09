# Four-Spark EXL3 + LMCache quickstart

This is SparkRing's default, main advertised, and currently running public-functional
configuration on four directly cabled DGX Sparks. It serves
`willfalco/GLM-5.2-EXL3-TR3-3.25bpw` at immutable revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` with TP4/DCP4, fixed MTP2,
Q4096/C8/Q32, 524,288 maximum model length, 4.5 GB KV/rank, native prefix
caching, and one LMCache CS512 server per rank. SparkCache is a separate
implementation and is disabled in this profile.

The public path was built from a clean checkout and live-validated on four
directly cabled Sparks. NF3 remains an accepted deterministic alternative
documented in [NF3_QUICKSTART.md](NF3_QUICKSTART.md). The EXL3 result is
bounded live validation, not blanket correctness, LMCache persistence, release
promotion, or complete public-functional acceptance.

## 1. Complete the prerequisites

Read [PREREQUISITES.md](PREREQUISITES.md) completely. You need four ARM64 DGX
Sparks with the direct 200-Gb/s cycle cabled and qualified, management SSH from
rank 0 to every rank, Docker with GPU access, enough storage for the 339-GB
model plus build/image headroom, and a filled ignored site configuration.

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

Do not commit `scripts/config/site.yaml`. It contains local identities and
paths. Review every resolved rank, NIC, GID, direct-ring peer, model path, and
storage path before proceeding.

## 2. Inspect the immutable recipe

```bash
python scripts/sparkring_recipe.py plan

python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

Both commands are offline. Confirm the immutable model revision, fixed-MTP2
policy, TP4/DCP4 topology, Q4096/C8/Q32 limits, packed-KV profile, and LMCache
CS512 topology. The recipe declares `default: true`; NF3 remains explicitly
selectable by recipe ID.

## 3. Build, verify, and distribute without launching

This step mutates all four named hosts, transfers hundreds of gigabytes, and
builds an ARM64 image. It does not replace the running model when `--no-launch`
is present.

```bash
python scripts/bootstrap_exl3.py execute \
  --site scripts/config/site.yaml \
  --no-launch \
  --confirmation BOOTSTRAP-EXL3-ALL-FOUR
```

The bootstrap verifies the fabric, adopts or resumes the exact 81-shard model,
hashes all runtime inputs, fans model bytes through the direct ring, reconstructs
the pinned source trees, builds one receipt-gated image on rank 0, verifies it,
and distributes that exact image ID. It writes ignored resolved artifacts to
`.sparkring/bootstrap-exl3/`.

Review them before serving:

```bash
cat .sparkring/bootstrap-exl3/site.yaml
cat .sparkring/bootstrap-exl3/launch.json

python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  plan
```

Do not bypass a receipt, model hash, image-ID, capacity, topology, or preflight
failure. Existing listeners on the API or rendezvous ports must be handled as
an intentional cutover, not ignored.

## 4. Start the reviewed profile

This command stops serving if it replaces a live stack. Obtain authorization
for the exact four hosts and preserve a rollback path first.

```bash
python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute \
  --confirmation START-EXL3-LMCACHE-CS512-ALL-FOUR \
  start
```

The launcher starts one host-local LMCache server per rank, waits for all four
servers, then starts the distributed engines. Watch the rank-0 log until model
load, KV allocation, and all configured graph captures complete. Do not send a
load test merely because the containers exist.

## 5. Verify the deployed bytes and processes

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
  --execute status
```

Require the same image ID on all ranks, four healthy LMCache servers, four
healthy engines, zero unexpected restarts, `/health` HTTP 200, and the exact
served model plus 524,288 maximum length from `/v1/models`.

## 6. Run the bounded API gate

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

The gate checks health before and after the workload, requires byte-identical
fixed-seed output, and enforces deliberately conservative C1/C2/C8 regression
floors. Repeat it before promoting a deployment. Retain the generated
site/profile, image and model receipts, bootstrap output, gate JSON, and all
four rank logs.

### Optional standard 16K sustained-decode matrix

Use `llm_decode_bench.py` v0.4.31 with these settings: 16K context;
concurrency 1,2,4,8; duration 25 seconds; maximum 2,048 tokens; temperature 0;
100% unique contexts; DCP4; KV budget 562688; three-second decode warmup; and
prefill skipped. Duration mode supplies `ignore_eos`. Set
`--cell-warmup-timeout-seconds 300`.

Equivalent invocation:

```bash
python /path/to/llm_decode_bench.py \
  --host <rank-0-management-address> \
  --port 8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --contexts 16k \
  --concurrency 1,2,4,8 \
  --duration 25 \
  --max-tokens 2048 \
  --temperature 0 \
  --unique-context-percent 100 \
  --dcp-size 4 \
  --kv-budget 562688 \
  --decode-warmup-seconds 3 \
  --skip-prefill \
  --cell-warmup-timeout-seconds 300 \
  --output .sparkring/bootstrap-exl3/decode-16k.json
```

The default automatic 60-second readiness allowance can be too short for the
unique-context C2/C4/C8 cells on this profile. A readiness timeout is not proof
of KV exhaustion: reject the cell, increase only the cell-warmup timeout to
300 seconds, and rerun. A valid result requires exact effective concurrency
and zero request errors. Do not quote the harness's suppressed raw cells.

## Validated public-path receipt

The 2026-08-03 clean-checkout run used image-source commit
`19523482c29860024c3a3cf51e793e8436e1c441` and launcher fix `cc9cc1e`. Image
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`
was identical on four ranks. Post-stop preflight passed 116/116 checks; four
engines and four LMCache servers started with zero restarts; 16/16 piecewise
and 12/12 full graphs captured. Five consecutive bounded gates passed, and ten
fixed-seed 128-token completions had SHA-256
`a310b67d304b36f5dea88cbbcb18ba7be640001cc463590fe4e8cbb31042131c`.
The standard unique-16K sustained matrix then returned 18.33 / 27.61 / 45.11 /
59.40 aggregate tok/s at C1/C2/C4/C8, with exact effective concurrency and zero
errors in every cell. These are public-functional clean-checkout live results,
not reference-lane measurements.

For the full source, model-manifest, launcher, and evidence boundaries, see
[EXL3_RECIPE.md](EXL3_RECIPE.md).

## Troubleshooting

For long double 315.78-GiB verification, page-cache pressure, bounded RM
allocation retries versus fatal signals, headless Wi-Fi, rollback failures,
PowerShell log tailing, and LMCache CS512 geometry verification, see
[EXL3_TROUBLESHOOTING.md](EXL3_TROUBLESHOOTING.md).

## Offline geometry verification

Before a live acceptance run, verify the LMCache CS512 block-256 geometry:

```bash
python scripts/exl3_cache_geometry_gate.py verify
python scripts/exl3_cache_geometry_gate.py plan
```

The first command verifies configuration facts from the recipe: chunk_size=512,
parent_chunk_size=256, lazy L1, LRU eviction, one-server-per-rank topology,
SparkCache disabled (SPARK_CONTEXT_CACHE_ENABLE=0), and APC enabled
(--enable-prefix-caching). It also lists planned live gates that require a
cluster: boundary token counts (511/512/513/1024/1025), DCP consensus
evidence (object counts + TTFT ratios, not hit counters), and capacity
metrics from /status (eviction_count is unavailable). The second discloses
the C1/C2/C4/C8 and 16K/64K cold/warm timing cells that require a live
cluster.

## Startup evidence classification

After capturing engine, kernel, and LMCache server logs, classify startup
evidence into four verdicts: clean, bounded_rm_retry, fatal, and
indeterminate:

```bash
python scripts/sparkring_startup_evidence.py \
  --engine-log engine-r0.log --kernel-log kernel-r0.log \
  --engine-log engine-r1.log --kernel-log kernel-r1.log \
  --engine-log engine-r2.log --kernel-log kernel-r2.log \
  --engine-log engine-r3.log --kernel-log kernel-r3.log \
  classify
```

Generic `CUDA out of memory` is fatal by default. Only kernel RM
`NV_ERR_NO_MEMORY` at `_memdescAllocInternal @ mem_desc.c:1359` during
EXL3 materialization can be `bounded_rm_retry`, with full cross-evidence.
NVIDIA errors are never globally ignored; each signature is classified
individually with provenance (line number, pattern label, SHA-256). The
`bounded_rm_retry` classification is evidence-scoped to the legacy EXL3
materialization callsite and does not prove all RM errors safe.
