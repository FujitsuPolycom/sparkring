# Four-Spark quickstart

SparkRing's public default is GLM-5.2 EXL3 3.25-bpw with LMCache CS512 on four
directly cabled NVIDIA DGX Sparks:

```text
willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d
TP4 / DCP4 / fixed MTP2 / Q4096 / C8 / Q32
524,288-token model limit / 4.5 GB KV per rank
native prefix cache + one LMCache CS512 server per rank
```

This public-functional configuration is **live-validated** on four directly
cabled DGX Sparks. Its clean-checkout receipt covers exact model verification,
an identical four-rank image, 116/116 preflight checks, startup and graph
capture, API health, repeated fixed-seed output, bounded concurrency gates, and
post-run health. It is not a blanket correctness, persistence, release, or full
acceptance claim.

The accepted deterministic NF3 configuration is available as an explicit
alternative in [NF3_QUICKSTART.md](NF3_QUICKSTART.md).

## 1. Prepare the four Sparks

Complete [PREREQUISITES.md](PREREQUISITES.md). You need four ARM64 DGX Sparks,
the qualified direct 200-Gb/s cycle, management SSH from rank 0 to all ranks,
Docker with GPU access, and storage for the 339-GB checkpoint plus image/build
headroom.

On rank 0:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml

python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py \
  --site scripts/config/site.yaml \
  --print-plan
```

Review every host, interface, direct-ring peer, model path, and storage path.
The real `site.yaml` is ignored and must not be committed.

## 2. Confirm the default recipe

No `--recipe` selection is necessary:

```bash
python scripts/sparkring_recipe.py plan
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

The first command must report `RECIPE: glm52-exl3-tr3-3.25bpw` and direct the
operator to `bootstrap_exl3.py`.

## 3. Prepare without interrupting serving

This mutates the four named hosts and transfers/builds hundreds of gigabytes,
but `--no-launch` does not replace a running model:

```bash
python scripts/bootstrap_exl3.py execute \
  --site scripts/config/site.yaml \
  --no-launch \
  --confirmation BOOTSTRAP-EXL3-ALL-FOUR
```

The resumable, fail-closed bootstrap verifies all 81 model shards, composes the
pinned runtime sources, builds one receipt-gated ARM64 image on rank 0, fans the
exact image and model bytes across the direct ring, and writes ignored resolved
artifacts under `.sparkring/bootstrap-exl3/`.

## 4. Review and launch

```bash
python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  plan

python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute \
  --confirmation START-EXL3-LMCACHE-CS512-ALL-FOUR \
  start
```

The second command is **STOPS SERVING** when it replaces an existing stack.
Inspect the generated plan and preserve a rollback path before running it.

## 5. Verify the deployment

Require four healthy engines, four healthy LMCache servers, an identical image
ID on every rank, zero unexpected restarts, `/health` HTTP 200, and the exact
served model from `/v1/models`:

```bash
python scripts/sparkring_exl3_lmcache_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute status

python scripts/exl3_live_gate.py \
  --base-url http://<rank-0-management-address>:8000 \
  --model glm-5.2-exl3-tr3-3.25bpw \
  --max-tokens 128 \
  --timeout-seconds 1800 \
  --min-c1-tps 15 \
  --min-c2-tps 24 \
  --min-c8-tps 35
```

Read [EXL3_QUICKSTART.md](EXL3_QUICKSTART.md) for model/image re-attestation,
log-tail commands, the standard 16K benchmark, readiness-timeout
troubleshooting, and the complete validated receipt.

The bounded gate is the normal deployment check. Maintainers preparing a
public acceptance result should continue with the dry-run-first
[EXL3 + LMCache acceptance runbook](EXL3_ACCEPTANCE_RUNBOOK.md). That workflow
adds broader token-ID correctness, repeated C1/C2/C4/C8 cells, cache restart
boundaries, resource/OOM monitoring, and exact-label rollback. It interrupts
serving only in explicitly confirmed execute mode.

## NF3 alternative

NF3 is published and supported. Select it explicitly:

```bash
python scripts/sparkring_recipe.py plan --recipe glm52-nf3-hybrid
python scripts/bootstrap_nf3.py plan --site scripts/config/site.yaml
```

Continue in [NF3_QUICKSTART.md](NF3_QUICKSTART.md).
