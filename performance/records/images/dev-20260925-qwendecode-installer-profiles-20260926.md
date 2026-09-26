# Installer image dev-20260925-qwendecode: six-profile installation with checkpoint adoption

Status: **implemented; functional checks passed; single-run timing; not
serving-qualified**.

The [machine-readable record](dev-20260925-qwendecode-installer-profiles-20260926.json)
covers one installation of each installer profile on image
`dev-20260925-qwendecode-cuda1342-nccl2323-status031` (configuration
`sha256:4100e1d2bd038f885d92f8c0021d482b23f9a38a003cd7bfc3e700e7e0afa971`)
by `sparkring install`, on one directly cabled Spark pair (TP2) and one
four-Spark ring (TP4). Every installation found the profile's checkpoint
already on each Spark, hard-linked its weight files into SparkRing's
checkpoint directory, copied the other files, and downloaded nothing.

- Source revision `aef20cd623de` (checkpoint discovery and adoption):
  `qwen38-flash-next-qad-tp4`, `glm53-flash-nvfp4-spark-tp2`,
  `glm53-flash-nvfp4-spark-tp4`.
- Source revision `5d184718ead6` (the same plus automatic ConnectX hairpin
  application): `mimo-v26-flash-rl-tp2`, `mimo-v26-flash-rl-tp4`, then both
  Qwen profiles again, reusing their checkpoint directories.

## Method

[`measure.sh`](../qwen38-flash-next/installer-tuning-20260925/programs/measure.sh)
ran against each installation: counting, arithmetic and code checks with
thinking disabled; 512-token single-stream decode by prompt type at
temperature 0 (two runs) and 1.0 (three runs); prefill as the time to the
first token of one cold prompt of about 4K, 16K and 64K tokens (second of two
runs); and three greedy 512-token decodes. The
[measurement directory](dev-20260925-qwendecode-installer-profiles-20260926/)
holds each run's output. GLM-5.3 writes its reasoning into the answer when
thinking is disabled, so its check answers begin with that reasoning.

## Results

| Profile | Source | Checkpoint steps | API readiness | Prose / code / JSON (tok/s) | Prefill 4K / 16K / 64K (tok/s) | Greedy median |
|---|---|---|---|---|---|---|
| `qwen38-flash-next-tp2` | `5d184718ead6` | 0.4–2.0 s (reused) | 267.6 s | 59.5 / 88.4 / 101.0 | 4,155 / 4,259 / 3,922 | 60.7 |
| `qwen38-flash-next-qad-tp4` | `aef20cd623de` | about 19 s per rank (adopted) | 234.3 s | 87.0 / 126.7 / 140.6 | 4,884 / 5,099 / 4,720 | 85.3 |
| `qwen38-flash-next-qad-tp4` | `5d184718ead6` | 0.2–2.4 s (reused) | 234.2 s | 87.9 / 126.8 / 142.2 | 4,789 / 5,003 / 4,723 | 86.4 |
| `glm53-flash-nvfp4-spark-tp2` | `aef20cd623de` | adopted | 590.2 s | 32.3 / 36.8 / 40.0 | 2,297 / 2,440 / 2,139 | 33.6 |
| `glm53-flash-nvfp4-spark-tp4` | `aef20cd623de` | about 32 s per rank (adopted) | 471.6 s | 59.0 / 69.5 / 73.2 | 2,732 / 2,900 / 2,868 | 60.2 |
| `mimo-v26-flash-rl-tp2` | `5d184718ead6` | adopted | 629.4 s | 25.7 / 49.8 / 62.8 | 3,810 / 3,802 / 2,949 | 26.2 |
| `mimo-v26-flash-rl-tp4` | `5d184718ead6` | about 31 s per rank (adopted) | 560.6 s | 44.8 / 93.3 / 111.5 | 4,190 / 4,253 / 3,636 | 45.7 |

Decode by prompt type is at temperature 0. At temperature 1.0 prose decoded
at 59.6 (Qwen TP2), 79.2 (Qwen TP4, `5d184718ead6`), 32.2 (GLM TP2), 59.8
(GLM TP4), 22.7 (MiMo TP2) and 40.5 (MiMo TP4) tokens per second. GLM and
MiMo API readiness includes the first kernel tuning of those models on this
image. Every installation passed the counting, arithmetic and code checks and
the installer's check that no checkpoint file changed during loading.

## Checkpoint adoption on hardware

On TP2, with a private copy of the Qwen checkpoint and fixtures made of hard
links to it (a `main`-branch Hugging Face cache, a folder with two shards
exchanged, and a download folder with an arbitrary name):

- The search took 4.6–7.0 s per Spark, completed, and did not enter the
  network automount `/mnt/synologytwo`.
- A folder with two exchanged shards stopped the installation with
  `needs_input` naming both shards before any download or model change.
- An installation that would have downloaded more than the plan reviewed with
  `--plan` stopped with `needs_input` and changed nothing.
- Adoption from the `main` cache linked the shards, downloaded only
  `config.json` (1.5 s, after the image checks), streamed 42.3 GiB to the
  worker over the fabric in 38.8 s, and served.
- The operator's own copy stayed unchanged in every recorded field on both
  Sparks, and the previous deployment's receipts stayed byte-identical; that
  deployment started again without re-hashing.

On TP4, adoption of the operator's copies on all four Sparks changed only the
link count and change time of their 36 weight files, and refreshed the
recorded file identities of the retained deployments. The search on the rank
with an NFS automount finished in 6.3 s without triggering it.

## ConnectX hairpin setting on hardware

On TP4, whose functions held the setting from a manual application,
`sudo sparkring hairpin --yes` (source `5d184718ead6`) updated the workers,
recorded the setting and enabled `sparkring-hairpin.service` on all four
Sparks in 82 s while a model kept serving. All 16 functions were in effect
before and after, and every `driver_reinit` counter stayed at 1, so no
driver restarted. The following four-Spark installation reported the
setting as kept. Applying the setting through a driver restart, during
installation or at boot, has no hardware evidence.

## Limits

Each profile has one installation and one measurement series at concurrency
1. No run started from Sparks that held neither the image nor the checkpoint,
repeated a failed preparation, rebooted a Spark, or measured media,
full-context pressure, concurrency or stability.
