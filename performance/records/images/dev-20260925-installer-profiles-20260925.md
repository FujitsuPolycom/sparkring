# Installer image dev-20260925: six-profile installation check

Status: **implemented; functional checks passed; single-run timing; not
serving-qualified**.

The [machine-readable record](dev-20260925-installer-profiles-20260925.json)
covers one installation of each installer profile on image
`dev-20260925-cuda1342-nccl2323-status031` (configuration
`sha256:5ce6ce267d8069215e08da87be3a8092175e63806cf97730847405eb04a2c51f`,
registry manifest
`sha256:9d8e3a6cd00b887ffd33496e43afa06cffeebbfc88008361d7569005c4ae49c6`).
Installer package revision `eb8ec2ba3b17` installed each profile with
`sparkring install` on one directly cabled Spark pair (TP2) and one four-Spark
ring (TP4). Each start was the profile's first on this image, so compile and
tuning caches were empty.

## Method

The [measurement directory](dev-20260925-installer-profiles-20260925/)
holds the two measurement programs and the raw benchmark output of each profile;
the record lists their SHA-256 values.

- `verify_api.py` sends three temperature-0 chat requests: count from 1 to 20,
  compute 17×23, and write `is_prime(n)`. A fourth request samples 1,024
  tokens at temperature 0.7 with `ignore_eos` and reports end-to-end tokens per
  second. Qwen and MiMo requests disable thinking with
  `chat_template_kwargs.enable_thinking=false`. GLM-5.3 requests set
  `reasoning_effort: low`, because with thinking disabled GLM-5.3 writes its
  reasoning into the answer.
- `bench.py` measures prefill as the time to the first streamed token of a
  one-token `/completions` request for prompts of about 4K, 16K and 64K
  tokens. A random prefix on every prompt prevents prefix-cache reuse; each
  size runs twice and the second run is reported. Greedy decode is three
  512-token chat completions with `ignore_eos` on a fixed story prompt; each
  rate is completion tokens divided by request time, which includes time to
  first token. All `bench.py` chat requests disable thinking, so GLM-5.3's
  counted tokens include the reasoning it writes into the answer.
- First-start readiness is the installer's "Wait for API readiness" step on
  Node A: from container start until the model API answers.

## Results

| Profile | First-start readiness | Checks | Prefill 4K / 16K / 64K (tokens/s) | Greedy decode median (tokens/s) | Sampled decode (tokens/s) |
|---|---|---|---|---|---|
| `qwen38-flash-next-tp2` | 546.8 s | passed | 4006 / 4060 / 3770 | 48.5 | 47.0 |
| `qwen38-flash-next-qad-tp4` | 476.9 s | passed | 4690 / 4484 / 4179 | 69.9 | 63.3 |
| `glm53-flash-nvfp4-spark-tp2` | 581.8 s | passed | 1215 / 2419 / 2337 | 33.5 | 29.3 |
| `glm53-flash-nvfp4-spark-tp4` | 483.6 s | passed | 2806 / 2947 / 2888 | 60.9 | 56.4 |
| `mimo-v26-flash-rl-tp2` | 640.0 s | passed | 3744 / 3806 / 2936 | 25.8 | 27.8 |
| `mimo-v26-flash-rl-tp4` | 543.6 s | passed | 4243 / 4235 / 3631 | 44.4 | 49.9 |

## Limits

Each profile has one installation and one measurement series at concurrency 1.
The runs include no media, full-context pressure, concurrency sweep, restart
cycle or soak. The series ran minutes after each first start, so a prompt size
used for the first time can include kernel compilation: the GLM-5.3 TP2 4K
request took 3.35 s. After a later restart of that deployment with its caches
filled, the best of three 3,907-token requests took 1.68 s (2,330 tokens/s),
in line with its other sizes. These observations do not establish serving
qualification or compare this image with another image.
