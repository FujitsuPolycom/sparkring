# EXL3 + LMCache public acceptance profile

The acceptance workflow for the default four-DGX-Spark GLM-5.2 EXL3 3.25-bpw
plus LMCache profile with 512-token cache chunks (CS512). It is a
`public-functional` candidate workflow: passing it does not upgrade the profile
from `live-validated` to `accepted`, which only a reviewed live evidence bundle
does. NF3 is the accepted deterministic alternative.

## The profile under test

| Setting | Required value |
|---|---|
| Hardware | four directly cabled DGX Sparks / GB10s |
| Model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| Parallelism | TP4 / DCP4 / PP1 |
| Speculation | fixed MTP2 |
| Context and KV | 524,288 tokens; 4,500,000,000 KV bytes/rank |
| Scheduling | maximum 8 sequences; 4,096 batched tokens; Q32 graphs |
| Cache | native prefix cache plus one LMCache CS512 RAM server per rank |

## Cache persistence boundary

LMCache CS512 in this recipe has RAM-only L1 storage, so:

- cache objects and warm reuse survive an **engine-only** restart, because the
  four LMCache servers stay alive;
- cache objects are absent immediately after a **server restart**;
- the first post-server-restart request is cold, a repeated request is warm,
  and all four shards repopulate.

That is lifecycle recovery and attribution, not NVMe durability. SparkCache is
a separate implementation and is disabled in this profile.

## Running it

`scripts/acceptance_gate.py` owns the stage sequence, the checks each stage
applies, and the exit codes. It is dry-run by default; `--execute` starts and
stops the configured stack and is therefore **STOPS SERVING**.

```bash
python scripts/acceptance_gate.py --site scripts/config/site.yaml \
  --gate scripts/config/gate.exl3.json
```

Inspect the printed plan before running anything against hosts. A reported
blocker is not a check to bypass, and a successful plan is not acceptance.

Behavior and exit codes live in the script rather than here; when this page and
the script disagree, the script is correct.
