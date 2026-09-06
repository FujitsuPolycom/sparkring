# GLM-5.3 Spark native-MTP3 long-context needle hunt

Status: **research-only** measured result. **Four of four checks passed**, with
exact answer equality, including cross-referencing at 507,367 prompt tokens.

## Conditions

Four DGX Sparks ran GLM-5.3 Flash NVFP4-Spark revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP3, TP4/DCP4/PP1,
8,192 batched tokens, and the SIRCL/RoCEnante hybrid mesh profile.
Image ID: `sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47`.
Serving source: `91c313d028877ada5fb1f04610f83c6465428657`.
SparkCache was enabled; all four requests reported zero cached prompt tokens.
Requests ran sequentially at temperature one, top-p one, request seed
20260905, and an output budget of 2,048 tokens.

## Measurement

The [numeric receipt](spark-mtp3-needle-20260905.json) retains fixture seeds,
prompt hashes, expected and returned identifiers, token usage, finish reasons,
source hashes, and full request durations. The fixtures are reproduced in the
[public retrieval harness](../../harnesses/validation/README.md): exact value
retrieval, superseding-revision selection, and joining separated records.
Depth is the fixture's percentage position in the document; actual context
length comes from server usage. Each case was run once.

## Result

| Task | Needle depth | Actual prompt tokens | Exact answer | Finish | Full request time |
|---|---:|---:|---|---|---:|
| Exact value | 5% | 252,125 | Pass | stop | 97.4 s |
| Superseding revision | 50% | 252,188 | Pass | stop | 113.1 s |
| Cross-reference | 95% | 252,176 | Pass | stop | 98.3 s |
| Cross-reference | 50% | 507,367 | Pass | stop | 202.9 s |

## Conclusion

The served profile correctly retrieved the identifiers, selected the
superseding record, and followed the cross-reference across these long
repository-style prompts. Every answer exactly matched its expected value
and completed within the output budget.

## Limitations

These four cases cover the listed contexts, depths, and tasks. The
[validation runbook](../../../docs/PROFILE_VALIDATION.md) defines the broader
depth/context matrix for a complete profile report. Durations here include
prefill and answer generation.
