# GLM-5.3 Spark native-MTP3: consolidated validation report

Status: **research-only** evidence summary. Completed runs cover prefill,
decode, Estonia accuracy, long-context retrieval, installation, and cache
restoration. This report indexes those results and identifies the remaining
work without scheduling a repeat of every benchmark.

## Conditions

The profile uses four DGX Sparks, GLM-5.3 Flash NVFP4-Spark revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP3, TP4/DCP4,
8,192 batched tokens, 16 sequences, and hybrid SIRCL/RoCEnante routing.
The managed image is
`sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47`.

Each source record identifies its measured configuration. Operator throughput
runs, routing-development screens, and managed-image checks remain separate
measurement groups. The three-pass prefill data below use the managed image
and the public runbook at `49cb4e9656664b509234642bd46c7f07251d7f41`.

## Measurement

The [numeric summary](spark-mtp3-validation-summary-20260905.json) retains
source hashes, coverage counts, the supplemental 8K matrix, and computed
prefill statistics. The [prefill receipt](../../receipts/glm53-flash/mtp3-prefill-temperature1-20260905.jsonl)
contains all 15 samples. It uses temperature one, unique prefixes, server
token counts, and client TTFT from a separate Linux benchmark controller.
All samples reported zero cached tokens.

Reuse existing completed runs for their measured conditions. A targeted
repeat is useful evidence without being counted as an entire matrix sweep.
Medians below pool only the three matching prefill samples at each context.

## Result

| Check | Existing result | Coverage |
|---|---|---|
| Public install, startup, restart | Passed on all four ranks | [Application-install record](spark-mtp3-public-application-install-20260905.md) |
| Native collective correctness | 80/80 passed | Repeated during the public installation rehearsal |
| Installer configuration guard | Four accepted configurations; 20 altered records rejected | [Guard receipt](../../../runtime/glm53-spark-mtp3-mesh/container-validation-receipt.json) |
| Prefill | 15/15 valid, cold-prefix samples | Three samples at each of five contexts |
| Broad decode matrix | 18 cells, peak 231.3 aggregate tok/s | [8K/32K/64K × C1/C2/C4/C8/C12/C16](spark-mtp3-mesh-20260905.md), one complete sweep |
| Supplemental decode | 8K C1–C8 sweep plus two C4–C8 follow-ups | Eight operator cells and ten targeted final-route observations |
| Additional 32K screens | Two five-cell native-MTP3 screens | Predecessor routing; retained separately |
| Estonia accuracy | 30/30 correct at C8, 133,208-token prompt | [Estonia record](spark-mtp3-country-recall-20260905.md); no rerun needed |
| Long-context needle hunt | 4/4 exact answers, through 507,367 tokens | [Exact/revision/cross-reference record](spark-mtp3-needle-20260905.md) |
| SparkCache restoration | Three verified restoration runs | Same 27,274-token fixture and seed; includes the managed-image install/restart test |
| Coding peak | Five native-MTP3 speed samples; median 56.7, peak 57.2 tok/s | Predecessor routing reference; four output-limit hits and one normal stop |
| Controlled mixed traffic | Remaining | Paired idle/decode-plus-prefill measurements |
| Controlled sustained load | Remaining | Dedicated 30-minute workload and health/capacity record |

### Three-pass prefill

| Context target | Samples | Median TTFT (s) | Median prompt tok/s | Min–max prompt tok/s |
|---:|---:|---:|---:|---:|
| 8K | 3 | 2.974 | 2,758 | 2,755–2,765 |
| 16K | 3 | 5.980 | 2,741 | 2,735–2,746 |
| 32K | 3 | 11.980 | 2,737 | 2,737–2,738 |
| 64K | 3 | 24.452 | 2,673 | 2,670–2,674 |
| 128K | 3 | 48.783 | 2,686 | 2,686–2,688 |

Actual prompt counts are preserved in the receipt and differ slightly from
target labels. The earlier integrated scouts remain in their original record;
they are not substituted for these confirmed cold-prefix samples.

### Cache restoration evidence

The three closed restoration records are the
[temperature-one image check](spark-mtp3-mesh-temperature-one-functional-20260905.md),
[managed-service check](spark-mtp3-managed-mesh-functional-20260905.md), and
[public application-install check](spark-mtp3-public-application-install-20260905.md).
They show correct recall after reload with 26,624 external-hit tokens and
per-rank restoration evidence. Repeating the same fixture establishes repeated
restoration; testing three distinct fixtures would add workload coverage.

### Coding reference

The retained native-MTP3 coding run asked for a Sieve of Eratosthenes script
at temperature one with a 2,000-token cap. It recorded five throughput samples.
Four reached the cap; three contained reasoning but no visible code before
the cap. Keep the speed measurements, and use a scored coding task when
evaluating whether complete generated code works. A short coding-peak
confirmation on the managed image is useful, rather than discarding this
reference.

## Remaining work

The smallest useful addition is:

1. **Three paired mixed-traffic trials** to measure how fresh prefill affects
   an ongoing decode workload.
2. **One documented 30-minute sustained run** with request-error, memory/cache,
   and transport-health observations.
3. **A short coding-peak confirmation**, if results specific to the managed
   image are required.

For the full repeatability targets in the
[runbook](../../../docs/PROFILE_VALIDATION.md), add matching decode repetitions
and broader needle/cache fixtures. The existing complete matrix used
20-second measurement windows; preserve that convention for its repeat series
instead of pooling it with unmatched 30-second sweeps. The four passed needle
cases remain useful results; the full depth grid and near-one-million-token
checks are additional coverage. Three distinct 128K cache fixtures would
extend the already demonstrated restoration behavior.

No new Estonia run is planned. No additional large benchmark is queued.

## Conclusion

Most report categories already have measured results. The main new coverage
gaps are controlled mixed traffic and sustained load. Further complete decode
sweeps, expanded needle grids, and additional cache seeds improve repeatability
or workload coverage; they do not replace the successful runs above.

## Limitations

Results apply to the configurations and workloads identified by their source
records. Three matching full decode sweeps, the full long-context depth grid,
and three distinct cache fixtures have not yet been assembled. Profile
recommendation remains an operator decision.
