# GLM native-MTP3 compute comparison

Status: research-only measurements. All five matrices contain 18 completed
decode cells with no request errors, underfilled cells, capacity-limited cells,
or warm-up timeouts.

## Conditions

Four DGX Sparks serve GLM-5.3-Flash-NVFP4-Spark with TP4/DCP4, native MTP3,
SIRCL/hardware-forwarded mesh, and SparkCache. The harness version is 0.4.32;
temperature is 1, the maximum output is 2,048 tokens, and each decode cell has
a 20-second measurement window. Contexts are 8K, 32K, and 64K tokens. Each
column below is their equally weighted arithmetic mean, not per-request
throughput and not a mean of repeated benchmark runs.

The [sanitized matrices](spark-mtp3-compute-matrices-20260905.json) retain all
cell values, acceptance lengths, prefill scouts, configuration labels, full
compute-image identities, and hashes of the original receipts. The historical
mesh reference is identified by its benchmark receipt; its filename contains
a DFlash label but the associated serving configuration used native MTP3.

## Aggregate decode tokens per second

The compute additions are cumulative within the four compute-image rows.

| Configuration | C1 | C2 | C4 | C8 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|
| Mesh MTP3 reference | 47.0 | 76.4 | 116.7 | 166.4 | 194.3 | 225.0 |
| NVFP4 MTP proposal head | 47.3 | 77.5 | 121.7 | 168.5 | 200.0 | 224.9 |
| Proposal head + loader/RNG fixes | 49.4 | 79.3 | 118.6 | 171.7 | 203.2 | 229.0 |
| Loader/RNG + MoE scale sharing | 49.3 | 79.9 | 117.6 | 171.4 | 200.7 | 228.5 |
| MoE scale sharing + top-k selector | 49.6 | 80.6 | 120.8 | 173.9 | 195.6 | 234.8 |

## MTP-normalized aggregate sequence steps per second

Normalization divides output throughput by observed accepted length. These
are aggregate sequence-step rates, not batched engine iterations per second.

| Configuration | C1 | C2 | C4 | C8 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|
| Mesh MTP3 reference | 17.7 | 27.6 | 42.4 | 59.8 | 69.2 | 79.9 |
| NVFP4 MTP proposal head | 18.7 | 29.2 | 43.2 | 61.0 | 71.0 | 81.0 |
| Proposal head + loader/RNG fixes | 18.7 | 29.4 | 43.4 | 61.5 | 72.7 | 81.3 |
| Loader/RNG + MoE scale sharing | 18.9 | 29.2 | 43.3 | 62.0 | 71.4 | 82.4 |
| MoE scale sharing + top-k selector | 18.5 | 29.2 | 43.8 | 62.8 | 70.4 | 83.2 |

The selector image's C4/C8/C16 normalized means exceed the scale-sharing
image's means, while C1/C12 are lower. This single matrix does not establish
repeatability of those differences. The independent-RNG correction changes
sampling behavior; raw token rates alone must not be interpreted as compute
speed when acceptance differs.

## Scope

The top-k matrix resumed ten cells after the benchmark client lost power;
the serving containers stayed up and the resumed harness matched the recorded
source hash. Its initial deployment followed a host reboot to restore memory
contiguity. The comparison does not isolate reboot effects.

These compute images use transport bundle `4204fabc`. They are not performance
qualification of the combined image containing stream-safety bundle
`69313e19`; that image requires its own deployment and validation receipts.
