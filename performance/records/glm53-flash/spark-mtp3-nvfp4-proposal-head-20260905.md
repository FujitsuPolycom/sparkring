# Native-MTP3 NVFP4 proposal-head comparison

Status: **research-only**. These measurements support the proposal-head default
inside the research-only GLM-5.3 native-MTP3 mesh profile. They are not a general
model-quality or production-serving claim.

## Conditions

The measured system used four NVIDIA DGX Spark GB10 nodes in TP4/DCP4/PP1,
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP depth three, FP8 KV,
24 GiB KV allocation per rank, SparkCache, and the profile's hybrid
SIRCL/RoCEnante transport. No DFlash model was loaded.

The measured proposal-head image is
`sha256:04d5a35b03e99f68c37a05514d221988a3eb70a5b8fdcfa859025ca1cbc25e74`.
Its image receipt has SHA-256
`aa8f1bf3d892af7b0a8059626ec91a83e2dbc5a39c8d58c0fb82c614e88dbaba`;
the completed model-readiness and live-control receipts have SHA-256
`fe786299a6459cd57dcf5362591b56d94569dd473b0d1fefb47ca374399983f3`
and `b0ebcf65be43998791f66da819ebaa98aa4b4323b9b76cc1e363d977750e3341`.
The image is layered on compute image
`sha256:f35ed3d1df1ee57f66ba571a491d6dd5575d1a2e619cbc63d065008894777ffe`,
which records CUDA 13.3.33, complete B12X revision
`b58f34eaf978277621efced6678e6713fd7122e4`, the uniform-speculation metadata
port from Local Inference Lab vLLM revision
`3512b066e7796128c0c380ccc558182960f2f0ea`, and dense wrapper changes from
revision `a8c796f3af74106b2d8d441e9ec54588936a5388`. The complete B12X tree also
contains MoE and other package changes in B12X revision `b58f34ea`, so these runs do not isolate a dense
kernel contribution.

The changed variable was a separate runtime-NVFP4 proposal head with BF16
activations. The target/verifier head kept its BF16 checkpoint representation.
The proposal head adds 85.08 MiB of persistent packed weight and scale storage
per rank while the retained BF16 target head remains allocated. It is not a net
85.08 MiB model-memory reduction. The shared-BF16-head control used the same
CUDA version, B12X kernels, metadata reuse, and dense-kernel integration without
the separate proposal allocation. Thus the proposal-head comparison preserves
the verifier implementation. Comparisons against the separate
[mesh matrix](spark-mtp3-mesh-20260905.md) also differ in CUDA, metadata,
dense-kernel integration, MoE, and B12X code; retaining a BF16 verifier head
does not imply that every target-side kernel matches between those configurations.

The benchmark used harness version 0.4.32, temperature 1.0, ignored EOS, a
20-second sustained-decode window, 8,192-token decode context, and concurrency
1, 2, 4, and 8. The proposal-head configuration has three repetitions and the
control has two. All compared 8K cells had
zero request errors, no warmup timeout, and no underfill flag.

The benchmark JSON does not embed the image ID. The image, launch, and completed
four-rank readiness receipts bind the benchmark endpoint to the tested private
image. The compute-equivalent public derivative is
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:1b97e1dc9cb93c39f887f40bab24359a9b6ec998c28d2417b160f2103cd5fd86`
with image ID
`sha256:dd6c51efaf4127df863ac85c3be3fe46f260b34c7ab2deb384669fffdbe857df`.
Its compute-equivalence record matches all vLLM, B12X, and SparkCache package
files and the selected environment to the tested image. Exact-image serving,
restart, and persistent-cache checks were not repeated on the public image ID.

## Measurement

[Sanitized per-run samples](spark-mtp3-nvfp4-proposal-samples-20260905.json)
include all decode windows, token counts, per-context prefill scouts, and
recorded settings for the two control and three proposal repetitions.
Each decode cell requests five seconds of warmup, allows 900 seconds to reach
the requested concurrency, and measures a 20-second client wall-clock window.
The harness version is 0.4.32; its exact historical source commit was not
captured. The recorded equivalent command is:

```bash
python llm_decode_bench.py --host http://RANK0 --port 8015 \
  --model glm-5.3-flash-spark --temperature 1.0 --token-targeting exact \
  --display-mode live --no-hw-monitor --dcp-size 4 \
  --concurrency 1,2,4,8 --contexts 8k,16k --max-tokens 2048 \
  --duration 20 --decode-warmup-seconds 5 \
  --cell-warmup-timeout-seconds 900 --output result.json
```

Variability is reported as the minimum and maximum across repetitions, not
as a confidence interval. At 8K/C1, control output throughput spans
46.61–48.84 tok/s and proposal throughput spans 49.35–52.80 tok/s;
normalized throughput spans 17.94–18.01 and 18.70–19.05 respectively.
The sample file contains the same ranges for every measured 8K concurrency.

Raw throughput is aggregate output tokens per second. Normalized throughput is
the harness's aggregate sequence steps per second, computed from drafted and
non-speculative request work. It is not a count of batched engine iterations.
Speculative acceptance can move raw tokens per second, so both metrics are
reported. The machine-readable values and receipt hashes are in
[`spark-mtp3-nvfp4-proposal-head-20260905.json`](spark-mtp3-nvfp4-proposal-head-20260905.json).

Proposal-head receipt hashes:

- `122857`: `05d846465153230f4312a59aefef12c499d81634855f3c84cb6757baf653c0e4`
- `123541`: `9664bc0410db423d3edc759106f10bc24a0fe1d15952102cdd74913ff96c2149`
- `124222`: `4213df0fc47ba53308ea7dc076e3e1c77e9097e5eaef95db4e09ca4cd404e434`

Shared-BF16-head control hashes:

- `114306`: `f7ae32cef4c9ea606477a28ff0465403e78f9519e2d771e4af0f13017b671677`
- `115629`: `7c3ff3210e805b8a801d4e8ea49152d0545a982707cc9010a174f77ff3d4e673`

The previously published native-MTP3 `233405` receipt is retained as historical context
with SHA-256 `f0916f6b72cb8256225169b44c4f11e3ca764a5dd854977b8963686197b843fa`.
Its basename contains a stale DFlash7 label, but the public performance record
identifies the measured runtime as native MTP3. Its matrix and decode warmup
differ, and its JSON does not attest an image identity. It is not used to
calculate the proposal-head delta.

## Result

| 8K concurrency | Control raw tok/s | NVFP4-head raw tok/s | Raw change | Control steps/s | NVFP4-head steps/s | Normalized change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.73 | 51.65 | **+8.22%** | 17.98 | 18.86 | **+4.90%** |
| 2 | 75.85 | 76.88 | +1.35% | 28.53 | 29.03 | +1.76% |
| 4 | 119.70 | 120.81 | +0.92% | 43.32 | 43.27 | -0.12% |
| 8 | 167.45 | 168.83 | +0.82% | 59.79 | 61.12 | +2.23% |

The C1 repetitions show a small-batch benefit: approximately 8%
raw throughput and 5% normalized throughput. C2, C4, and C8 are mixed and do
not support a monotonic concurrency-wide speedup claim.

Mean prefill scouts were unchanged within ±0.36%: 8K 2,668 versus 2,670.3
prompt tok/s, 16K 2,709 versus 2,707, 64K 2,782.5 versus 2,778.3, and 128K
2,749 versus 2,739.3. The head change is decode-facing; these observations do
not establish a prefill improvement.

## Conclusion

The bounded evidence supports making runtime NVFP4/BF16 proposal-head execution
the default within this opt-in, research-only profile. It improves repeated C1
decode while remaining approximately flat or mixed at the measured higher
concurrencies and prefill contexts. Relative to the compute-image control, the
verifier implementation and BF16 head are fixed; the proposal change can alter
speculative acceptance but not the target distribution under standard
rejection sampling.

## Limitations

The two control and three proposal-head repetitions are a small sample. They
cover only C1/C2/C4/C8 at 8K for
the direct comparison, not the full C1–C16 and 8K/32K/64K matrix. They do not
include isolated dense or MoE attribution, a general
accuracy evaluation, host reboot, long soak, failure containment, or unattended
high availability. A full C1–C16 and 8K/32K/64K matrix requires a separate
completed receipt before it can extend this record.
