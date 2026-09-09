# GLM native-MTP3 compute and stream-safety image validation

Status: **qualified** for the bounded checks listed below. The serving profile
remains **research-only**.

## Configuration

Four NVIDIA DGX Sparks serve GLM-5.3-Flash-NVFP4-Spark with TP4/DCP4, native
MTP3, an NVFP4/BF16 proposal head, a BF16 verifier head, 24 GiB FP8 KV per
rank, an 8,192-token scheduling limit, and 16 sequences. SIRCL and RoCEnante
use the hardware-forwarded physical ring. SparkCache uses the profile's
compute-specific namespace.

- Tested and published config-image ID:
  `sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f`.
- Registry manifest:
  `sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987`.
- Compute source lock:
  `139f36701e0e47f45bf99fba2cc2fa59b417f2ee801dad3a064455d5b464a459`.
- Transport manifest:
  `69313e19e881ec93e9ed3bd150d2f24fc6b444488ac729a69f45d038e2243500`.

The [numeric record](spark-mtp3-compute-stream-safety-20260906.json) includes
per-rank collective evidence, stream checks, all serving cells, source hashes,
and cache measurements. Host startup used the memory gate proposed in
[PR 222](https://github.com/FujitsuPolycom/sparkring/pull/222).

## Results

| Check | Result |
|---|---|
| Installed compute package comparison | 4,891 vLLM, 385 B12X, and 150 SparkCache files match the compute-tested image; selected environment values match |
| Four-rank native all-reduce | Exact BF16 results at 4, 20, 28, and 64 token rows; graph input mutations and per-QP completion counters pass |
| GPU stream safety | At 4 and 64 rows, 16 alternating-stream calls with misaligned input/output, two changed-input graph replays, and second-stream capture rejection pass |
| Full serving matrix | 18 cells complete without errors, capacity limitation, underfilling, or warm-up timeout |
| Focused repetitions | Two four-cell 8K/64K C1/C2 runs complete cleanly; 64K prefill measures 2,770 and 2,767 tok/s |
| Idle model-rank loss | Deliberately stopping rank 3 causes all model ranks to stop within 10.063 seconds |
| Recovery and restart | Same image and container identities pass managed recovery, startup memory checks, and four-rank readiness |
| Persistent prefix restore | Correct phrase recall after restart, 26,624 external-hit tokens, matching restore logs on all four ranks |
| Public image access | Anonymous manifest/config verification and Docker pull pass; pull reused local image layers |

The serving harness is version 0.4.32, pinned by source hash. It uses
temperature 1, 2,048 maximum output tokens, 20-second cells, 8K/32K/64K
contexts and C1/C2/C4/C8/C12/C16. The full matrix's initial 64K prefill scout
measured 2,687 tok/s; that lower result did not repeat in the two focused runs.
Individual decode cells vary. These results establish a functioning composed
image, not a guaranteed performance improvement over every tested configuration.

The persistent-cache fixture contains 27,274 prompt tokens. Both requests
returned `cobalt orchard lantern` with a normal stop. Full request time was
11.859 seconds cold and 2.922 seconds after restart; restore logs reported
26,624 tokens on every rank, and the engine reported 26,624 external-hit
tokens. This proves persistent restoration for that fixture rather than merely
fast GPU-prefix reuse within one process.

## Scope

The rank-loss test used an idle model. It does not establish containment of
an in-flight stalled GPU collective or resolve upstream RoCEnante issue #313.
One recall fixture does not qualify every cache boundary, concurrent cache
workload, or general model accuracy. The recorded checks do not exercise
unattended high availability. Watchdog and memory-protection thresholds were
not relaxed. All four model containers were healthy after restoration.
