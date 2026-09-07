# Token-sharded mHC prefill evidence

Status: **research-only**. The original sharded-mHC composition produced about
4.59–5.26% higher prompt throughput than its recorded control at 8K, 16K, and
64K. These sequential measurements include the changes and limits below; they
do not qualify a rebuilt image containing additional request attribution.

## Conditions

Four DGX Spark GB10 GPUs with a virtual mesh over the physical ring, TP4/DCP4,
PP1, native MTP3, 8,192 maximum batched tokens, 16 sequences, and 24 GiB KV cache
per rank. Model: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.

The control image config is
`sha256:489d1975619e9083d14f14bcd1c6cbb4a96c41e3ab3978af048dc3ed2bb452a8`;
the sharded image is
`sha256:65f2b9181acd77db660f9c105554c4fca5c4df89d87d1374276e17c6831d1359`.
The latter enables `SPARK_MHC_PREFILL_SHARD=1`, replacing eight runtime files
and adding one helper. Both retain continuation coalescing, native mesh,
graph-only decode geometry, linked posting, B12X compute, and SparkCache with
periodic full captures off. The sharded run uses a separate initially empty
cache root.

Control preceded sharded measurements. Rank zero was rebooted between arms to
recover launch memory; driver/kernel, CPU governor, and reported GPU application
and maximum clocks matched peers afterward. The first sharded startup failed;
the unchanged image passed on retry. Requests use concurrency one, temperature
zero, one output token, unique prefixes, exact prompt usage, and zero cached
tokens. Full conditions and source identities are in [the record](mhc-evidence.json).

## Measurement

TTFT spans request submission to the first nonempty streamed delta, including
client/network overhead. Both arms use Windows `time.perf_counter`, backed by
QueryPerformanceCounter with nominal 100 ns resolution. Startup/API warmup and
idle gates passed; one shape-warmup cycle is excluded. Three measurements per
shape produce median, min, and max TTFT. Throughput is prompt tokens divided by
median TTFT; change is `100 * (control TTFT / sharded TTFT - 1)`.

Raw [control rows](mhc-control-prefill.json), [sharded rows](mhc-candidate-prefill.json),
and [all-rank activation records](mhc-activation.json) are retained exactly.
[Provenance](provenance.json) binds the public files and retained historical
harness copy; it is not a signed execution-time attestation.

## Result

| Prompt tokens | Sharded median TTFT | Sharded min–max TTFT | Sharded tok/s | Control tok/s | Change |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 2.682 s | 2.677–2.707 s | 3,055 | 2,902 | +5.26% |
| 16,384 | 5.393 s | 5.387–5.410 s | 3,038 | 2,893 | +5.00% |
| 32,768 | 10.832 s | 10.822–10.857 s | 3,025 | Not measured | — |
| 65,536 | 21.755 s | 21.731–21.767 s | 3,012 | 2,880 | +4.59% |
| 131,072 | 44.274 s | 44.027–44.427 s | 2,960 | Not measured | — |

All four ranks logged 8,192 full rows, 2,048 owner rows, and 90 reduce-scatters
plus 90 all-gathers per chunk, with zero auxiliary captures. Nine fresh,
repeated, or extended answer/cache checks passed. A 16K exact-answer prefill
overlapped two 192-token streaming decodes. Two saved 8K/16K prompts selected
the same first token as control, with different probabilities.

## Conclusion

The recorded sharded composition has higher cold-prefill throughput at the three
compared sizes under these conditions. The 32K and 128K rows describe only the
sharded image. Existing NCCL reduce-scatter/all-gather carries owner-row mHC;
this is no evidence for a new native split transport or decode improvement.

## Limitations

Sequential order, three samples, synthetic prompts, reboot, fresh cache root,
and startup history prevent a strictly isolated causal claim. First-token
agreement and bounded semantic/cache cases do not establish full-model numerical
identity or comprehensive quality. Different collective reduction order changes
probabilities. Request overlap was observed, but per-step mixed-batch fallback
was not instrumented. Other row counts, parallel layouts, and capture paths are
outside this feature's scope.

The first startup failed in Triton `load_binary` during MTP W4A16 warmup before
completed sharded-mHC activation. An unchanged retry passed. Its cause remains
unresolved; no startup fix is claimed. The source-build draft also composes
request attribution and therefore requires separate build, GPU, startup, and
serving validation.

## Reproduction

Set `BENCH_API_BASE`, `BENCH_MODEL`, and `BENCH_RANK0_SSH` for an explicitly
authorized idle deployment, verify all-rank startup separately, then use the
[portable harness](../../../harnesses/vllm/prefill_checks/mhc_precise_checks.py):

```bash
python performance/harnesses/vllm/prefill_checks/mhc_precise_checks.py prefill \
  --container-prefix sparkring-model --sizes 8192,16384,32768,65536,131072 \
  --samples 3 --output /path/to/absent-results.json
```

This sends inference and reads rank-zero Docker health over SSH. The public
copy preserves historical timing/request logic and additionally rejects missing
or malformed cache-token accounting. Every retained timing row contains explicit
integer cache accounting, so this check does not change the record. The public
copy has only offline validation; it has not been rerun on a GPU.
