# vLLM plugin four-Spark live validation, 2026-08-20

Status: **live-validated** for eager transport admission at width 768 on
four directly cabled DGX Sparks. This record covers the
`sparkring_plugin` integration path only. It establishes that the
pip-installed plugin reaches the SparkRing transport and serves; it
establishes no numerical, performance, or profile claim.

## Conditions

| Item | Value |
|---|---|
| Hardware | Four directly cabled DGX Sparks, GB10 |
| Container image | `sparkring/glm52-exl3-r7-3.5bpw:r34-sm121a-flat2-20260810` |
| vLLM | `0.1.dev1+ge2666d9a6.d20260810` |
| Plugin | Wheel built on aarch64 from this branch, `libspark_transport_capi_w4096.so` packaged as `_native/` payload |
| Model | `facebook/opt-125m`, hidden width 768 |
| Serving | TP4, four nodes, `--enforce-eager`, BF16, `--max-model-len 2048`, `--max-num-seqs 4` |
| Transport | `VLLM_SPARK_TP4_MODE=shadow`, `VLLM_SPARK_TP4_EAGER_WIDTHS=768` |
| Overlay | Neutralized: an empty file bind-mounted over `/opt/spark-vllm/sitecustomize.py` |

The overlay is neutralized deliberately. The image bakes
`sitecustomize.py`, and with a mode variable set both installers run, so
a result from an unmodified image is not attributable to either path.

## Measurement

The plugin installed and patched the integration point on every rank:

```
installed Spark TP4 vLLM backend in shadow mode
```

Every rank opened both eager sessions the width-768 shapes require:

```
rank 0  Spark TP4 eager session ready for 12288 bytes (control ports 11100/11101)
rank 0  Spark TP4 eager session ready for  6144 bytes (control ports 12130/12131)
rank 1  Spark TP4 eager session ready for 12288 bytes (control ports 11100/11101)
rank 1  Spark TP4 eager session ready for  6144 bytes (control ports 12130/12131)
rank 2  Spark TP4 eager session ready for 12288 bytes (control ports 11100/11101)
rank 2  Spark TP4 eager session ready for  6144 bytes (control ports 12130/12131)
rank 3  Spark TP4 eager session ready for 12288 bytes (control ports 11100/11101)
rank 3  Spark TP4 eager session ready for  6144 bytes (control ports 12130/12131)
```

Both payload sizes follow from the shapes: 6,144 bytes is four query rows
of 768 BF16 elements, and 12,288 bytes is eight.

`sparkring-preflight` passed every required check on a serving rank,
including `hca-devices` resolving `rocep1s0f0` and `rocep1s0f1` at GID 3,
`native-library` resolving the packaged payload, `vllm-compat` resolving
against the deployed fork, and `vendor-integrity` matching all sixteen
vendored modules against the manifest in the installed wheel.

A greedy completion returned correct text:

```
prompt "The capital of France is" -> " the capital of the French Republic."
system_fingerprint vllm-0.1.dev1+ge2666d9a6.d20260810-tp4-nohash
```

## Result

The pip-installed plugin admits width-768 eager all-reduce through the
SparkRing transport on four ranks and serves an OpenAI-compatible
completion. Shadow mode reported no mismatch.

## Limitations

- One width, one model, one bounded request. No sustained load, no
  concurrency sweep, and no performance number.
- Shadow mode compares against the stock path; `custom` mode, in which
  the transport carries the result, is unexercised here.
- Graph capture is unexercised: the run is `--enforce-eager`.
- The absence of a reported mismatch is not a numerical qualification.
  A shadow window per the eager-width runbook remains outstanding.
- The overlay was neutralized. Running both installers together is
  unqualified, and section "Coexistence" in the plugin README states
  what is known about it.

## Conditions required for a profile claim

1. A shadow-comparison window closed per
   [eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md)
   at each width a profile admits.
2. `custom` mode serving the same shapes with the transport carrying the
   result, compared against the overlay path on the same model.
3. A width the qualified profiles actually use. Width 768 is a
   validation instrument, not a serving width.
