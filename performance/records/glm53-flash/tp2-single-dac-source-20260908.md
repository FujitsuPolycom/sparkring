# Original NVFP4 TP2 single-DAC source qualification

Status: **qualified** for the bounded text workloads described here.
A shared image assembled from different source components remains unqualified.

## Conditions

Two GB10 nodes served original `local-inference-lab/GLM-5.3-Flash-NVFP4`,
revision prefix `520de24`, with TP2/DCP1, managed B12X loading, static MTP3,
Humming draft MoE, mHC sharding, and recurrent-checkpoint coalescing. FP8 KV
was 6.75 GiB per node. Context was 262144, maximum sequences eight, batch
8192, and prefill interval eight. Graph capture maximum was 32. SparkCache
was disabled and both host-memory guards used a 2 GiB floor.

The [source dependency record](https://github.com/FujitsuPolycom/sparkring/blob/2f01b6ee8f6173745c4b6b165498bbef82fc03f1/runtime/profiles/glm53-flash-nvfp4-tp2/dependencies.json)
pins the exact source archives and both rank image IDs. The image IDs differ;
checked source hashes matched. The RoCEnante package is source commit
`60d8d68486540ce9ddb2702dd545fda6b347c087`. NCCL 2.30.4 used eight channels.

The single-DAC arm selected HCA indices 0/2, both PCI functions of physical
cage p0. The two-DAC control selected indices 2/1, spanning cages p0/p1.
Only the HCA map, NCCL HCA selection, and container names changed. Both
physical cables remained connected. The benchmark was
`local-inference-lab/llm-inference-bench` revision
`ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`, version 0.6.2.

## Measurement

[Per-pass observations](tp2-single-dac-source-20260908.json) preserve the
selected raw benchmark fields and hashes of their original files. The copied
observations omit host inventory and freeform logs. Decode used C1/C8 at
context 0/8192, 512-token output limits, and nominal 15-second windows after
warmup. Its rate is continuous-usage completion tokens divided by the measured
window. Prefill used one 8K/32K/64K scout per pass and reports client prompt
tokens divided by TTFT. Each arm ran twice; pass 2 is the warm repeat.

Relative decode differences compare the two-pass arithmetic means. The
records contain measured durations, token counts, rates, and error/loop gates.
No independent-run confidence interval was estimated. The first single-DAC
32K scout overlapped logged attention and mHC JIT compilation; it is retained
in the observations and is not averaged into the warm-prefill comparison.

## Result

| Context | Warm single DAC | Warm two DACs |
|---|---:|---:|
| 8K | 2311 tok/s | 2300 tok/s |
| 32K | 2176 tok/s | 2175 tok/s |
| 64K | 2283 tok/s | 2272 tok/s |

Two-pass mean decode differences were +1.7%, -1.2%, -2.8%, and -0.4% for
0/C1, 8K/C1, 0/C8, and 8K/C8 respectively. All measured cells completed
without request errors, underfilling, or detected loops. The semantic smoke
returned `FINAL=42`; both guards remained active. Measured traffic used both
p0 functions and neither p1 function.

## Conclusion

The software-selected single-DAC configuration showed practical short-run
parity with the two-DAC control on these text workloads.

## Limitations

Two passes do not establish statistical equivalence or long-duration
stability. JIT cost was not isolated. Multimodal accuracy, physical cable
removal, and maximum context/concurrency capacity were not qualified. These
source measurements do not qualify a different shared image or a SparkCache
composition, even when the image contains the same transport bundle.
