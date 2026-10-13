# GLM-5.3-Flash CSF on a pair: decode-context parallelism 2 against 1

Lane: **public-functional**. Status: **research-only**; one start of each
configuration. Evidence scope: **the settings of
`glm53-flash-nvfp4-spark-tp2-dcp2` and the installed `csf` deployment of
`glm53-flash-nvfp4-spark-tp2`, each started once on positions 6 and 7 of an
eight-Spark ring with installer image `d52737a109e0` on 2026-10-13 and
measured through its API with GPU clocks not locked; not serving-qualified.**

## Conditions

- **Checkpoint.** `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at
  `dec48abd33ef`, `--quantization nvfp4_csf --load-format nvfp4_csf`.
- **Image.** Installer image
  `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`,
  release 2026.10.2 (`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`),
  whose vLLM is SIRCL's pinned build `sparkring-kraken-beta-20261007-bc9ea774`.
- **Hardware.** Eight DGX Sparks (GB10) cabled as one ConnectX-7 RoCE ring.
  Both configurations ran on positions 6 and 7, a cabled pair, on SIRCL ring
  sessions with NCCL never allowed. Another model served on positions 0-3
  throughout; positions 4 and 5 were idle.
- **DCP1 configuration.** `sudo sparkring install --profile glm53-flash-tp2
  --checkpoint csf --on 6-7 --no-auto-recover --yes` with SparkRing package
  revision `a9d0ef61`: every setting of the pair profile's `csf` entry, 10 GiB
  of KV cache per Spark, a 1,048,576-token context, mHC prefill row ownership
  and KDA prefill coalescing on. Served name `GLM-5.3-Flash-CSF-TP2`.
- **DCP2 configuration.** The DCP1 deployment's two containers, re-created
  with the differences that this repository's installer renders for
  `glm53-flash-nvfp4-spark-tp2-dcp2`
  (`runtime/common/test_transport.py`,
  `test_the_pair_dcp2_variant_is_the_pair_profiles_csf_deployment_with_its_kv_cache_split`):
  `--decode-context-parallel-size 2`, `--cp-kv-cache-interleave-size 4`,
  `--kv-cache-memory-bytes 15032385536` (14 GiB), `SIRCL_GROUPS=tp,dcp`,
  `VLLM_GLM53_MHC_PREFILL_SHARD=0`, `VLLM_B12X_KDA_PREFILL_COALESCING=0`,
  served name `GLM-5.3-Flash-CSF-DCP2-TP2` and compile caches under the
  namespace `glm53-flash-csf-dcp2`, which were empty. The program that
  re-created them is [`dcp2_argv.py`](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013/dcp2_argv.py).
  The installed package does not contain the profile, and SIRCL's serve
  launcher runs a profile's release image, not this one.
- **Order.** DCP1 installed, measured and stopped with `sparkring down`; then
  DCP2 started and measured. API ready after 817 s (DCP1, installer's wait)
  and 880 s (DCP2, from container start).
- **Client.** A workstation on the Sparks' LAN. llm-inference-bench 0.7.6,
  temperature 1: `--concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k
  --duration 20 --max-tokens 2048 --coding-peak`, then `--concurrency 1
  --contexts 256k,512k`. The profile acceptance harness
  ([`accept_profile.py`](../../harnesses/acceptance/README.md)) with
  `--skip-install --steps readiness,functional,stress`.

## Measurement

- **KV capacity:** rank 0's startup line `GPU KV cache size: N tokens`.
- **Prefill:** the benchmark's cold scout requests at 8K-128K: time to first
  token and prompt tokens per second, one sample each.
- **Decode:** aggregate output tok/s and engine steps/s (output tokens divided
  by MTP accept length) over 20 s cells, one cell per context and stream
  count. 256K and 512K: one stream; their warm-up time is a cold prefill of
  the cell's prompt plus its first tokens.
- **Coding peak:** median of five single-stream coding requests.
- **Correctness:** the harness's seven functional checks (counting,
  arithmetic, code, automatic and forced tool calls, an image, thinking on)
  and its screen of 256 greedy requests (degenerate, wrong, errors).
- **Memory:** `MemAvailable` every 2 s on both Sparks; for DCP1 during a
  repeat of the 256K and 512K cells only, for DCP2 over the whole run.
- **Transport:** DCP2's SIRCL receipts of every rank and both containers'
  logs; DCP1's from the installer's transport check.
- **Raw files:** [summary.json, startup lines and the refused start](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013/).

## Result

| | DCP1 | DCP2 |
|---|---|---|
| KV cache per Spark | 10 GiB | 14 GiB |
| KV tokens | 1,472,548 | 4,126,165 |
| Lowest `MemAvailable`, rank 0 / rank 1 | 10.49 / 12.06 GiB | 6.48 / 7.99 GiB |
| Functional checks | 7 of 7 | 7 of 7 |
| Screen, 256 requests | no degenerate, wrong or failed response | no degenerate, wrong or failed response |
| Prefill 8K / 64K / 128K, tok/s | 2,370 / 2,349 / 2,327 | 1,725 / 1,941 / 1,932 |
| Decode 32K, 1 / 4 / 8 streams, tok/s | 35.8 / 79.7 / 111.4 | 36.2 / 73.1 / 107.9 |
| Decode 32K, 1 / 4 / 8 streams, steps/s | 14.9 / 30.8 / 41.4 | 14.5 / 29.4 / 40.0 |
| Coding peak, tok/s | 48.2 | 46.1 |
| 256K, 1 stream: tok/s, steps/s, warm-up | 38.3, 14.4, 122.6 s | 34.2, 14.3, 141.4 s |
| 512K, 1 stream: tok/s, steps/s, warm-up | 35.6, 14.2, 257.0 s | 33.9, 14.1, 284.8 s |

A repeat of DCP1's 256K and 512K cells gave 35.1 tok/s and 14.5 steps/s at
256K, 35.1 tok/s and 14.2 steps/s at 512K, and warm-ups of 122.3 and 256.4 s.
Across 8K-128K, DCP2's steps/s were 14.2-14.6 at one stream and 39.5-42.6 at
eight, DCP1's 14.8-14.9 and 40.7-43.1.

Transport, DCP2: every rank's `dcp:0`, `tp:0` and `ep:0` receipt shows
`nccl=none pynccl=skipped`; `dcp:0` and `tp:0` ran on ring sessions of their
own and `ep:0` shared `tp:0`'s; every decision row is on SIRCL; neither log
shows an NCCL communicator.

A first DCP2 start with KDA prefill coalescing on stopped at model
construction: the image's vLLM admits `VLLM_B12X_KDA_PREFILL_COALESCING=1`
only at TP2 with decode-context parallelism 1 or TP4 with 1, 2 or 4
([refusal](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013/kda-coalescing-refusal.txt)).

## Conclusion

On one pair with this image and checkpoint, decode-context parallelism 2 holds
2.00 times the KV tokens per byte of KV cache of decode-context parallelism 1
and 2.80 times the tokens of the DCP1 profile at the settings compared. Its
decode steps per second were 1-6 % lower, its prefill 17 % lower at 64K and
128K and 27 % lower at 8K, and both configurations passed the same functional
checks and correctness screen.

## Limitations

- One start per configuration; no repeat of the DCP2 measurements. GPU clocks
  were not locked.
- The DCP2 configuration differs from DCP1 in two prefill settings besides
  decode-context parallelism (mHC row ownership, KDA prefill coalescing) and in
  KV bytes; the result does not separate their effects.
- DCP2 ran as re-created installer containers, not through `sparkring
  install`; the installer's rendering of the profile is CPU-tested only.
- Memory was not measured under image or video loads at 14 GiB of KV, and
  DCP1's lowest `MemAvailable` covers only its long-context cells.
- No request longer than 515,076 tokens ran; the 1,048,576-token context was
  configured, not exercised.
