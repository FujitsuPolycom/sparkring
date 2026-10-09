# GLM-5.3 TP8 speculative decoding: MTP3, MTP3 with adaptive draft counts and DFlash2 against MTP2

Status: **research-only; serving A/B runner measurements on image `27e9f75c0d09`; two measured starts per
measured arm; not an installer run; not serving-qualified**. The DFlash2 arm is research-only under its
drafter's licence (cc-by-nc-nd-4.0); no DFlash2 configuration is a profile default.

Profile `glm53-nvfp4-tp8` ([guide](../../../profiles/glm53-nvfp4-tp8/README.md)) speculates with GLM-5.3's
multi-token-prediction (MTP) layer and two draft tokens. This record compares three alternatives with it on the
same image and Sparks on one afternoon: three MTP draft tokens; three MTP draft tokens with the scheduler's
acceptance-length adaptation; and the separate DFlash2 drafter with seven draft tokens.

## Conditions

- **Image:** `sha256:27e9f75c0d09716764c468ea06fd9cbd2762843d2b0abb344dc6a64de329dffa`, image lock
  `dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp` (v3; SIRCL 0.3.1, libsircl snapshot `a3477af2`, vLLM
  plugins `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.0). The
  lock is not in the repository; each run's `plan.json` records its name and SHA-256.
- **Hardware:** eight DGX Sparks (GB10) cabled as a ring of eight with ConnectX-7 RoCE; every run used all
  eight.
- **Host:** GPU SM clocks locked to 2,418 MHz on every Spark, which the installer does not set. Before every
  start the runner wrote back dirty pages, dropped the page cache and waited for stable available memory.
- **Serving configuration:** the profile's settings (tensor parallelism 8, decode-context parallelism 4, 16
  sequences, utilization 0.91, MXFP8 linears on the B12X linear backend, the GLM-5.3 plugins with the DCP
  decode switches at `0`) with each run's speculative configuration below; arm `S+` (SIRCL ring sessions,
  NCCL off), `SIRCL_LINK_SLOTS=16` and `SIRCL_LINK_SLOT_BYTES=524288` at the container level, capacity and
  dispatch 1 MiB, no SIRCL tuning table.
- **Starts:** one warm-up start and two measured starts per run, measured with the
  [serving A/B runner](../../harnesses/serving_ab/README.md)'s `phase1-16k` metrics: output tokens per
  second and accepted tokens per step at 0, 16K and 32K tokens of context and 1, 2, 4 and 8 streams
  (temperature 0, 30 s cells, at most 1,024 output tokens), and the time to first token of a cold 8K, 16K
  and 32K prompt (three samples each). Each run directory holds `tables.txt`, `summary.json` and the
  identifying fields of its `plan.json`
  ([directory](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009/)).

| Run | Speculative configuration |
|---|---|
| `mx-t8-mtp2` | the profile's: MTP, 2 draft tokens (B12X draft attention, draft tensor parallelism 8, probabilistic draft sampling, MXFP8 draft linears, W4A16 draft experts) |
| `mx-t8-mtp3` | as `mx-t8-mtp2` with 3 draft tokens |
| `mx-t8-mtp3-al` | as `mx-t8-mtp3` with `adaptive_speculative_tokens_window` 16: the scheduler sets each step's draft count, at most 3, from the accepted lengths of the last 16 verification steps |
| `mx-t8-dflash2` | method `dflash`, 7 draft tokens, drafter `incoai/GLM-5.3-DFlash2` at revision `425aa615` (six sliding-window layers, BF16) mounted read-only from every Spark (`--mount`), draft KV cache `fp8` and `--kv-cache-dtype-skip-layers sliding_window` beside the target's `fp8_ds_mla` |

The image's draft-aware verification (`enable_adaptive_verification`) is not among the runs: it trims each
request's verification on the device, and GLM-5.3's B12X sparse-MLA and DSA indexer attention backends
declare that they cannot run with device query lengths that differ from the host's
(`supports_device_cpu_query_lens_mismatch` is false), so vLLM refuses it for this model.

## Result

Output tokens per second / accepted tokens per step, median of the two measured starts:

| Run | Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|---|
| `mx-t8-mtp2` | 0K | 50.7 / 2.56 | 73.9 / 2.48 | 102.4 / 2.46 | 151.4 / 2.48 |
| | 16K | 44.2 / 2.27 | 64.0 / 2.34 | 89.8 / 2.32 | 135.7 / 2.33 |
| | 32K | 47.9 / 2.48 | 65.9 / 2.41 | 99.8 / 2.42 | 143.7 / 2.43 |
| `mx-t8-mtp3` | 0K | 48.6 / 2.94 | 72.4 / 2.85 | 101.9 / 2.82 | 149.6 / 2.86 |
| | 16K | 44.1 / 2.59 | 63.3 / 2.59 | 95.8 / 2.67 | 127.1 / 2.59 |
| | 32K | 46.6 / 2.75 | 64.5 / 2.71 | 98.1 / 2.75 | 137.0 / 2.77 |
| `mx-t8-mtp3-al` | 0K | 47.5 / 2.69 | 72.4 / 2.80 | 105.1 / 2.82 | 154.8 / 2.83 |
| | 16K | 44.5 / 2.59 | 62.3 / 2.39 | 86.8 / 2.58 | 129.4 / 2.31 |
| | 32K | 49.2 / 2.87 | 65.4 / 2.71 | 90.6 / 2.69 | 138.3 / 2.69 |

Against `mx-t8-mtp2` over the twelve cells: `mx-t8-mtp3` output tokens per second -1.9 % (median; -6.3 % to
+6.7 %), engine steps per second -13.3 % (-16.6 % to -7.2 %); `mx-t8-mtp3-al` output tokens per second
-2.4 % (-9.1 % to +2.7 %), engine steps per second -11.6 % (-18.5 % to -3.6 %). The third draft token raises
the accepted tokens per step by 11-15 % and costs about as much in steps per second.

Time to first token, median of the samples, 8K / 16K / 32K prompt: `mx-t8-mtp2` 10.03 / 11.85 / 24.40 s;
`mx-t8-mtp3` 8.00 / 12.02 / 27.43 s; `mx-t8-mtp3-al` 10.15 / 12.17 / 24.75 s. The 8K samples take one of two
levels (about 6 s or about 10 s) in every run.

`mx-t8-dflash2` did not start: rank 0's engine core refused its KV cache layout while sizing the KV cache,
"KV cache layout LBNHC cannot express this model's mixed page sizes ([16896, 83968, 147456])"
([refusal.txt](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009/mx-t8-dflash2/refusal.txt)).
The drafter's sliding-window layers form a second KV cache group whose page size differs from the target's;
vLLM expresses mixed page sizes across several groups only in a layout without the layer dimension
outermost, and GLM-5.3's B12X DSA sparse-MLA backend accepts only `LBNHC`, which has it outermost. No
other DFlash2 configuration ran.

Every measured start passed the runner's arm checks; prompt log-probabilities were identical between the
two measured starts of each run.

## Conclusion

On this image and hardware none of the alternatives improves on two-token MTP: three MTP draft tokens give
no gain, MTP3 with acceptance-length adaptation gives no gain, and the DFlash2 drafter does not start with
GLM-5.3's B12X attention backends. Profile `glm53-nvfp4-tp8` keeps two-token MTP.

## Limitations

- Two measured starts per run; the same-day `mx-t8-mtp2` starts differed by up to 12 % in one cell, so
  single-cell differences of a few percent are within start-to-start spread.
- The adaptation window (16 verification steps) was the only one tried.
- DFlash2 ran in one configuration; a configuration that keeps one KV cache group (for example with the
  hybrid KV cache manager off) did not run.
- GPU clocks were locked; an installer deployment runs at the Sparks' default clocks unless the operator
  locks them.
- The serving A/B runner ran the installer's container settings without the installer.
