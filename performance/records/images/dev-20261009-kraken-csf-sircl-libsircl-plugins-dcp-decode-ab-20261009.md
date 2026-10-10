# GLM-5.3 TP8 decode with the DCP decode collectives plugin on and off

Status: **research-only; serving A/B runner measurements on image `27e9f75c0d09`; one measured start per
arm; not an installer run; not serving-qualified**.

The vLLM general plugin `glm_dcp_decode_comm` 2.0.0
([README](../../../integrations/vllm/glm_dcp_decode_comm/README.md)) changes how GLM-5.3's decode steps carry
their decode-context-parallel (DCP) collectives. Five switches select its items, each `0` by default in
profile `glm53-nvfp4-tp8` ([guide](../../../profiles/glm53-nvfp4-tp8/README.md)):
`GLM_DCP_DECODE_QUERY_PACK`, `GLM_DCP_DECODE_OVERLAP`, `GLM_DCP_DECODE_WK_OVERLAP`,
`GLM_DCP_DECODE_SELECTION_REUSE` and `GLM_DCP_DECODE_A2A_FUSED`. `GLM_DCP_DECODE_AUDIT=1` also runs the
image's own path beside each patched item and counts the words that differ. This record holds one audit of
the five items and one decode comparison of the five items against the profile's default, on the same image
and Sparks.

## Conditions

- **Image:** `sha256:27e9f75c0d09716764c468ea06fd9cbd2762843d2b0abb344dc6a64de329dffa`, image lock
  `dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp` (v3, SHA-256 `7ff35670…`; SIRCL 0.3.1, libsircl
  snapshot `a3477af2`, vLLM plugins `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and
  `glm_dcp_decode_comm` 2.0.0), built from commit `68934e24`. The lock is `installer-image-68934e24.json` of the
  [image record](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009.md); each
  run's `plan.json` records its name and SHA-256.
- **Hardware:** eight DGX Sparks (GB10) cabled as a ring of eight with ConnectX-7 RoCE; every run used all
  eight.
- **Host:** GPU SM clocks locked to 2,418 MHz on every Spark (`nvidia-smi -lgc 2418,2418`), which the
  installer does not set. Before every start the runner wrote back dirty pages, dropped the page cache and
  waited for stable available memory.
- **Serving configuration:** profile `glm53-nvfp4-tp8` as committed at `68934e24`: tensor parallelism 8,
  DCP 4, two-token MTP with the MXFP8 draft, MXFP8 linears on the B12X linear backend, the plugins above,
  16 sequences, utilization 0.91.
- **Transport:** arm `S+` (SIRCL ring sessions on every collective, NCCL off, the fused all-reduce and
  RMSNorm and the column gathers the profile pins); `SIRCL_LINK_SLOTS=16` and
  `SIRCL_LINK_SLOT_BYTES=524288` at the container level; capacity and dispatch 1 MiB; no SIRCL tuning
  table.

| Run | Starts | DCP decode switches (besides the profile) |
|---|---|---|
| `mx-glm53-final` | warm-up `W-S+`, measured `S+1` | none: all five `0` (the profile's default) |
| `mx-glm53-dcp-audit` | warm-up `W-S+` only | all five `1` and `GLM_DCP_DECODE_AUDIT=1` |
| `mx-glm53-dcp-on` | warm-up `W-S+`, measured `S+1` | all five `1` |

## Measurement

- **Audit:** the warm-up start's requests (fingerprints, prompt log-probabilities, short decode and prefill
  cells), then every rank's container stopped at once with `docker stop -t 180` and its log saved. The
  plugin logs each check's counters, `audit cuda:N: <check>: C calls, W words compared, D differ`, when they
  change; [audit.json](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-decode-ab-20261009/mx-glm53-dcp-audit/audit.json)
  holds each rank's enable line, `PatchRefused` count and last counters.
- **Decode comparison:** the [serving A/B runner](../../harnesses/serving_ab/README.md)'s `phase1-16k`
  metrics: decode engine steps per second from vLLM's speculative-decoding counters, in 30 s cells at
  temperature 0 with at most 1,024 output tokens, at 0, 16K and 32K tokens of context and 1, 2, 4 and 8
  concurrent streams; each cell's aggregate output rate and accepted tokens per step; three samples of the
  time to first token of a cold 8K, 16K and 32K prompt. Each run directory holds `tables.txt` and
  `summary.json` as the runner wrote them and the identifying fields of its `plan.json`
  ([directory](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-decode-ab-20261009/)).

## Result

**Audit.** Every rank logged `glm_dcp_decode_comm 2.0.0: enabled` with the five items and the audit `on
SIRCL 0.3.1`; no rank logged `PatchRefused`. All 488 audit lines of the eight ranks report 0 differing
words. The last counters, the same on every rank:

| Check | Calls | Words compared | Differ |
|---|---|---|---|
| `query` | 96,571 | 3,664,055,808 | 0 |
| `gathered_query` | 2,776 | 2,211,655,680 | 0 |
| `wk` | 26,550 | 35,294,560 | 0 |
| `selection` | 47,386 | 881,553,564 | 0 |
| `combine` | 3,252 | 127,107,072 | 0 |

**Decode.** Engine steps per second, switches on / off, and the change:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|
| 0K | 20.59 / 20.24 (+1.7 %) | 30.55 / 29.56 (+3.3 %) | 43.28 / 42.74 (+1.3 %) | 60.19 / 61.46 (-2.1 %) |
| 16K | 19.72 / 20.04 (-1.6 %) | 27.87 / 28.13 (-0.9 %) | 42.05 / 40.59 (+3.6 %) | 58.55 / 58.39 (+0.3 %) |
| 32K | 19.70 / 19.16 (+2.8 %) | 27.88 / 28.17 (-1.0 %) | 40.62 / 40.94 (-0.8 %) | 56.32 / 60.02 (-6.2 %) |

Aggregate output tokens per second, switches on / off: 0K 53.2 / 51.6, 77.0 / 73.9, 107.0 / 106.9, 147.9 /
153.7; 16K 48.0 / 46.4, 64.7 / 63.4, 97.7 / 94.8, 135.9 / 135.2; 32K 48.6 / 46.7, 67.0 / 67.0, 95.9 / 99.3,
137.3 / 144.8 (1, 2, 4 and 8 streams). Accepted tokens per step differ by at most 0.12 in any cell.

**Time to first token,** three samples, switches on / off:

| Prompt | On | Off |
|---|---|---|
| 8K | 5.80, 5.82, 5.95 s | 5.84, 10.38, 9.95 s |
| 16K | 12.66, 38.68, 23.80 s | 11.71, 11.77, 11.81 s |
| 32K | 24.40, 26.52, 26.09 s | 24.04, 23.86, 23.89 s |

Both measured starts passed the runner's arm checks. With one measured start per run, the runner's
comparison of output fingerprints and prompt log-probabilities between starts has nothing to compare.

## Conclusion

The five items are exact against the image's own path in this audit: 0 differing words in every check on
every rank. In this single pair of measured starts they give no consistent decode gain (engine steps per
second -6.2 % to +3.6 % of the default across the twelve cells, slower at 8 streams at 0K and 32K), and
the time to first token at 16K and 32K is higher with them on. The profile therefore keeps all five
switches at `0`. Settling their effect needs a warm-up and two measured starts of each arm on the same
Sparks.

## Limitations

- One measured start per arm. Repeated starts of other configurations on these Sparks differed by a few
  percent, and this profile's 8K time to first token takes one of two levels per sample (about 6 s or
  about 10-11 s, in the switches-off run here and in an earlier run of these settings with 8 sequences on
  image `af06e272`), so neither the decode differences nor the 16K and 32K time-to-first-token differences are
  established by this pair.
- The audit's counters at worker exit were not logged: about 1 s after the stop vLLM's executor reported
  workers still running after SIGTERM and killed them on seven ranks. The last counters were logged at the
  end of the warm-up's decode requests; the requests of its prefill cells, after that report, are not in any
  audit line.
- The audit ran with CUDA graphs on, as the measured start did; it did not run with `--enforce-eager`.
- GPU clocks were locked; an installer deployment runs at the Sparks' default clocks unless the operator
  locks them.
- The serving A/B runner ran the installer's container settings without the installer: no `sparkring
  install` run, functional check or correctness screen beyond the audit is part of this record.
