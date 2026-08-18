# GLM-5.2 EXL3 performance campaign, 2026-08-02

## Status and evidence boundary

Status: **completed external performance campaign; B0 remains the default**

This document records a controlled configuration campaign on four directly
cabled NVIDIA DGX Sparks / GB10s. The target is the unchanged legacy
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
checkpoint at revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`.
The baseline serving configuration, campaign arm `B0`, is documented in
[`EXL3_FIXED_MTP2_RECIPE_20260802.md`](EXL3_FIXED_MTP2_RECIPE_20260802.md),
and the canonical fixed-MTP3 source recipe remains
[`EXL3_RECIPE.md`](EXL3_RECIPE.md).

This is a **public-functional-lane, external-evidence, live-validated
configuration campaign candidate**. The performance campaign is complete: B0
remains the default, while M3 remains a Pareto alternative for the measured
C2/C4 workload in the custom matched harness only. It is not a reference-lane result, not
public-bootstrap acceptance, not a clean-checkout receipt, and not a result
from the proposed `shared_h_v1` format. The running model uses legacy
`per_expert_v1`. The campaign used ignored operator-local profiles, plans, and
a benchmark harness from repository HEAD
`2f367c8c66f6b4fb54fed9dadf85250906ea40ce`; the checkout was dirty. No claim
is made that the tracked launcher reproduces these results unchanged.

The sanitized machine-readable campaign record is
[`docs/configurations/glm52-exl3-ab-campaign-20260802.json`](configurations/glm52-exl3-ab-campaign-20260802.json).
Raw JSON, profiles, plans, launcher output, metrics, and logs remain uncommitted
because they contain site identities and paths. Their hashes are retained
below. Results for unrun arms remain explicit placeholders rather than inferred
outcomes.

## Sealed hardware and serving invariants

| Setting | Campaign value |
|---|---|
| hardware | 4x DGX Spark / GB10, direct 200-Gb/s cycle |
| platform | `linux/arm64`, `sm_121` |
| model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| local image ID on all ranks | `sha256:314d75abdcbd65433fa9e4a744caf8fa31bdc108e9292df65603a6fe823766ad` |
| topology | TP4 / PP1 / DP1 / DCP4 |
| attention / DCP | `B12X_MLA_SPARSE` / `ag_rs` |
| model length | 1,048,576 tokens |
| KV | `nvfp4_ds_mla`, 9,000,000,000 bytes/rank, 1,125,632 tokens reported |
| batching | 8 sequences, 4,096 maximum batched tokens |
| graphs | `FULL_AND_PIECEWISE`, Q32 ceiling |
| loading | `safetensors` |
| cache / scheduling | prefix cache and chunked prefill on; SparkCache off |

Except for each arm's declared delta, those values were held constant. The
older image's explicit-unset compatibility wrapper described in the baseline
recipe remained part of the launch path.

## Matched benchmark method

The opening B0, M3, Q8, and closing B0 arms used the same ignored streaming
harness, SHA-256
`12bb68c2ebb033a144f3b9da7415e15917f8aeadd9c5e27f19d607c4384bc133`,
with schema `sparkring-exl3-ab-streaming-benchmark/v1` and seed `20260802`.

Prefill was measured as true streaming time to first token: request-body send
to the first nonempty SSE text. Each of the nominal 16K, 32K, and 64K sizes had
three repeats. A deterministic per-cell nonce avoided prefix-cache hits while
keeping each prompt byte-identical between arms. The server's final usage count
provided actual prompt tokens, and the reported rate is actual prompt tokens
divided by streaming TTFT. The cells used `max_tokens=1` and `ignore_eos=true`.

Decode used nominal 16K contexts at C1, C2, C4, and C8. Streams synchronized at
a barrier, discarded a three-second coordinated warmup interval, then used
continuous streaming usage deltas inside one 25-second measurement window.
`ignore_eos=true` and `max_tokens=4096` kept requests alive. Aggregate throughput
is the sum of measured completion tokens divided by 25 seconds. There is only
one decode sample per concurrency per arm; it is a matched sustained cell, not
a multi-run performance distribution.

A 64K prefill warmup preceded the measured cells. All six completed matched
full-matrix artifacts reported healthy APIs before and after, the expected served model,
and zero invalid prefill or decode cells.

The harness file was later extended with a decode-only repeat mode. The
extended file's SHA-256 is
`b276f7f1a54afe4732b204c0b14f0ab1600a8d3cbde718f6ecfec186d6ee67e9`.
That later hash was used by R8 and ADR8 in repeat mode and by AD and M1 in
full-matrix mode. It must not be retroactively assigned to the four earlier
full-matrix artifacts.

## Evidence hashes

| Logical artifact | SHA-256 |
|---|---|
| B0 benchmark JSON | `f41abbc02d3c6eb5c7d478bb385d962185f1f79c5c86edabd0dd84fa9d89e9cb` |
| M3 benchmark JSON | `5eb887ee14fd6e59a590910e27d96ebb5d897c64b5082d14a96080c98f935092` |
| Q8 benchmark JSON | `76727e2386b4909f09d3af744891402e8e6e8e0483709679ef9277b15de43174` |
| closing B0 benchmark JSON | `f9743c013f849edf6ba112361a03f05bfbcc2258703b0dbbb25938e4b77e6aa0` |
| B0 repeated-C8 benchmark JSON | `2b2ab86e4b8e11698082837856dc142b8fc5a9b9d5e65d7a78aa9955f271e4e7` |
| AD adaptive MTP2/3 benchmark JSON | `658c4803922bef0a78266778e86c3af861d48c6ef1ffcdc593ab48145ee095d3` |
| ADR8 clean-restart adaptive repeated-C8 benchmark JSON | `1c613a671315ce7f9bbbd2e367a7c1a6cee40e07718eb47bc7a4f0a6bba82ac0` |
| M1 fixed-MTP1 benchmark JSON | `e4272aa4efe4e0c377bd1326aa404458b1dd92f985ef50866d7c1d0647c73669` |
| ADR8 clean-restart B0 repeated-C8 benchmark JSON | `271f626f5500d8d1189e14665b6627ee2959884c4156151f14669a1ad86c7971` |
| B0 `llm_decode_bench` v0.4.31 benchmark JSON | `f02b659f798ce2b2a2afb8103008e85ea5185d49bb1c341d583ddf9a13abd69d` |
| initial B0 `llm_decode_bench` default-timeout JSON | `d3025f37e8fdd2060184ba4c3b85012f5547c9d7c6e304f0a8b442d550b0d8e2` |
| opening B0 standard quick 16K JSON | `09de1306cd765560701f837ff93c6f724c506ea0d4ade9da441c1df1aab85e34` |
| M3 standard quick 16K JSON | `373ded09e49bb706b36244b7bc67eb1e1d08774d08f94cad835f060545e055fd` |
| adaptive standard quick 16K JSON | `c17a4703821fb0b4fc06cada1731c6489e9d6f9f6fecca995f9b75cdf4bb2aad` |
| closing B0 standard quick 16K JSON | `74a5188e36892196c6629df8f21f2d60113875ccd982704ff269b3c3fa778f5a` |
| incomplete old-salt adaptive repeated-C8 JSON | `9cf2f5f6c30fb6c19bbc506e003864e9db9052fbb80d4ab198dda3510777022e` |
| original full-matrix harness used by B0/M3/Q8/closing B0 | `12bb68c2ebb033a144f3b9da7415e15917f8aeadd9c5e27f19d607c4384bc133` |
| final extended streaming/decode-repeat harness, used by R8, AD, M1, and both ADR8 halves | `b276f7f1a54afe4732b204c0b14f0ab1600a8d3cbde718f6ecfec186d6ee67e9` |
| A/B arm launcher, final version after local arm additions | `89318a90dae5df92a208bb18f779799287c3cbdc727e3dc4cd2bc13dea6b9d34` |
| B0 profile | `7dc65df82f0bc165104bffa5c786b8aaf4ba5f9ee3c3b77c3afb33c007870bbd` |
| M3 profile | `5874988de835a78712af83492e01bc5a30041ae7465d3938d8034a65f1fc41c3` |
| AS profile | `a441d87f95a172cd3153ece132099e8d0bcfd0c215c025b47493b0892690ecc8` |
| Q8 profile | `5a31fe6d55d5ac24bf084f0ea5f254641ea8ebb21f0626118f3ebb6c64908765` |
| Q8 executed plan | `b5c2ccbb3fad23ba662418ded500015e2c4854fd636b49207f320563bd484c8e` |
| independent B0/M3 review | `8173299634beafd659eb85146a4e5fc15912ddf83099bb57fc4ba689d47f7873` |

The hash list identifies external evidence; it does not make those uncommitted
artifacts part of the public source bootstrap.

## B0: fixed-MTP2 synchronous baseline

B0 used fixed MTP2, `num_spec_tokens=2`, synchronous scheduling, and 4,096
maximum batched tokens. Its artifact timestamp was 2026-08-02 13:15:03 UTC.

### Streaming prefill

| Nominal size | Actual prompt tokens per repeat | Mean tok/s |
|---:|---|---:|
| 16K | 19,478 / 19,472 / 19,477 | **552.29** |
| 32K | 38,933 / 38,935 / 38,933 | **541.25** |
| 64K | 77,848 / 77,848 / 77,848 | **533.39** |
| unweighted mean of all nine cells | — | **542.31** |

### Sustained decode

| Concurrency | Completion tokens in window | Aggregate tok/s |
|---:|---:|---:|
| C1 | 579 | **23.16** |
| C2 | 936 | **37.44** |
| C4 | 1,359 | **54.36** |
| C8 | 1,863 | **74.52** |

The C8 per-stream token counts were 253, 253, 255, 225, 255, 255, 112,
and 255. The 112-token lane is a material imbalance and limits how confidently
small C8 deltas can be attributed.

## M3: fixed-MTP3 synchronous arm

M3 changed only the speculative depth:

```text
VLLM_SPARK_MTP_MODE_ID: fixed-mtp2 -> fixed-mtp3
VLLM_SPARK_MTP_TOKENS: 2 -> 3
num_speculative_tokens: 2 -> 3
```

Scheduling remained synchronous and the batch-token limit remained 4,096. Its
artifact timestamp was 2026-08-02 14:06:58 UTC.

### Matched streaming prefill delta

| Nominal size | B0 mean tok/s | M3 mean tok/s | Delta |
|---:|---:|---:|---:|
| 16K | 552.29 | 536.98 | **-2.77%** |
| 32K | 541.25 | 524.67 | **-3.06%** |
| 64K | 533.39 | 508.59 | **-4.65%** |
| all nine cells, unweighted mean | 542.31 | 523.42 | **-3.48%** |

M3 was slower in every prefill cell. The size means are descriptive averages of
three matched prompt hashes, not confidence intervals.

### Matched sustained-decode delta

| Concurrency | B0 tok/s | M3 tok/s | Delta |
|---:|---:|---:|---:|
| C1 | 23.16 | 22.20 | **-4.15%** |
| C2 | 37.44 | 40.84 | **+9.08%** |
| C4 | 54.36 | 60.04 | **+10.45%** |
| C8 | 74.52 | 75.40 | **+1.18%** |

M3's C8 lanes produced 252, 264, 261, 241, 264, 261, 79, and 263 tokens.
Both arms therefore had a severely underproducing C8 lane. The +1.18% aggregate
delta is below the campaign's predeclared 5% target and is small relative to the
observed lane imbalance.

Against the later opening/closing B0 midpoint, M3 prefill is -4.11% overall;
C2 and C4 are +9.49% and +10.25%; and C8 is only +1.67%. C1 is +1.37%
against the midpoint, but the opening and closing controls differ by 10.9% at
C1, so neither the raw opening-only -4.15% nor the midpoint value is a stable
C1 effect.

M3 did not meet the promotion rule: prefill regressed and C8 improved by less
than 5%. It remains a **Pareto alternative** for the measured C2/C4 workload
in the custom matched harness, where the gain survives the baseline bracket. That is not a general MTP3
advantage and does not promote M3 to the baseline.

A supplemental live metrics reading recorded 89.43% draft-token acceptance,
2.683 accepted tokens per draft, and 93.20% conditional acceptance at position
three. Those figures came through the external review path rather than either
sealed benchmark JSON, so they are diagnostic evidence, not an additional
artifact-attested performance result.

## AS: asynchronous-scheduling failed cold-start gate

AS changed only scheduling by removing `--no-async-scheduling` from B0. Startup
completed with the same model and KV settings, fixed MTP2 with
`num_spec_tokens=2` and `next_n=3`, all graph captures complete, and an explicit
log that asynchronous scheduling was enabled. Its deterministic-output canary
passed.

The bounded gate then returned:

| Cell | Observed tok/s | Gate floor | Result |
|---:|---:|---:|---|
| C1 | 22.43096 | 15 | pass |
| C2 | 16.53733 | 24 | **fail** |
| C8 | 41.63180 | 35 | pass |

These are short gate cells, not the matched streaming campaign matrix. In
particular, they must not be compared numerically with the B0/M3 25-second
decode cells.

Rank 0 logged first-use Triton JIT warnings for
`_map_global_topk_to_gathered_ckv_kernel` at log clock 14:30:24 and
`_pack_topk_routes_post_prefix_kernel` at 14:30:46 while the gate was running.
Cold first-use compilation likely contaminated the failed C2 cell, but that is
a diagnosis, not proof. The gate was correctly treated as failed; no full
matched prefill/decode matrix was run and AS is not promotable.

No durable AS gate JSON was written. The values and warnings above are
operator-recorded live-console evidence only, which is a material evidence
limitation. AS was stopped cleanly. At the time this draft was written, the
exact original fixed-MTP2 baseline was restarting; post-rollback verification
was still pending. The later closing B0 arm completed with healthy APIs before
and after, resolving that at-capture rollback uncertainty.

## Q8: 8,192 maximum batched tokens

Q8 changed only `max_num_batched_tokens` from 4,096 to 8,192. It preserved B0's
fixed MTP2, synchronous scheduling, model, KV allocation, topology, and graph
ceiling. Its matched artifact timestamp was 2026-08-02 15:45:21 UTC.

Operator-recorded startup logs proved the Q8192 compile range, a 1,939.8 MiB
mixed-buffer allocation, a 2,584.3 MiB rank arena, and the unchanged
1,125,632-token KV capacity. All four graph families - decode piecewise, decode
full, prefill piecewise, and prefill full - completed. The measured cells
produced no JIT warning, error, or graph fallback. Those raw logs are not
committed; the throughput cells below are artifact-attested by the Q8 JSON.

### Full streaming-prefill repeats

| Cell | Prompt tokens | TTFT | Q8 tok/s | Matched B0 delta |
|---|---:|---:|---:|---:|
| 16K r0 | 19,478 | 43.3769 s | **449.04** | **-17.67%** |
| 16K r1 | 19,472 | 34.6549 s | **561.88** | +1.45% |
| 16K r2 | 19,477 | 35.8330 s | **543.55** | -2.52% |
| 32K r0 | 38,933 | 71.8817 s | **541.63** | +0.16% |
| 32K r1 | 38,935 | 71.1079 s | **547.55** | +2.08% |
| 32K r2 | 38,933 | 70.6486 s | **551.08** | +0.82% |
| 64K r0 | 77,848 | 144.5049 s | **538.72** | +1.97% |
| 64K r1 | 77,848 | 145.2678 s | **535.89** | -0.08% |
| 64K r2 | 77,848 | 145.5368 s | **534.90** | -0.11% |

### Prefill summaries

| Nominal size | B0 mean | Q8 mean | Q8 median | Mean delta |
|---:|---:|---:|---:|---:|
| 16K | 552.29 | 518.16 | 543.55 | **-6.18%** |
| 32K | 541.25 | 546.75 | 547.55 | **+1.02%** |
| 64K | 533.39 | 536.51 | 535.89 | **+0.58%** |
| all nine cells, unweighted | 542.31 | 533.81 | 541.63 | **-1.57%** |

The first 16K sample, 449.04 tok/s, is an obvious outlier relative to the two
later Q8 repeats and all three matched B0 repeats. No measured-cell JIT warning,
error, or graph fallback explains it. Because the harness used fixed ordering,
the evidence cannot determine whether it was a transient first-cell effect or
something caused by Q8. It remains in every reported mean; discarding it after
seeing the result would be post-hoc selection. Even outside that sample, the
32K and 64K gains are only 1.02% and 0.58%, well below the 5% promotion target.

### Matched sustained decode

| Concurrency | B0 tok/s | Q8 tok/s | Delta | Q8 per-lane tokens |
|---:|---:|---:|---:|---|
| C1 | 23.16 | 22.20 | **-4.15%** | 555 |
| C2 | 37.44 | 33.60 | **-10.26%** | 420 / 420 |
| C4 | 54.36 | 51.40 | **-5.45%** | 351 / 348 / 235 / 351 |
| C8 | 74.52 | 73.36 | **-1.56%** | 258 / 249 / 258 / 187 / 258 / 258 / 108 / 258 |

Q8 regressed in every decode cell. C4 and C8 again contain underproducing
lanes, so a single arm cannot establish a stable distribution, but none of the
aggregate cells supports promotion.

A live MTP2 metrics snapshot recorded 2,447 drafts, 4,894 draft tokens, 4,143
accepted tokens, and accepted-position counts of 2,198 and 1,945. That is an
84.65% draft-token acceptance rate and 1.693 accepted tokens per draft. These
counters were operator-recorded live metrics and are not contained in the
sealed Q8 benchmark JSON, so they are diagnostic rather than artifact-attested
performance evidence.

Q8 is **not promoted**. Its small 32K/64K prefill changes did not reach the
target, its retained 16K outlier made overall mean prefill slower, and every
decode concurrency regressed.

The closing control sharpens that conclusion. Relative to the opening/closing
B0 midpoint, Q8 prefill is -6.36% at 16K, +0.11% at 32K, -0.29% at 64K,
and -2.21% overall. The apparent opening-only 32K/64K gains therefore collapse
to near zero. Midpoint decode deltas are +1.37% at noisy C1, -9.92% at C2,
-5.62% at C4, and -1.08% at C8. The raw opening-only comparisons above remain
part of the evidence; bracketing does not overwrite them.

## BR: completed closing fixed-MTP2 baseline

The closing B0 control used the same fixed-MTP2/Q4096 profile, prompt hashes,
original harness bytes, warmup, and benchmark order as opening B0. Its artifact
timestamp was 2026-08-02 16:30:01 UTC. Health passed before and after, all cells
were valid, and operator-observed measured-cell logs contained no JIT warning,
error, or graph fallback.

### Full closing prefill repeats

| Cell | Prompt tokens | TTFT | Tok/s |
|---|---:|---:|---:|
| 16K r0 | 19,478 | 35.2446 s | **552.65** |
| 16K r1 | 19,472 | 35.2053 s | **553.10** |
| 16K r2 | 19,477 | 34.9339 s | **557.54** |
| 32K r0 | 38,933 | 71.2134 s | **546.71** |
| 32K r1 | 38,935 | 71.4770 s | **544.72** |
| 32K r2 | 38,933 | 69.2898 s | **561.89** |
| 64K r0 | 77,848 | 142.6169 s | **545.85** |
| 64K r1 | 77,848 | 143.0907 s | **544.05** |
| 64K r2 | 77,848 | 144.5957 s | **538.38** |

### Opening-to-closing prefill drift

| Nominal size | Opening mean | Closing mean | Drift |
|---:|---:|---:|---:|
| 16K | 552.29 | 554.43 | **+0.39%** |
| 32K | 541.25 | 551.11 | **+1.82%** |
| 64K | 533.39 | 542.76 | **+1.76%** |
| all nine cells, unweighted | 542.31 | 549.43 | **+1.31%** |

### Opening-to-closing decode drift

| Concurrency | Opening tok/s | Closing tok/s | Drift | Closing per-lane tokens |
|---:|---:|---:|---:|---|
| C1 | 23.16 | 20.64 | **-10.88%** | 516 |
| C2 | 37.44 | 37.16 | **-0.75%** | 465 / 464 |
| C4 | 54.36 | 54.56 | **+0.37%** | 342 / 340 / 340 / 342 |
| C8 | 74.52 | 73.80 | **-0.97%** | 255 / 249 / 255 / 216 / 255 / 255 / 105 / 255 |

Prefill drifted about +0.4%, +1.8%, and +1.8% by size. C2, C4, and C8
held within about 1%; C1's -10.9% movement is noisy and should not be used as a
fine-grained arm discriminator. The systemic C8 imbalance also persisted: the
slow lane produced 105 tokens against a median of 255. Closing B0 therefore
bounds ordinary prefill and C2/C4/C8 campaign drift, but it does not resolve
C1 variance or the C8 lane pathology.

## Standard `llm_decode_bench` v0.4.31 result

A final standard-harness run measured the restored fixed-MTP2 B0 default with
`llm_decode_bench.py` v0.4.31. This is separate from the campaign's custom
matched harness and repeat artifacts. It used duration-based sustained decode,
fully unique 16K contexts, C1/C2/C4/C8, 25-second measurement windows, a
three-second decode warmup, `max_tokens=2048`, `temperature=0`,
`ignore_eos=true`, DCP4, and the observed 1,125,632-token KV budget.

The run required `--cell-warmup-timeout-seconds 300`. An earlier attempt with
the original 60-second timeout invalidated C2, C4, and C8 while waiting for
readiness/admission; it recorded no request errors. Its artifact SHA-256 is
`d3025f37e8fdd2060184ba4c3b85012f5547c9d7c6e304f0a8b442d550b0d8e2`.
That attempt's prefill scouts were 555/545/536 tok/s at 16K/32K/64K, and its
only valid decode cell was C1 at 16.62 tok/s. C2 reached two running requests
but still exceeded the readiness deadline; C4 and C8 were also underfilled.
Their partial rates are not throughput results and are not retained. With the
300-second timeout, every requested concurrency reached effective concurrency
and all cells were valid.

### Standard-harness prefill

| Requested context | Observed prompt tokens | TTFT | Prefill tok/s | Method |
|---:|---:|---:|---:|---|
| 16K | 16,249 | 28.969 s | **561** | integrated scout |
| 32K | 32,321 | 58.750 s | **550** | scout-only |
| 64K | 64,513 | 119.016 s | **542** | scout-only |

### Standard-harness sustained decode at 16K

| Concurrency | Aggregate tok/s | Effective concurrency | Errors | Valid |
|---:|---:|---:|---:|---|
| C1 | **15.9** | 1 | 0 | yes |
| C2 | **28.9** | 2 | 0 | yes |
| C4 | **43.1** | 4 | 0 | yes |
| C8 | **62.1** | 8 | 0 | yes |

The full run took 808.141 seconds, approximately 13 minutes 28 seconds, with
326 hardware samples across four GPUs. Total GPU power averaged 280.42 W and
peaked at 322.58 W. These are run-wide telemetry summaries, not normalized
energy or per-cell efficiency claims.

This standard result is **public-functional-lane, external-evidence,
live-validated**. It is not a reference-lane result, public-bootstrap
acceptance, or a correctness pass. The raw artifact remains uncommitted because
its diagnostics contain private workstation and cluster identities; only the
sanitized results and artifact hash are published here.

## Standard quick 16K arm comparison

Four later `llm_decode_bench.py` v0.4.31 quick matrices used the same standard
decode settings: fully unique 16K contexts, C1/C2/C4/C8, 25-second windows,
three-second warmup, `max_tokens=2048`, temperature zero, `ignore_eos=true`,
the 300-second readiness timeout, and no prefill phase. Every cell reached its
requested concurrency with zero request errors.

| Arm | C1 | C2 | C4 | C8 | Artifact SHA-256 |
|---|---:|---:|---:|---:|---|
| opening B0, fixed MTP2 | **18.45** | **26.39** | **43.15** | **61.91** | `09de1306cd765560701f837ff93c6f724c506ea0d4ade9da441c1df1aab85e34` |
| M3, fixed MTP3 | **17.93** | **27.51** | **41.01** | **58.32** | `373ded09e49bb706b36244b7bc67eb1e1d08774d08f94cad835f060545e055fd` |
| AD, adaptive MTP2/3 | **14.79** | **29.04** | **41.75** | **58.17** | `c17a4703821fb0b4fc06cada1731c6489e9d6f9f6fecca995f9b75cdf4bb2aad` |
| closing B0, fixed MTP2 | **17.46** | **28.36** | **43.40** | **61.43** | `74a5188e36892196c6629df8f21f2d60113875ccd982704ff269b3c3fa778f5a` |

Opening-to-closing B0 drift was -5.34% at C1, +7.45% at C2, +0.59% at
C4, and -0.77% at C8. Against the B0 midpoint, M3 was -0.13%/+0.50%/-5.24%/
-5.43% at C1/C2/C4/C8. Adaptive was -17.60%/+6.11%/-3.52%/-5.68%.

These standard quick matrices do **not** reproduce the custom matched harness's
M3 C2/C4 gains: M3 is effectively flat at C1/C2 and slower at C4/C8, while
adaptive improves C2 but regresses the other three cells. B0 remains the
default. M3's Pareto label is restricted to the custom matched harness and its
measured workload; it is not a standard-harness C2/C4 claim.

These artifacts used different prompts and ordering from the custom campaign
matrices. Their values and deltas must not be pooled with custom-harness cells.
Raw diagnostics remain uncommitted because they contain private site identity.

## Runtime correctness caveat

The stock [`scripts/exl3_live_gate.py`](../scripts/exl3_live_gate.py)
deterministic check is **not passing** at `max_tokens=128`. Two greedy
fixed-seed B0 outputs first diverged at token index 124. A six-run localization
produced two output hashes with a 4/2 frequency split, again localized to token
124.

A later ten-run logprob sample at that position produced the token ` if` five
times and ` in` five times. Some runs reported an exact tie between the top
logprobs; the non-tied runs had only a 0.125-0.25 top-logprob margin. This is a
narrow repeatability caveat near a tied or low-margin choice, not evidence of
checkpoint corruption. It does not invalidate the health or performance cells.

It does mean the deterministic 128-token gate **fails**, not merely that
equivalence is unproven. Earlier fixed-seed divergence was also observed on Q8,
so cross-arm output equivalence remains unproven. B0 may remain the performance
campaign default, but it is not correctness-accepted. Release-quality use still
requires a passing or explicitly tolerance-bounded correctness policy with
durable inputs, token IDs, and comparison artifacts.

As a bounded diagnostic, the same stock gate passed when limited to 120 output
tokens, before the observed branch point. Its deterministic output SHA-256 was
`6ebd2106f275ecccceeb8ecb01d538c404727eab9df009f2a4815dee246e2849`,
and its C1/C2/C8 cells were 21.31/30.65/44.14 tok/s against thresholds of
15/24/35. The uncommitted gate artifact SHA-256 is
`a38441afbee2c22bfe362dc8924cc96aa53530d9a7c48a44b5468ca463021038`.
This bounds the observed failure; it does not replace or waive the failing
128-token release gate.

## AD: adaptive MTP2/3

AD is a **live-validated, matched-benchmark-complete configuration arm**, but it
is not promoted. It used adaptive depths 2/3, maximum depth 3, a 32-round
window, `num_spec_tokens=3`, true adaptive draft, index reuse, synchronous
scheduling, and the unchanged 4,096 batch-token limit. The artifact timestamp
was 2026-08-02 17:35:40 UTC and it used the final extended harness hash.

Operator-recorded startup evidence showed `next_n=4`, index reuse activated,
the unchanged 1,125,632-token KV capacity, and every graph family complete.
All four ranks ran the exact sealed image with zero restart or OOM events.
These startup and metrics observations come from uncommitted live logs; the
throughput cells below are attested by the benchmark JSON.

### Adaptive streaming prefill

| Cell | Prompt tokens | TTFT | Tok/s |
|---|---:|---:|---:|
| 16K r0 | 19,478 | 35.6540 s | **546.31** |
| 16K r1 | 19,472 | 36.3194 s | **536.13** |
| 16K r2 | 19,477 | 35.3891 s | **550.37** |
| 32K r0 | 38,933 | 71.8560 s | **541.82** |
| 32K r1 | 38,935 | 72.4754 s | **537.22** |
| 32K r2 | 38,933 | 70.8599 s | **549.44** |
| 64K r0 | 77,848 | 145.0657 s | **536.64** |
| 64K r1 | 77,848 | 146.1947 s | **532.50** |
| 64K r2 | 77,848 | 147.7089 s | **527.04** |

| Nominal size | B0 mean tok/s | AD mean tok/s | Delta |
|---:|---:|---:|---:|
| 16K | 552.29 | **544.27** | -1.45% |
| 32K | 541.25 | **542.82** | +0.29% |
| 64K | 533.39 | **532.06** | -0.25% |
| all nine cells, unweighted | 542.31 | **539.72** | -0.48% |

At log clock 17:13:16, during the measured phase, rank 0 emitted a Triton JIT
warning for `_pack_topk_routes_post_prefix_kernel`. The artifact still has nine
valid prefill cells, healthy before/after checks, and no cell error, but the
evidence does not identify which sample was affected or quantify contamination.
Prefill promotion therefore remains caveated rather than discarding any cell.

### Adaptive sustained decode

| Concurrency | B0 tok/s | AD tok/s | Delta | AD per-lane tokens |
|---:|---:|---:|---:|---|
| C1 | 23.16 | **25.84** | +11.57% | 646 |
| C2 | 37.44 | **40.68** | +8.65% | 509 / 508 |
| C4 | 54.36 | **61.36** | +12.88% | 386 / 372 / 384 / 392 |
| C8 | 74.52 | **68.72** | -7.78% | 263 / 124 / 272 / 168 / 272 / 272 / 89 / 258 |

Against the opening/closing B0 midpoint, AD is -1.13% in overall prefill,
+17.99% at C1, +9.06% at C2, +12.67% at C4, and -7.34% at C8. C2/C4 gains
survive the baseline bracket. C1 exceeds both controls but remains a single
sample in the campaign's noisy C1 lane. C8 materially regresses and contains
three low lanes: 89, 124, and 168 tokens against a 260.5-token median.

Final live counters recorded 1,688 rounds, 5,064 draft tokens, 4,319 accepted
tokens, and accepted-position counts of 1,578 / 1,428 / 1,313. That is 85.29%
draft-token acceptance and 2.559 accepted tokens per round. More importantly,
5,064 is exactly three draft tokens for every one of the 1,688 rounds. The
controller never selected depth 2 and behaved as depth 3 throughout this
observation. AD therefore measures the declared configuration, but does not
demonstrate a benefit from adaptive depth selection.

AD is **not promoted**: adaptive selection did not vary, C8 regressed about 7%
against the bracket, prefill promotion is JIT-caveated, and a controlled
repeated-C8 pair later showed no robust adaptive median gain. It remains useful
evidence for the measured custom-harness C2/C4 tradeoff, not a general
adaptive-MTP win.

## M1: fixed-MTP1 synchronous

M1 is a **public-functional-lane, live-validated,
matched-benchmark-complete configuration arm**, but it is not promoted. It
changed only fixed speculative depth from two to one. The live engine reported
`num_spec_tokens=1` and `next_n=2`; scheduling remained synchronous, the batch
limit remained 4,096, and adaptive control plus index reuse were disabled. Its
artifact timestamp was 2026-08-02 19:11:31 UTC.

Operator-recorded startup evidence showed the exact sealed image on all four
ranks, zero restarts or OOMs, unchanged 1,125,632-token KV capacity, and all
graph families complete. EXL3 buffers were 979.7 MiB. Graph capture took 92
seconds and the reported memory change was -4.33 GiB. These startup facts and
the final counters below come from uncommitted live logs; the benchmark JSON
attests the throughput cells.

### M1 streaming prefill

| Cell | Prompt tokens | TTFT | Tok/s |
|---|---:|---:|---:|
| 16K r0 | 19,478 | 34.7861 s | **559.94** |
| 16K r1 | 19,472 | 35.5762 s | **547.33** |
| 16K r2 | 19,477 | 34.8904 s | **558.23** |
| 32K r0 | 38,933 | 70.3697 s | **553.26** |
| 32K r1 | 38,935 | 70.1446 s | **555.07** |
| 32K r2 | 38,933 | 69.8062 s | **557.73** |
| 64K r0 | 77,848 | 142.6579 s | **545.70** |
| 64K r1 | 77,848 | 144.5360 s | **538.61** |
| 64K r2 | 77,848 | 145.5857 s | **534.72** |

| Nominal size | B0 mean tok/s | M1 mean tok/s | Delta |
|---:|---:|---:|---:|
| 16K | 552.29 | **555.17** | +0.52% |
| 32K | 541.25 | **555.35** | +2.61% |
| 64K | 533.39 | **539.68** | +1.18% |
| all nine cells, unweighted | 542.31 | **550.07** | +1.43% |

During the final measured 16K cell, log clock 18:49:32-18:49:33, rank 0
emitted Triton JIT warnings for `_w4a16_route_count_kernel`,
`_pack_topk_routes_post_prefix_kernel`, and
`_pack_topk_routes_sort_kernel`. All nine artifact cells remain valid and are
retained, but the small apparent prefill gain is promotion-caveated. Against
the opening/closing B0 midpoint, overall prefill is only +0.77%.

### M1 sustained decode

| Concurrency | B0 tok/s | M1 tok/s | Delta | M1 per-lane tokens |
|---:|---:|---:|---:|---|
| C1 | 23.16 | **17.36** | **-25.04%** | 434 |
| C2 | 37.44 | **28.80** | **-23.08%** | 360 / 360 |
| C4 | 54.36 | **43.08** | **-20.75%** | 282 / 282 / 231 / 282 |
| C8 | 74.52 | **65.60** | **-11.97%** | 216 / 216 / 216 / 216 / 216 / 216 / 128 / 216 |

Against the opening/closing B0 midpoint, decode remains -20.73% at C1,
-22.79% at C2, -20.90% at C4, and -11.54% at C8. The regression is therefore
far larger than baseline drift at C2/C4/C8; C1 is also below both baseline
controls. The C8 aggregate retains a 128-token slow lane against a 216-token
median, but the arm is already decisively slower at every concurrency.

Final MTP counters recorded 2,390 rounds, 2,390 draft tokens, 2,194 accepted
tokens, and 2,194 position-0 acceptances: exactly one draft per round, 91.80%
draft-token acceptance, and 0.918 accepted tokens per round.

M1 is **not promoted**. Its roughly flat-to-slightly-higher, JIT-caveated
prefill does not compensate for 12-25% decode regressions. The decode result is
decisive enough that no M1 repeat is required for this campaign disposition.

## R8: repeated C8 windows

**Completed B0 repeat; the later fresh-salt pair is also complete.** The decode-only
artifact uses schema `sparkring-exl3-ab-decode-repeat/v1`, timestamp
2026-08-02 16:46:28 UTC, seed `20260802`, and fresh run salt
`b0-c8-closing-20260802-a`. The decode-repeat harness hash is
`b276f7f1a54afe4732b204c0b14f0ab1600a8d3cbde718f6ecfec186d6ee67e9`.
Health passed before and after, all eight lanes returned in each window, and
there were no invalid cells or lane errors. All imbalanced lanes are retained.

The proposed fairness check requires the minimum lane count to reach at least
80% of that window's median lane count.

| Repeat | Aggregate tok/s | Total tokens | Per-lane tokens | Lane median | Minimum | 80% threshold | Fairness |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | **61.48** | 1,537 | 85 / 255 / 253 / 255 / 85 / 94 / 255 / 255 | 254 | 85 | 203.2 | fail; 3 lanes below |
| 1 | **73.04** | 1,826 | 252 / 252 / 252 / 252 / 252 / 227 / 252 / 87 | 252 | 87 | 201.6 | fail; 1 lane below |
| 2 | **77.68** | 1,942 | 261 / 261 / 261 / 261 / 261 / 261 / 261 / 115 | 261 | 115 | 208.8 | fail; 1 lane below |

Aggregate C8 was strong at a **73.04 tok/s median** and **70.73 tok/s mean**,
but its **16.20 tok/s range** is material. Every repeat failed the proposed
minimum-lane fairness criterion, and repeat 0 had multiple slow lanes. The
median is near opening B0's 74.52 tok/s and closing B0's 73.80 tok/s, yet the
wide repeat range prevents treating a single C8 window as stable. C8 remains
highly window-, prompt-, and lane-sensitive.

### Incomplete old-salt adaptive attempt

The first adaptive attempt with the same `b0-c8-closing-20260802-a` salt was
reported as timing out, but a benchmark artifact was subsequently recovered.
It contains two valid windows and one invalid window, not a complete pair.

| Repeat | Aggregate tok/s | Per-lane tokens | Fairness |
|---:|---:|---|---|
| 0 | **58.84** | 65 / 260 / 260 / 247 / 65 / 67 / 247 / 260 | fail; 3 lanes below 197.6 |
| 1 | **74.08** | 260 / 260 / 259 / 251 / 259 / 241 / 254 / 68 | fail; 1 lane below 205.2 |
| 2 | invalid | 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 | no continuous usage stats for four lanes |

The artifact SHA-256 is
`9cf2f5f6c30fb6c19bbc506e003864e9db9052fbb80d4ab198dda3510777022e`.
The two valid aligned AD-minus-B0 window deltas are -2.64 and +1.04 tok/s.
Because repeat 2 is invalid, no three-window median, mean, range, or promotion
delta is reported from this attempt. The later fresh-salt pair below is the
complete paired result.

## Fresh-salt paired C8 result

### AD-R8: fresh-salt, clean-restart paired C8

**Pair complete; AD not promoted.** This replacement pair followed the
incomplete old-salt attempt above. It used a fresh salt and clean restart for
each side; the incomplete attempt is not pooled into this result.

The replacement AD service then started cleanly with zero prefix-cache queries
and hits before the benchmark. Its artifact uses salt
`exl3-c8-pair-b-20260802`, schema
`sparkring-exl3-ab-decode-repeat/v1`, timestamp 2026-08-02 18:24:12 UTC,
and the current repeat harness. Health passed before and after; all three cells
and all 24 lanes were valid and error-free.

| Repeat | Aggregate tok/s | Total tokens | Per-lane tokens | Lane median | Minimum | 80% threshold | Fairness |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | **83.04** | 2,076 | 259 / 255 / 262 / 251 / 259 / 264 / 262 / 264 | 260.5 | 251 | 208.4 | pass; 0 below |
| 1 | **58.76** | 1,469 | 90 / 81 / 282 / 76 / 284 / 270 / 111 / 275 | 190.5 | 76 | 152.4 | fail; 4 below |
| 2 | **60.48** | 1,512 | 259 / 268 / 92 / 266 / 208 / 67 / 268 / 84 | 233.5 | 67 | 186.8 | fail; 3 below |

The aggregate median was **60.48 tok/s**, the mean **67.43 tok/s**, and the
range **24.28 tok/s**. Only one of three windows passed the fairness criterion.
Across all 24 lane observations, the median was 259 tokens, the 80% threshold
207.2, and the minimum 67; seven lanes were below threshold, so combined-lane
fairness also fails. The balanced 83.04 tok/s first window is real, but it does
not cancel the two later imbalanced windows or justify quoting the peak as
representative C8 throughput.

Final live counters recorded 2,256 rounds, 6,768 draft tokens, 4,651 accepted
tokens, and position counts of 1,678 / 1,543 / 1,430. Acceptance was 68.72%
of draft tokens and 2.062 tokens per round. Exactly 3.0 draft tokens were
generated per round, proving that depth 2 was never selected; the controller
again behaved as depth 3 throughout.

The B0 control then used the identical `exl3-c8-pair-b-20260802` salt after its
own clean restart. All ranks used the exact sealed image, the engine reported
fixed MTP2 with `num_spec_tokens=2` and `next_n=3`, and prefix-cache queries
and hits were both zero before the benchmark. Health passed before and after;
all three cells and all 24 lanes were valid and error-free.

| Repeat | Aggregate tok/s | Total tokens | Per-lane tokens | Lane median | Minimum | 80% threshold | Fairness |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | **76.88** | 1,922 | 252 / 252 / 158 / 252 / 252 / 252 / 252 / 252 | 252 | 158 | 201.6 | fail; 1 below |
| 1 | **61.04** | 1,526 | 124 / 104 / 273 / 99 / 273 / 266 / 114 / 273 | 195 | 99 | 156.0 | fail; 4 below |
| 2 | **60.56** | 1,514 | 249 / 249 / 132 / 249 / 202 / 100 / 249 / 84 | 225.5 | 84 | 180.4 | fail; 3 below |

B0's aggregate median was **61.04 tok/s**, its mean **66.16 tok/s**, and its
range **16.32 tok/s**. The 60.56 tok/s value is repeat 2, not the median. None
of the three B0 windows passed fairness. Across its 24 lane observations, the
median was 249 tokens, the 80% threshold 199.2, and the minimum 84; eight lanes
were below threshold, so combined fairness fails.

Final B0 counters recorded 2,736 rounds, 5,472 draft tokens, 3,736 accepted
tokens, and position counts of 1,980 / 1,756. That is exactly 2.0 drafts per
round, 68.27% draft-token acceptance, and 1.365 accepted tokens per round.

### Paired comparison

| Repeat | B0 tok/s | AD tok/s | AD - B0 |
|---:|---:|---:|---:|
| 0 | 76.88 | 83.04 | **+6.16** |
| 1 | 61.04 | 58.76 | **-2.28** |
| 2 | 60.56 | 60.48 | **-0.08** |

| Summary | B0 tok/s | AD tok/s | AD - B0 |
|---|---:|---:|---:|
| median | **61.04** | **60.48** | **-0.56 (-0.92%)** |
| mean | **66.16** | **67.43** | **+1.27 (+1.91%)** |
| range | **16.32** | **24.28** | **+7.96 wider for AD** |

There is **no robust adaptive median gain**. The positive adaptive mean is
driven by repeat 0; the two later aligned windows are slower, and adaptive
median is 0.56 tok/s below B0. Adaptive passed one of three per-window fairness
checks versus zero of three for B0, but both combined lane sets fail fairness:
seven adaptive and eight B0 lanes fall below their respective thresholds.
Prompt-sensitive lane collapse dominates both arms, and AD is more variable.

The same salt and clean restarts make this a matched configuration pair for
these measured prompts, but they do not establish a general controller effect.
AD again generated exactly three drafts per round and never selected depth 2.
AD is not promoted. Fixed-MTP2 B0 remains the campaign default, while M3
remains only a Pareto alternative for measured C2/C4 workloads in the custom
matched harness. The older R8
artifact remains useful variance evidence but is not part of this matched pair.

## Optional follow-ups

### CX: combined selected settings

**Pending individual-arm results.** A combined arm may test whether independently
successful settings compose. It cannot attribute a gain to any one setting and
must list its exact parent arms. Do not run or publish it as the winner before
the fresh-salt B0/AD repeated-C8 pair establishes the remaining effect and
variance.

## Proposed 512K capacity profile

The deployment only requires approximately 500K tokens of context and KV
capacity if the reduced allocation creates useful serving headroom. A coherent
candidate is:

```text
max_model_len=524288
kv_cache_memory_bytes_per_rank=4500000000
projected_kv_tokens=562816
```

The 562,816-token capacity is a linear half-scale projection from the observed
9,000,000,000-byte, 1,125,632-token profile. It would retain approximately the
same 1.07x capacity ratio for a 524,288-token maximum request. It is **not a live
measurement**.

Reducing capacity alone is not claimed to speed attention, EXL3, or other
kernels. It would reclaim approximately 4.5 GB decimal (4.19 GiB) per rank for
larger batch, concurrency, or CUDA-graph headroom. Test the reduced-capacity
profile first with no other throughput change, then test use of the reclaimed
memory in separately declared arms. Otherwise a speed change cannot be
attributed to capacity reduction.

The 512K profile must pass model/KV readback, startup and graph capture, at least
one near-limit request, the matched benchmark, all-rank health, and rollback
checks before it becomes an alternative recipe.

## Final disposition

**Performance campaign complete.** The opening and closing controls bracketed
ordinary drift, and the fresh-salt C8 pair completed with durable artifacts.
B0 remains the default. M3 remains a measured Pareto alternative for C2/C4
workloads in the custom matched harness only; the standard quick matrix did not
reproduce that gain. Its prefill regression and negligible custom-harness C8
change prevent general promotion. M1, Q8, AS, and AD are not promoted.

The incomplete old-salt adaptive repeat remains bounded to its two valid
windows and one invalid window; it is not a paired summary. The stock 128-token deterministic gate fails at token 124,
so the release-quality correctness gate remains open. Therefore this
disposition is a completed **public-functional-lane external performance campaign**, not
public-bootstrap acceptance, a release correctness pass, or a reference-lane
result. Combined and 512K arms remain optional follow-ups rather than required
campaign gates.

## Operational safety

Every arm requires a four-rank restart and therefore **STOPS SERVING**. Inspect
the dry-run plan and obtain explicit authorization for the named hosts before a
cutover. Capture the current four-rank state before each arm. On any startup,
graph, correctness, gate, benchmark, temperature, OOM, restart, NCCL, or RDMA
failure, remove only the explicitly named arm containers and restore the sealed
baseline. Verify every rank and both `/health` and `/v1/models` after rollback.
Never place management addresses, SSH identities, NIC assignments, host paths,
or credentials in tracked campaign evidence.
