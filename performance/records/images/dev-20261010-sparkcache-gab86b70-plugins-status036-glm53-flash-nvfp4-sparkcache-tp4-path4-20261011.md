# SparkCache variant of GLM-5.3-Flash CSF on four Sparks of the ring of eight

Lane: **public-functional**. Status: **research-only**; maturity
**live-validated** for the stages below, not `implemented`. Evidence scope:
**`glm53-flash-nvfp4-sparkcache-tp4` installed with `sparkring install` on
positions 0 to 3 of an eight-Spark ring, on the SparkCache lease layer's
derived image `e1607979b7f0`, 2026-10-10 and 2026-10-11: Stage 1 of the
[hardware test plan](../../sparkcache-flash-csf-hardware-test-plan.md), steps
1-7 and 9 and a single-start speed check; step 8 and the two-start A/B did
not run; Stage 2, `glm53-flash-nvfp4-sparkcache-tp2` on positions 0 and 1,
stopped both Sparks during its first start; not serving-qualified.** This
record takes the test operator's notes as they stood at 2026-10-11 06:32 UTC
and the coordinator's report of Stage 2's start.

## Conditions

| Item | Value |
|---|---|
| Profile | `glm53-flash-nvfp4-sparkcache-tp4`: `glm53-flash-nvfp4-spark-tp4`'s settings on the CSF checkpoint plus the SparkCache connector: model profile `glm53-flash-hybrid`, lease mode `connector-jobs` with contract `vllm-connector-jobs-gab86b70.json`, publication `tail-cow-v2`, 64K-token spans, 8 GiB of cache, CUDA restore |
| Connector arenas | `spark_cache_cuda_placement_arena_bytes` 67108864 inside a `spark_cache_cuda_restore_arena_budget_bytes` of 268435456 (commit `a6d1e309`); `spark_cache_load_threads` 8, which that budget caps at 2. `config.json` SHA-256 `15fd4f45e5c8…` |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at `dec48abd33ef`, 45 files, in place on every rank; SparkCache checkpoint identity `6af0343976ee…` |
| Cache root | `/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef` in the containers, created empty by the 2026-10-11 start |
| Image | Derived image `sha256:e1607979b7f0f00217c8deae170185a57f693721f1bc88368dcc536181a495f7`, built on a Spark from the recipe [installer-sparkcache-gab86b70](../../../runtime/images/compositions/installer-sparkcache-gab86b70/README.md) over release 2026.10.2's image `d52737a109e0`, on all eight Sparks before the 2026-10-11 run. Its lock, which `derived_layer.py record` wrote on Node A (SHA-256 `1c2fee63…`), is not in this repository |
| Packages | SparkRing revision `46eeb18b` (branch `claude/ring-test-20261010`, with `a6d1e309`) on every Spark for 2026-10-11; revision `0291b9cb` for 2026-10-10 |
| Hardware | Eight DGX Sparks (GB10), one ConnectX-7 RoCE ring of eight (`cycle-8`). The deployment ran on positions 0 to 3 as a `path-4` group, rank `r` on position `r`, on SIRCL ring sessions with the default tuning table. GPU clocks 2405-2411 MHz at step 3, not locked |
| Other groups | GLM-5.3-Flash CSF at TP2 on positions 4 and 5, `qwen38-flash-next-tp2 --checkpoint csf` on positions 6 and 7, both serving and not touched; a separate measurement ran against the Qwen API during the run ([record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-qwen38-flash-next-tp2-csf-pair-20261011.md)) |
| Client | A workstation on the owner's network against rank 0's API, Python 3.12.3 and httpx 0.28.1; for speed, the serving A/B runner's `bench_run` and `prefill_probe.py` of SparkRing revision `0291b9cb` with llm-inference-bench `ca84f36` |
| Automatic recovery | On throughout |

## Measurement

- **Requests.** Temperature-0 chat completions, streamed with usage, with
  `reasoning_effort` `low`. Prompts are filler sentences with "vault" records
  whose six-digit codes are the exact answers, sized on vLLM's `/tokenize`
  token IDs. Each result records the request SHA-256, `prompt_tokens`,
  `cached_tokens`, time to first token (first content or reasoning delta),
  finish reason and exact-answer equality.
- **Per rank:** every rank's container log: the `sparkcache: config` line,
  commits (`committed N tokens`), hits, restores (`restore tokens=N total=Xms
  … bytes=Y MiB`), rejections and invalidations.
- **Restarts.** Step 6: `sudo sparkring down --on 0-3` and `up --on 0-3`, each
  planned and then executed. Step 7: `docker kill` of all four rank containers
  at once (restart policy `no`), then `up --on 0-3`. Automatic recovery's
  journal and record show whether it acted.
- **Step 9.** With the workers stopped, every chunk in rank 2's cache root had
  one bit flipped at its midpoint (`cache_fault.py corrupt`); ranks 0, 1 and 3
  were left intact.
- **Speed.** Decode with `bench_run` at 32K, 1, 4 and 8 streams, temperature 0,
  at most 1,024 output tokens, 30 s cells after 10 s of warm-up, a unique
  prefix per stream, the dashboard held quiet; cold time to first token with
  `prefill_probe.py` at 32K, three repeats, unique prefixes.
- **Neighbours.** `GET /health` of the APIs on positions 4 and 6 every 5-9 s
  through the 2026-10-11 run.
- **Raw files** stay with the operator on Node A; they hold host names and
  addresses.

## Result

### 2026-10-10, package `0291b9cb`: 256 MiB placement arenas

The start reached kernel compilation; then every worker refused its connector
configuration at 01:32:53 UTC:
`spark_cache_cuda_restore_arena_budget_bytes must be at least 536870912 bytes
for one lane's two arenas`. The installer restarted the deployment that served
before it, `deepseek-v41-flash-tp4`. Automatic recovery did not act.

### 2026-10-11, package `46eeb18b`: Stage 1 steps

| Step | Result | Evidence |
|---|---|---|
| 1. Image probe, no GPU | pass (2026-10-10, same image) | 69 of 69 SparkCache receipt entries, the four layer files and two receipt hashes match; the contract verifies 11 of 11 vLLM files |
| 2. Cold start, empty root | pass | API ready after 467.5 s; every rank logs the connector config (read-write, store, restore and CUDA restore on, `max_span=65536`, `load_threads=2`), two 64 MiB arenas and an empty capacity |
| 3. Exact answers | pass | `sparkring check --on 0-3`: functional 7 of 7, transport as expected (SIRCL on every collective, NCCL absent), connector state enabled with the profile's settings; known answers `391`, `Paris`, `gnir` exact |
| 4. Cold prefill | pass | 37,029 and 37,034 tokens: `cached_tokens` 0, 14.54 s and 13.81 s to first token, exact answers; every rank committed 36,864 tokens in 604-720 ms |
| 5. Prefix hits and suffix cases | pass | below |
| 6. Planned restart restore | pass | below |
| 7. SIGKILL restore | pass; automatic recovery did not act | below |
| 8. Restore fixture (`needle_hunt.py`) | not run | |
| 9. Failure path | pass | below |
| Single-start speed | ran; 4-stream cell flagged | below |
| Two-start A/B | not run | |

**Step 5.** Prompts sharing a prefix of exactly 32,768 tokens (32 pages of
1,024) with a 32,787-token base, each adding N tokens:

| Case | Prompt tokens | `cached_tokens` | Scheduled | Time to first token, s | Answer |
|---|---|---|---|---|---|
| base, cold | 32,787 | 0 | — | 11.388 | exact |
| identical repeat | 32,787 | 31,744 | — | 0.897 | exact |
| N=512 | 33,280 | 31,744 | 1,536 | 1.053 | exact |
| N=513 | 33,281 | 31,744 | 1,537 | 1.496 | exact |
| N=515 | 33,283 | 31,744 | 1,539 | 1.651 | exact |
| N=531 | 33,299 | 31,744 | 1,555 | 1.464 | exact |
| N=1027 | 33,795 | 31,744 | 2,051 | 1.531 | exact |
| N=2048 | 34,816 | 31,744 | 3,072 | 1.582 | exact |

These hits are vLLM's in-process GPU prefix cache, one 1,024-token page short
of the shared pages; no `sparkcache: hit` line appeared. SparkCache committed
the new entries meanwhile.

**Steps 6 and 7.** Writers were cold, and every rank committed their pages
(12,288 tokens for each 12K entry, 32,768 for the 32K entry). The planned
restart (`up` ready after 379.9 s) kept the cache root, 1,559,929,009 bytes in
35 files per rank. After the SIGKILL, `up` found every rank stopped with exit
code 137 and took the full start path (ready after 369.3 s); automatic
recovery reported the group `unknown`, then `busy` while `up` ran, and made no
attempt.

| Reader | After | `cached_tokens` | Time to first token, s (writer's cold) | Answer |
|---|---|---|---|---|
| A, 12K | planned restart | 12,288 | 0.514 (4.451) | exact |
| 32K | planned restart | 32,768 | 0.588 (11.149) | exact |
| B, 12K | SIGKILL | 12,288 | 0.545 | exact |
| 32K | SIGKILL | 32,768 | 0.617 | exact |

Restores per rank, tokens / ms / MiB:

| Rank | Position | Planned, A | Planned, 32K | SIGKILL, B | SIGKILL, 32K |
|---|---|---|---|---|---|
| 0 | 0 | 12,288 / 243.5 / 115.3 | 32,768 / 334.2 / 246.8 | 12,288 / 241.8 / 115.3 | 32,768 / 326.4 / 246.8 |
| 1 | 1 | 12,288 / 228.4 / 115.3 | 32,768 / 317.6 / 246.8 | 12,288 / 227.4 / 115.3 | 32,768 / 316.3 / 246.8 |
| 2 | 2 | 12,288 / 229.7 / 115.3 | 32,768 / 318.8 / 246.8 | 12,288 / 219.7 / 115.3 | 32,768 / 295.3 / 246.8 |
| 3 | 3 | 12,288 / 204.0 / 115.3 | 32,768 / 269.0 / 246.8 | 12,288 / 228.6 / 115.3 | 32,768 / 301.9 / 246.8 |

The API's `cached_tokens` equalled the restored tokens on every rank in every
case.

**Step 9.** A cold 12K writer committed 12,288 tokens on every rank. With the
workers stopped, all 30 chunks (1,801,673,998 bytes) of rank 2's root were
corrupted. After `up` (ready after 365.3 s) every rank offered all 10 entries.
Ranks 0, 1 and 3 restored 12,288 tokens (218.3-232.3 ms, 115.3 MiB); rank 2
rejected a page object on its SHA-256 and invalidated the entry. The API
returned `cached_tokens` 0 and a first token after 5.641 s (writer 5.007 s),
so the request was recomputed, with the writer's answer.

**Speed**, one start, after step 9. Decode output tok/s at 32K:

| Streams | Variant | Cache-off reference | Ratio | Notes |
|---|---|---|---|---|
| 1 | 62.9 | 64.2 | 0.98 | 0 errors |
| 4 | 121.9, flagged | 126.6 | 0.96 | the benchmark marked the cell `capacity_limited`: streams that finish resubmit unique 32K prefills, which queue behind `--max-parallel-prefills 1` |
| 8 | 201.6 | 218.7 | 0.92 | 0 errors; no request finished inside the window |

The reference is A1 of the
[ring-of-four installer record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md):
image `1a8c10354eb0`, a separate ring of four with the measured `cycle-4`
table, shared-prefix cells and another benchmark revision.

Time to first token at 32K: cold, 10.39 / 19.19 / 18.32 s and 10.27 /
20.74 / 14.08 s in two sets; a prefix restored from disk after a restart,
0.588 and 0.617 s; an in-process GPU prefix hit, 0.897 s.

**Neighbours.** All 528 health polls of each of the two other APIs returned
200 through the 2026-10-11 run, and the install log has no image-transfer
step.

### Two Sparks with 8 GiB capture slots

- **Conditions.** `glm53-flash-nvfp4-sparkcache-tp2` installed on positions 0
  and 1 of the same ring, on the same derived image, 2026-10-11, with the
  profile's settings at the time: `--gpu-memory-utilization` 0.87 (105.87 of
  121.69 GiB), 10 GiB of KV cache, two 8 GiB capture slots and a 256 MiB
  restore budget.
- **Observed.** During kernel compilation, at about 01:32 CDT as reported, both Sparks
  stopped answering: position 0 answered ping, but its SSH banner timed out;
  position 1 accepted no TCP connection on port 22. When this record was
  written the Sparks were still unreachable, and their recovery is not
  confirmed.
- **Likely cause (inference, not measured on this start).** Memory
  exhaustion. The cache-off profile's CSF starts at TP2 (image
  `816c6d6a7e96`, serving A/B runner logs, not a repository record) show, per
  Spark: 116.4-116.7 GiB `MemAvailable` with caches dropped before the start;
  111.8-113.6 GiB free in vLLM's worker before it loads 84.53 GiB of weights;
  10.0 GiB of KV cache and 0.16-0.36 GiB of CUDA graphs. With 16 GiB of
  pinned capture slots and 0.25 GiB of restore arenas beside them, about
  0.7 GiB is left for compilation, activations and the API server. The TP4
  variant, which ran, holds 42.56 GiB of weights and 37 GiB of KV per Spark,
  leaving about 15.6 GiB with the same slots.
- **Fix.** The TP2 variant takes two 512 MiB capture slots, the value of the
  qualified shared-2026.09.3 profile `glm53-flash-spark-tp2-dcp1-sparkcache`:
  1.25 GiB of pinned memory, which leaves about 15.7 GiB. Its KV cache and
  speed settings are unchanged. The installer's memory settle now refuses a
  start whose vLLM share plus SparkCache's pinned memory exceeds a Spark's
  total less 6 GiB, naming the numbers; the 8 GiB slots at 0.87 needed
  122.12 GiB against 115.69 GiB. No start with the 512 MiB slots has run.

## Conclusion

With 64 MiB placement arenas the variant starts on four Sparks of the ring of
eight on the derived image. It stores pages on every rank, restores them after
a planned restart and after SIGKILL of every rank (32,768 tokens in 269-334 ms
per rank, first token in about 0.6 s against about 11 s cold), and recomputes
correctly when one rank's cache is corrupt. With 256 MiB arenas inside a
256 MiB restore budget it does not start: the budget must hold one lane's two
arenas. Its single-start decode at 32K was 2-8 % below a cache-off reference
measured under different conditions.

## Limitations

- Step 8 and the two-start A/B did not run, so by the test plan the variant
  stays research-only. Stage 2, the TP2 variant, stopped both Sparks with 8 GiB
  capture slots; its cause is inferred, and no start with 512 MiB slots has
  run.
- The speed comparison crosses image, topology, tuning table, prefix sharing
  and benchmark revision; the cold-TTFT spread (10.3-20.7 s) is unexplained.
- The 2 effective restore load threads are below the configured 8; a larger
  restore budget would allow more.
- No spans beyond 64K tokens, no full-context restore, no more than two
  restarts.
- The derived image's lock and the raw files are not in this repository.
