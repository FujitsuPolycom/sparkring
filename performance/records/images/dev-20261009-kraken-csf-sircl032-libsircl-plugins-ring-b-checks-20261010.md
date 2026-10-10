# Installer checks on a ring of four with the SIRCL 0.3.2 image: pairs, screens, transports, lifecycle

Status: **research-only record of installer runs; single runs; not serving-qualified**.

One ring of four DGX Sparks (GB10, ConnectX-7 RoCE; positions 0-3, recorded as in the
[ring-of-four installer record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md))
served the installer profiles below with image `sha256:1a8c10354eb0…` and the recorded lock
[installer-image-c7c35fe0.json](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json),
unless a row names another image. Every plan listed no file to download. Packages: `6555b89b` for the first
pair installations, then `7013807b` (`6555b89b` with this record's repeated-installation fix). GPU clocks
were not locked. Checks are `sparkring check`'s 7 functional checks and three temperature-0 questions with
one exact answer each; screens are [`accept_profile.py`](../../harnesses/acceptance/accept_profile.py)'s
256-request correctness screen, as in the [CSF screen record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010.md).

## Outside client

A machine on the owner's network sent a short chat request to each served API about every 31 s throughout
these runs. A profile that serves 16 sequences absorbs it. GLM-5.3-Flash with CSF on two Sparks serves at most
8 (`--max-num-seqs 8`): in an 8-stream decode cell the outside request waits behind the cell's streams, and
the [serving A/B runner](../../harnesses/serving_ab/README.md)'s benchmark, whose readiness gate requires no
waiting request, then opens its 30 s window only when a stream completes, so the window covers that wave's
last tokens and the next wave's prefills.

## Result

| Check | Sparks | Result |
|---|---|---|
| `glm53-flash-nvfp4-spark-tp2`, CSF | 0, 1 | installed in 492 s; transport as expected, NCCL absent |
| `qwen38-flash-next-tp2` | 2, 3 | installed in 607 s at the same time; as expected; screen 256 responses, 0 degenerate, 1 wrong (`a7`, "the remainder when 1000 is divided by 7", answered 5 in 1 of 8 rounds), 0 errors |
| `qwen38-flash-next-qad-tp4` | 0-3 | 263 s; 7 of 7; screen 0 degenerate, 1 wrong (`a7`, 5), 0 errors |
| `deepseek-v41-flash-tp4` | 0-3 | 354 s; 7 of 7; screen 0 degenerate, 0 wrong, 0 errors |
| `qwen38-flash-next-tp2 --transport prepared` | 2, 3 | 301 s; 7 of 7 and 3 of 3; transport `prepared` (RoCEnante profile `tp2-rocenante-adaptive-prepared`) |
| `qwen38-flash-next-tp2 --transport nccl` | 2, 3 | 273 s; 7 of 7 and 3 of 3; transport `nccl` (vLLM's PyNccl on NCCL 2.32.3, RoCEnante off) |
| the same installation repeated while it serves | 2, 3 | package `6555b89b`: failed after 28 s, "Not starting: available memory after dropping caches is below what vLLM asks for" (the running model held it); package `7013807b`: complete in 40 s, the containers kept running |
| `sparkring down --on 2,3 --execute`, then `up --execute` | 2, 3 | 15 s, then 210 s; 7 of 7 and 3 of 3 |
| `docker kill` of rank 1's container | 2, 3 | `status --refresh`: "The model runs on rank 0 … but stopped on rank 1 …", exit code 137; automatic recovery began 2 min after the kill and reported "the model started" 3.5 min later; 7 of 7 and 3 of 3 |
| `sudo sparkring fabric verify` | 0-3 | "Fabric verified: 4 cables on 4 Sparks (cycle-4), relays 16 rules and 16 routes" in 3 s |
| `qwen38-flash-next-tp2 --image 2026.10.1` | 2, 3 | image `aba309e4610c` from the Sparks' own copies, transport `prepared` (that image has no SIRCL layer); 487 s; 7 of 7 and 3 of 3 |

Earlier screens of these Qwen profiles missed `a7` too
([TP2](dev-20260927-b12xcache-qwen38-flash-next-tp2-20260927.md),
[TP4](dev-20260927-b12xcache-qwen38-flash-next-tp4-20260927.md)).

### Measurements

The serving A/B runner's `phase1-16k` metrics against the installed APIs, after its warm-up requests; output
tokens per second at 1 / 2 / 4 / 8 streams:

| Deployment | 0K | 16K | 32K | Time to first token 8K / 16K / 32K |
|---|---|---|---|---|
| CSF TP2 | 41.2 / 57.4 / 79.1 / 125.3 | 36.3 / 59.1 / 83.6 / 123.2 | 36.9 / 58.9 / 83.9 / 123.2 | 4.00 / 8.12 / 16.12 s |
| Qwen TP2 | 53.0 / 90.4 / 137.8 / 197.7 | 40.3 / 64.2 / 114.4 / 157.6 | 42.9 / 69.4 / 109.9 / 162.9 | 2.35 / 4.58 / 8.98 s |
| DeepSeek-V4.1-Flash TP4, second measured start | 59.7 / 90.4 / 128.2 / 181.7 | 60.6 / 94.2 / 126.4 / 175.2 | 61.2 / 91.6 / 128.6 / 169.5 | 1.87 / 3.82 / 7.73 s |

The CSF TP2 row comes from a run of the runner with commit `55e37428`, which starts every benchmark cell from
an idle server. CSF TP2's 32K, 8-stream cell is the one the outside client changes: 14 repeats of that cell
alone measured 123.2 and 123.3 tok/s in the 2 whose readiness gate opened before an outside request arrived,
and 41.9-86.5 tok/s in the 12 others (gate after 84-96 s, effective concurrency 6.8-7.6), against 115-135
tok/s of generation in the engine's statistics while all 8 streams decoded. Qwen TP2's first table ran beside
CSF TP2's from the same client. DeepSeek's first measured start is in the
[ring-of-four record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md).

## Limits

Each check ran once. An 8-stream cell of an 8-sequence profile is reliable only on an API that no other
client uses, or with a benchmark gate that accepts the cell's own streams running while other requests wait.
