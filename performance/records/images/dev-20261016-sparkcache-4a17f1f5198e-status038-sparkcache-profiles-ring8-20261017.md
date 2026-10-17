# The five SparkCache profiles on release 2026.10.5's image: installation, restart and corruption recovery

Status: **research-only**; single runs. This record gives, for each SparkCache
profile the installer image `dev-20261016-sparkcache-4a17f1f5198e-status038`
(configuration ID `sha256:f625053225d4…`) admits, one installation with the
payload check `fast`, a session saved to the cache, its restore after a
restart of the deployment, and its recovery after one Spark's cache files
were damaged. The release record
([README](../../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md))
cites it.

## Conditions

- A ring of eight DGX Sparks (GB10), ConnectX-7 RoCE, cycle-8 fabric, SIRCL
  ring sessions with NCCL off, GPU clocks not locked. Positions 0-3 and 4-7
  are the two four-Spark groups; 4-5 and 6-7 the two pairs.
- Package built from the release candidate tree at `a1ec4dcc` on every Spark;
  the image selected with `--image-lock` naming this image's lock. Every
  other option the installer's default except where stated:
  `--sparkcache-integrity fast` is the default.
- 2026-10-17, 00:16 to 02:16 CDT, with automatic recovery off during the run.
- Per deployment: `sparkring check`; a four-turn session through the API
  (300 lines of text growing by 120 per turn); an immediate replay of turn 4
  with `prompt_logprobs` set, which bypasses vLLM's own prefix cache so a
  `cached_tokens` credit comes from SparkCache; `sparkring down` and `up`,
  then replays of turns 4 and 3 twice each; with the deployment down, one
  bit flipped in every chunk file of rank 1's cache root; `up`, then turns 4,
  3 and 2 read twice each.

## Measurement

`cached_tokens` of `prompt_tokens` in each replay's usage, the request's
time, the scheduler's `restore hit`, `restore failed` and `restore miss`
lines and rank 1's `invalidated damaged entry` lines in the container logs.

## Result

| Profile | Sparks | Install | Immediate replay | After restart | After rank-1 corruption |
|---|---|---|---|---|---|
| `qwen38-flash-next-csf-sparkcache-tp2` | 4-5 | rc 0, 640 s; check passed | 17,088 of 18,530 tokens in 1.2 s | 17,088 in 1.45 s | first read recomputes (4 damaged entries invalidated and saved again); second read restores |
| `glm53-flash-nvfp4-sparkcache-crc32-tp2` | 6-7 | rc 0, 612 s; check passed | 14,336 of 15,681 in 2.1 s | 14,336 in 3.5 and 1.5 s | as above |
| `glm53-flash-nvfp4-sparkcache-crc32-tp4` (first run on hardware) | 4-7 | rc 0, 599 s; check passed | 15,360 of 15,756 in 1.4 s | 15,360 in 1.8 and 0.9 s | as above |
| `qwen38-flash-next-step5500-sparkcache-tp4` | 4-7 | rc 0, 271 s; check passed | 17,280 of 18,532 in 0.9 s | 17,280 in 1.1 and 0.7 s | as above |
| `deepseek-v41-flash-sparkcache-tp4`, with `--sparkcache-size 48G` (the Install Builder's command) | 0-3 | rc 0, 385 s; check passed; the rank commands carry `spark_cache_max_bytes=51539607552` after the install and after the restart | 14,592 of 14,884 in 0.6 s | 14,592 in 0.7 s | as above |

The two pairs were installed at the same time. The two TP2 profiles were then
installed again on their existing caches, and the first replay restored
17,088 and 14,336 tokens from the previous deployment's cache. Every replay
after a restart and every second read after corruption was a scheduler
`restore hit` of the same token count; every first read after corruption
logged `restore failed … recomputing` on the damaged entries and then
`restore miss rank=0 reason=quorum`. The plain `deepseek-v41-flash-tp4` also
installed on this image (rc 0, 404 s; check passed; a smoke request answered).

## Conclusion

Each of the five SparkCache profiles installs, checks, saves, restores across
a restart and recovers from rank-local corruption on this image with the
default payload check. Restores after a restart took 0.5 to 1.5 s for
12,000 to 17,000 tokens, against 4 to 11 s recomputed.

## Limitations

- Single runs; no soak, no correctness screen, no speed measurement.
- The DeepSeek installation used a non-default cache budget; the default
  budget's installation is the release record's earlier evidence.
- The other fourteen profiles the lock admits were not started on this image.
- The deployments' records are on the ring's Node A, not in this repository.
