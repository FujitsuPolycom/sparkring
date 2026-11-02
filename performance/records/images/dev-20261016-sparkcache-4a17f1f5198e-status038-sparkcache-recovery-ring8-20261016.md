# SparkCache installer profiles on release 2026.10.5's image: installation and recovery on a ring of eight

Status: **research-only**; single runs. This record gives the hardware
evidence behind release 2026.10.5's image record
([release record](../../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)):
the SparkCache installer profiles install, serve, survive rank-local cache
corruption and keep the installer's cluster name. No performance was
measured.

## Conditions

- A ring of eight DGX Sparks (GB10), ConnectX-7 RoCE, SIRCL ring sessions
  with NCCL off, GPU clocks not locked; date 2026-10-16.
- Package built from the release 2026.10.5 tree
  (`sparkring 0.1.0~dev.1792185588+gita1ec4dccd169`) installed on all eight
  Sparks with `sparkring setup`.
- Two images of the same composition,
  [installer-sparkcache-4a17f1f5198e](../../../runtime/images/compositions/installer-sparkcache-4a17f1f5198e/README.md):
  - the **recheck build**, image
    `sha256:c6fb5c0f4abb25835cdd93f87fa38864c34451b13015eada41939c6f97b6c8da`,
    built with SparkCache `6ef875ecd94b` and installed with `--image-lock`;
  - the **release image**, `sha256:f625053225d4b57bfd7d0bb2e8348fea38e69472b10cef87c4701928abc7c4a8`
    (`dev-20261016-sparkcache-4a17f1f5198e-status038`), built with SparkCache
    `4a17f1f5198e`, whose only change over `6ef875ecd94b` is the operator log
    lines and counter names, installed with the installer's defaults.
- Every `sparkring install` ran with `--yes` and without research options;
  the Qwen sha256 run added `--sparkcache-integrity sha256`.
- A session of four turns (300 lines, growing by 120 per turn) was saved
  through the API of each deployment; a saved turn was replayed with the
  same prefix to exercise restore.
- Corruption: with the deployment down, every chunk file of rank 1's cache
  root (position 1 of the group) was overwritten in place, then the
  deployment was started again with `sparkring up`.

## Measurement

Installer return codes and elapsed times from the install logs;
`sparkring check` output; the scheduler and per-rank SparkCache log lines of
the serving containers; the cluster name in `cluster.json` after each setup
run; the API's time to answer the replayed turn.

## Result

| Check | Image | Result |
|---|---|---|
| Setup without `--name` keeps the cluster name | recheck build | two `sparkring setup` runs and a plan: the name stays |
| Three disjoint installations at once: `deepseek-v41-flash-sparkcache-tp4` on 0-3, `qwen38-flash-next-csf-sparkcache-tp2` on 4-5, `glm53-flash-nvfp4-sparkcache-crc32-tp2` on 6-7 | recheck build | all rc=0 (405, 531 and 512 s); no refusal for Sparks held by another installation |
| DeepSeek TP4, integrity `fast`: rank 1 corrupted (356 chunks), restart, same prompt twice | recheck build | first read recomputes (nothing restored, 7.6 s); rank 1 invalidates 4 damaged entries and re-saves; second read restores 14,592 of 14,875 tokens from SparkCache on all four ranks (0.54 s) |
| Qwen step 5500 TP4, integrity `sha256`: rank 1 corrupted (186 chunks), same sequence | recheck build | first read recomputes (6.26 s); second read restores 17,280 of 18,533 tokens (0.79 s) |
| Integrity schema separation: a `sha256` installation replays a session saved in `fast` mode | recheck build | first lookup `restore miss reason=integrity_schema_mismatch`; then it saves and restores in the `sha256` schema |
| `deepseek-v41-flash-sparkcache-tp4` on 0-3 and `qwen38-flash-next-step5500-sparkcache-tp4` on 4-7, installed at once with default settings | release image | rc=0 (522 s and 487 s); `sparkring check` passed on both groups; both served requests |
| Request-level log line after a restart of the release image | release image | one scheduler line per request (`restore hit tokens=14592 ms=100.3`), no per-rank restore line at INFO |

## Limitations

- Single runs; the restores are attributed to SparkCache by the scheduler
  and per-rank log lines, not by timing alone.
- `glm53-flash-nvfp4-sparkcache-crc32-tp4` was not started on either image;
  the other 15 profiles of the lock were not started on the release image.
- The dashboard's per-request output rate was seen by the operator during
  live use; the harness polls never caught a request mid-decode.
