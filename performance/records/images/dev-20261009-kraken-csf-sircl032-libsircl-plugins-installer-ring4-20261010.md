# Installer runs on two rings of four with the measured cycle-4 SIRCL row

Status: **research-only record of installer runs; every installation on SIRCL ring sessions passed the
installer's functional and transport checks; the libsircl pair did not start; single-run timing; not
serving-qualified**.

`sparkring setup` re-formed eight DGX Sparks into two separate rings of four, and `sparkring install` then
installed six installer scenarios on them with the image lock of the
[SIRCL 0.3.2, libsircl-from-source and GLM-5.3 plugin image](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009.md).
Three four-Spark deployments were measured in place through their API.

## Conditions

- **Hardware:** eight DGX Sparks (GB10) cabled as two rings of four with ConnectX-7 RoCE. Ring A and ring B
  each have their own Node A. Positions below are ring positions 0-3.
- **Setup:** `sudo sparkring setup --re-form --ssh-user ACCOUNT --no-share-internet --name ring-a` (and
  `ring-b`) with package revision `e2590e79`, a dry run (`--plan`) first. Each Spark's state of the earlier
  ring of eight moved to `/var/lib/sparkring/retired/<UTC time>-ring-a/` (or `-ring-b/`) with a receipt,
  including that fabric record's relay table (12 ingress rules, 12 relay routes and 12 permanent neighbor
  entries per Spark) and its relay markers. Setup then renumbered the fabric, recorded the hairpin
  setting without a driver restart (it was in effect), installed the relay table and verified the fabric:
  "4 cables on 4 Sparks (cycle-4), relays 16 rules and 16 routes", every cable 212.7-213.4 Gb/s in both
  directions. Fabric documents: ring A `sha256:097062777e17…`, ring B `sha256:f1cb76795938…`. On every Spark
  the live ingress rules, relay routes, permanent neighbor entries and marker processes equal the new table
  (4 of each) and no object of the earlier table remains.
- **Installer:** package revision `6555b89b` on every Spark, run on each ring's Node A. Its default SIRCL
  tuning table (`runtime/common/sircl-tuning-defaults.json`, digest `92bd95e6…`) has the measured cycle-4
  row, which names the SIRCL table `runtime/common/sircl-tuning/cycle-4.json` (hash `df333d97acb1a4b2`)
  ([evidence](../transport/sircl-cycle4-tune-two-rings-20261009.md)).
- **Image:** configuration ID `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  selected with `--image-lock` and the recorded lock
  [installer-image-c7c35fe0.json](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json)
  (SHA-256 `3bbcdfe378d7b0ad1577bb1e4e83517a428310a00fe5f388a35b79a4f2c95958`). No image was pulled.
- **Checkpoints:** every plan listed no file to download (`hub_files` empty, `hub_bytes` 0); the installer
  hard-linked the weight files from copies on each Spark.
- **Installs:** `--plan`, then `--yes`. Each scenario's first start on a ring had empty compile and tuning
  caches. GPU clocks were not locked.
- **Checks:** after each installation, `sparkring check` (counting, arithmetic, code, a tool call, a forced
  tool call, an image and thinking, then the transport check and the GPU clock reading) and three
  temperature-0 questions with one exact answer each.
- **Measurement:** the [serving A/B runner](../../harnesses/serving_ab/README.md)'s `phase1-16k` metrics
  (`measure.measure`), run from the operator's machine against the installed deployment's API, after the
  runner's warm-up requests: output tokens per second and accepted tokens per step at 0, 16K and 32K tokens of
  context and 1, 2, 4 and 8 streams (temperature 0, at most 1,024 output tokens, 30 s cells), and the time to
  first token of a cold 8K, 16K and 32K prompt (median of three). The runner started nothing.

## Result

| Scenario | Ring and positions | Profile and checkpoint | Install time | API readiness | Checks | Transport |
|---|---|---|---|---|---|---|
| A1 | A, 0-3 | `glm53-flash-nvfp4-spark-tp4`, CSF (`dec48abd`) | 920 s | 791.7 s | 7 passed; 3 of 3 answers | SIRCL, NCCL absent, as expected |
| A2, switch | A, 0-3 | `qwen38-flash-next-qad-tp4` (`60215d26`), replacing A1 | 690 s | 533.4 s | 7 passed; 3 of 3 | SIRCL, NCCL absent, as expected |
| A2, switch back | A, 0-3 | `glm53-flash-nvfp4-spark-tp4`, CSF, A1's deployment | 433 s | 351.4 s | 7 passed; 3 of 3 | SIRCL, NCCL absent, as expected |
| A3 | A, 0-1 and 2-3 | `glm53-flash-nvfp4-spark-tp2`, CSF; `qwen38-flash-next-tp2` (`60215d26`) | 917 s; 629 s | 835.6 s; 544.5 s | 7 and 7 passed; 3 of 3 each | SIRCL pair row, NCCL absent, as expected |
| B1 | B, 0-3 | `deepseek-v41-flash-tp4` (`dba1be0a`) | 1,335 s | 1,034.4 s | 7 passed; 3 of 3 | SIRCL, NCCL absent, as expected |
| B2 | B, 0-3 | `qwen38-flash-next-qad-tp4` (`60215d26`) | 698 s | 535.9 s | 7 passed; 3 of 3 | SIRCL, NCCL absent, as expected |
| B3 | B, 0-1 | `qwen38-flash-next-tp2 --transport libsircl` | 160 s, failed | stopped at 71 s | not reached | libsircl (research-only) |

Every four-Spark plan and receipt named the measured cycle-4 row: each tensor-parallel and expert-parallel
session decided from table `df333d97acb1a4b2` and reported its 8 link slots of 1,048,576 bytes. The A3 pairs
took the measured pair row. Each A2 plan named the deployment it replaced among the deployments it stops;
the switch back reused A1's deployment and its compile caches, and its checkpoint verification took 0.5 s.
The two A3 pairs served at the same time. Every `sparkring check` read GPU clocks of 2,392-2,411 MHz with no
clock event reason other than idle.

B3's containers exited during vLLM's tensor-parallel setup: on both ranks `ncclCommInitRank` of vLLM's PyNccl
on libsircl 0.6.0 returned `invalid usage` (`RuntimeError: NCCL error: invalid usage` in
`initialize_model_parallel`), after libsircl printed its identification line; no libsircl receipt was
written. The routing settings the installer passed (`LIBSIRCL_POSITION`, `SIRCL_PEER_ROUTES`,
`LIBSIRCL_CHAIN_ORDER`, `LIBSIRCL_RING_WINDOW`) equal the ones `spark_transport/libsircl/tools/site_routes.py
--layout path:0-1 --lanes 2` gives. In the same image on the same pair, with the deployment's LIBSIRCL_*,
SIRCL_* and NCCL_* environment, one process per Spark that set CUDA device 0, created a unique ID on rank 0
and called `ncclCommInitRank` inside `torch.cuda.device(0)` created the communicator on both ranks. The cause
is therefore in vLLM's process setup, not in the routing settings; vLLM reports only the error's name, and
libsircl keeps its message for `ncclGetLastError`, which vLLM does not call.

### Measurements

Output tokens per second, one measured series per deployment:

| Deployment | Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|---|
| A1, CSF TP4 | 0K | 67.5 | 104.3 | 152.4 | 225.9 |
| | 16K | 65.6 | 93.1 | 128.7 | 214.5 |
| | 32K | 64.2 | 94.4 | 126.6 | 218.7 |
| B1, DeepSeek-V4.1-Flash TP4 | 0K | 63.6 | 95.6 | 132.0 | 190.6 |
| | 16K | 61.9 | 92.8 | 130.4 | 181.9 |
| | 32K | 63.8 | 93.6 | 127.5 | 182.8 |
| B2, Qwen3.8-Flash-Next QAD TP4 | 0K | 81.3 | 120.5 | 201.8 | 292.7 |
| | 16K | 61.1 | 105.7 | 155.3 | 238.6 |
| | 32K | 62.1 | 97.1 | 149.4 | 234.0 |

Accepted tokens per step: A1 2.61-2.75; B1 2.39-2.64; B2 2.40-2.58 at 0K and 1.95-2.08 at 16K and 32K. Engine
steps per second at one stream, 0K / 16K / 32K: A1 24.6 / 24.1 / 23.5, B1 25.4 / 24.4 / 24.2, B2 32.5 /
31.3 / 30.7. B2's lower output rate at 16K and 32K comes from its acceptance; its engine step rate changes
by at most 6 %. Time to first token, 8K / 16K / 32K: A1 2.60 / 5.10 / 10.48 s, B1 1.85 / 3.76 / 7.57 s, B2
1.75 / 3.41 / 7.02 s. Every series' four output fingerprints were identical.

## Plan-only checks

Each ran `sparkring install ... --plan --json`; each refusal changed nothing.

| Check | Result |
|---|---|
| `--checkpoint csf` on a pair of ring A with the default image lock | refused (exit 3): the CSF checkpoint needs SIRCL's pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`; image `dev-20261004-kraken-cuda1342-nccl2323-status034` records none |
| `glm53-flash-nvfp4-spark-tp2` on that pair with the default image lock | planned: NVFP4-Spark (`a6082410`) on image `aba309e4610c`, prepared transport, nothing to download |
| `qwen38-flash-next-qad-tp8` with the SIRCL 0.3.2 lock (ring of eight) | refused (exit 3): the profile is not admitted on that lock |
| `qwen38-flash-next-tp2` on four Sparks (ring of eight) | refused (exit 3): the profile serves two Sparks |
| `qwen38-flash-next-tp2 --checkpoint no-such-checkpoint` (ring of eight) | refused (exit 3) with the profile's checkpoint names |
| `--image <stock vLLM image> --transport libsircl` on a pair (ring of eight) | refused (exit 3) in order: without `--libsircl-library` (the host build every Spark holds), without `--model-path`, then because the image is absent on one Spark; no pull |

## Limits

Each scenario ran once, on one image, with one measured series per measured deployment. The runs include no
correctness screen, full-context request, restart cycle or soak, and the pairs were not measured. B3's
cause is not identified. These results do not establish serving qualification.
