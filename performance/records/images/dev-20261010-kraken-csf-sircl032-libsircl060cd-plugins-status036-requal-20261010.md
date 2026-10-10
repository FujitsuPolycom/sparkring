# Requalification of installer image `d52737a109e0` on two rings of four Sparks

Lane: **public-functional**. Status: **live-validated** (not accepted).
Evidence scope: **installer image
`sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`
on eight NVIDIA DGX Sparks (GB10) cabled as two independent rings of four,
2026-10-10 09:50–11:57 UTC: libsircl's image gate on each ring, installer
smoke tests of a SIRCL and a libsircl deployment with known-answer requests and
`sparkring check`, and two measured starts of the CSF checkpoint at
tensor-parallel size 4 against two starts of image `1a8c10354eb0`.** The
image is built from commit `d3d33158` of branch
`claude/release-2026.10.2-rc2`, release 2026.10.2 with libsircl's
current-device fix; image `1a8c10354eb0` is the image that
[release 2026.10.2's record](../../../runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)
names.

## Image

| Field | Value |
|---|---|
| Name | `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036` |
| Image ID | `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b` |
| v3 lock | [installer-image.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010/installer-image.json), SHA-256 `ed4332d92f03086f60679f7e4e78e053a5c08c899ecd56a4c809afccbee8598d`, 13 profiles |
| Source | commit `d3d3315863a63e3e5848acbc47d788251a32278d` |
| Base | `sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f` (release 2026.10.1's image); delta archive 38.9 MB, SHA-256 `63a3d98e…` |

The layers above the base, against image `1a8c10354eb0`:

| Layer | Against `1a8c10354eb0` |
|---|---|
| SIRCL 0.3.2 (ABI 9), image `eb1d893535ee` | the same image: its inputs, `spark_transport/sircl` tree `8a54c1ab` and `runtime/images/sircl_layer.py` blob `4a084d2f`, are the Git objects it was built from |
| libsircl 0.6.0 | rebuilt from source tree `030419b8a2b61acc1a74010308a1c3e758863e2f`: a communicator created on a thread without a CUDA context uses the CUDA runtime's current device (or the only visible device) through that device's primary context. Library SHA-256 `ffdba5b24a2b22680465a2f130473bc6197cb3cacc2c0590b0d6fe884cc170e9` (was `8b180879…`) |
| GLM-5.3 plugins | the same files (`derive_glm53_plugins.py` differs in its docstring only); the image of this layer is `467bf63f36ad` |
| runtime-status | 0.3.6 replaces 0.3.4 through a `derived_layer.py` descriptor: wheel SHA-256 `d3142703…`, source archive `60190947…`, both built reproducibly from component tree `de6358a8` of `integrations/vllm/runtime_status` |

## Conditions

- **Rings:** ring A (fabric `097062777e17`) and ring B (fabric
  `f1cb76795938`), each four Sparks at positions 0 to 3. Ring B's positions 0
  and 1 run GPU driver 580.178.04 with kernel 7.0.0-1019, its positions 2 and
  3 and all of ring A driver 580.173.02 with kernel 6.17.0-1029. One rank per
  Spark.
- **Installer on each ring's Node A:** ring A's runs sparkring
  `0.1.0~dev.1791594745+git6555b89b5042`, ring B's
  `0.1.0~dev.1791614917+git7013807b3036`. Neither is the installer of commit
  `d3d33158`; both validate a v3 lock and the libsircl transport with the same
  code (`runtime/common/image_lock.py`, `runtime/common/libsircl.py`).
- **Every installation:** `sparkring install --plan --json` first, refused
  unless it downloads nothing and stops nothing; then `--yes`, three
  known-answer requests (`17 × 23`, the capital of France, `ring` reversed),
  `sparkring check --json` and `sparkring down --execute`.
- **CSF starts:** each an install-measure-remove cycle of
  `glm53-flash-nvfp4-spark-tp4` served as the CSF checkpoint
  (`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at `dec48abd`) on
  ring A, measured through the API by the serving A/B runner's
  `phase1-16k` metrics after its warm-up requests: decode at contexts 0, 16k
  and 32k with 1, 2, 4 and 8 streams (30 s cells, temperature 0, at most 1,024
  tokens) and time to first token at 8k, 16k and 32k (median of 3). The
  starts of image `1a8c10354eb0` ran at 01:31 and 06:26 UTC, those of
  `d52737a109e0` at 11:10 and 11:45 UTC, with the same installer and the same
  procedure.

## Results

### libsircl image gate

| Run | Ring | Verdict | nccl-tests `off-coll` at 256 MiB, bus GB/s: all-reduce, all-gather, reduce-scatter, broadcast |
|---|---|---|---|
| `gate-ring4-image-d52737a109e0-20261010T100127Z` | B alone | PASS | 24.30, 24.16, 23.26, 5.41 |
| `gate-ring4-image-d52737a109e0-20261010T105303Z` | A, both rings at once | PASS | 24.33, 24.16, 23.21, 5.26 |
| `gate-ring4-image-d52737a109e0-20261010T105303Z` | B, both rings at once | PASS | 24.30, 24.12, 23.18, 5.39 |
| [`gate-ring4-image-1a8c10354eb0-20261010T005453Z`](../transport/libsircl-ring4-image-1a8c10354eb0-20261010.md) (image `1a8c10354eb0`) | A and B at once | PASS | A 24.31, 24.20, 23.29, 5.28; B 24.25, 24.12, 23.21, 5.36 |

Each pass: every rank's library and kernel packs match the lock and the layer
receipt (4 of 4 ranks); every communicator receipt names the packs (136); the
bit-exact check passes with point-to-point channels off and on; nccl-tests
`off-coll`, `on-coll`, `ring-coll` and `on-p2p` report no wrong value and no
refusal; every four-rank default communicator takes the cycle plan (48 of 48)
and runs within 1.15 times the ring schedules from 8 MiB (24 of 24 sizes).

### Installer smoke tests

| Ring | Deployment | Install | Known answers | `sparkring check` |
|---|---|---|---|---|
| B | `qwen38-flash-next-tp2 --transport libsircl` on positions 0,1 | complete, 395 s | 3 of 3 | functional 7 of 7; transport verdict `unknown` (this installer does not judge libsircl's receipts), so `ok` false |
| B | `glm53-flash-nvfp4-spark-tp4` (CSF) | complete, 874 s | 3 of 3 | `ok` true: functional 7 of 7, SIRCL carries every collective, NCCL absent |
| A | `glm53-flash-nvfp4-spark-tp4` (CSF), two starts | complete, 583 s and 434 s | 3 of 3 each | `ok` true each |
| A | `qwen38-flash-next-tp2 --transport libsircl` on positions 0,1 | complete, 747 s | 3 of 3 | functional 7 of 7; transport verdict `unknown`, `ok` false |

libsircl's receipts of the two libsircl deployments, per rank: the
tensor-parallel communicator carried 7,808 all-reduces (3,420 in CUDA graphs)
and 161 all-gathers on ring B, 7,776 (3,420) and 124 on ring A; a second
communicator carried no call. Every receipt is healthy, records no error, no
refused and no forwarded call, names kernel pack `16033e7d…` and the startup
flag-wait regime. With the CSF checkpoint and thinking off, the known-answer
replies hold the model's reasoning before the answer in `content`; each
contains the expected answer.

### CSF TP4 starts

Decode engine steps/s
([starts](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010/),
`compare_starts.py`):

| Context | Streams | `1a8c10354eb0` start 1 | `1a8c10354eb0` start 2 | `d52737a109e0` start 1 | `d52737a109e0` start 2 | Mean against mean |
|---|---|---|---|---|---|---|
| 0k | 1 | 24.57 | 24.68 | 24.19 | 24.25 | −1.6% |
| 0k | 2 | 38.03 | 36.99 | 36.40 | 35.84 | −3.7% |
| 0k | 4 | 56.25 | 52.82 | 52.65 | 54.63 | −1.6% |
| 0k | 8 | 82.12 | 80.26 | 81.06 | 80.23 | −0.7% |
| 16k | 1 | 24.07 | 23.28 | 23.68 | 22.87 | −1.7% |
| 16k | 2 | 35.34 | 34.67 | 33.63 | 34.24 | −3.1% |
| 16k | 4 | 49.30 | 48.35 | 46.18 | 46.83 | −4.8% |
| 16k | 8 | 78.17 | 79.82 | 75.72 | 80.03 | −1.4% |
| 32k | 1 | 23.46 | 23.09 | 23.67 | 22.89 | +0.0% |
| 32k | 2 | 35.53 | 34.88 | 34.01 | 34.41 | −2.8% |
| 32k | 4 | 47.07 | 48.63 | 46.26 | 46.76 | −2.8% |
| 32k | 8 | 80.22 | 78.59 | 76.53 | 76.50 | −3.6% |

Time to first token, s: 8k 2.60 and 2.61 against 2.63 and 2.64; 16k 5.10 and
5.29 against 5.25 and 5.44; 32k 10.48 and 10.66 against 10.68 and 10.77. GPU
clocks after each measurement were 2405–2411 MHz with no throttle reason.

## Conclusion

- The image passes libsircl's gate on both rings, with the bandwidth of
  image `1a8c10354eb0`'s gate within 0.1 GB/s at 256 MiB in every collective,
  and every installation on it completes, downloads nothing, answers the
  known-answer requests and passes `sparkring check`'s functional checks; the
  SIRCL deployments' transport verdicts are as expected.
- Its CSF TP4 decode is 2.2% below image `1a8c10354eb0`'s in the median cell
  (mean 2.3%; 9 of 12 cells below both of its starts), where those two starts
  differ from each other by a median of 2.1% (at most 6.5%). The SIRCL layer
  is the same image, the libsircl layer is not loaded by a SIRCL deployment,
  and runtime-status works only when its status endpoint is requested (a
  worker RPC at most every 5 s, reading the SIRCL receipt as one bounded file).
  Two starts per image do not separate this difference from start-to-start
  variation; no start of image `467bf63f36ad` (runtime-status 0.3.4) ran.
- The libsircl deployments serve with their communicators in the startup
  flag-wait regime (600 s); this image's libsircl plugin does not select the
  serving regime, and its installer does not judge libsircl's receipts.
