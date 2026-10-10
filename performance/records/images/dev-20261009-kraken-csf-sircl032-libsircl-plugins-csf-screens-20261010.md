# GLM-5.3-Flash CSF checkpoint: correctness screens on four Sparks and on a pair

Status: **implemented; all 7 functional checks passed and a 256-request correctness screen returned no
degenerate, wrong or failed response on each of two installations; single runs; not serving-qualified**.

`sparkring install` installed the GLM-5.3-Flash installer profiles of four and of two Sparks with their
default checkpoint on this image, the CSF checkpoint (`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`
at `dec48abd33efa73c3bb7c95b74eee10cad34f9be`), and the acceptance harness checked both while they served at
the same time. The screen is the one the [NVFP4-QAD pair record](dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001.md)
used, with the same requests and pass rule.

## Conditions

- **Hardware:** eight DGX Sparks (GB10) cabled as two rings of four with ConnectX-7 RoCE, set up as in the
  [ring-of-four installer record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md).
- **Installations:** package revision `6555b89b` on every Spark, image
  `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952` selected with `--image-lock` and
  the recorded lock
  [installer-image-c7c35fe0.json](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json).
  `--plan`, then `--yes`, with no other option; each plan named the CSF checkpoint and listed no file to
  download.
  - `glm53-flash-nvfp4-spark-tp4` on one whole ring of four (SIRCL ring sessions on the measured cycle-4
    row): 424 s, of which API readiness 357.7 s, with compile caches from an earlier start of the same
    deployment on that ring.
  - `glm53-flash-nvfp4-spark-tp2` on two cabled Sparks of the other ring (SIRCL ring sessions on the measured
    pair row): 1,009 s, of which API readiness 909.5 s.
  - Both installation results named the CSF checkpoint; the transport check of each was "as expected" with
    NCCL absent.
- **Harness:** [`accept_profile.py`](../../harnesses/acceptance/accept_profile.py) at commit `760f3cdb`,
  `--skip-install --steps readiness,functional,stress --checkpoint csf`, from a separate machine on Node A's
  network, with the profile's thinking-off request settings
  (`{"chat_template_kwargs": {"reasoning_effort": "low"}}`).

## Measurement

- **Functional checks:** counting, arithmetic and code with the thinking-off settings, an automatic and a
  forced tool call, a description of a generated two-color image, then the arithmetic question with the chat
  template's default thinking, which must return reasoning text.
- **Correctness screen:** 8 rounds of 32 requests (24 short questions with known answers and 8 questions about
  a code hidden in about 6K tokens) through 16 threads at temperature 0 with the thinking-off settings. A
  failed request is an error; a response in which one word repeats 8 or more times in a row is degenerate; any
  other response that misses the expected answer is wrong.

## Result

| Installation | Functional checks | Screen: responses / degenerate / wrong / errors | Screen time |
|---|---|---|---|
| `glm53-flash-nvfp4-spark-tp4`, four Sparks | 7 of 7 ([output](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010/functional-tp4.txt)) | 256 / 0 / 0 / 0 ([summary](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010/stress-tp4.json)) | 126.3 s |
| `glm53-flash-nvfp4-spark-tp2`, two Sparks | 7 of 7 ([output](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010/functional-tp2.txt)) | 256 / 0 / 0 / 0 ([summary](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010/stress-tp2.json)) | 203.5 s |

## Limits

Each installation was screened once. The functional checks and the screen test correctness, not output
quality against a reference checkpoint. No soak, restart cycle or full-context request ran. The four-Spark
deployment's decode was measured in the
[ring-of-four installer record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md);
the pair's was not.
