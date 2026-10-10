# Image d52737a109e0: SIRCL 0.3.2, libsircl with the current-device fix, the GLM-5.3 plugins and runtime-status 0.3.6

Status: **implemented**. Evidence scope: **the image build, its layer checks and its v3 lock; no serving
measurement**. This is the installer image of release 2026.10.2
([release record](../../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)).

## Conditions

- **Source:** commit `d3d3315863a63e3e5848acbc47d788251a32278d`: the release branch with libsircl's
  current-device fix (`c4f3a1af`), whose `spark_transport/libsircl` is tree
  `030419b8a2b61acc1a74010308a1c3e758863e2f`. Its `spark_transport/sircl` (tree `8a54c1ab`) and
  `runtime/images/sircl_layer.py` are the same Git objects as at `c7c35fe0`, the source of image
  `1a8c10354eb0` ([image record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009.md)).
- **Build:** on one Spark, 2026-10-10 from 09:51 UTC, from a `git archive` of that commit committed
  into a scratch repository, which gave the same three Git object IDs. The layers below were built in
  order, each recording a v3 lock that lists the 13 profiles of image `1a8c10354eb0`'s lock.

## Layers

| Step | Builder | Result |
|---|---|---|
| Parent | SIRCL layer of image `1a8c10354eb0`, reused | image `sha256:eb1d8935…`, the CSF sources of 2026.10.1 with SIRCL 0.3.2 (native ABI 9) |
| libsircl | [libsircl_layer.py](../../../runtime/images/libsircl_layer.py) | libsircl 0.6.0 built from tree `030419b8` with its kernel packs and fail-stop mode, library SHA-256 `ffdba5b2…`; image `sha256:941b8d2b…`, lock [libsircl-lock-d3d33158.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010/libsircl-lock-d3d33158.json) (`7d473e6e…`) |
| GLM-5.3 plugins | [derive_glm53_plugins.py](../../../runtime/images/derive_glm53_plugins.py) | `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1; image `sha256:467bf63f…`, lock [plugins-lock-d3d33158.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010/plugins-lock-d3d33158.json) (`c22c941e…`) |
| runtime-status 0.3.6 | [derived_layer.py](../../../runtime/images/derived_layer.py) with [the descriptor](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010/status036-descriptor-d3d33158.json) | runtime-status 0.3.6 replaces 0.3.4: the wheel and source archive built from `integrations/vllm/runtime_status` tree `de6358a8` (commit `0520de9f`); the dashboard names the collective transport and the SIRCL version of the tensor-parallel group |

## Result

- **Image:** configuration ID `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`,
  tag `sparkring-dev/kraken:csf-sircl032-libsircl060cd-plugins-status036-20261010`, release name
  `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`; 31,884,969,464 bytes unpacked.
- **Lock:** [installer-image-d3d33158.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010/installer-image-d3d33158.json),
  SHA-256 `ed4332d92f03086f60679f7e4e78e053a5c08c899ecd56a4c809afccbee8598d`: transports `libsircl`,
  `prepared` and `sircl`; `status_version` 0.3.6; `tuning_defaults_sha256` `c6f82805…`, the default
  tuning table of the build, which has no `cycle-4` row (installations apply the installing package's
  table). Its `image_reference` is the configuration ID until publication records the registry digest.
- **Delta:** against 2026.10.1's image `aba309e4610c`, 6 added layers of 38,721,536 bytes; the delta
  archive is 38,871,040 bytes.

## Limitations

- The build ran once, on one Spark.
- No serving measurement in this record. Image `1a8c10354eb0` has the same SIRCL layer and was built
  from the same CSF and plugin sources; this image's libsircl (tree `030419b8` against `dbf36074`) and
  dashboard (0.3.6 against 0.3.4) differ.
