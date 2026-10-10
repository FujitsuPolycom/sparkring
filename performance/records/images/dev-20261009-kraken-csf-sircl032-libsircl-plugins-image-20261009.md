# Image lock of the SIRCL 0.3.2, libsircl-from-source and GLM-5.3 plugin image

Status: **image record; the image was built from commit `c7c35fe0` and loaded on eight Sparks by its
configuration ID; it is not in a registry and not serving-qualified**. This record holds the image's v3
installer image lock (`sparkring-installer-image/v3`,
[runtime/common/image_lock.py](../../../runtime/common/image_lock.py)) byte for byte, so the installer
scenarios run on this image can name the exact lock.

## Image

- **Configuration ID:** `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`; tag
  `sparkring-dev/kraken:csf-sircl032-libsircl-plugins-20261009`; lock name
  `dev-20261009-kraken-csf-sircl032-libsircl-plugins`, line `kraken`. Its image reference is the
  configuration ID, so `sparkring images` does not list it; `--image-lock` selects the lock below on Sparks
  that hold the image under that ID.
- **Sources:** the Kraken CSF sources (vLLM `bc9ea774` and B12X `cc36aa6f`; composition digest `10454550…`,
  the same as images `816c6d6a7e96` and `27e9f75c0d09`).
- **SIRCL layer:** SIRCL 0.3.2, native ABI 9, wheel `sparkring_sircl-0.3.2-py3-none-any.whl`, prebuilt native
  libraries from C sources `09db871538819191` (ring proxy) and `c8ccfcb93ffa2d6a` (point-to-point proxy),
  pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`. The default SIRCL tuning table, keyed to
  0.3.2 ([sircl-tuning-defaults.json](../../../runtime/common/sircl-tuning-defaults.json)), applies to its
  sessions.
- **libsircl layer:** libsircl 0.6.0 built from the committed source `spark_transport/libsircl` (git tree
  `dbf3607484dd47df4cf6c8238eb5b3272466effb`) with its four kernel packs built from CUDA source, the
  fail-stop mode and NCCL API level 22705.
- **vLLM plugins:** `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1
  ([derive_glm53_plugins.py](../../../runtime/images/derive_glm53_plugins.py)).
- **Runtime status:** 0.3.4. **Size:** 31,883,141,118 bytes unpacked; 15,308,236,724 bytes to download.

## Lock

| File | SHA-256 | Profiles |
|---|---|---|
| [installer-image-c7c35fe0.json](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json) | `3bbcdfe378d7b0ad1577bb1e4e83517a428310a00fe5f388a35b79a4f2c95958` | 13 |

It lists `deepseek-v41-flash-tp4`, `deepseek-v41-flash-tp8`, `glm53-flash-csf-tp8`,
`glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `glm53-nvfp4-tp8`, `glm53-nvfp4-tp8-dcp1`,
`mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`,
`swift15-qwen38-flash-next-tp2` and `swift15-qwen38-flash-next-tp4`, and validates with
`image_lock.validate` for each. It holds no host name, address or local path. Its
`tuning_defaults_sha256` (`c6f82805e102…`) is the default tuning table of the build commit; the installer
applies the default table of the installing package.
