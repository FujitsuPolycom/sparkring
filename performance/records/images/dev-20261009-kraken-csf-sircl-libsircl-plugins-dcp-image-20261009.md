# Image locks of the SIRCL 0.3.1, libsircl and GLM-5.3 plugin image

Status: **image record; the image was built locally and loaded on eight Sparks by its configuration ID; it is
not in a registry and not serving-qualified**. This record holds the image's two v3 installer image locks
(`sparkring-installer-image/v3`, [runtime/common/image_lock.py](../../../runtime/common/image_lock.py)) byte
for byte, so the records and profile guides that ran this image can name the exact lock.

## Image

- **Configuration ID:** `sha256:27e9f75c0d09716764c468ea06fd9cbd2762843d2b0abb344dc6a64de329dffa`; lock name
  `dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp`, line `kraken`. Its image reference is the
  configuration ID, so `sparkring images` does not list it; `--image-lock` selects either lock below on
  Sparks that hold the image under that ID.
- **Sources:** the Kraken CSF sources (vLLM `bc9ea774` and B12X `cc36aa6f`; the same composition digest,
  `10454550…`, as the SIRCL 0.3.0 and libsircl image `816c6d6a7e96`).
- **SIRCL layer:** SIRCL 0.3.1, native ABI 9, wheel `sparkring_sircl-0.3.1-py3-none-any.whl`, prebuilt
  native libraries from C sources `09db871538819191` (ring proxy) and `c8ccfcb93ffa2d6a` (point-to-point
  proxy), pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`. The default SIRCL tuning table's rows
  apply to its sessions, because the table names SIRCL 0.3.1
  ([sircl-tuning-defaults.json](../../../runtime/common/sircl-tuning-defaults.json)).
- **libsircl layer:** libsircl 0.6.0 from vendored snapshot `a3477af2`
  ([spark_transport/libsircl](../../../spark_transport/libsircl/README.md)), with the fail-stop mode and NCCL
  API level 22705.
- **vLLM plugins:** `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.0
  ([derive_glm53_plugins.py](../../../runtime/images/derive_glm53_plugins.py)).
- **Size:** 31,883,175,482 bytes unpacked; 15,308,271,088 bytes to download.

## Locks

| File | SHA-256 | Profiles | Written by |
|---|---|---|---|
| [installer-image-68934e24.json](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009/installer-image-68934e24.json) | `7ff356706dac78efec626de33186e6b737fdfd999b6190ed2880e7c2daad9581` | 12 | the image build at commit `68934e24` |
| [installer-image-ddcd1ae6.json](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009/installer-image-ddcd1ae6.json) | `3e9055ef08f1ad6804ed7e744abbaf3c9dcb9d9fead38df5b915b37aa94da931` | 13 | commit `ddcd1ae6`, which adds `glm53-nvfp4-tp8-dcp1` |

The two locks differ only in `profiles`. Both list `deepseek-v41-flash-tp4`, `deepseek-v41-flash-tp8`,
`glm53-flash-csf-tp8`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `glm53-nvfp4-tp8`,
`mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`,
`qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2` and `swift15-qwen38-flash-next-tp4`; the second
also lists `glm53-nvfp4-tp8-dcp1`. Each validates with `image_lock.validate` for every profile it lists.

Both record `tuning_defaults_sha256` `ce319a5421e67f58…`, the default tuning table of the build commits, which
named SIRCL 0.2.0. The installer does not compare that field with its own table: a deployment applies the
default table of the installing package (its transport section records that table's digest).

## Records that ran this image

- [GLM-5.3 TP8 DCP decode plugin audit and on/off runs](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-decode-ab-20261009.md),
  with `installer-image-68934e24.json`.
- The final-image measurement in the [glm53-nvfp4-tp8 guide](../../../profiles/glm53-nvfp4-tp8/README.md).
