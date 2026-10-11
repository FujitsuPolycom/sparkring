# GLM-5.3 NVFP4-CSF on eight Sparks

This profile runs GLM-5.3-NVFP4-CSF, created by Local Inference Lab, Inc., a
non-profit organization, available at
https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF.
GLM-5.3-NVFP4-CSF is licensed under the Local Inference Lab License, Version
1.0.

[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF/tree/a4d406932f2db96db8e852867c7939f448e0cd6f)
(revision `a4d406932f2d`) is Local Inference Lab's lossless container of
[GLM-5.3-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/b472e4ee53f6a9862da5486c56c6ca21be3dab70)
at revision `b472e4ee53f6`, the checkpoint of
[glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md), itself an NVFP4 quantization
of the routed experts of Z.ai's [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
a 744B mixture-of-experts model. The weights are the same; only the routed
experts' block scales are stored compressed (CSF). This profile runs it on all
eight Sparks of an eight-Spark ring with every setting of glm53-nvfp4-tp8
except those the format needs. Node A serves the API on port 8015 as
`GLM-5.3-NVFP4-CSF-TP8`, with no API key. Status: Experimental; no
installation of it has run on hardware.

## Install

1. No image lock this package carries lists this profile. Give a lock of
   release 2026.10.2's image that lists it with `--image-lock FILE`.
   The checkpoint needs an image whose vLLM is SIRCL's pinned build
   `sparkring-kraken-beta-20261007-bc9ea774` (the lock's `sircl.vllm_pins`),
   and the profile the three GLM-5.3 plugins of
   [glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md#image); the installer
   refuses any other image.
2. Lock the GPU clocks on every Spark of the ring
   (`sudo nvidia-smi -lgc 2418,2418`); the installer does not.
3. Install:

   ```bash
   sudo sparkring install --profile glm53-nvfp4-csf-tp8 --image-lock FILE
   ```

Each Spark holds the whole checkpoint, 412.5 GiB in 188 files. A blank Spark
needs about 485 GiB free with the serving image and the compile cache
allowance. The plan states that the checkpoint is research-only.

A copy downloaded beforehand with `hf download --local-dir` is used without a
download when it lies at
`/srv/sparkring/downloads/local-inference-lab--GLM-5.3-NVFP4-CSF/a4d406932f2db96db8e852867c7939f448e0cd6f`
or in any folder the installer searches
([where the installer looks](../../docs/operations/install-reference.md#where-the-installer-looks)).
After their SHA-256 is checked, its weight files are hard-linked into
SparkRing's checkpoint directory when they share its filesystem and copied
otherwise; the other Sparks receive the files along the cables. Never upload
the checkpoint anywhere: its license forbids reuploads.

## Serving directory

vLLM reads this checkpoint only from a serving directory: the source
checkpoint's metadata with a `config.json` whose `quantization_config` names
`nvfp4_csf` and the checkpoint's root. The installer builds it on every Spark
as a derived checkpoint without a donor
([derived checkpoints](../../docs/operations/install-reference.md#derived-checkpoints)),
beside the checkpoint's own directory:

- every file of the checkpoint as a hard link, which takes no space; the
  files of its `metadata/` are linked at the root, where vLLM reads them;
- `config.json` with `checkpoint_root` `/models/target`, where the rank
  container mounts the serving directory, written by
  [csf_serving.py](../../runtime/common/csf_serving.py) on Node A in the
  installer image and sent to the other Sparks; and `derivation.json`.

The serving directory's
[manifest](../../profiles/checkpoints/sparkring-derived--GLM-5.3-NVFP4-CSF-Serving/bbd8f71def6bde14a201887796692cb1cc4828a8.json)
pins all 189 files; each is placed only after its SHA-256 equals the
manifest's, and the directory is verified again before the model starts.
`sudo sparkring checkpoints` and `sudo sparkring storage` list it as derived
from the checkpoint; releasing it frees about 12 KB.

## Settings

The settings of [glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md#settings)
([config.json](config.json)) with these differences:

| Setting | This profile | glm53-nvfp4-tp8 |
|---|---|---|
| Checkpoint | GLM-5.3-NVFP4-CSF `a4d406932f2d`, served from its serving directory | GLM-5.3-NVFP4 `b472e4ee53f6` |
| `--quantization` | `nvfp4_csf` | `modelopt_fp4` |
| `--load-format` | `nvfp4_csf`, which the method requires | `b12x` |
| `--hf-overrides` | not set | the checkpoint's NVFP4 configuration with the MTP layer excluded |
| Served model name | `GLM-5.3-NVFP4-CSF-TP8` | `GLM-5.3-NVFP4-TP8` |
| Cache namespace | `glm53-nvfp4-csf` | `glm53-nvfp4` |

The `nvfp4_csf` method takes the NVFP4 recipe only for the modules that store
weight scales, so the BF16 MTP layer stays unquantized without an override;
glm53-nvfp4-tp8's `--hf-overrides` would replace the serving directory's
`nvfp4_csf` configuration. Everything else is unchanged: TP8 with
decode-context parallelism 4, MTP2, the B12X DSA attention backend,
`fp8_ds_mla` KV, the three GLM-5.3 plugins, SIRCL ring sessions, the online
MXFP8 linears and every expert setting.

## Start checks

No run shows that these settings work through the `nvfp4_csf` quantization
and loader. The profile keeps them, and its `start_checks` make the first
start record whether each took effect
([start_checks.py](../../runtime/host/start_checks.py)):

| Check | Setting | Evidence |
|---|---|---|
| `mxfp8-linears` | Online MXFP8 dense linears and the MXFP8 LM head | the status endpoint's `target_head_mxfp8`; weights per rank under 53.07 GiB |
| `draft-quantization` | The MTP draft's load-time MXFP8 and `nvfp4_a16` quantization | weights per rank under 51.9 GiB and accepted draft tokens |
| `column-split-loading` | `glm53full_speedups`' column-split latent projection and `eh_proj` | the plugin's own log lines: `patched for` both flags and rank 0's split `fused_qkv_a_proj` and `eh_proj`, without a refusal or a replicated fallback |
| `csf-w4a16-a4-prefill` | W4A16 decode with A4 prefill on the CSF experts | 75 `NVFP4-CSF layer … W4A16 decode, A4 prefill` log lines on rank 0 |
| `csf-scale-prefetch` | `VLLM_B12X_CSF_SCALE_PREFETCH`, not set, so the image's default | the variable evaluated in the container |
| `w4a16-host-barrier-reset` | `B12X_W4A16_SMALL_M_HOST_BARRIER_RESET=0` | the image's B12X W4A16 code reads the variable |

Each verdict is `took effect`, `did not take effect` or `unknown`. The verdicts
appear in the install result, the smoke action's receipt and
`start-checks.json` in rank 0's installer directory; they never stop the
installation. The weight bounds are estimates from glm53-nvfp4-tp8's
measured 53.07 GiB per rank with MXFP8 linears and 57.34 GiB without; an A/B
start of both profiles on one image settles a value near a bound.

## Evidence and open items

No installation of this profile has run on hardware; it is not
serving-qualified. The checkpoint's pins, the serving directory's manifest,
the rendered rank commands and the installer's operations are checked by CPU
tests only. Expected effects, from the checkpoint's file sizes and the
expert kernels' storage format (research arithmetic, no measurement): about
1.9-2.1 GiB less GPU memory per rank for the experts' scales, which reaches
the KV cache only if `--kv-cache-memory-bytes` is raised, and decode within
about -0.5 % to +3 % of glm53-nvfp4-tp8.
