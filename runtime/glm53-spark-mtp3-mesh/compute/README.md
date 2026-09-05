# GLM-5.3 compute source composition

Status: **implemented**. These files reproduce the compute source used by the
GLM-5.3 NVFP4-Spark native-MTP3 mesh image. Hardware qualification belongs to
the profile's validation records, not to this source-preparation tooling.

`source-lock.json` is the authoritative input. It binds:

- the vLLM source revision already present in the parent image;
- a reviewable vLLM patch, a byte-exact 14-file replacement archive, and every
  base and resulting file hash;
- B12X revision `b58f34eaf978277621efced6678e6713fd7122e4`, its Git tree,
  source archive, and all 385 installed package-file hashes;
- seven NVIDIA CUDA 13.3 SBSA redistributable archives; and
- the five environment settings that select metadata reuse, dense-kernel
  policy, and the NVFP4 native-MTP proposal head.

The proposal head receives a separate tensor-parallel copy of the checkpoint's
unquantized `lm_head.weight`, converts that copy to NVFP4 during loading, and
uses BF16 activations. `VLLM_MXFP8_LM_HEAD=0` leaves the target/verifier head
unchanged. Rejection sampling therefore retains the target model's sampling
contract, while proposal-head quantization can change acceptance length.

The image builder calls `prepare_compute_source.prepare(destination, cache)`
while network access is available. The prepared directory contains the pinned
B12X source and CUDA archives. Docker copies that directory into the build and
runs `apply_compute.py` with network access disabled. The installer verifies
the parent hashes before extracting the replacement archive; this preserves the
mixed line endings bound by the source lock without requiring Git in the image.
`verify_compute.py`
requires exact installed hashes and rejects missing or partial source maps.

The B12X source is obtained from
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x) and is
licensed under Apache License 2.0. The downloaded source archive includes its
`LICENSE` file. The vLLM patch derives from
[`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
revision `3512b066e7796128c0c380ccc558182960f2f0ea`, with dense-kernel integration
from revision `a8c796f3af74106b2d8d441e9ec54588936a5388`; vLLM is licensed under
Apache License 2.0.

B12X source archives use LF endings. Source preparation converts Python and C
files to CRLF to reproduce the installed package hashes in `source-lock.json`.
Markdown and compressed profile data retain the archive bytes. This byte-level
contract makes package-content verification independent of checkout settings.
