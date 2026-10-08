# vLLM and B12X sources that read the GLM-5.3-Flash CSF checkpoint

Status: **implemented** as a build input. These are the source files of the
layer that [derive_kraken_csf_sources.py](../../derive_kraken_csf_sources.py)
(`installer-kraken-csf-sources`) adds to
`dev-20261004-kraken-cuda1342-nccl2323-status034`. The image they enter is
described by its [release recipe](../../../releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md).

The CSF checkpoint of GLM-5.3-Flash,
`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at `dec48abd33ef`,
stores its routed experts' NVFP4 block scales losslessly compressed (CSF).
vLLM reads it with `--quantization nvfp4_csf --load-format nvfp4_csf`, whose
loader and B12X kernels are in the sources below and not in the parent image.

## Sources

| Package | Commit | Branch | Merge of |
|---|---|---|---|
| vLLM | `bc9ea7744bc73d1f45cf3232d5de1b8f823919ae` | FujitsuPolycom/vllm `sparkring/kraken-beta-20261007` | the parent's `d51b4181` (`sparkring/kraken-beta-20261004`) and Local Inference Lab `integration/karmic-kraken-beta` at `89f1ceecd713` |
| B12X | `cc36aa6f1e93c31eb51668f7bfcc59046990548b` | FujitsuPolycom/b12x `sparkring/kraken-beta-20261007` | the parent's `a850d8eb` (`sparkring/kraken-beta-20261004`) and Local Inference Lab `integration/karmic-kraken-beta` at `236ddff04a5f` |

No public remote holds either `sparkring/kraken-beta-20261007` branch. The
files of both merges that the layer needs exist as the payload archive
`csf_payload.tar` (SHA-256 `05ba9d1c6e5ac005c0279901b911afc78b7d74a792c06a359ee2aed854c53c84`)
and as the read-only CSF source overlay that the payload's `build_overlay.sh`
wrote on each Spark of the owner's eight-Spark ring
(`/srv/sparkring/overlays/csf-05ba9d1c6e5a`, tree SHA-256
`d02d7dcb87b0ee8bc0c7341d40836529b3713d816e8af446a439bc5d7e2365b8` over 5,719
files). The layer reads either one; [sources.json](sources.json) pins every
file it takes, so both give the same bytes.

## What changes

[sources.json](sources.json) (`sparkring-source-payload/v1`) lists every
file under `vllm/` and `b12x/` whose content differs between each parent
commit and its merge: 47 changed and 2 added Python files, 3,594,218 bytes.
For each it records the package, the SHA-256 in the parent image (null for
an added file), the merged SHA-256 and the size. No file is deleted or
renamed. Outside the two packages the merges change only tests,
documentation, benchmarks and wheel-bundle CI scripts: no file under
`csrc/` or `cmake/`, no `CMakeLists.txt`, `setup.py`, `pyproject.toml` or
requirements file. The parent's compiled vLLM extensions therefore stay. Each
recorded parent SHA-256 equals the parent image's external-base receipt
entry.

The layer keeps the parent's `vllm/distributed/device_communicators/shm_broadcast.py`,
whose shared-memory readers poll for `SPARKRING_SHM_BUSY_LOOP_S` seconds
after a read, so the image has the `shm_reader_window` capability that
`--save-cpu` needs ([installer-capabilities.json](../../../releases/installer-capabilities.json)).
The six B12X sources that the prepared RoCEnante transport hashes at startup
(`transport_sources`) are byte-identical in both trees and are not written;
the prepared transport, its manifest `fee0680b6dc6` and the receipt's
transport fields stay those of the parent. The Qwen prefill feature hooks hash
`vllm/models/qwen4_exp/nvidia/hyperconnection.py` and
`b12x/sequence/mtp_feedback/_kernels.py`, which the layer does not write.

In the merged vLLM, the pinned files of SIRCL's adapter are those of
`sparkring-kraken-beta-20261007-bc9ea774`
([pins.py](../../../../spark_transport/sircl/sparkring_sircl/vllm/pins.py)),
the build `image_lock.CHECKPOINT_BUILDS` requires for the CSF checkpoint. The
vLLM merge also carries upstream commit `e3e03644` ("pass the index-group
builder to the B12X DSA attention"): its `DeepseekV32B12xAttention` accepts
the argument that `GlmMoeDsaForCausalLM`'s decoder layers pass, which the
parent's does not.

## Regenerating the manifest

The payload holds the merged blob of every path that
`git diff --name-status --no-renames IMAGE_COMMIT MERGE_COMMIT -- PACKAGE/`
lists, read with `git -c core.autocrlf=false cat-file blob`; each parent
SHA-256 is that of the image commit's blob. A merge of other commits needs a
distinct manifest directory and image release.

## Evidence

Conditions: offline replay of `prepare` with the parent's external-base and
toolchain receipts, read from the image layers of
`sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f`
(SHA-256 `1c047d1e8583…` and `2d030d097131…`, the values its lock records),
the parent files from the image commits and the merged files from the merge
commits. No image was built.

Result: every pinned parent SHA-256 matched the receipt and the image commit;
the layer wrote 49 files and produced external-base receipt
`5dca988070da4c25caaecc5b296b7e270c1189d7f3a419567717ed1a0943983f` and
toolchain receipt
`7ab8ee6fac079e37f6558ecd407c749f9175d58901aa74b9dc1d19504b532c55`. Its vLLM
hook files matched only the pinned build
`sparkring-kraken-beta-20261007-bc9ea774`; the parent's matched only
`lil-image-aba309e4610c`.

Conclusion: `prepare` on the build host, with these inputs, prints the same
two receipt digests. Serving correctness of the merged sources is not
established by this replay.
