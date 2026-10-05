# Installer image on LIL's Karmic Kraken beta of 2026-10-04

Status: **implemented**. These are the build inputs of installer image
`dev-20261004-kraken-cuda1342-nccl2323-status034`
([release lock](../../../releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json)).
It is built exactly as
[`dev-20261001-kraken-cuda1342-nccl2323-status034`](../external-kraken-20261001/README.md) is, from
the same base, integration assets, add-ons, runtime-status package and
toolchain; only the vLLM and B12X source archives differ.

| Layer | Image ID | Builder | Inputs |
|---|---|---|---|
| Base | `sha256:b094a6744a9b11da9366575745245c9c81d31c565030d1acd226c0018d1a774f` | `eugr/spark-vllm-b12x@sha256:141f46a4a2c3751798f16759cc859648784be430be852a84a21f0c4c427b4052` (nightly-20261001) | As for [`external-kraken-20261001`](../external-kraken-20261001/README.md) |
| Software | `sha256:1ecf0d8c6b846a262daaa2ea57be80243d905731499d0e2723c808366569fd09`, composition `104545500d1a7d0999b393fecdd2af3b281f94d4a883faea4736a132bd3b81f1` | `external-arm64`, [external_context.py](../../external_context.py) | [inputs.json](inputs.json) |
| CUDA 13.4.2 and NCCL 2.32.3 | `sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f` | `cuda134-nccl232-assembly`, [toolchain_assembly.py](../../toolchain_assembly.py) | [`external-kraken-20261001/toolchain-lock.json`](../external-kraken-20261001/toolchain-lock.json) |

## Sources

The software layer replaces the base's vLLM and B12X Python sources with
SparkRing's merges of Local Inference Lab's `integration/karmic-kraken-beta`
branches. Neither merge changes a file under `csrc/`, `cmake/`, `setup.py` or
the package requirements, so the base's compiled vLLM extensions stay.

| Package | Source commit | Branch | Merged upstream commit |
|---|---|---|---|
| vLLM | `d51b4181acc06622357c81dc8e0be46ad96a0779` | [FujitsuPolycom/vllm `sparkring/kraken-beta-20261004`](https://github.com/FujitsuPolycom/vllm/tree/sparkring/kraken-beta-20261004) | `ad3743e35876d895a26a04e88702a260bc79b07b` |
| B12X | `a850d8ebde9b23674e0b76949c2e87161baf3ec4` | [FujitsuPolycom/b12x `sparkring/kraken-beta-20261004`](https://github.com/FujitsuPolycom/b12x/tree/sparkring/kraken-beta-20261004) | `dfe61efc19862d764872bf2d831200e40b3ce561` |

Each branch is the `sparkring/kraken-beta-20261001` source of
`dev-20261001-kraken-cuda1342-nccl2323-status034` merged with the upstream
commit. Where both sides changed the same code, the merge keeps:

- B12X's preparation session reads its selection cache only after the ranks
  reconcile it (SparkRing), with upstream's per-session collective-barrier
  timeout.
- B12X's W4A16 route packing keeps SparkRing's `w4a16_stable_route_pack`
  control (`B12X_W4A16_STABLE_ROUTE_PACK`, off by default) beside upstream's
  A4-prefill and compressed-scale changes. Because the decode tuning query
  carries both sides' controls, its schema versions are 18 (query) and 11
  (configuration), distinct from either side's.
- vLLM's DFlash context projection follows upstream: it shards at TP > 1 when
  the width divides evenly, and pads other widths only with
  `VLLM_DFLASH_SHARD_AUX_PROJECTION`. The MiMo-V2.6-Flash DFlash draft is 4096
  wide, so its TP2 and TP4 sharding is unchanged.

The vLLM branch also accepts compute capability 12.1 (GB10) for GLM-5.3's
independent NVFP4 MTP draft vocabulary head (`VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4`),
which upstream accepts only on 12.0. No installer profile selects that head.

## Rebuild

Follow the [`external-kraken-20261001` rebuild](../external-kraken-20261001/README.md#rebuild) with
two changes: in step 1, export `vllm-kraken-d51b4181.tar` from `d51b4181` and
`b12x-kraken-a850d8eb.tar` from `a850d8eb` (the baseline archives are the same),
and in step 5 use this directory's [make_inputs.py](programs/make_inputs.py),
which names those archives and commits. Its other programs and its
`parent-cache-contract.json` are used unchanged. The toolkit stage is the one
that record describes (`sha256:0dd26ae86cf63c0d2b68a2ff3d59d34a1d7367f9408044e9e9ade7aa5d9a6115`).
