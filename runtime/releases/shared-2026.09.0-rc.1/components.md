# Shared candidate components and provenance

Status: **implemented integration; publication and profile validation pending**.
This index identifies the source inputs of `shared-2026.09.0-rc.1`, not a
published image digest. Its [release record](README.md) owns availability and
qualification. Components retain their own licenses; the combined image is not
licensed solely under Apache-2.0.

The [frozen source patches and manifest](sources/README.md) reconstruct the
vLLM/B12X inputs. [Qualification inputs](qualification/README.md) pin the bounded
GPU harness and reference fixtures separately from runtime sources.

| Component and role | Source identity | License / attribution |
|---|---|---|
| vLLM inference runtime | [LIL vLLM](https://github.com/local-inference-lab/vllm), `35bab057b1751a6076a457803bcc4b78809689cf`, plus reconciled SparkRing patches | Apache-2.0; upstream contributor notices retained |
| B12X model kernels and prepared execution | [LIL B12X](https://github.com/local-inference-lab/b12x), `a83336581a3a907076e60797df69ab66df5a2ff1`, plus reconciled checkpoint/scoring patches | Apache-2.0; upstream contributor notices retained |
| SparkCache persistent hybrid cache | Base `f220230a5a85b94af8a296187241b6aacc3ed724`, plus the [Qwen/cache64 composition](../../images/compositions/lil-r37-cache64/descriptor.json): integrated source `76598d1d368d7a8fad2ebd03b7a0550e09eb7f99`, tree `ac45a7029c19df8d8752d87627d5c8b8ded34444` | Apache-2.0; [source patch](../../../integrations/sparkcache/qwen38-r37.patch) and native hashes retained |
| SGLang isolated inference runtime | `e087e662ba1ac4ef7747537e2a9141085efd4561`, with [recorded overlays](../../deepseek-v41-sglang/patches/manifest.json) | Apache-2.0; complete source and `LICENSE` retained under `/sgl-workspace/sglang` |
| Mia NVMe Engram adapter and launcher | [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks), `e59e6eb67479aa68f6fa700c600dc90a0729b5ec`; [composition and integration edits](../../deepseek-v41-sglang/combined-image/README.md) | Upstream notice specifies AGPL-3.0-or-later; retains 0xSero's MIT attribution and SGLang's Apache-2.0 wrapper notice |
| NCCL collective communication for vLLM | Installed 2.31.2 (`2.31.2+cuda13.3`); `/opt/local-inference/nccl/lib/libnccl.so.2.31.2`, SHA256 `84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47`. vLLM selects its `libnccl.so.2` link through `VLLM_NCCL_SO_PATH`. | NCCL's complete license/notices remain applicable; [patch attribution](../../../THIRD_PARTY_NOTICES.md#1-nvidia-nccl-portions-included) |
| NCCL for isolated SGLang | Separate retained 2.30.7; `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`, SHA256 `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3`; [combined manifest](../../deepseek-v41-sglang/combined-image/manifest.json) | NCCL license/notices and retained source/build attribution apply independently of the vLLM copy |
| RoCEnante RDMA communication | [Pinned source origins and adaptations](../../../third_party/b12x_roce/provenance.json); selected transport manifests identify installed variants | Apache-2.0; Luke and Local Inference Lab provenance retained |
| SIRCL and SparkRing integration | Exact repository revision and transport receipts pending finalized build | Apache-2.0; SparkRing contributor notices retained |

The Qwen HC/coalescing and paired-scoring adaptations retain
[original-el8/Jason's PR779/386/387 provenance](../../images/compositions/lil-r37-qwen-prefill/provenance.json).
That record describes their source origins, not unchanged application to Kraken
or candidate performance evidence. The finalized build must record each patch's
disposition and installed result hashes.

## License locations

The retained comparison image contains vLLM, B12X and SparkCache license texts
under their `/opt/venv/lib/python3.12/site-packages/*.dist-info/licenses/`
directories. Replacement wheels must preserve those texts. RoCEnante's selected
transport retains its `LICENSE`; NCCL retains package notices and a native build
receipt. `/opt/dsv41` contains the complete Mia adapter source, `LICENSE`,
`NOTICE`, `LICENSE.upstream-MIT` and `LICENSE.sglang`.

CUDA, PyTorch, FlashInfer and other inherited dependencies retain their own
terms and package notices. Model weights are not bundled. See the repository's
[third-party inventory](../../../THIRD_PARTY_NOTICES.md) for detailed origins.

## Publication payload — pending

The release should provide:

- A checksummed source bundle containing the frozen pins, carried/reconciled
  patches, build recipe, license texts and Mia integration changes.
- Build and installed-payload receipts: source/wheel/native hashes, runtime
  inventories and the source-matched SparkCache lease/migration contract.
- Component and profile-scoped test results, with unresolved limitations and
  untested profiles explicit; then the registry manifest digest and pull proof.

The existing receipt structure under `/opt/sparkring/receipts` and isolated
SGLang `{manifest.json,context.json,installed.json}` should remain authoritative,
not be replaced by a second inventory with independent version selections.
A minimal embedded entry point is `/opt/sparkring/licenses/components.md`,
linking those receipts and retained license locations. Embedding and release
attachments are recommendations here, not completed packaging steps.
