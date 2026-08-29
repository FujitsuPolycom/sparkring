# GLM-5.3 Flash with BF16 DFlash2 on four DGX Spark systems

Status: **qualified** for persistent SparkCache store and restore under the
exact artifacts and settings in
[`recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json`](../../recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json).
The otherwise identical cache-disabled profile is **implemented** and has no
standalone live qualification receipt.

The deployment serves the ModelOpt mixed-precision GLM-5.3 Flash target with
the public BF16 DFlash2 drafter over a four-rank direct-cable cycle. Tensor
parallelism is four, decode-context parallelism is one, and pipeline
parallelism is one. The runtime keeps asynchronous scheduling, native vLLM
prefix caching, and chunked prefill enabled in both cache modes.

## Serving contract

| Setting | Value |
|---|---|
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Parallelism | TP4 / DCP1 / PP1, four nodes |
| Model limit | 524,288 tokens |
| Scheduler | 8,192 batched tokens, 32 sequences, asynchronous |
| Target cache | 12 GiB FP8 per rank, block size 256, aligned recurrent state |
| Prefill and reuse | FlashKDA prefill, chunked prefill, native prefix caching |
| Speculation | DFlash2, seven draft tokens, BF16 draft weights, draft TP4 |
| Graph execution | `FULL_AND_PIECEWISE` target; capture rows `8,16,32,64,128,256`; vLLM-selected DFlash FULL graphs |
| Transport | SparkRing-patched NCCL 2.30.7 over two direct RoCE links per rank |
| External cache | `SparkContextCacheConnector`, `kv_both`, failure policy `recompute` |

The cache identity binds the target revision-derived digest
`a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9`
and the DFlash weight digest
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.
SparkCache persists target sparse-MLA/C4 pages and KDA/GDN recurrent
checkpoints. It recomputes external DFlash state after restore.

## Runtime validation

Both runtime templates require one exact Docker image ID on every rank. The
generic launcher verifies that image ID and these inherited or derived image
labels before it runs any container:

- vLLM commit `da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- B12X commit `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`;
- SparkRing NCCL 2.30.7 transport label;
- SparkCache profile `glm53-flash-hybrid`; and
- SparkCache source-tree SHA-256
  `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.

An in-container attestation then verifies the target configuration and weight
index, DFlash configuration and weights, patched NCCL binary, normalized
SparkCache source tree, vLLM configuration postimage, and complete seven-file
vLLM key-value block lease contract before serving starts. Any mismatch stops
the launch. The immutable Hugging Face revision and the verified 59-file
inventory remain the complete target artifact boundary.

## Qualification evidence

Conditions: four directly cabled NVIDIA DGX Spark systems used the exact
artifacts and serving contract above, dedicated rank-local cache roots, and
the four parent/derived image pairs recorded in the composition recipe.

Measurement: all ranks stored one 8,192-token reusable span, all serving
containers were replaced without removing the cache roots, every rank
discovered three manifests with zero rejected, and a repeated request restored
after the scheduler received an all-rank inventory checkpoint. The validation also
sent an uncached semantic canary, verified the 59 expected target repository
files, and checked process, log, speculation, and RDMA state.

Result: all ranks committed the same cache digest. The restored request
reported 8,192 external-prefix hits and completed in 1.509 seconds. Per-rank
restore times were 155.6, 147.2, 194.0, and 151.8 milliseconds. DFlash
produced 301 draft tokens from 43 drafts and accepted 112 tokens. The canary
completed in 1.176 seconds with `stop` and suffix `SPARKCACHE_GLM53_OK`.
Ranks 0, 2, and 3 passed strict 59-file target verification; rank 1 matched
the same 59 files and also contained `.cache/huggingface` metadata. Each rank
retained 24 RTS `VLLM::Worker` queue pairs with zero preemptions, restarts,
out-of-memory events, or fatal-log matches.

Conclusion: the exact cache-enabled composition restores persistent target
context after a coordinated engine restart and continues correct DFlash7
generation. The record is
[`performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md`](../../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md).

Limitations:

- The first request after API startup recomputed because the scheduler had not
  received a complete four-rank inventory checkpoint. The following identical
  request formed quorum and restored.
- The evidence covers an 8,192-token restored span. It does not establish
  throughput neutrality or larger-span restore performance.
- A rebuilt or redistributed image has implemented status until the same live
  qualification records its image identity and result.
- Streaming snapshots, native direct restore, MTP drafting, and other DFlash
  checkpoints are outside the evidence scope.
- A no-extra-files target-checkout claim applies only to ranks 0, 2, and 3.
  Rank 1 verified all expected files but contained Hugging Face cache metadata.
- The cache-disabled template has implemented status; the cache-enabled result
  does not qualify its performance or long-duration behavior.

## Provenance

All facts in this section were verified from the pinned repositories, source
locks, image labels, or qualification receipts. No unrecorded pull-request,
base-checkpoint, or binary-build lineage is inferred.

| Component | Verified source and role | Limitation |
|---|---|---|
| Target quantization | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`, repository owner `local-inference-lab`, uploaded by `lukealonso`, MIT. Its config names NVIDIA ModelOpt `0.39.0.dev290+gf9d9a71de.d20260407`, `MIXED_PRECISION`, NVFP4 target expert layers 3-44, and MXFP8 MTP expert layer 45. | The repository does not record a base-checkpoint revision. |
| Target artifact verification | `config.json` SHA-256 `676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`; `model.safetensors.index.json` SHA-256 `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb`; 59 expected repository files matched on all ranks. | Rank 1 also contained `.cache/huggingface` metadata; strict no-extra-files verification passed only on ranks 0, 2, and 3. |
| Public drafter | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, produced by Inco AI and uploaded by `zhijianliu`; BF16 config SHA-256 `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`; weights SHA-256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`; CC BY-NC-ND 4.0. | The model card limits use to research and evaluation and directs commercial licensing inquiries to Inco AI. |
| vLLM runtime | `local-inference-lab/vllm`, `dev/jovian-judgement@da4d7be6c97434f6942292ed8abbf4b32dc44355`. Direct commits used by that revision are `e0db84abedb4a85f93d130252e54b73c0f3ed695` (GLM model), `0c878821cf46c99729c7936bcbd4d868ad40e44e` (B12X integration), `4dbd82b9ced13114f90e93b8b6fae0966c942a3b` (C4 state), `1036123e935177900122c14d3cf02ad67b5422aa` (C4 behavior), and `e7097feb6fcdf57911cd68884420af2d80600dd7` (DFlash speculation). Merged pull requests are `#486@15d3f79439eadc396a57e253c955aa149def94ea` (C4 DCP), `#489@015dcd423d6aabf843c8ad69074ff67d35c2a395` (MoE router gate), `#493@067c37d974ca2b775d95e51e8fec234929f4e2c4` (capture-resource lifetime), `#494@e91c7e68f5863a27c79d2773205678be7d8ff132` (target/draft KV formats), `#497@05d85f603097fe7678d7dda2d522613d9dc61f46` (processor revision), and `#499@da4d7be6c97434f6942292ed8abbf4b32dc44355` (serialized MXFP8 projections); `#499` declares dependencies on `#493` and `#494`. | No lineage beyond the verified fork history is claimed. |
| B12X | `local-inference-lab/b12x`, branch `master`, commit `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`, Apache-2.0, titled `Accept runtime QSA cache page sizes`. | No associated pull request was found. |
| Collective transport | NVIDIA NCCL 2.30.7; `nccl-2.30.7-skip-tree-pat.patch` SHA-256 `097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2`; `nccl-2.30.7-advertise-all-listener-gids.patch` SHA-256 `dccfce86d14c15c39f0e0a742863960205a3d9823c464b31a7f7389354844178`; qualified binary SHA-256 `ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`. | The binary receipt does not bind the binary to an NVIDIA NCCL source commit and patch-build receipt. |
| SparkCache | `FujitsuPolycom/sparkcache@2d6a222f04fcb7b903cb899aba3ed3fdc75edc11` on branch `codex/glm53-flash`, source-tree SHA-256 `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`, profile `glm53-flash-hybrid`, vLLM lease contract SHA-256 `2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`, VMM exemption patch SHA-256 `370b498eebf44b4e52a2d2751fa249ad4bd3d0b6fd951b063a161fb06febbe99`, patch-preimage manifest SHA-256 `e0eb1b64d15812f122450f2e32323f0c907c640b8f8ccc270c77037bb9909b85`, Containerfile SHA-256 `ccc6b39173df80f604820959c3f19f8bc363f79d11f7d4f2d913054a4161b3f5`, and image-builder SHA-256 `c130e5c2fdd5f33e73f90f04ef85fa1247d93bfe6db409cd99508841f8d84547`. | The immutable commit and the source-tree, contract, and build-recipe digests are authoritative. |
| SparkRing deployment profile | `FujitsuPolycom/sparkring@d45572dbd2adc7afa1d3208fb801c8ad9eac7864` on branch `codex/glm53-flash-sparkcache-tp4`, based on `510556275ed3b77fc56a14367d319417072eeb8c`. | The branch has no pull request, release tag, or distributed image artifact. |
| SparkRing profile adaptation | The serving arguments were adapted into SparkRing's validated site/runtime schemas from a four-rank launcher snapshot (`fef84dda87bab36f36f993f21a3e582438f3b0d1e3239b292ef0ef39e8c44b23`) and service-settings snapshot (`2c4d81d04060d92f4419d3f17d3c51b2f195d66376c9271617a167c18de14df1`). Source lock SHA-256: `913d54bd68fdea1280a8dd2baf15cf3461e04645f50be5bda9eafc027d03e4a8`. | The source snapshots were uncommitted operator artifacts. Their hashes identify the adapted inputs; base Git revision `f3ba67fa476fd28109868811d6edbb4085c8f0a0` alone does not reproduce them. |

The machine-readable provenance contract is
[`runtime/glm53-flash/pins.json`](../../runtime/glm53-flash/pins.json).

## Docker image publication contract

Any container published for this profile is a **FujitsuPolycom community
derivative**. It is not an official image from vLLM, Z.AI,
local-inference-lab, NVIDIA, or Inco AI. Qualification status describes the
tested profile and exact artifacts; it does not transfer ownership or support
responsibility for upstream projects.

The recorded qualification used these immutable parent and derived image IDs:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

The qualification rebuilt the derivative independently from four rank-local
parents, so its image IDs are not one distributable artifact. A publishable
build must produce one derived image from one recorded parent and distribute
that same image ID to all ranks before repeating the live qualification.

Before publication, attach a receipt containing every checklist item:

- derived registry reference and manifest digest, local image ID, platform,
  size, and creation timestamp;
- parent registry reference and manifest digest plus the resolved local parent
  image ID;
- `deploy/glm53_flash/Containerfile` from the SparkCache source digest in the
  provenance table, all build arguments, the Docker/BuildKit versions, and the
  exact build command;
- vLLM and B12X commits, the verified vLLM direct commits and pull requests,
  both SparkRing NCCL patch hashes, the loaded NCCL binary hash, the
  SparkCache VMM patch/preimage hashes, and the vLLM lease-contract hash;
- target and DFlash repositories, revisions, uploaders, hashes, producers,
  and licenses, including the missing target base-checkpoint revision;
- inherited content: the parent image's vLLM, B12X, CUDA/toolchain, model
  architecture support, DFlash implementation, and patched NCCL;
- SparkCache-owned changes: the connector source, `glm53-flash-hybrid`
  profile, narrow VMM compatibility exemption, lease-contract verifier, and
  source/profile OCI labels. The derivative does not alter or include model
  weights and does not replace the NCCL binary;
- the tested TP4/DCP1 serving and cache configuration from this document;
- CPU validation commands and pass/skip counts, image-label and in-container
  attestation output, all-rank launch command, store/restart/restore/canary
  commands, results, and sanitized receipts;
- unsupported configurations: MTP, another DFlash checkpoint or depth,
  another target revision, TP2, DCP other than one, another context/scheduler/
  cache geometry, streaming snapshots, native direct restore, spans above the
  tested restore length, throughput-neutrality claims, and commercial use of
  the DFlash checkpoint without an applicable license; and
- support owner `FujitsuPolycom`, with SparkRing deployment issues reported at
  [FujitsuPolycom/sparkring issues](https://github.com/FujitsuPolycom/sparkring/issues)
  and SparkCache connector/image issues at
  [FujitsuPolycom/sparkcache issues](https://github.com/FujitsuPolycom/sparkcache/issues).

Run and record at least these public validation commands:

```bash
python -m ruff check --select E,F,W --ignore E501 spark_transport runtime scripts performance
python -m pytest spark_transport runtime/exl3-r7 runtime/deepseek0731-gb10 runtime/qwen38 runtime/test_public_overlay.py performance/harnesses scripts -q -rs
python scripts/sparkring_generic_launcher.py --site <site-yaml> --profile <profile-json> validate
python scripts/sparkring_generic_launcher.py --site <site-yaml> --profile <profile-json> plan
```

The SparkRing CPU suite for this profile change passed 1,877 tests with nine
skips, and Ruff passed. Those results verify repository contracts only; the
GPU/RDMA claim is the separately scoped live record above.

Use this minimal announcement only after the publication receipt is complete:

> **FujitsuPolycom community image — GLM-5.3 Flash TP4/DCP1 with BF16
> DFlash2 and optional SparkCache**
> Image: `<repository>@sha256:<manifest-digest>`; local ID:
> `sha256:<derived-image-id>`; parent:
> `<parent-repository>@sha256:<parent-manifest-digest>` / local ID
> `sha256:<parent-image-id>`. Built from the source and contract hashes in
> `runtime/glm53-flash/pins.json`. Qualified scope and unsupported
> configurations are documented in
> `docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md`. This is not an official
> vLLM, Z.AI, local-inference-lab, NVIDIA, or Inco AI image. Support:
> `https://github.com/FujitsuPolycom/sparkring/issues` and
> `https://github.com/FujitsuPolycom/sparkcache/issues`.

## Operator entry points

- [SparkCache-enabled quickstart](../GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md)
- [Cache-disabled quickstart](../GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md)
