# Shared serving image: build and runtime checks

Status: **qualified for the bounded checks below; serving qualification pending**.
This record identifies an ARM64/GB10 runtime build for the
[shared serving candidate](../../../runtime/releases/shared-2026.09.0-rc.1/README.md).
It is not a registry publication, pull reference or profile recommendation.

## Identity and conditions

| Artifact | Identity |
|---|---|
| Runtime-build Docker configuration ID | `sha256:98d5a51e2e0ed8f7cee14ff1420ab9d2413f4bba88b70cff9b40ab0b1ea00deb` |
| Build-input SHA256 | `d0eb323000696e6473a2855e63b225dad6b2e518720f258bcc2fd0c0475aee66` |
| Native installed-payload receipt SHA256 | `85655482b7ff4a2ce9248a1af7729f1ec47a66730e2b9c1b9e152a7c3a2a0e1c` |
| Operator build-report SHA256 | `72ff07854fd09122ae1e96af25331f456a493c5c2d7e1d0d1e9d6d6a2d286c9f` |
| Runtime-check report SHA256 | `8861dbdc13810b1803985ef8721714271076a92ac5a8cb2a273b9c38f73ec390` |

The build uses `linux/arm64`, PyTorch 2.13.0 and CUDA 13.3. Runtime checks detected
one NVIDIA GB10 GPU with compute capability 12.1 and 48 SMs. The
[source manifest](../../../runtime/releases/shared-2026.09.0-rc.1/sources/manifest.json)
binds vLLM commit `35bab057b1751a6076a457803bcc4b78809689cf` and B12X commit
`a83336581a3a907076e60797df69ab66df5a2ff1` plus their reconciled patches. Accepted
snapshot SHA256 values are:

- vLLM: `34351e28ad097d1d6d88561d921759197c7c2f6ecc0d6b61803f7d0ee4cac025`.
- B12X: `e1de3a479781ca34e94c83daa72f8ed7e673e016dfe24e48473de7eaaf19aa33`.

These snapshot hashes are not Git tree IDs. The build reports
`native_rebuilt=false`: it reused native artifacts and verified their installed
payload; this run is not evidence of a fresh native compilation. The embedded
feature inventory includes `qwen-collectives` and `qwen4-prefill`; their presence
does not qualify a model or enable every feature for every profile.

## Verification results

Installed-payload verification passed for 235,822 files. The isolated SGLang
inventory additionally reports 177,742 verified files and receipt SHA256
`8b6046fdcecfdada52eb7e0adf799cbfadfb6fb0989c3c4ab1951d54b268c9d0`;
these counts are not additive inventories of distinct files.

The active cache-source contract is
`/opt/sparkring/contracts/vllm-connector-jobs-source-34351e28ad097d1d.json`, SHA256
`20a837a07b55b4f1879f8f62b61c8fb5f0d0badc9b104c366a08b193339fcd79`.
All 45 source-file hashes match the installed-payload receipt. This proves the
selected source binding, not persistence correctness. The contract requires a
fresh cache namespace and does not claim compatibility with parent-image entries.

All 10 runtime checks passed with exit code 0 and no timeout:

| Check | Verified condition |
|---|---|
| GPU | CUDA is available; device is GB10/SM121 |
| Native vLLM | Stable-libtorch extension and custom-ops modules import |
| Qwen model | Conditional-generation and MTP modules/classes import |
| GLM model | Conditional-generation and MTP modules/classes import |
| B12X | Loader registration and prepared-execution modules import |
| FlashInfer | Runtime, JIT environment and JIT-cache modules import |
| SparkCache | Connector module and `SparkContextCacheConnector` import |
| Serving CLI | `vllm serve --help=all` exits successfully |
| Package constraints | `pip check` passes without retained or foundation exceptions |
| Isolated SGLang | Its PyTorch, kernel, FlashInfer and Rust multimodal modules import from the isolated environment |

## Qualification boundary

No model-generation or cache-restore result is asserted for this image. Profile
startup, correctness, distributed restore, performance, media and soak checks
remain separate.

The metadata-only publication candidate has Docker configuration ID
`sha256:5252600fb91bd1f999856f31cc382ca91934d45fa3043429344aa8fd18ad01d3`.
Its [equivalence summary](shared-5252600fb91b-metadata-equivalence-20260918.json)
records identical ordered filesystem layers (123) and identical raw/runtime
image configuration except labels. Installed payloads and the native receipt
are unchanged. These facts associate the runtime-build evidence with the
candidate's unchanged runtime; they are not additional executed tests. The
underlying proof SHA256 is
`d2775027029bcbd3da0095293942aec2eaddd3f9584e038722ee1b2ca8c06e3c`.
Neither image has a public registry digest in this record.

The [prepared RoCEnante transport record](../transport/rocenante-prepared-35cf12b2-20260918.md)
qualifies bounded two/four-rank operations on image `35cf12b2d644`, not this image.
Both contain transport-manifest SHA256
`e8577c447a69ac75253758a0964791e862ccca1ad13168e7addefa6ec96369c9`.
The GDN constructor fix has separate component evidence on that preceding image
with a read-only source overlay; neither result substitutes for serving tests of
the configuration ID recorded above.
