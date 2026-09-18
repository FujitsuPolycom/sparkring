# SparkRing image with vLLM and SGLang

Status: **Development**. This local composition adds an isolated SGLang runtime
to the Qwen-capable SparkRing image identified by [manifest.json](manifest.json).
The [local build](local-build.json) passed four-rank collective checks and
[bounded SGLang serving checks](../../../performance/records/deepseek-v41-flash/sglang-shared-bounded-prefill.json),
including exact retrieval from a 639831-token prompt. It has no published digest.

| Component | vLLM | SGLang |
|---|---|---|
| Python environment | Inherited `/opt/venv` | `/opt/sglang` |
| CUDA tools | Inherited CUDA 13.3 | CUDA 13.0 |
| NCCL | Inherited SparkRing library | Pinned SparkRing 2.30.7 in Torch's dependency directory |
| Entry point | Inherited source verifier and vLLM CLI | `/opt/sparkring/bin/sglang-python` |
| Model configuration | Existing Qwen HC, checkpoint, QSA and transport options | DeepSeek-V4.1-Flash TP4/EP4 with Mia's NVMe Engram adapter |

The default image entrypoint and environment retain vLLM behavior. SGLang's
wrapper selects its own Python, CUDA tools and cache directories and removes
vLLM's library preload. The Engram adapter uses the CUDA runtime shipped beside
SGLang's Torch package. The two engines do not share Python packages or NCCL
libraries. A profile starts one engine; this composition does not schedule two
models concurrently on the same GPU.

The build preserves Mia's source and license in `/opt/dsv41`. Two recorded
integration edits select the matching CUDA runtime and correct the checkpoint
revision written to launch metadata. The [SGLang overlays](../patches/README.md)
add compressed-state verification kernels and bound dense-indexer allocation.
The image also records input hashes and an installed SGLang payload inventory.

Use the [build and launch procedure](../README.md#shared-sparkring-image).
The existing standalone image builder and 262K profile default remain available.
The recorded image passed the 655360-context configuration with 1499904 allocated
KV tokens, 4096-token chunks and eight allowed requests. Its minimum sampled
host memory headroom was 22.68 GiB. The retrieval check used one request; it does
not establish concurrent long-context capacity, general quality or soak stability.

Verify each installed runtime independently:

```bash
docker run --rm --network none --entrypoint /opt/venv/bin/python IMAGE \
  /opt/sparkring/bin/source-extension.py verify
docker run --rm --network none --entrypoint /opt/sparkring/bin/sglang-python IMAGE \
  -S /opt/sparkring/sglang/verify.py verify
```

Replace `IMAGE` with the exact built image ID. These commands check source and
package receipts without GPU access. The SGLang record covers startup,
authenticated generation, near-limit retrieval and per-rank memory. Qwen's
57247 inherited files verify unchanged. The [Qwen TP2 record](../../../performance/records/qwen38-flash-next/combined-image-tp2-sparkcache.json)
covers startup, generation and SparkCache restoration after restarting both
workers, with 33 GiB KV per rank. Both retrieval fixtures restored 5696 tokens
on each rank and returned the expected answer.

The same TP2 check measured cold prefill and bounded C1/C8 completion against
the parent image. The 8K prefill medians were 2.606 s and 2.607 s; the 32K medians
were 15.752 s and 13.129 s. Combined-image completion throughput was lower and
variable, so performance parity is not established. The public 24 GiB TP2
default and GLM serving are not qualified by this record.

The [Qwen TP4 record](../../../performance/records/qwen38-flash-next/combined-image-tp4-sparkcache.json)
covers startup, generation and SparkCache restoration with 40 GiB KV per rank,
HC sharding and prefill coalescing selected. Both fixtures restored 7200 tokens
on every rank after all four workers restarted and returned the expected answer.
Startup reported 5218792 logical KV tokens. This bounded check does not measure
performance, full-context capacity or soak stability.

SGLang restoration exposed a retained-container restart limitation: two ranks
failed before server initialization, and a rank 3 probe reproduced a bus error
in `torch.cuda.is_available()`. Fresh containers with equivalent settings passed
CUDA initialization. The TP4 record preserves that recovery separately from the
successful Qwen cache check; the underlying restart failure remains unresolved.

## Local GLM admission

The explicit local GLM source adapter accepts TP2/DCP1 and TP4/DCP1/DCP4 with
SparkCache disabled by default. A separate receipt option admits their SparkCache
variants for isolated local trials. It verifies the complete source receipt chain
and disables Qwen feature selection. This is structural admission, not GLM serving evidence.
Existing R33/R37 registrations and public selections are unchanged.

On a Docker host containing the selected image and the registered R37 base used
for ancestry verification, create a private receipt without starting a model:

```bash
python3 -m runtime.common.glm_source_candidate \
  --local-source-extension lil-r37-qwen-prefill \
  --image-id sha256:b03062b032bb147f5255463d4f068bfc476bf966c78c82ddeb66df4fb1b0b37d \
  --output /private/glm-source-receipt.json
```

The TP2 planner accepts this file through `--runtime-receipt`. For TP4, pass it
to the mesh profile renderer through `--image-receipt`, then use
`python3 -m runtime.common.glm_launch plan` for structured Docker/Compose creation.
The generated legacy shell launcher deliberately refuses source-image trials;
its historical image gate does not verify the complete source ancestry.

To create an isolated SparkCache trial receipt, add `--allow-sparkcache-trial`
to the admission command and choose a new private output path. Its boolean
`allow_sparkcache_trial` field defaults to `false`; a missing or false value
rejects cache profiles, and non-boolean values are invalid. The TP2 planner then
accepts `--sparkcache`; TP4 sites may select `tp4-dcp1-sparkcache` or
`tp4-dcp4-sparkcache`.

The trial selects the packaged
`/opt/sparkring/contracts/vllm-connector-jobs-r37-qwen-prefill.json` lease and the
placement/snapshot library hashes verified in the child image. The inherited
parent lease does not match the changed scheduler and allocator sources.
Persistent cache namespaces include the source-trial prefix, image ID, TP/DCP
profile and model revision, keeping them distinct from parent-profile entries.

The focused [allocator checks](../../images/compositions/lil-r37-qwen-prefill/checks/mamba_checkpoint_reservation.py)
passed 35 CPU cases on the exact child source. Three GLM cases also passed on
the parent: planned four-checkpoint continuation, short packed fallback and
sparse fallback beyond checkpoint capacity. With four checkpoint slots and
three MTP scratch pages, those normal aligned GLM cases do not select the
occupied-scratch preservation branch. This supports the explicit local trial;
GPU checkpoint contents, restart/restore correctness, serving quality and
stability still require qualification on the selected topology. Qwen cache
measurements do not establish those GLM results.
