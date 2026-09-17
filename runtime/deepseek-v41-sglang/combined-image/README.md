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
57247 inherited files verify unchanged; its prefill, decode and cache restoration
on this combined image remain unmeasured. Keep that distinction when selecting
or publishing a profile.
