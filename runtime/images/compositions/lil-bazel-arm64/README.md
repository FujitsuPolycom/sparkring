# Published ARM64 LIL-source foundation

Status: **Experimental** foundation candidate. This evaluates the immutable
`randomvariable/vllm-b12x-multi` image recorded in [evaluation.json](evaluation.json).
It is a source-pinned alternative foundation for a SparkRing composition; no
serving profile selects it directly.

## Observed compatibility

The image was pulled on a GB10 host and inspected in an isolated container with
no GPU devices, networking or host mounts. Python 3.12.12, Torch 2.13.0/CUDA 13.3,
vLLM and B12X imported successfully. The installed Qwen HC, MTP and RoCEnante
adapter files match the three source preimages used by the tested SparkRing
feature hooks. This establishes compatibility for those file interfaces, not
the complete runtime or GPU behavior.

The packaged NCCL reports 2.30.4 and lacks SparkRing's PCI-domain-preservation
and subnet-aware-routing switch strings. SparkCache, SIRCL and SparkRing's
installed-image receipts are absent. The [evaluation](evaluation.json) retains
the actual package versions, library hash and source hashes.

## Composition requirements

1. Keep the immutable external foundation as the Docker parent. Reconcile the
   selected LIL vLLM/B12X sources with SparkRing's carried GLM mHC, KDA coalescing
   and source-lease changes. Remove patches already supplied equivalently upstream.
2. Add the pinned SparkRing transport libraries and selectors. Verify the actual
   loaded NCCL library and qualify mesh/dual-domain behavior; setting flags on the
   external library does not add the missing implementation.
3. Add SparkCache and generate its lease contract against the reconciled vLLM
   source. Add only the model features selected by the profile.
4. Generate a complete installed inventory and use the shared image-admission
   boundary. The parent does not contain a SparkRing receipt, so the existing
   feature-extension installer cannot consume it unchanged.
5. Compare native build inputs and ABI before reusing compiled components. Test
   the composed image through the same profiles and bounded acceptance checks.

The existing R37 composition remains the operational reference. Adopting a
foundation does not require adopting the publisher's entire build system or
changing profile settings. Image publication and profile promotion remain
separate review decisions.

## Reproduce the CPU inspection

On an ARM64 Docker host, run from the repository root:

```bash
BASE=ghcr.io/randomvariable/vllm-b12x-multi@sha256:72431d1f54507c621d7b8c4140c1620867b8a171bdacfb7fa76f4c075db90561
docker pull --platform linux/arm64 "$BASE"
docker run --rm -i --runtime runc --network none --read-only \
  --tmpfs /tmp:rw,size=536870912 --memory 4g --cpus 2 --pids-limit 256 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  --entrypoint /opt/venv/bin/python "$BASE" - \
  < runtime/images/compositions/lil-bazel-arm64/probe.py
```

The output line beginning `SPARKRING_BASE_PROBE=` contains the observations.
This command deliberately cannot qualify GPU serving. The
[publisher's build design](https://github.com/randomvariable/vllm-multiarch-oci/blob/36b70acaa5e108657c121c744617dee12538d96d/docs/explanation/build-and-cache-design.md)
describes its dependency and application layers.
