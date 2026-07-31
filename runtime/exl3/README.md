# Public EXL3 derived-image layer

This directory reconstructs and verifies the ARM64/SM121 runtime required by
the non-default `glm52-exl3-tr3-3.25bpw` recipe. It is a thin, receipt-gated
layer over SparkRing's public NF3 NVFP4/FP8-RoPE image; it does not rebuild
Torch, vLLM, CUDA, or FlashInfer from scratch.

The build is source-reproducible rather than tag-reproducible:

- `pins.json` owns every Git base, expected tree, patch hash, vLLM overlay
  hash, installed-runtime preimage/postimage, and model-manifest digest;
- `prepare_context.py` fetches the public bases, applies only the pinned
  patches, verifies the exact Git trees, and emits a byte manifest;
- `verify_build_context.py` rejects missing, extra, or changed context bytes;
- `build-image.sh` requires the exact NF3 parent image ID and an ARM64 host;
- `verify_exl3_runtime.py` checks the installed runtime at build and launch;
- `entrypoint.sh` refuses to start vLLM unless runtime verification passes.

Users should not invoke these files individually. The supported entry point is:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

See [`docs/EXL3_RECIPE.md`](../../docs/EXL3_RECIPE.md) for the full preparation,
review, and launch sequence. The source bootstrap is offline-validated; its
clean-checkout four-Spark acceptance run remains pending.
