# Public EXL3 derived-image layer

This directory reconstructs and verifies the ARM64/SM121 runtime required by
the `glm52-exl3-tr3-3.25bpw` recipe. The recipe's executable profile is
TP4/DCP4 with fixed MTP2, a 524,288-token model limit, 4.5 GB of KV per rank,
a 4,096-token batch budget, C8, a Q32 graph ceiling, and one local LMCache
server per rank with 512-token chunks. It is a thin, receipt-gated layer over SparkRing's public NF3
NVFP4/FP8-RoPE image; it does not rebuild Torch, vLLM, CUDA, or FlashInfer from
scratch.

The build is source-reproducible rather than tag-reproducible:

- `pins.json` owns every Git base, expected tree, patch hash, vLLM overlay
  hash, installed-runtime preimage/postimage, and model-manifest digest;
- `prepare_context.py` fetches the public bases, applies only the pinned
  patches, verifies the exact Git trees, and emits a byte manifest;
- `verify_build_context.py` rejects missing, extra, or changed context bytes;
- `build-image.sh` requires the exact NF3 parent image ID and an ARM64 host;
- `verify_exl3_runtime.py` checks the installed runtime at build and launch;
- `model_manifest.py` verifies all 88 execution inputs while deliberately
  excluding nine manifested release-only sidecars from content verification;
- `entrypoint.sh` refuses to start vLLM unless runtime verification passes.

At the pinned model revision, the served `README.md` bytes disagree with the
README digest in the immutable `MANIFEST.sha256`. README content is release
metadata, not a runtime input. The manifest itself remains hash-pinned, and the
verifier still hashes all 81 weight shards plus seven execution metadata files
and rejects unmanifested non-cache files. The README and eight other
release-only sidecars are the only manifested files excluded from content
verification.

Users should not invoke these files individually. The supported entry point is:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

See [`docs/EXL3_RECIPE.md`](../../docs/EXL3_RECIPE.md) for the full preparation,
review, and LMCache launch sequence. The source bootstrap is clean-checkout
live-validated on four directly cabled DGX Sparks. The identical deployed image
ID was
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`.
This validates the bounded bootstrap/startup/API gate; it does not establish
blanket correctness, NVMe persistence, release promotion, or full
public-functional acceptance.
