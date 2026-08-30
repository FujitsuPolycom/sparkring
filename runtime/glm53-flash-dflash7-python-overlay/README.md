# GLM-5.3 DFlash7 public-base Python overlay

Status: **implemented**, not qualified. The builder constructs and verifies an
ARM64 image but no image digest from this path has completed four-rank serving
qualification.

The image combines these exact roles:

- retained vLLM native extensions and wheel metadata from
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- the 31-file vLLM Python delta at
  `0b67266a0f37d6146a8403fb8482403c62f412d5`;
- B12X `b1d541f9e71a35f030d45fae437630fff7507c2a`;
- SparkCache reconstructed-page placement source
  `19e2ec8b59c84ef359c2a3290f86962e3ff71d96`, Git tree
  `d8b417bb4b6d734c4403c0a73e7e42b95abd8343`, and deployable source SHA-256
  `bc7cae86732c869ee8b2205d48ac5be6f580ee8b77a3e4ffd4c69dcd4f1bfae5`;
- external BF16 DFlash2 weights with SHA-256
  `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

The image builder shares the byte allowlist, retained-native verifier, exact
SparkCache patch chain, and eleven-file lease contract with
`runtime/glm53-flash-adaptive-mtp-python-overlay/`. Prepared image metadata is
rendered for external DFlash7; it does not claim adaptive MTP.

Build on Linux ARM64:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-dflash7-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-dflash7-python-overlay/build-image.sh
```

The script does not push the image. Its receipt verifies mixed vLLM
provenance, B12X, target loader dependencies, NCCL, SparkCache CUDA placement,
the clean SparkCache source receipt, and the vLLM lease contract. DFlash model
files remain operator-mounted and are verified by the runtime profile.

Two executable profiles use external DFlash at depth seven and TP4, FP8 target
KV, 256-token vLLM blocks, 32 sequences, and SparkCache page-tail copy-on-write
publication with CUDA restore. Both are implemented but unqualified on the
composed 0b image. The conservative profile uses global safetensors. The mixed
profile uses global fastsafetensors for the target and an exact
`draft_load_config` selecting safetensors for DFlash. The image applies and
verifies the draft-loader patch before installing SparkCache patches. See
`docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md`.

The SparkCache source accepts the canonical CUDA configuration keys directly.
No PR25 compatibility profile or legacy-key translation is required by these
profiles.

The pinned SparkCache source from
[pull request #29](https://github.com/FujitsuPolycom/sparkcache/pull/29)
accepts canonical CUDA configuration keys and replaces a partial terminal HMA
page when the authenticated cache boundary falls inside that page. It does not
change cache identity, page-tail wire schemas, record geometry, vLLM patch
bytes, the lease contract, or the CUDA placement ABI. Compatible entries
produced by the source at commit
`5d571018de5b63a9a90e5c11e6d6e86bbff4a957` remain in the same namespace.
Null-block publication failures remain unsupported by this source contract.
