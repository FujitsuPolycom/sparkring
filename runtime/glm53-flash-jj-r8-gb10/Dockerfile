ARG PARENT_IMAGE=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5
FROM ${PARENT_IMAGE}

ARG PARENT_IMAGE
ARG PARENT_IMAGE_ID
ARG SPARKRING_REVISION
ARG SOURCE_RECEIPT_SHA256

COPY bundle/ /opt/sparkring/overlays/jj-r8-sparkcache-arm64/
COPY install_overlay.py /opt/sparkring/bin/install-jj-r8-sparkcache-overlay.py
COPY verify_image.py /opt/sparkring/bin/verify-jj-r8-sparkcache-image.py
COPY warmup_dflash.py /opt/sparkring/bin/warmup_dflash.py
COPY serve_with_warmup.py /opt/sparkring/bin/serve-with-warmup.py

RUN python3 /opt/sparkring/bin/install-jj-r8-sparkcache-overlay.py \
      --root /opt/sparkring/overlays/jj-r8-sparkcache-arm64 && \
    mkdir -p /opt/sparkring/receipts/jj-r8-sparkcache-arm64 && \
    cp /opt/sparkring/overlays/jj-r8-sparkcache-arm64/receipts/* \
      /opt/sparkring/receipts/jj-r8-sparkcache-arm64/ && \
    python3 /opt/sparkring/bin/verify-jj-r8-sparkcache-image.py --inside-image

HEALTHCHECK --interval=2s --timeout=1s --start-period=600s --retries=1 \
  CMD test -f /tmp/sparkring-engine-ready || exit 1

ENV VLLM_B12X_MOE_FP4_FORCE_A16=0 \
    CUTE_DSL_ARCH=sm_121a \
    TORCH_CUDA_ARCH_LIST=12.1a \
    CMAKE_CUDA_ARCHITECTURES=121 \
    FLASHINFER_CUDA_ARCH_LIST=12.1f

LABEL org.opencontainers.image.title="SparkRing GLM-5.3 ARM64 operator image with optional SparkCache" \
      org.opencontainers.image.description="Source-pinned vLLM and B12X runtime for four GB10 systems with DFlash2 and DCP-aware SparkCache" \
      org.opencontainers.image.source="https://github.com/FujitsuPolycom/sparkring" \
      org.opencontainers.image.revision="${SPARKRING_REVISION}" \
      org.opencontainers.image.base.name="${PARENT_IMAGE}" \
      org.sparkring.runtime.status="implemented-live-verification-required" \
      org.sparkring.parent.image="${PARENT_IMAGE_ID}" \
      org.sparkring.source-receipt-sha256="${SOURCE_RECEIPT_SHA256}" \
      org.sparkring.vllm.proven-base="331573d20bd47e78327ed8d8b4d2e6d350bbb1ab" \
      org.sparkring.vllm.python-commit="22ffe1401ca9bd3e4503e62de7b414deca7661a1" \
      org.sparkring.vllm.python-tree="1bb7f10a5838d348ca2fcb0134b05ad768d3340b" \
      org.sparkring.vllm.sparkcache-composition="22ffe1401ca9bd3e4503e62de7b414deca7661a1" \
      org.sparkring.vllm.tree="1bb7f10a5838d348ca2fcb0134b05ad768d3340b" \
      org.sparkring.vllm.community-release="Jovian Judgement Community R10" \
      org.sparkring.vllm.community-parent="55969c16d4da57da76ee5729f3102d4b2003833c" \
      org.sparkring.vllm.sparse-pooled-index="adb69eac865a1a37081fa4edb9f7599a351f7aac" \
      org.sparkring.vllm.fwht-scaling="ae37fd6ed8df72d4bd8cdc067c7c241c93408235" \
      org.sparkring.vllm.prefill-cadence-component="f1191b9090cd02ac49238c8e4f371050759703b6" \
      org.sparkring.vllm.prefill-cadence-pr-head="2412a6f34ab1412f86ed3e4cdd355271a082d93d" \
      org.sparkring.vllm.delta-patch-id="44ad5586548465c002d0195d3739992795233ffe" \
      org.sparkring.vllm.retained-native-build="3633d61c3c7b04bb4d598cadbdc342f3be40482d" \
      org.sparkring.vllm.retained-native-parent="da4d7be6c97434f6942292ed8abbf4b32dc44355" \
      org.sparkring.b12x.composition="6255090a03b12c3f7d552102a02fac0b542fb8c9" \
      org.sparkring.b12x.tree="0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad" \
      org.sparkring.b12x.package-tree="6de9871d15dab093340695518fec0f744289e676" \
      org.sparkring.nccl.sha256="5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3" \
      org.sparkring.loader="fastsafetensors" \
      org.sparkring.diagnostics="compact-startup-no-deep-ep" \
      org.sparkcache.commit="737ed1399f559ba036fb0e358541744011afd47d" \
      org.sparkcache.tree="decc0e042a4b1807e960551ffa3ef12c8c9114a7" \
      org.sparkcache.source-sha256="3cfb8d66db3a437a8b3a886633e64b7006af4c50cccb3ddbf75eb8d73eda5de6" \
      org.sparkcache.cuda-placement-sha256="d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c" \
      org.sparkcache.cuda-snapshot-sha256="4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5" \
      org.sparkcache.startup-inventory="connector-handshake-all-rank" \
      org.sparkcache.cache-geometry="manager-pages-v2" \
      org.sparkcache.publication-schema="snapshot-v1,page-tail-cow-v1,page-tail-cow-v2" \
      org.sparkcache.dcp-layouts="1,2,4" \
      org.sparkcache.default="disabled-until-explicitly-configured"
