ARG PARENT_IMAGE=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f
FROM ${PARENT_IMAGE}

ARG PARENT_IMAGE
ARG PARENT_IMAGE_ID
ARG BUNDLE_MANIFEST_SHA256
ARG SOURCE_RECEIPT_SHA256

COPY bundle/ /opt/spark-sircl/
COPY receipts/ /opt/sparkring/receipts/glm53-spark-mtp3-mesh/
COPY verify_mesh_image.py /opt/sparkring/bin/verify-mtp3-mesh-image.py
COPY marker.c /opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c
COPY warmup_dflash.py /opt/sparkring/bin/warmup_dflash.py
ENV SPARKRING_WARMUP_TEMPERATURE=1

RUN cc -O2 -Wall -Wextra \
      /opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c \
      -o /opt/sparkring/bin/mlx5-rdma-tx-marker -libverbs -lmlx5 && \
    python3 -I /opt/sparkring/bin/verify-mtp3-mesh-image.py --inside-image

LABEL org.opencontainers.image.title="SparkRing GLM-5.3 Spark MTP3 mesh runtime" \
      org.opencontainers.image.description="Pinned ARM64 runtime with SIRCL and RoCEnante virtual-diagonal transport; host fabric provisioning is separate" \
      org.opencontainers.image.source="https://github.com/FujitsuPolycom/sparkring" \
      org.opencontainers.image.base.name="${PARENT_IMAGE}" \
      org.sparkring.runtime.status="research-only" \
      org.sparkring.mesh.parent-image="${PARENT_IMAGE_ID}" \
      org.sparkring.mesh.bundle-manifest-sha256="${BUNDLE_MANIFEST_SHA256}" \
      org.sparkring.mesh.source-receipt-sha256="${SOURCE_RECEIPT_SHA256}" \
      org.sparkring.mesh.default-speculation="mtp3" \
      org.sparkring.sircl.manifest-sha256="${BUNDLE_MANIFEST_SHA256}"
