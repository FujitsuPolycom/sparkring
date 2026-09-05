ARG PARENT_IMAGE=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f
FROM ${PARENT_IMAGE}

ARG PARENT_IMAGE
ARG PARENT_IMAGE_ID
ARG BUNDLE_MANIFEST_SHA256
ARG SOURCE_RECEIPT_SHA256
ARG COMPUTE_SOURCE_LOCK_SHA256

COPY compute/ /opt/sparkring-compute/
ENV CUDA_HOME=/opt/cuda-13.3 CUDA_PATH=/opt/cuda-13.3
ENV PATH="/opt/cuda-13.3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda-13.3/lib:/opt/cuda-13.3/lib64:${LD_LIBRARY_PATH}"
ENV TRITON_PTXAS_PATH=/opt/cuda-13.3/bin/ptxas
ENV VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1 VLLM_B12X_DENSE_ACTIVATION_MODE=auto
ENV VLLM_MTP_NVFP4_LM_HEAD=1 VLLM_LM_HEAD_A16=1 VLLM_MXFP8_LM_HEAD=0
RUN python3 /opt/sparkring-compute/apply_compute.py \
      --prepared /opt/sparkring-compute \
      --site-packages /usr/local/lib/python3.12/dist-packages

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
      org.sparkring.compute.source-lock-sha256="${COMPUTE_SOURCE_LOCK_SHA256}" \
      org.sparkring.b12x.composition="b58f34eaf978277621efced6678e6713fd7122e4" \
      org.sparkring.vllm.compute-source="${COMPUTE_SOURCE_LOCK_SHA256}" \
      org.sparkring.mesh.proposal-head="nvfp4-a16" \
      org.sparkring.sircl.manifest-sha256="${BUNDLE_MANIFEST_SHA256}"
