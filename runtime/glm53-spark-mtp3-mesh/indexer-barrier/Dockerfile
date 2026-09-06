FROM ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987
ARG SOURCE_RECEIPT_SHA256
ARG COMPUTE_SOURCE_LOCK_SHA256
ARG SPARKRING_REVISION
COPY patch_indexer_barrier.py install.py /opt/sparkring/mesh-indexer-barrier/
RUN python3 /opt/sparkring/mesh-indexer-barrier/install.py --expected-receipt ${SOURCE_RECEIPT_SHA256}
LABEL org.opencontainers.image.revision="${SPARKRING_REVISION}" \
      org.opencontainers.image.base.name="ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987" \
      org.sparkring.mesh.parent-image="sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f" \
      org.sparkring.mesh.source-receipt-sha256="${SOURCE_RECEIPT_SHA256}" \
      org.sparkring.compute.source-lock-sha256="${COMPUTE_SOURCE_LOCK_SHA256}" \
      org.sparkring.vllm.compute-source="${COMPUTE_SOURCE_LOCK_SHA256}" \
      org.sparkring.mesh.indexer-barrier="35a7564ec8bf18f1b7dc5562b21616437b65e7b0689c56341b6436c8d202637b"
