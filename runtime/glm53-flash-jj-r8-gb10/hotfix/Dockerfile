ARG PARENT_IMAGE=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f
FROM ${PARENT_IMAGE}
ARG SOURCE_RECEIPT_SHA256
ARG SPARKRING_REVISION
COPY patch_indexer_barrier.py install_hotfix.py serve_with_warmup.py scheduler_liveness.py /opt/sparkring/issue224/
RUN python3 /opt/sparkring/issue224/install_hotfix.py --expected-source-receipt ${SOURCE_RECEIPT_SHA256}
LABEL org.opencontainers.image.revision="${SPARKRING_REVISION}" \
      org.opencontainers.image.base.name="ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f" \
      org.sparkring.parent.image="sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075" \
      org.sparkring.source-receipt-sha256="${SOURCE_RECEIPT_SHA256}" \
      org.sparkring.runtime.status="implemented-indexer-gpu-tested-no-model-soak" \
      org.sparkring.indexer-barrier.sha256="49f6fd916fd1ccf94311ee99427551edbd0dc3a5de23aeeb426418370f76f66d" \
      org.sparkring.indexer-barrier.issue="https://github.com/FujitsuPolycom/sparkring/issues/224" \
      org.sparkring.sampling-warmup="temperature1-thinking-enabled-before-readiness"
