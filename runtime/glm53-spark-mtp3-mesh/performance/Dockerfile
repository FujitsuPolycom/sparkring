ARG PARENT_IMAGE=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987
FROM ${PARENT_IMAGE}
RUN --mount=type=bind,target=/performance python3 -S -B /performance/install.py
ENV SPARK_GDN_PREFILL_CHECKPOINTS=2
LABEL org.sparkring.runtime.status="research-only" \
      org.sparkring.runtime.profile="glm53-mtp3-cache-checkpoints" \
      org.sparkcache.commit="48bbd2be4a7b972e56632a2d7b934bac5460f272" \
      org.sparkcache.cuda-placement-sha256="2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d" \
      org.sparkring.sircl.manifest-sha256="c0fd5567442b08b908cc193f36d0864e262573c7e5d232509479a823cface742"
ENTRYPOINT ["python3", "-S", "-B", "/opt/sparkring/bin/start-performance.py"]
