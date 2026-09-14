ARG BASE_IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:de885a8a3f687d1966b918f913ab95b0da33a84422313ed4c10ba5477c66f523
FROM ${BASE_IMAGE}
ENV SPARKRING_FEATURES=""
COPY . /tmp/sparkring-feature-context/
RUN /opt/venv/bin/python /tmp/sparkring-feature-context/feature_extension.py install --context /tmp/sparkring-feature-context && rm -rf /tmp/sparkring-feature-context
ARG DESCRIPTOR_SHA256
LABEL org.sparkring.status="Development" \
      org.sparkring.feature-descriptor-sha256="${DESCRIPTOR_SHA256}" \
      org.sparkring.features="qwen-collectives,qwen-prefill"
