FROM ghcr.io/fujitsupolycom/sparkring@sha256:26c366af994cf42e38e4596db4d611342a3466fd8ca49d6037237d04249e5132

# Keep the parent's native extensions and SparkRing transport; replace Python
# sources with the mutually compatible revisions in b12x-image.json.
COPY vllm/ /opt/venv/lib/python3.12/site-packages/vllm/
COPY b12x/ /opt/venv/lib/python3.12/site-packages/b12x/
COPY provenance/ /opt/sparkring/mimo-provenance/

LABEL org.sparkring.candidate.vllm-commit="6afb99982576a7a2eb53d667189e859629e22739" \
      org.sparkring.candidate.b12x-commit="4f3028b19c1d8290dc72b6f483aba40de23eae5a" \
      org.sparkring.candidate.status="research-only"

RUN /opt/venv/bin/python -m py_compile \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/b12x.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/b12x_noncausal.py \
    /opt/venv/lib/python3.12/site-packages/b12x/attention/paged/api.py
