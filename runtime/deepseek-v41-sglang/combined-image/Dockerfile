ARG SGLANG_BASE=lmsysorg/sglang@sha256:e56358a68b06427362283c8c8a9d7d706448082ae53098ea1131b5d51aa1fd62
ARG VLLM_PARENT=sparkring-local:lil-r37-qwen-prefill-f01854c21da9
FROM ${SGLANG_BASE} AS sglang
COPY mia/ /opt/dsv41/
COPY patches/ /opt/sparkring/sglang-patches/
RUN /opt/sglang/bin/python3 -S /opt/sparkring/sglang-patches/apply.py \
      --source-root /sgl-workspace/sglang --receipt /opt/sparkring/sglang-overlay.json \
 && g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
      /opt/dsv41/adapter/row_store.cpp -o /opt/dsv41/adapter/librow_store.so \
 && cp /opt/dsv41/runtime/flash_mla_sm120.py \
      /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py

FROM ${VLLM_PARENT}
# Keep the parent's vLLM entrypoint, dependencies, CUDA selection and receipts.
COPY --from=sglang /opt/sglang/ /opt/sglang/
COPY --from=sglang /sgl-workspace/sglang/ /sgl-workspace/sglang/
COPY --from=sglang /usr/local/cuda-13.0/ /usr/local/cuda-13.0/
COPY --from=sglang /opt/dsv41/ /opt/dsv41/
COPY --from=sglang /opt/sparkring/sglang-patches/ /opt/sparkring/sglang-patches/
COPY --from=sglang /opt/sparkring/sglang-overlay.json /opt/sparkring/sglang-overlay.json
COPY libnccl.so.2 /opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
COPY runtime/ /opt/sparkring/sglang/
RUN chmod 755 /opt/sparkring/sglang/sglang-python \
 && ln -s /opt/sparkring/sglang/sglang-python /opt/sparkring/bin/sglang-python \
 && /opt/sglang/bin/python3 -S /opt/sparkring/sglang/verify.py install
LABEL org.sparkring.runtimes="vllm,sglang" \
      org.sparkring.sglang.composition="sglang-deepseek-bounded-prefill"
