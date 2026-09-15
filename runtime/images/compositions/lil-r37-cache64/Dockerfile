ARG BASE_IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270
FROM ${BASE_IMAGE} AS native_build
COPY . /opt/sparkring-build/cache-extension/
COPY source/sparkcache/native /tmp/qwen-native-source/native
RUN /opt/venv/bin/python /opt/sparkring-build/cache-extension/cache_extension.py verify-parent --descriptor /opt/sparkring-build/cache-extension/descriptor.json
RUN cmake -S /tmp/qwen-native-source/native -B /tmp/qwen-native-build -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_CUDA_FLAGS=--frandom-seed=spark-cache-snapshot -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON && \
    cmake --build /tmp/qwen-native-build --target spark_cache_snapshot spark_cache_page_capture_layout_test -j 8 && \
    /tmp/qwen-native-build/spark_cache_page_capture_layout_test
FROM native_build
RUN /opt/venv/bin/python /opt/sparkring-build/cache-extension/cache_extension.py install --context /opt/sparkring-build/cache-extension --native-build /tmp/qwen-native-build
LABEL org.sparkring.cache-extension="lil-r37-cache64"
