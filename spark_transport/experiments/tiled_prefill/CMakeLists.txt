cmake_minimum_required(VERSION 3.24)

project(spark_tiled_prefill_research LANGUAGES CXX CUDA)

option(SPARK_TILED_PREFILL_ENABLE_CUDA_SMOKE
    "Build the single-GPU tiled-kernel numerical smoke test" OFF)
option(SPARK_TILED_PREFILL_ENABLE_BIDIRECTIONAL_CUDA_SMOKE
    "Build the fixed Q2048x4096 bidirectional RS/AG CUDA smoke test" OFF)
option(SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_PROXY_SMOKE
    "Build the probe-only fused Q8192/N4 mapped-host proxy smoke test" OFF)
option(SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_VERBS_PROXY
    "Build the probe-only fused Q8192/N4 real-verbs proxy" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# This project is intentionally separate from spark_transport/CMakeLists.txt.
# It compiles the research qualification ABI without linking it into the
# production transport library or any serving binary.
add_library(spark_tiled_prefill_cuda STATIC
    tiled_bulk_kernels.cu
    tiled_correctness_kernels.cu)
target_include_directories(spark_tiled_prefill_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(spark_tiled_prefill_cuda PRIVATE CUDA::cudart)
target_compile_options(spark_tiled_prefill_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if(SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_VERBS_PROXY)
    find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
    add_library(spark_fused_prefill_verbs_proxy STATIC
        fused_prefill_verbs_proxy.cpp)
    target_include_directories(spark_fused_prefill_verbs_proxy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_link_libraries(spark_fused_prefill_verbs_proxy PRIVATE
        ${IBVERBS_LIBRARY})
    target_compile_options(spark_fused_prefill_verbs_proxy PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(tp4_fused_prefill_layout_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/tp4_fused_prefill_layout_test.cpp)
    target_include_directories(tp4_fused_prefill_layout_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_options(tp4_fused_prefill_layout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_fused_prefill_layout_test
        COMMAND tp4_fused_prefill_layout_test)

    add_executable(tp4_tiled_executor_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/tp4_tiled_executor_test.cpp
        tiled_executor.cpp)
    target_include_directories(tp4_tiled_executor_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_options(tp4_tiled_executor_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_executor_test COMMAND tp4_tiled_executor_test)

    add_executable(tp4_tiled_signaled_work_gate_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/tp4_tiled_signaled_work_gate_test.cpp)
    target_include_directories(tp4_tiled_signaled_work_gate_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(tp4_tiled_signaled_work_gate_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_signaled_work_gate_test
        COMMAND tp4_tiled_signaled_work_gate_test)

    add_executable(tp4_tiled_dual_rail_edge_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/tp4_tiled_dual_rail_edge_test.cpp
        tiled_dual_rail_edge.cpp)
    target_include_directories(tp4_tiled_dual_rail_edge_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_options(tp4_tiled_dual_rail_edge_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_dual_rail_edge_test
        COMMAND tp4_tiled_dual_rail_edge_test)

    add_executable(tp4_tiled_correctness_oracle_test
        tiled_correctness_oracle_test.cpp)
    target_include_directories(tp4_tiled_correctness_oracle_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(tp4_tiled_correctness_oracle_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_correctness_oracle_test
        COMMAND tp4_tiled_correctness_oracle_test)

    if(SPARK_TILED_PREFILL_ENABLE_CUDA_SMOKE)
        add_executable(tp4_tiled_cuda_smoke_test
            tiled_cuda_smoke_test.cu)
        target_include_directories(tp4_tiled_cuda_smoke_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
        target_link_libraries(tp4_tiled_cuda_smoke_test PRIVATE
            spark_tiled_prefill_cuda CUDA::cudart)
        target_compile_options(tp4_tiled_cuda_smoke_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        add_test(NAME tp4_tiled_cuda_smoke_test
            COMMAND tp4_tiled_cuda_smoke_test)
    endif()

    if(SPARK_TILED_PREFILL_ENABLE_BIDIRECTIONAL_CUDA_SMOKE)
        add_executable(tp4_bidirectional_bulk_cuda_smoke_test
            bidirectional_bulk_cuda_smoke_test.cu
            bidirectional_bulk_kernels.cu)
        target_include_directories(tp4_bidirectional_bulk_cuda_smoke_test
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
        target_link_libraries(tp4_bidirectional_bulk_cuda_smoke_test
            PRIVATE CUDA::cudart)
        target_compile_options(tp4_bidirectional_bulk_cuda_smoke_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        add_test(NAME tp4_bidirectional_bulk_cuda_smoke_test
            COMMAND tp4_bidirectional_bulk_cuda_smoke_test)
    endif()

    if(SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_PROXY_SMOKE)
        add_executable(tp4_fused_prefill_proxy_smoke_test
            fused_prefill_proxy_smoke_test.cu
            fused_prefill_kernels.cu)
        target_include_directories(tp4_fused_prefill_proxy_smoke_test
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
        target_link_libraries(tp4_fused_prefill_proxy_smoke_test
            PRIVATE CUDA::cudart)
        target_compile_options(tp4_fused_prefill_proxy_smoke_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        add_test(NAME tp4_fused_prefill_proxy_smoke_test
            COMMAND tp4_fused_prefill_proxy_smoke_test --operations=100)
    endif()
endif()
