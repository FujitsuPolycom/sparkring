cmake_minimum_required(VERSION 3.24)

project(spark_transport LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)

option(SPARK_TILED_PREFILL_ENABLE_BIDIRECTIONAL_CUDA_SMOKE
    "Build the fixed Q2048x4096 bidirectional RS/AG CUDA smoke test" OFF)
option(SPARK_TP4_ENABLE_FUSED_STREAM_SWITCH_SMOKE
    "Build the fused-session alternating caller-stream smoke test" OFF)
option(SPARK_TP4_ENABLE_FUSED_RING_OCCUPANCY_PROBE
    "Build the Q8192 fused-ring kernel occupancy probe" OFF)
option(SPARK_TP4_ENABLE_FUSED_PREFILL_PROBE
    "Build the research-only four-node Q8192 fused verbs probe" OFF)

add_library(spark_transport
    src/control_channel.cpp
    src/gpu_doorbell.cu
    src/gpu_tp4.cu
    src/gpu_tp4_tensor.cu
    src/gpu_tp4_vocab_allgather.cu
    src/memory_buffer.cu
    src/statistics.cpp
    src/topology.cpp
    src/tp4_schedule.cpp
    src/tp4_graph_command.cpp
    src/tp4_bidirectional_prefill_session.cpp
    src/tp4_fused_prefill_session.cpp
    src/tp4_session.cpp
    src/tp4_vocab_allgather_session.cpp
    src/verbs_endpoint.cpp
    experiments/tiled_prefill/bidirectional_ring_executor.cpp
    experiments/tiled_prefill/bidirectional_bulk_kernels.cu
    experiments/tiled_prefill/fused_prefill_kernels.cu
    experiments/tiled_prefill/fused_prefill_verbs_proxy.cpp
)

target_include_directories(spark_transport PUBLIC include)
target_include_directories(spark_transport PRIVATE
    experiments/tiled_prefill)
target_link_libraries(spark_transport PUBLIC CUDA::cudart ${IBVERBS_LIBRARY})
target_compile_options(spark_transport PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(spark_transport PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(spark_transport_capi SHARED
    src/tp4_bidirectional_prefill_c_api.cpp
    src/tp4_fused_prefill_c_api.cpp
    src/tp4_c_api.cpp
    src/tp4_vocab_allgather_c_api.cpp)
target_include_directories(spark_transport_capi PUBLIC include)
target_link_libraries(spark_transport_capi PRIVATE spark_transport)

add_executable(spark_transport_probe app/transport_probe.cpp)
target_link_libraries(spark_transport_probe PRIVATE spark_transport)


add_executable(spark_tp4_probe app/tp4_probe.cpp)
target_link_libraries(spark_tp4_probe PRIVATE spark_transport)

add_executable(spark_tp4_tensor_probe app/tp4_tensor_probe.cu)
target_link_libraries(spark_tp4_tensor_probe PRIVATE spark_transport)

add_executable(spark_tp4_graph_q1_probe app/tp4_graph_q1_probe.cu)
target_link_libraries(spark_tp4_graph_q1_probe PRIVATE spark_transport)


add_executable(spark_tp4_vocab_allgather_probe
    app/tp4_vocab_allgather_probe.cu)
target_link_libraries(spark_tp4_vocab_allgather_probe
    PRIVATE spark_transport_capi spark_transport)

add_executable(spark_tp4_vocab_graph_probe
    app/tp4_vocab_graph_probe.cu)
target_link_libraries(spark_tp4_vocab_graph_probe
    PRIVATE spark_transport_capi spark_transport)

# Research-only standalone tiled-prefill qualification probe. This target is
# intentionally excluded from the production library and C ABI.
add_executable(spark_tp4_tiled_prefill_probe
    app/tp4_tiled_prefill_probe.cu
    experiments/tiled_prefill/tiled_executor.cpp
    experiments/tiled_prefill/tiled_bulk_kernels.cu
    experiments/tiled_prefill/tiled_correctness_kernels.cu)
target_include_directories(spark_tp4_tiled_prefill_probe PRIVATE
    include experiments/tiled_prefill)
target_link_libraries(spark_tp4_tiled_prefill_probe PRIVATE spark_transport)
target_compile_options(spark_tp4_tiled_prefill_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Research-only fixed Q2048x4096 bidirectional-ring hardware probe. It owns
# standalone verbs/CUDA resources and is not linked into the C ABI or vLLM.
add_executable(spark_tp4_bidirectional_prefill_probe
    app/tp4_bidirectional_prefill_probe.cu)
target_include_directories(spark_tp4_bidirectional_prefill_probe PRIVATE
    include experiments/tiled_prefill)
target_link_libraries(spark_tp4_bidirectional_prefill_probe
    PRIVATE spark_transport)
target_compile_options(spark_tp4_bidirectional_prefill_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if(SPARK_TP4_ENABLE_FUSED_PREFILL_PROBE)
    add_executable(spark_tp4_fused_prefill_probe
        app/tp4_fused_prefill_probe.cu
        experiments/tiled_prefill/fused_prefill_kernels.cu
        experiments/tiled_prefill/fused_prefill_verbs_proxy.cpp)
    target_include_directories(spark_tp4_fused_prefill_probe PRIVATE
        include experiments/tiled_prefill)
    target_link_libraries(spark_tp4_fused_prefill_probe PRIVATE
        spark_transport CUDA::cudart)
    target_compile_options(spark_tp4_fused_prefill_probe PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

include(CTest)
if(BUILD_TESTING)
    # These tests intentionally use assert() as their fail-fast contract.
    # Release builds define NDEBUG, which would otherwise compile every
    # assertion away and can also turn assertion-only values into -Werror
    # failures. Keep the tests active in every build configuration.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>")

    if(SPARK_TP4_ENABLE_FUSED_STREAM_SWITCH_SMOKE)
        add_executable(tp4_fused_stream_switch_smoke
            app/tp4_fused_stream_switch_smoke.cu)
        target_link_libraries(tp4_fused_stream_switch_smoke PRIVATE CUDA::cudart)
        add_test(NAME tp4_fused_stream_switch_smoke
            COMMAND tp4_fused_stream_switch_smoke)
    endif()

    if(SPARK_TP4_ENABLE_FUSED_RING_OCCUPANCY_PROBE)
        add_executable(tp4_fused_ring_occupancy_probe
            app/tp4_fused_ring_occupancy_probe.cu)
        target_link_libraries(tp4_fused_ring_occupancy_probe PRIVATE
            CUDA::cudart)
        add_test(NAME tp4_fused_ring_occupancy_probe
            COMMAND tp4_fused_ring_occupancy_probe)
    endif()

    add_executable(statistics_test tests/statistics_test.cpp)
    target_link_libraries(statistics_test PRIVATE spark_transport)
    add_test(NAME statistics_test COMMAND statistics_test)

    add_executable(wire_protocol_test tests/wire_protocol_test.cpp)
    target_link_libraries(wire_protocol_test PRIVATE spark_transport)
    add_test(NAME wire_protocol_test COMMAND wire_protocol_test)

    add_executable(verbs_completion_test
        tests/verbs_completion_test.cpp)
    target_include_directories(verbs_completion_test PRIVATE
        tests/syntax_stubs include)
    target_compile_options(verbs_completion_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME verbs_completion_test COMMAND verbs_completion_test)

    add_executable(topology_test tests/topology_test.cpp)
    target_link_libraries(topology_test PRIVATE spark_transport)
    add_test(NAME topology_test COMMAND topology_test)

    add_executable(gpu_doorbell_layout_test tests/gpu_doorbell_layout_test.cpp)
    target_link_libraries(gpu_doorbell_layout_test PRIVATE spark_transport)
    add_test(NAME gpu_doorbell_layout_test COMMAND gpu_doorbell_layout_test)

    add_executable(graph_poll_policy_test tests/graph_poll_policy_test.cpp)
    target_include_directories(graph_poll_policy_test PRIVATE include)
    target_compile_options(graph_poll_policy_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME graph_poll_policy_test COMMAND graph_poll_policy_test)


    add_executable(tp4_schedule_test tests/tp4_schedule_test.cpp)
    target_link_libraries(tp4_schedule_test PRIVATE spark_transport)
    add_test(NAME tp4_schedule_test COMMAND tp4_schedule_test)

    add_executable(tp4_allreduce_protocol_test
        tests/tp4_allreduce_protocol_test.cpp)
    target_include_directories(tp4_allreduce_protocol_test PRIVATE include)
    target_compile_options(tp4_allreduce_protocol_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_allreduce_protocol_test
        COMMAND tp4_allreduce_protocol_test)

    add_executable(tp4_graph_kernel_strategy_test
        tests/tp4_graph_kernel_strategy_test.cpp)
    target_include_directories(tp4_graph_kernel_strategy_test PRIVATE include)
    target_compile_options(tp4_graph_kernel_strategy_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_graph_kernel_strategy_test
        COMMAND tp4_graph_kernel_strategy_test)

    add_executable(tp4_dual_port_striped_allreduce_test
        tests/tp4_dual_port_striped_allreduce_test.cpp)
    target_include_directories(tp4_dual_port_striped_allreduce_test
        PRIVATE include)
    target_compile_options(tp4_dual_port_striped_allreduce_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_dual_port_striped_allreduce_test
        COMMAND tp4_dual_port_striped_allreduce_test)

    add_executable(tp4_bidirectional_prefill_test
        tests/tp4_bidirectional_prefill_test.cpp)
    target_include_directories(tp4_bidirectional_prefill_test PRIVATE include)
    target_compile_options(tp4_bidirectional_prefill_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_bidirectional_prefill_test
        COMMAND tp4_bidirectional_prefill_test)

    add_executable(tp4_fused_ring_contract_test
        tests/tp4_fused_ring_contract_test.cpp)
    target_include_directories(tp4_fused_ring_contract_test PRIVATE include)
    target_compile_options(tp4_fused_ring_contract_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_fused_ring_contract_test
        COMMAND tp4_fused_ring_contract_test)

    add_executable(tp4_bidirectional_ring_executor_test
        tests/tp4_bidirectional_ring_executor_test.cpp
        experiments/tiled_prefill/bidirectional_ring_executor.cpp)
    target_include_directories(tp4_bidirectional_ring_executor_test PRIVATE
        include experiments/tiled_prefill)
    target_compile_options(tp4_bidirectional_ring_executor_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_bidirectional_ring_executor_test
        COMMAND tp4_bidirectional_ring_executor_test)

    add_executable(tp4_bidirectional_prefill_c_api_layout_test
        tests/tp4_bidirectional_prefill_c_api_layout_test.cpp)
    target_include_directories(tp4_bidirectional_prefill_c_api_layout_test
        PRIVATE include)
    target_compile_options(tp4_bidirectional_prefill_c_api_layout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_bidirectional_prefill_c_api_layout_test
        COMMAND tp4_bidirectional_prefill_c_api_layout_test)

    add_executable(tp4_fused_prefill_c_api_layout_test
        tests/tp4_fused_prefill_c_api_layout_test.cpp)
    target_include_directories(tp4_fused_prefill_c_api_layout_test PRIVATE
        include)
    target_compile_options(tp4_fused_prefill_c_api_layout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_fused_prefill_c_api_layout_test
        COMMAND tp4_fused_prefill_c_api_layout_test)

    add_executable(tp4_dual_port_striped_host_test
        tests/tp4_dual_port_striped_host_test.cpp)
    target_include_directories(tp4_dual_port_striped_host_test
        PRIVATE include src)
    target_compile_options(tp4_dual_port_striped_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_dual_port_striped_host_test
        COMMAND tp4_dual_port_striped_host_test)

    add_executable(tp4_tiled_session_test
        tests/tp4_tiled_session_test.cpp)
    target_include_directories(tp4_tiled_session_test PRIVATE include)
    target_compile_options(tp4_tiled_session_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_tiled_session_test COMMAND tp4_tiled_session_test)

    add_executable(tp4_tiled_executor_test
        tests/tp4_tiled_executor_test.cpp
        experiments/tiled_prefill/tiled_executor.cpp)
    target_include_directories(tp4_tiled_executor_test PRIVATE
        include experiments/tiled_prefill)
    target_compile_options(tp4_tiled_executor_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_executor_test COMMAND tp4_tiled_executor_test)

    add_executable(tp4_tiled_signaled_work_gate_test
        tests/tp4_tiled_signaled_work_gate_test.cpp)
    target_include_directories(tp4_tiled_signaled_work_gate_test PRIVATE
        experiments/tiled_prefill)
    target_compile_options(tp4_tiled_signaled_work_gate_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_signaled_work_gate_test
        COMMAND tp4_tiled_signaled_work_gate_test)

    add_executable(tp4_tiled_dual_rail_edge_test
        tests/tp4_tiled_dual_rail_edge_test.cpp
        experiments/tiled_prefill/tiled_dual_rail_edge.cpp)
    target_include_directories(tp4_tiled_dual_rail_edge_test PRIVATE
        include experiments/tiled_prefill)
    target_compile_options(tp4_tiled_dual_rail_edge_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME tp4_tiled_dual_rail_edge_test
        COMMAND tp4_tiled_dual_rail_edge_test)

    if(SPARK_TILED_PREFILL_ENABLE_BIDIRECTIONAL_CUDA_SMOKE)
        add_executable(tp4_bidirectional_bulk_cuda_smoke_test
            experiments/tiled_prefill/bidirectional_bulk_cuda_smoke_test.cu
            experiments/tiled_prefill/bidirectional_bulk_kernels.cu)
        target_include_directories(tp4_bidirectional_bulk_cuda_smoke_test
            PRIVATE include experiments/tiled_prefill)
        target_link_libraries(tp4_bidirectional_bulk_cuda_smoke_test
            PRIVATE CUDA::cudart)
        target_compile_options(tp4_bidirectional_bulk_cuda_smoke_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        add_test(NAME tp4_bidirectional_bulk_cuda_smoke_test
            COMMAND tp4_bidirectional_bulk_cuda_smoke_test)
    endif()

    add_executable(tp4_c_api_layout_test tests/tp4_c_api_layout_test.cpp)
    target_include_directories(tp4_c_api_layout_test PRIVATE include)
    target_compile_options(tp4_c_api_layout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_c_api_layout_test COMMAND tp4_c_api_layout_test)

    add_executable(tp4_graph_command_test
        tests/tp4_graph_command_test.cpp
        src/tp4_graph_command.cpp)
    target_include_directories(tp4_graph_command_test PRIVATE include)
    target_compile_options(tp4_graph_command_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_graph_command_test COMMAND tp4_graph_command_test)


    add_executable(eager_staging_timeout_test
        tests/eager_staging_timeout_test.cpp)
    target_include_directories(eager_staging_timeout_test PRIVATE include)
    target_compile_options(eager_staging_timeout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME eager_staging_timeout_test
        COMMAND eager_staging_timeout_test)

    add_executable(cuda_event_gate_test
        tests/cuda_event_gate_test.cu)
    target_include_directories(cuda_event_gate_test PRIVATE src)
    target_link_libraries(cuda_event_gate_test PRIVATE CUDA::cudart)
    add_test(NAME cuda_event_gate_test
        COMMAND cuda_event_gate_test)


    add_executable(tp4_c_api_test tests/tp4_c_api_test.cpp)
    target_link_libraries(tp4_c_api_test PRIVATE spark_transport_capi)
    add_test(NAME tp4_c_api_test COMMAND tp4_c_api_test)


    add_executable(tp4_vocab_allgather_layout_test
        tests/tp4_vocab_allgather_layout_test.cpp)
    target_link_libraries(tp4_vocab_allgather_layout_test
        PRIVATE spark_transport)
    add_test(NAME tp4_vocab_allgather_layout_test
        COMMAND tp4_vocab_allgather_layout_test)

    add_executable(tp4_vocab_allgather_c_api_test
        tests/tp4_vocab_allgather_c_api_test.cpp)
    target_link_libraries(tp4_vocab_allgather_c_api_test
        PRIVATE spark_transport_capi)
    add_test(NAME tp4_vocab_allgather_c_api_test
        COMMAND tp4_vocab_allgather_c_api_test)
endif()
