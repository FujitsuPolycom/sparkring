cmake_minimum_required(VERSION 3.24)

project(spark_transport LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)

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
    src/tp4_session.cpp
    src/tp4_vocab_allgather_session.cpp
    src/verbs_endpoint.cpp
)

target_include_directories(spark_transport PUBLIC include)
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

include(CTest)
if(BUILD_TESTING)
    # These tests intentionally use assert() as their fail-fast contract.
    # Release builds define NDEBUG, which would otherwise compile every
    # assertion away and can also turn assertion-only values into -Werror
    # failures. Keep the tests active in every build configuration.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>")

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

    add_executable(tp4_dual_port_striped_host_test
        tests/tp4_dual_port_striped_host_test.cpp)
    target_include_directories(tp4_dual_port_striped_host_test
        PRIVATE include src)
    target_compile_options(tp4_dual_port_striped_host_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME tp4_dual_port_striped_host_test
        COMMAND tp4_dual_port_striped_host_test)

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
