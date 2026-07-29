cmake_minimum_required(VERSION 3.24)

project(spark_cache_placement LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spark_cache_placement_layout STATIC
    src/spark_cache_placement_parse.cpp)
target_compile_definitions(spark_cache_placement_layout
    PRIVATE SPARK_CACHE_PLACEMENT_BUILD=1)
target_include_directories(spark_cache_placement_layout PUBLIC include)
target_compile_options(spark_cache_placement_layout PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
set_target_properties(spark_cache_placement_layout PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

add_library(spark_cache_snapshot_layout STATIC
    src/spark_cache_snapshot_layout.cpp)
target_compile_definitions(spark_cache_snapshot_layout
    PRIVATE SPARK_CACHE_SNAPSHOT_BUILD=1)
target_include_directories(spark_cache_snapshot_layout PUBLIC include)
target_compile_options(spark_cache_snapshot_layout PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
set_target_properties(spark_cache_snapshot_layout PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

include(CTest)
if(BUILD_TESTING)
    add_executable(spark_cache_placement_layout_test tests/layout_test.cpp)
    target_link_libraries(spark_cache_placement_layout_test
        PRIVATE spark_cache_placement_layout)
    target_compile_options(spark_cache_placement_layout_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(
        NAME spark_cache_placement_layout_test
        COMMAND spark_cache_placement_layout_test)

    add_executable(spark_cache_snapshot_ring_test
        tests/snapshot_ring_test.cpp)
    target_link_libraries(spark_cache_snapshot_ring_test
        PRIVATE spark_cache_snapshot_layout)
    target_compile_options(spark_cache_snapshot_ring_test PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(
        NAME spark_cache_snapshot_ring_test
        COMMAND spark_cache_snapshot_ring_test)
endif()

option(
    SPARK_CACHE_PLACEMENT_ENABLE_CUDA
    "Build the mapped/managed/staged CUDA placement engine"
    ON)
option(
    SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE
    "Test seam: force every snapshot completion-event record to fail"
    OFF)
option(
    SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE
    "Test seam: force every quarantined producer-stream drain to fail"
    OFF)

if(SPARK_CACHE_PLACEMENT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_library(spark_cache_placement SHARED
            src/spark_cache_placement.cu
            src/spark_cache_placement_parse.cpp)
        target_compile_definitions(spark_cache_placement
            PRIVATE SPARK_CACHE_PLACEMENT_BUILD=1)
        target_include_directories(spark_cache_placement PUBLIC include)
        target_link_libraries(spark_cache_placement PRIVATE CUDA::cudart)
        target_compile_options(spark_cache_placement PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr;
                # nvcc 13 emits GCC-extension line directives in generated
                # cudafe sources. Keep warnings fatal, but do not apply
                # -Wpedantic to nvcc-generated host code.
                -Xcompiler=-Wall,-Wextra,-Werror>)
        set_target_properties(spark_cache_placement PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)

        add_executable(
            spark_cache_native_probe
            app/spark_cache_native_probe.cu)
        target_link_libraries(spark_cache_native_probe
            PRIVATE spark_cache_placement CUDA::cudart)

        add_library(spark_cache_snapshot SHARED
            src/spark_cache_snapshot.cu)
        target_compile_definitions(spark_cache_snapshot
            PRIVATE SPARK_CACHE_SNAPSHOT_BUILD=1)
        if(SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE)
            target_compile_definitions(spark_cache_snapshot PRIVATE
                SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE=1)
        endif()
        if(SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE)
            target_compile_definitions(spark_cache_snapshot PRIVATE
                SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE=1)
        endif()
        target_include_directories(spark_cache_snapshot PUBLIC include)
        target_link_libraries(spark_cache_snapshot
            PRIVATE spark_cache_snapshot_layout CUDA::cudart)
        target_compile_options(spark_cache_snapshot PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr;
                -Xcompiler=-Wall,-Wextra,-Werror>)
        set_target_properties(spark_cache_snapshot PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)

        add_executable(
            spark_cache_snapshot_probe
            app/spark_cache_snapshot_probe.cu)
        target_link_libraries(spark_cache_snapshot_probe
            PRIVATE spark_cache_snapshot CUDA::cudart)

        add_executable(
            spark_cache_snapshot_matrix_probe
            app/spark_cache_snapshot_matrix_probe.cu)
        target_link_libraries(spark_cache_snapshot_matrix_probe
            PRIVATE spark_cache_snapshot CUDA::cudart)
        target_compile_options(spark_cache_snapshot_matrix_probe PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr;
                -Xcompiler=-Wall,-Wextra,-Werror>)
    else()
        message(STATUS
            "CUDA compiler unavailable: building layout/parser tests only")
    endif()
endif()
