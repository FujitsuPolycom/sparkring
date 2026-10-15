cmake_minimum_required(VERSION 3.20)
# Optimized with debug information unless a build type is given, as the Makefile's -O2 -g.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "CMake build type" FORCE)
endif()
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION LIBSIRCL_VERSION LIMIT_COUNT 1)
project(libsircl VERSION ${LIBSIRCL_VERSION} LANGUAGES C)
if(NOT UNIX OR APPLE)
  message(FATAL_ERROR "libsircl targets Linux. On Windows build inside WSL.")
endif()
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The transport, fold, link and point-to-point packs: nvcc (LIBSIRCL_NVCC, CUDA 13.3 or later for sm_121)
# builds each fatbin from its CUDA C++ source with the Makefile's flags, and the library embeds it for
# cuModuleLoadData.
find_program(LIBSIRCL_NVCC NAMES nvcc HINTS ENV CUDA_HOME PATH_SUFFIXES bin PATHS /usr/local/cuda/bin)
if(NOT LIBSIRCL_NVCC)
  message(FATAL_ERROR "libsircl's kernel packs need nvcc (CUDA 13.3 or later): set LIBSIRCL_NVCC")
endif()
set(LIBSIRCL_NVCC_EXTRA_FLAGS "" CACHE STRING "flags added to nvcc's (for example -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK)")
set(LIBSIRCL_NVCCFLAGS -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121)
# The architectures each pack holds, for the kernel-entry check: the number of -gencode flags.
set(LIBSIRCL_ARCHITECTURES 0)
foreach(flag ${LIBSIRCL_NVCCFLAGS})
  if(flag STREQUAL "-gencode")
    math(EXPR LIBSIRCL_ARCHITECTURES "${LIBSIRCL_ARCHITECTURES} + 1")
  endif()
endforeach()
separate_arguments(LIBSIRCL_NVCC_EXTRA UNIX_COMMAND "${LIBSIRCL_NVCC_EXTRA_FLAGS}")
file(GLOB LIBSIRCL_KERNEL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cuh)
foreach(pack kernels fold links p2p)
  set(FATBIN ${CMAKE_CURRENT_BINARY_DIR}/packs/sircl_${pack}.fatbin)
  add_custom_command(
    OUTPUT ${FATBIN}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/packs
    COMMAND ${LIBSIRCL_NVCC} ${LIBSIRCL_NVCC_EXTRA} ${LIBSIRCL_NVCCFLAGS} -fatbin -o ${FATBIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/sircl_${pack}.cu
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/sircl_${pack}.cu ${LIBSIRCL_KERNEL_HEADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${pack}_fatbin.c
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_fatbin.py ${FATBIN}
            ${CMAKE_CURRENT_BINARY_DIR}/${pack}_fatbin.c --symbol sccl_${pack}_fatbin
    DEPENDS ${FATBIN} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_fatbin.py)
endforeach()

# The engine's setup check of the library's local feature word (p2p_local_features bit 0: p2p_destroy counts
# failed verbs calls; SIRCL change LF), on by default as in the Makefile (P2P_FEATURES).
option(LIBSIRCL_P2P_FEATURES "check the point-to-point library's local feature word at setup" ON)
# SIRCL's native proxy and point-to-point library are vendored byte-identical (LF, SHA-256 recorded beside
# each, with change LF's feature words): checked at configuration and again before every build of the library.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_vendored.cmake)
add_custom_target(vendored_sources_guard
                  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                          -DP2P_FEATURES=${LIBSIRCL_P2P_FEATURES} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_vendored.cmake
                  COMMENT "checking the vendored SIRCL native sources")

# Hardening of every object and of the library, as in the Makefile: stack protector, fortified libc calls,
# full RELRO with immediate binding, and a non-executable stack.
# _FORTIFY_SOURCE needs an optimized build, so a Debug build (-O0) goes without it.
set(LIBSIRCL_HARDENING_COMPILE -fstack-protector-strong $<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE>
                               $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>)
set(LIBSIRCL_HARDENING_LINK -Wl,-z,relro,-z,now,-z,noexecstack)

# SIRCL's native proxy and point-to-point library twice: over the real libibverbs (loaded at run time) and
# over the shared-memory verbs stand-in (emulation), each with its own include path.
add_library(transport_hw OBJECT src/transport/proxy_hw.c src/transport/p2p_hw.c src/transport/verbs_dl.c)
add_library(transport_emu OBJECT src/transport/proxy_emu.c src/transport/p2p_emu.c src/transport/shm_verbs.c)
target_include_directories(transport_emu BEFORE PRIVATE src/transport/fake_verbs src/transport)
foreach(target transport_hw transport_emu)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wno-format-truncation ${LIBSIRCL_HARDENING_COMPILE})
  add_dependencies(${target} vendored_sources_guard)
endforeach()

add_library(sircl SHARED src/api.c src/bootstrap.c src/stubs.c src/aliases.c src/device_api.c src/fabric.c src/engine.c
            src/cuda_api.c src/kernelpack.c ${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c
            ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c
            ${CMAKE_CURRENT_BINARY_DIR}/p2p_fatbin.c $<TARGET_OBJECTS:transport_hw> $<TARGET_OBJECTS:transport_emu>)
target_compile_features(sircl PRIVATE c_std_11)
target_compile_definitions(sircl PUBLIC LIBSIRCL_CPU_ONLY NCCL_OS_LINUX
                           PRIVATE LIBSIRCL_VERSION="${PROJECT_VERSION}")
if(LIBSIRCL_P2P_FEATURES)
  target_compile_definitions(sircl PRIVATE SCCL_P2P_LOCAL_FEATURES)
endif()
target_include_directories(sircl PUBLIC include PRIVATE src)
target_compile_options(sircl PRIVATE -Wall -Wextra -Werror -Wno-format-truncation ${LIBSIRCL_HARDENING_COMPILE})
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c
                            ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/p2p_fatbin.c
                            PROPERTIES COMPILE_OPTIONS "-O0;-w")
target_link_libraries(sircl PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(sircl PROPERTIES NO_SONAME TRUE SUFFIX ".so.${PROJECT_VERSION}")
target_link_options(sircl PRIVATE -Wl,-soname,libnccl.so.2 -Wl,-Bsymbolic-functions -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map
                    ${LIBSIRCL_HARDENING_LINK})
add_dependencies(sircl vendored_sources_guard)

# The kernel-entry check (tests/check_entries.c): the pack loader runs on the embedded fatbins through a
# stand-in CUDA driver (tests/fake_cuda.c) that reads their cubins offline; the library links after it
# passes, so a loader that names an entry missing from a pack fails the build.
add_library(fake_cuda SHARED tests/fake_cuda.c)
set_target_properties(fake_cuda PROPERTIES OUTPUT_NAME cuda NO_SONAME TRUE SUFFIX ".so.1" C_STANDARD 11
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fake-cuda C_VISIBILITY_PRESET hidden)
target_link_options(fake_cuda PRIVATE -Wl,-soname,libcuda.so.1)
target_compile_options(fake_cuda PRIVATE -Wall -Wextra -Werror)
add_executable(check_entries tests/check_entries.c src/kernelpack.c src/cuda_api.c
               ${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c
               ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/p2p_fatbin.c)
target_include_directories(check_entries PRIVATE include src)
target_compile_definitions(check_entries PRIVATE LIBSIRCL_CPU_ONLY NCCL_OS_LINUX)
target_link_libraries(check_entries PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/entries.ok
                   COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fake-cuda
                           $<TARGET_FILE:check_entries> ${LIBSIRCL_ARCHITECTURES}
                   COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/entries.ok
                   DEPENDS check_entries fake_cuda)
add_custom_target(kernel_entries DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/entries.ok)
add_dependencies(sircl kernel_entries)

enable_testing()
add_executable(test_fabric tests/test_fabric.c src/fabric.c)
target_include_directories(test_fabric PRIVATE include tests)
target_compile_options(test_fabric PRIVATE -Wall -Wextra -Werror)
add_test(NAME fabric COMMAND test_fabric)
add_executable(test_link_plan tests/test_link_plan.c)
target_include_directories(test_link_plan PRIVATE src)
target_compile_options(test_link_plan PRIVATE -Wall -Wextra -Werror)
add_test(NAME link_plan COMMAND test_link_plan)
add_executable(test_kernelpack_cache tests/test_kernelpack_cache.c src/kernelpack.c src/cuda_api.c
               ${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c
               ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/p2p_fatbin.c)
target_include_directories(test_kernelpack_cache PRIVATE include src)
target_compile_definitions(test_kernelpack_cache PRIVATE LIBSIRCL_CPU_ONLY NCCL_OS_LINUX)
target_link_libraries(test_kernelpack_cache PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
add_test(NAME kernelpack_cache COMMAND ${CMAKE_COMMAND} -E env
         LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fake-cuda $<TARGET_FILE:test_kernelpack_cache>)
foreach(name abi api lifecycle engine_cpu build_flags)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.py --library $<TARGET_FILE:sircl>)
endforeach()
foreach(name bootstrap shm_verbs vendored_sources p2p_native site_routes mpi_shim check_nccl_tests)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.py)
endforeach()
add_test(NAME kernel_entries COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernel_entries.py
         --build ${CMAKE_CURRENT_BINARY_DIR})
