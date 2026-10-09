cmake_minimum_required(VERSION 3.20)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION LIBSIRCL_VERSION LIMIT_COUNT 1)
project(libsircl VERSION ${LIBSIRCL_VERSION} LANGUAGES C)
if(NOT UNIX OR APPLE)
  message(FATAL_ERROR "libsircl targets Linux. On Windows build inside WSL.")
endif()
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The prebuilt transport, fold and link packs, embedded for cuModuleLoadData and checked against their
# recorded SHA-256.
foreach(pack kernels fold links)
  set(FATBIN ${CMAKE_CURRENT_SOURCE_DIR}/kernels/prebuilt/sircl_${pack}.fatbin)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${pack}_fatbin.c
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_fatbin.py ${FATBIN}
            ${CMAKE_CURRENT_BINARY_DIR}/${pack}_fatbin.c --check-sha256 ${FATBIN}.sha256 --symbol sccl_${pack}_fatbin
    DEPENDS ${FATBIN} ${FATBIN}.sha256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_fatbin.py)
endforeach()

# SIRCL's native proxy twice: over the real libibverbs (loaded at run time) and over the shared-memory
# verbs stand-in (emulation), each with its own include path.
add_library(transport_hw OBJECT src/transport/proxy_hw.c src/transport/verbs_dl.c)
add_library(transport_emu OBJECT src/transport/proxy_emu.c src/transport/shm_verbs.c)
target_include_directories(transport_emu BEFORE PRIVATE src/transport/fake_verbs src/transport)
foreach(target transport_hw transport_emu)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wno-format-truncation)
endforeach()

add_library(sircl SHARED src/api.c src/bootstrap.c src/stubs.c src/aliases.c src/device_api.c src/fabric.c src/engine.c
            src/cuda_api.c src/kernelpack.c ${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c
            ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c
            $<TARGET_OBJECTS:transport_hw> $<TARGET_OBJECTS:transport_emu>)
target_compile_features(sircl PRIVATE c_std_11)
target_compile_definitions(sircl PUBLIC LIBSIRCL_CPU_ONLY NCCL_OS_LINUX
                           PRIVATE LIBSIRCL_VERSION="${PROJECT_VERSION}")
target_include_directories(sircl PUBLIC include PRIVATE src)
target_compile_options(sircl PRIVATE -Wall -Wextra -Werror -Wno-format-truncation)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c
                            ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c
                            PROPERTIES COMPILE_OPTIONS "-O0;-w")
target_link_libraries(sircl PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(sircl PROPERTIES NO_SONAME TRUE SUFFIX ".so.${PROJECT_VERSION}")
target_link_options(sircl PRIVATE -Wl,-soname,libnccl.so.2 -Wl,-Bsymbolic-functions -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map)

# The kernel-entry check (tests/check_entries.c): the pack loader runs on the embedded fatbins through a
# stand-in CUDA driver (tests/fake_cuda.c) that reads their cubins offline; the library links after it
# passes, so a loader that names an entry missing from a prebuilt pack fails the build.
add_library(fake_cuda SHARED tests/fake_cuda.c)
set_target_properties(fake_cuda PROPERTIES OUTPUT_NAME cuda NO_SONAME TRUE SUFFIX ".so.1" C_STANDARD 11
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fake-cuda C_VISIBILITY_PRESET hidden)
target_link_options(fake_cuda PRIVATE -Wl,-soname,libcuda.so.1)
target_compile_options(fake_cuda PRIVATE -Wall -Wextra -Werror)
add_executable(check_entries tests/check_entries.c src/kernelpack.c src/cuda_api.c
               ${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c ${CMAKE_CURRENT_BINARY_DIR}/fold_fatbin.c
               ${CMAKE_CURRENT_BINARY_DIR}/links_fatbin.c)
target_include_directories(check_entries PRIVATE include src)
target_compile_definitions(check_entries PRIVATE LIBSIRCL_CPU_ONLY NCCL_OS_LINUX)
target_link_libraries(check_entries PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/entries.ok
                   COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fake-cuda
                           $<TARGET_FILE:check_entries> 2
                   COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/entries.ok
                   DEPENDS check_entries fake_cuda)
add_custom_target(kernel_entries DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/entries.ok)
add_dependencies(sircl kernel_entries)

enable_testing()
add_executable(test_fabric tests/test_fabric.c src/fabric.c)
target_include_directories(test_fabric PRIVATE include tests)
target_compile_options(test_fabric PRIVATE -Wall -Wextra -Werror)
add_test(NAME fabric COMMAND test_fabric)
foreach(name abi api lifecycle engine_cpu)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.py --library $<TARGET_FILE:sircl>)
endforeach()
foreach(name bootstrap shm_verbs mpi_shim check_nccl_tests)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.py)
endforeach()
add_test(NAME kernel_entries COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernel_entries.py
         --build ${CMAKE_CURRENT_BINARY_DIR})
