cmake_minimum_required(VERSION 3.20)

# Allow VERSION for projects, as expected in CMake 3.0+
cmake_policy(SET CMP0048 NEW)
project(rocMLIR VERSION 2.0.0 LANGUAGES CXX C)

# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "")

# Adapted from https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set policy CMP0057 to support IN_LIST operators
cmake_policy(SET CMP0057 NEW)

#===----------------------------------------------------------------------===//
# Parallel build configuration
#===----------------------------------------------------------------------===//

set(ROCMLIR_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(ROCMLIR_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${ROCMLIR_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
elseif(ROCMLIR_PARALLEL_LINK_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

set(ROCMLIR_PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compile jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(ROCMLIR_PARALLEL_COMPILE_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${ROCMLIR_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
elseif(ROCMLIR_PARALLEL_COMPILE_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

#===----------------------------------------------------------------------===//
# rocMLIR Options
#===----------------------------------------------------------------------===//

# Enable tests by default so that check-rocmlir-build-only target is available
set(MLIR_INCLUDE_TESTS ON CACHE BOOL "Include MLIR tests")
# Disable E2E tests by default since they require JIT runner
set(ROCMLIR_DRIVER_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build E2E tests for Rock driver")
set(ROCMLIR_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(ROCMLIR_GEN_FLAGS "" CACHE BOOL "Set feature flag for rocmlir-gen")
set(ROCMLIR_DRIVER_TEST_GPU_VALIDATION 1 CACHE BOOL "Enable E2E tests with GPU validation")
# Disable E2E tests by default since they require JIT runner
set(ROCK_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build rock E2E tests")
set(ROCMLIR_ENABLE_BENCHMARKS "" CACHE STRING "List of enabled benchmarks")

set(ROCMLIR_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(ROCMLIR_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "ROCMLIR_BIN_DIR: ${ROCMLIR_BIN_DIR}")

if(ROCMLIR_ENABLE_BENCHMARKS STREQUAL "all")
  set(ROCMLIR_ENABLE_BENCHMARKS CACHE STRING "ck;hipblaslt" FORCE)
endif()

#===----------------------------------------------------------------------===//
# Library type and linkage settings
#===----------------------------------------------------------------------===//

if(BUILD_MIXR_TARGET)
  set(BUILD_FAT_LIBROCKCOMPILER ON CACHE BOOL "Build static rockCompiler library")
endif()

if(NOT DEFINED BUILD_FAT_LIBROCKCOMPILER)
  set(BUILD_FAT_LIBROCKCOMPILER OFF CACHE BOOL "Build fat rockCompiler library to link into Rock driver")
endif()

if(BUILD_FAT_LIBROCKCOMPILER)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build PR-triggered E2E tests for Rock driver")
  if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")
    list(APPEND CMAKE_IGNORE_PATH /opt/conda)
  endif()
else()
  set(BUILD_SHARED_LIBS ON CACHE BOOL "")
  set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build PR-triggered E2E tests for Rock driver")
endif()

#===----------------------------------------------------------------------===//
# Triton paths for RPATH
#===----------------------------------------------------------------------===//

set(TRITON_EXTERNAL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")
set(TRITON_EXTERNAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_EXTERNAL_BIN_DIR "${TRITON_EXTERNAL_BUILD_DIR}/bin")
set(TRITON_EXTERNAL_LIB_DIR "${TRITON_EXTERNAL_BUILD_DIR}/lib")

message(STATUS "TRITON_EXTERNAL_BIN_DIR: ${TRITON_EXTERNAL_BIN_DIR}")
message(STATUS "TRITON_EXTERNAL_LIB_DIR: ${TRITON_EXTERNAL_LIB_DIR}")

#===----------------------------------------------------------------------===//
# Address sanitizer workaround
#===----------------------------------------------------------------------===//

if(DEFINED ENV{ADDRESS_SANITIZER})
  if($ENV{ADDRESS_SANITIZER} MATCHES "ON")
    message("Address sanitizer workarounds")
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
      OUTPUT_VARIABLE clang_asan_lib_file
      ERROR_VARIABLE clang_stderr
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE clang_exit_code)
    if(NOT "${clang_exit_code}" STREQUAL "0")
      message(FATAL_ERROR
      "Unable to invoke clang to find asan lib dir: ${clang_stderr}")
    endif()
    file(TO_CMAKE_PATH "${clang_asan_lib_file}" clang_asan_lib_file_cmake)
    get_filename_component(clang_asan_lib_dir "${clang_asan_lib_file_cmake}" DIRECTORY)
    message(STATUS "Asan lib directory ${clang_asan_lib_dir}")
    list(APPEND CMAKE_BUILD_RPATH "${clang_asan_lib_dir}")
  endif()
endif()

#===----------------------------------------------------------------------===//
# Set up Triton (includes LLVM/MLIR via find_package)
# NOTE: MLIR_DIR must be set to a pre-built LLVM/MLIR installation
#===----------------------------------------------------------------------===//

include(cmake/triton.cmake)

# TODO: Re-enable mlir-hal when updated for newer LLVM (22+)
# mlir-hal has API incompatibilities with LLVM 22 (getArgAttrsAttr, etc.)
# include(cmake/mlir-hal.cmake)

#===----------------------------------------------------------------------===//
# RPATH configuration
#===----------------------------------------------------------------------===//

if(NOT WIN32)
  # Add library paths to RPATH
  list(PREPEND CMAKE_BUILD_RPATH "${TRITON_EXTERNAL_LIB_DIR}")
    list(PREPEND CMAKE_BUILD_RPATH "${ROCMLIR_LIB_DIR}")
  # Also add LLVM library dir if available
  if(DEFINED LLVM_LIBRARY_DIR)
    list(PREPEND CMAKE_BUILD_RPATH "${LLVM_LIBRARY_DIR}")
  endif()
    message(STATUS "Final CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
endif()

#===----------------------------------------------------------------------===//
# Build rocMLIR dialects
#===----------------------------------------------------------------------===//

add_subdirectory(mlir)
