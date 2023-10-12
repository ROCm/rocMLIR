cmake_minimum_required(VERSION 3.15.1)

# Allow VERSION for projects, as expected in CMake 3.0+
cmake_policy(SET CMP0048 NEW)
project(rocMLIR VERSION 2.0.0 LANGUAGES CXX C)

# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "")

# Adapted from https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set policy CMP0057 to support IN_LIST operators
cmake_policy(SET CMP0057 NEW)

# Rock dialect.
set(ROCMLIR_DRIVER_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for Rock driver")
set(ROCMLIR_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(ROCMLIR_GEN_FLAGS "" CACHE BOOL "Set feature flag for rocmlir-gen")
set(ROCMLIR_DRIVER_TEST_GPU_VALIDATION 1 CACHE BOOL "Enable E2E tests with GPU validation")
set(ROCK_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build rock E2E tests")
set(ROCMLIR_ENABLE_BENCHMARKS "" CACHE STRING "List of enabled benchmarks")

set(ROCK_E2E_TEST_SUITES "" CACHE STRING "Enable partial rock E2E tests")

set(ROCMLIR_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(ROCMLIR_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "ROCMLIR_BIN_DIR: ${ROCMLIR_BIN_DIR}")

if( ROCMLIR_ENABLE_BENCHMARKS STREQUAL "all" )
  set(ROCMLIR_ENABLE_BENCHMARKS CACHE STRING "ck;rocblas" FORCE)
endif()

# BUILD_MIXR_TARGET is deprecated but still available
if(BUILD_MIXR_TARGET)
  set(BUILD_FAT_LIBROCKCOMPILER ON CACHE BOOL "Build static rockCompiler library")
endif()

# Library type and linkage settings
if( NOT DEFINED BUILD_FAT_LIBROCKCOMPILER )
  set(BUILD_FAT_LIBROCKCOMPILER OFF CACHE BOOL "Build fat rockCompiler library to link into Rock driver")
endif()
if( BUILD_FAT_LIBROCKCOMPILER )
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
  # rocm-runner is not supported with static libraries
  set(MLIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "")
  set(MLIR_INCLUDE_INTEGRATION_TESTS OFF CACHE BOOL "")
  set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build PR-triggered E2E tests for Rock driver")
  set(MHAL_ENABLE_HOST_RUNNER OFF CACHE BOOL "Enable MHAL host runner")
  if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")
    # Note, this is a hack to ignore Pytorch added conda path
    list(APPEND CMAKE_IGNORE_PATH /opt/conda)
  endif()
else()
  set(BUILD_SHARED_LIBS ON CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
  set(LLVM_BUILD_EXAMPLES ON CACHE BOOL "")
  set(MLIR_ENABLE_ROCM_RUNNER 1 CACHE BOOL "")
  set(MLIR_INCLUDE_INTEGRATION_TESTS ON CACHE BOOL "")
  set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED 1 CACHE BOOL "Enable build PR-triggered E2E tests for Rock driver")
endif()

# Pointers to external llvm build, needed here to set up rpath project-wide
set(LLVM_EXTERNAL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project")
set(LLVM_EXTERNAL_BIN_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/bin")
set(LLVM_EXTERNAL_LIB_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/lib")

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "LLVM_EXTERNAL_LIB_DIR: ${LLVM_EXTERNAL_LIB_DIR}")
list(PREPEND CMAKE_BUILD_RPATH "${LLVM_EXTERNAL_LIB_DIR}")
### Workaround ROCm address sanitizer build not being able to propagate LD_LIBRARY_PATH
### Remove when https://github.com/pfultz2/cget/issues/110 is fixed.
if (DEFINED ENV{ADDRESS_SANITIZER})
  if ($ENV{ADDRESS_SANITIZER} MATCHES "ON")
    message("Address sanitizer workarounds")
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
      OUTPUT_VARIABLE clang_asan_lib_file
      ERROR_VARIABLE clang_stderr
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE clang_exit_code)
    if (NOT "${clang_exit_code}" STREQUAL "0")
      message(FATAL_ERROR
      "Unable to invoke clang to find asan lib dir: ${clang_stderr}")
    endif()
    file(TO_CMAKE_PATH "${clang_asan_lib_file}" clang_asan_lib_file_cmake)
    get_filename_component(clang_asan_lib_dir "${clang_asan_lib_file_cmake}" DIRECTORY)
    message(STATUS "Asan lib directory ${clang_asan_lib_dir}")
    list(APPEND CMAKE_BUILD_RPATH "${clang_asan_lib_dir}")
  endif()
endif()
### End workaround

# Set up the build for the LLVM/MLIR git-submodule
include(cmake/llvm-project.cmake)
include(cmake/mlir-hal.cmake)

if(NOT WIN32)
    # Update the build-tree RPATH
    list(PREPEND CMAKE_BUILD_RPATH "${ROCMLIR_LIB_DIR}")
    message(STATUS "Final CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
endif()

# Set up the build for the rocMLIR dialects
add_subdirectory(mlir)
