cmake_minimum_required(VERSION 3.15.1)

# Allow VERSION for projects, as expected in CMake 3.0+
cmake_policy(SET CMP0048 NEW)
project(miopen-mlir VERSION 0.202108 LANGUAGES CXX C)

# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

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

# MIOpen dialect.
set(MLIR_MIOPEN_DRIVER_ENABLED 1 CACHE BOOL "Enable build MIOpen driver")
set(MLIR_MIOPEN_DRIVER_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_MISC_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build miscellaneous E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_PR_E2E_TEST_ENABLED 1 CACHE BOOL "Enable build PR-triggered E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(MLIR_MIOPEN_DRIVER_XDLOPS_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for XDLops")
set(MLIR_MIOPEN_DRIVER_TEST_GPU_VALIDATION 1 CACHE BOOL "Enable E2E tests with GPU validation")
set(MLIR_MIOPEN_DRIVER_TIMING_TEST_ENABLED 0 CACHE BOOL "Enable printing individual test timing")
set(MLIR_MIOPEN_SQLITE_ENABLED 0 CACHE BOOL "Enable SQLite integration")
set(MLIR_ENABLE_SQLITE "${MLIR_MIOPEN_SQLITE_ENABLED}")
set(LLVM_ENABLE_ZLIB "OFF" CACHE STRING "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")

# LLVM settings that have an effect on the MLIR dialect
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")

# Pointers to: 1) external LLVM bins/libs, and 2) MIOpen Dialect bins/libs
set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm" CACHE PATH "Path to LLVM sources")
set(LLVM_EXTERNAL_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/bin" CACHE PATH "")
set(LLVM_EXTERNAL_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/lib" CACHE PATH "")
set(MLIR_MIOPEN_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(MLIR_MIOPEN_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "MLIR_MIOPEN_BIN_DIR: ${MLIR_MIOPEN_BIN_DIR}")

# Update the build-tree RPATH
set(CMAKE_BUILD_RPATH "${MLIR_MIOPEN_LIB_DIR};${LLVM_EXTERNAL_LIB_DIR}")
message(STATUS "CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")

if(BUILD_MIXR_TARGET)
  set(BUILD_FAT_LIBMLIRMIOPEN ON CACHE BOOL "Build static libMLIRMIOpen")
  set(EXPORT_ALL_HEADERS ON CACHE BOOL "Export internal headers")
endif()

# Library type and linkage settings
if( NOT DEFINED BUILD_FAT_LIBMLIRMIOPEN )
  set(BUILD_FAT_LIBMLIRMIOPEN OFF CACHE BOOL "Build fat libMLIRMIOpen to link into MIOpen driver")
endif()
if( BUILD_FAT_LIBMLIRMIOPEN )
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
  # rocm-runner is not supported with static libraries
  set(MLIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "")
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")
else()
  set(BUILD_SHARED_LIBS ON CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
  set(LLVM_BUILD_EXAMPLES ON CACHE BOOL "")
  set(MLIR_ENABLE_ROCM_RUNNER 1 CACHE BOOL "")
endif()

# Disable other runners
set(MLIR_CUDA_RUNNER_ENABLED 0 CACHE BOOL "")
set(MLIR_SPIRV_CPU_RUNNER_ENABLED 0 CACHE BOOL "")
set(MLIR_VULKAN_RUNNER_ENABLED 0 CACHE BOOL "")
set(MLIR_BINDINGS_PYTHON_ENABLED 0 CACHE BOOL "")

# Set up the build for the LLVM/MLIR git-submodule
include(cmake/llvm-project.cmake)

# Set up the build for the official MLIR 'standalone' example
#add_subdirectory(mlir-standalone-example)

# Set up the build for the MLIR MIOpen dialect
add_subdirectory(mlir)
