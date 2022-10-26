cmake_minimum_required(VERSION 3.15.1)

# Allow VERSION for projects, as expected in CMake 3.0+
cmake_policy(SET CMP0048 NEW)
project(rocMLIR VERSION 1.0.0 LANGUAGES CXX C)

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

# Rock dialect.
set(ROCMLIR_DRIVER_ENABLED 1 CACHE BOOL "Enable build Rock driver")
set(ROCMLIR_DRIVER_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for Rock driver")
set(ROCMLIR_DRIVER_MISC_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build miscellaneous E2E tests for Rock driver")
set(ROCMLIR_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(ROCMLIR_GEN_FLAGS "" CACHE BOOL "Set feature flag for rocmlir-gen")
set(ROCMLIR_DRIVER_TEST_GPU_VALIDATION 1 CACHE BOOL "Enable E2E tests with GPU validation")
set(ROCMLIR_SQLITE_ENABLED 0 CACHE BOOL "Enable SQLite integration")
set(MLIR_ENABLE_SQLITE "${ROCMLIR_SQLITE_ENABLED}")
set(LLVM_ENABLE_ZLIB "OFF" CACHE STRING "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(ROCK_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build rock E2E tests")

# LLVM settings that have an effect on the MLIR dialect
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")

# Pointers to: 1) external LLVM bins/libs, and 2) Rock Dialect bins/libs
set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm" CACHE PATH "Path to LLVM sources")
set(LLVM_EXTERNAL_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/bin" CACHE PATH "")
set(LLVM_EXTERNAL_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/lib" CACHE PATH "")
set(ROCMLIR_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(ROCMLIR_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "ROCMLIR_BIN_DIR: ${ROCMLIR_BIN_DIR}")

# Update the build-tree RPATH
set(CMAKE_BUILD_RPATH "${ROCMLIR_LIB_DIR};${LLVM_EXTERNAL_LIB_DIR}")
message(STATUS "CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")

function(_maybe_force_set var val type docs force)
  if (${force})
    set(${var} ${val} CACHE ${type} "${docs}" FORCE)
  else()
    set(${var} ${val} CACHE ${type} "${docs}")
  endif()
endfunction()

function(_set_librockcompiler_varying_bools value force)
  _maybe_force_set(BUILD_SHARED_LIBS ${value} BOOL "" ${force})
  _maybe_force_set(LLVM_BUILD_LLVM_DYLIB ${value} BOOL "" ${force})
  _maybe_force_set(LLVM_BUILD_EXAMPLES ${value} BOOL "" ${force})
  _maybe_force_set(MLIR_ENABLE_ROCM_RUNNER ${value} BOOL "" ${force})
  _maybe_force_set(MLIR_INCLUDE_INTEGRATION_TESTS ${value} BOOL "" ${force})
  _maybe_force_set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED ${value}
    BOOL "Enable build PR-triggered E2E tests for Rock driver" ${force})
endfunction()

# Set up defaults for not using librockcompiler either way
_set_librockcompiler_varying_bools(ON OFF)

if(BUILD_MIXR_TARGET)
  set(BUILD_FAT_LIBROCKCOMPILER ON CACHE BOOL "Build static librockCompiler")
  set(EXPORT_ALL_HEADERS ON CACHE BOOL "Export internal headers")
endif()

# Library type and linkage settings
if(NOT DEFINED BUILD_FAT_LIBROCKCOMPILER)
  set(BUILD_FAT_LIBROCKCOMPILER OFF CACHE BOOL "Build fat librockCompiler to link into Rock driver")
endif()

if(BUILD_FAT_LIBROCKCOMPILER)
  _set_librockcompiler_varying_bools(OFF ON)
  set(FAT_LIBROCKCOMPILER_OPTS_APPLIED ON CACHE INTERNAL "was fat librockcompiler used before?" FORCE)
elseif(FAT_LIBROCKCOMPILER_OPTS_APPLIED)
  _set_librockcompiler_varying_bools(ON ON)
  set(FAT_LIBROCKCOMPILER_OPTS_APPLIED OFF CACHE INTERNAL "was fat librockcompiler used before?" FORCE)
endif()

# Set up the build for the LLVM/MLIR git-submodule
include(cmake/llvm-project.cmake)

# Set up the build for the official MLIR 'standalone' example
#add_subdirectory(mlir-standalone-example)

# Set up the build for the rocMLIR dialects
add_subdirectory(mlir)
