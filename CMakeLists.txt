project(miopen-mlir)
cmake_minimum_required(VERSION 3.13.4)

# Set policy CMP0057 to support IN_LIST operators
cmake_policy(SET CMP0057 NEW)

# MIOpen dialect.
set(MLIR_MIOPEN_DRIVER_ENABLED 0 CACHE BOOL "Enable build MIOpen driver")
set(MLIR_MIOPEN_DRIVER_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_MISC_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build miscellaneous E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_PR_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build PR-triggered E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(MLIR_MIOPEN_DRIVER_XDLOPS_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for XDLops")
set(MLIR_MIOPEN_SQLITE_ENABLED 0 CACHE BOOL "Enable SQLite integration")

# Pointers to: 1) external LLVM bins/libs, and 2) MIOpen Dialect bins/libs
set(LLVM_EXTERNAL_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/bin" CACHE PATH "")
set(LLVM_EXTERNAL_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/lib" CACHE PATH "")
set(MLIR_MIOPEN_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(MLIR_MIOPEN_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "MLIR_MIOPEN_BIN_DIR: ${MLIR_MIOPEN_BIN_DIR}")

# Update the build-tree RPATH
set(CMAKE_BUILD_RPATH "${MLIR_MIOPEN_LIB_DIR};${LLVM_EXTERNAL_LIB_DIR}")
message(STATUS "CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")

# Library type and linkage settings
if( NOT DEFINED BUILD_FAT_LIBMLIRMIOPEN )
  set(BUILD_FAT_LIBMLIRMIOPEN OFF CACHE BOOL "Build fat libMLIRMIOpen to link into MIOpen driver")
endif()
if( BUILD_FAT_LIBMLIRMIOPEN )
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")
else()
  set(BUILD_SHARED_LIBS ON CACHE BOOL "")
  set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
  set(LLVM_BUILD_EXAMPLES ON CACHE BOOL "")
  set(MLIR_ROCM_RUNNER_ENABLED 1 CACHE BOOL "")
endif()

# Python's diff_match_patch package is currently required to
# patch external/llvm-project
execute_process(
  COMMAND pip show diff_match_patch
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET
)

# Python's pathlib2 package is currently required to
# patch external/llvm-project
execute_process(
  COMMAND pip show pathlib2
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET
)

if (NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "The \"diff_match_patch\" python package is not installed. \
       The following command can be used to install it: \
       \"pip install diff_match_patch\""
  )
endif()

# Patch 'external/llvm-project/mlir' with mods in 'mlir-patch'
# FIXME: multi-line cmd + args...
#set(PY_CMD "python utils/src-tree-patcher/patcher.py ")
#set(PY_ARG1 "--infile utils/src-tree-patcher/filelist.txt ")
#set(PY_ARG2 "--srcdir mlir-patch ")
#set(PY_ARG3 "--dstdir external/llvm-project/mlir")
#set(PY_ARGS ${PY_ARG1} ${PY_ARG2} ${PY_ARG3})
#execute_process(COMMAND ${PY_CMD} ${PY_ARGS}
execute_process(COMMAND python utils/src-tree-patcher/patcher.py --infile utils/src-tree-patcher/filelist.txt --srcdir mlir-patch --dstdir external/llvm-project/mlir
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

# Set up the build for the MLIR git-submodule
include(cmake/mlir.cmake)

# Set up the build for the official MLIR 'standalone' example
#add_subdirectory(mlir-standalone-example)

# Set up the build for the MLIR MIOpen dialect
add_subdirectory(mlir)
