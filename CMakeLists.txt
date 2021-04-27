project(miopen-mlir)
cmake_minimum_required(VERSION 3.13.4)

if( NOT DEFINED BUILD_FAT_LIBMLIRMIOPEN )
  set(BUILD_FAT_LIBMLIRMIOPEN OFF CACHE BOOL "Build fat libMLIRMIOpen to link into MIOpen driver")
endif()

set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "List of default llvm targets")
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(MLIR_ENABLE_SQLITE ON CACHE BOOL "")
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

# Set policy CMP0057 to support IN_LIST operators
cmake_policy(SET CMP0057 NEW)

#set (git_cmd "git")
#set (git_arg "status" "--help")
#message(STATUS "git cmd: ${git_cmd}")
#execute_process(COMMAND ${git_cmd} ${git_arg})

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
