project(miopen-mlir)
cmake_minimum_required(VERSION 3.13.4)

set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "List of default llvm targets")
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(MLIR_ENABLE_SQLITE ON CACHE BOOL "")

add_subdirectory(llvm)

