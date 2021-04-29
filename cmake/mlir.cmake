message(STATUS "Adding LLVM git-submodule src dependency")

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Build the ROCm conversions and run according tests if the AMDGPU backend
# is available
if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ROCM_CONVERSIONS_ENABLED 1)
else()
  set(MLIR_ROCM_CONVERSIONS_ENABLED 0)
endif()
add_definitions(-DMLIR_ROCM_CONVERSIONS_ENABLED=${MLIR_ROCM_CONVERSIONS_ENABLED})
set(MLIR_ROCM_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir ROCm runner")

# LLVM settings
set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE INTERNAL "")
set(LLVM_BUILD_EXAMPLES ON CACHE INTERNAL "")
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE INTERNAL "")
set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
set(LLVM_BUILD_LLVM_DYLIB ON CACHE INTERNAL "")
set(LLVM_INSTALL_UTILS ON CACHE INTERNAL "")
set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "List of default llvm targets")
set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(MLIR_ENABLE_SQLITE ON CACHE BOOL "")

set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(LLVM_PROJ_SRC "${CMAKE_SOURCE_DIR}/external/llvm-project")

add_subdirectory("${LLVM_PROJ_SRC}/llvm" "external/llvm-project/llvm" EXCLUDE_FROM_ALL)

# Cmake module paths
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir"
)
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/lib/cmake/llvm/"
)

# Include dirs for MLIR and LLVM
list(APPEND MLIR_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/mlir/include
  ${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/tools/mlir/include
)
list(APPEND LLVM_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/include
)

# Linker flags
list(APPEND CMAKE_EXE_LINKER_FLAGS
  " -Wl,-rpath -Wl,${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project/llvm/lib"
)
