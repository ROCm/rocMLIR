cmake_minimum_required(VERSION 3.13.4)  # fixme: double-check versions

project(mlir-miopen)



message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

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

# MIOpen dialect.
set(MLIR_MIOPEN_DRIVER_ENABLED 0 CACHE BOOL "Enable build MIOpen driver")
set(MLIR_MIOPEN_DRIVER_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_MISC_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build miscellaneous E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_PR_E2E_TEST_ENABLED 0 CACHE BOOL "Enable build PR-triggered E2E tests for MIOpen driver")
set(MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(MLIR_MIOPEN_DRIVER_XDLOPS_TEST_ENABLED 0 CACHE BOOL "Enable build E2E tests for XDLops")
set(MLIR_MIOPEN_DRIVER_TEST_GPU_VALIDATION 0 CACHE BOOL "Enable E2E tests with GPU validation")

set(MLIR_MIOPEN_SQLITE_ENABLED 0 CACHE BOOL "Enable SQLite integration")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})



add_subdirectory(docs)
add_subdirectory(examples)
add_subdirectory(include/mlir)
add_subdirectory(lib)
#add_subdirectory(test)
add_subdirectory(tools)
