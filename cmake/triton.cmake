message(STATUS "Adding Triton src dependency")

#===----------------------------------------------------------------------===//
# Triton Project Configuration
# NOTE: Triton requires a pre-built LLVM/MLIR. Set MLIR_DIR before configuring.
#===----------------------------------------------------------------------===//

set(TRITON_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")

#===----------------------------------------------------------------------===//
# LLVM/MLIR Configuration
# Triton uses find_package(MLIR) - must be provided externally
#===----------------------------------------------------------------------===//

# User must provide MLIR_DIR (e.g., from a built LLVM or system installation)
if(NOT DEFINED MLIR_DIR)
  # Try common locations
  if(DEFINED ENV{MLIR_DIR})
    set(MLIR_DIR $ENV{MLIR_DIR} CACHE PATH "Path to MLIR CMake config")
  elseif(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake")
    # Default: Use LLVM built by Triton's build-llvm-project.sh script
    set(MLIR_DIR "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
  elseif(DEFINED LLVM_LIBRARY_DIR)
    set(MLIR_DIR "${LLVM_LIBRARY_DIR}/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
  else()
    message(FATAL_ERROR 
      "MLIR_DIR must be set to the path containing MLIRConfig.cmake\n"
      "Example: cmake -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir ..\n"
      "You can build LLVM/MLIR using: cd external/triton && ./scripts/build-llvm-project.sh")
  endif()
endif()

message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# Find MLIR package (this also sets up LLVM variables)
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

# Set up LLD_DIR based on MLIR_DIR location
get_filename_component(_llvm_cmake_dir "${MLIR_DIR}" DIRECTORY)
set(LLD_DIR "${_llvm_cmake_dir}/lld" CACHE PATH "Path to LLD CMake config")
message(STATUS "LLD_DIR: ${LLD_DIR}")
message(STATUS "Found MLIR ${MLIR_VERSION} at ${MLIR_DIR}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

# Set up CMake module paths from found MLIR/LLVM
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include LLVM/MLIR CMake utilities
include(TableGen)
include(AddLLVM)
include(AddMLIR)

#===----------------------------------------------------------------------===//
# ROCm Configuration
#===----------------------------------------------------------------------===//

if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
  endif()
endif()
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")

#===----------------------------------------------------------------------===//
# Triton Build Options (matching external/triton/CMakeLists.txt)
#===----------------------------------------------------------------------===//

# Disable Python module - we're using C++ API only
set(TRITON_BUILD_PYTHON_MODULE OFF CACHE BOOL "Don't build Python bindings")

# Disable Proton profiler
set(TRITON_BUILD_PROTON OFF CACHE BOOL "Don't build Proton profiler")

# Disable unit tests (can enable later)
set(TRITON_BUILD_UT OFF CACHE BOOL "Don't build Triton unit tests")

# Enable AMD backend via TRITON_CODEGEN_BACKENDS
set(TRITON_CODEGEN_BACKENDS "amd" "nvidia" CACHE STRING "Enable AMD codegen backend")

#===----------------------------------------------------------------------===//
# Include Directories
#===----------------------------------------------------------------------===//

# Triton includes
list(APPEND TRITON_INCLUDE_DIRS
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
)

#===----------------------------------------------------------------------===//
# For lit testing configuration
#===----------------------------------------------------------------------===//

set(MLIR_CMAKE_CONFIG_DIR "${MLIR_DIR}")
set(MLIR_TABLEGEN_EXE mlir-tblgen)

#===----------------------------------------------------------------------===//
# Add Triton subdirectory
#===----------------------------------------------------------------------===//

add_subdirectory("${TRITON_PROJECT_DIR}" "external/triton" EXCLUDE_FROM_ALL)

#===----------------------------------------------------------------------===//
# Create dummy targets for MLIR tablegen dependencies
# When using pre-built MLIR, tablegen targets don't exist but headers do
#===----------------------------------------------------------------------===//

if(NOT TARGET MLIRConversionPassIncGen)
  add_custom_target(MLIRConversionPassIncGen)
endif()

#===----------------------------------------------------------------------===//
# Helper Functions for rocMLIR Libraries
#===----------------------------------------------------------------------===//

function(add_rocmlir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_DIALECT_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_dialect_library)

function(add_rocmlir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_CONVERSION_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_conversion_library)

function(add_rocmlir_test_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_TEST_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_test_library)

function(add_rocmlir_public_c_api_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_PUBLIC_C_API_LIBS ${name})
  add_mlir_library(${name}
    ${ARGN}
    EXCLUDE_FROM_LIBMLIR
    ENABLE_AGGREGATION
    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir-c
  )
  set_target_properties(obj.${name}
    PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(obj.${name}
    PRIVATE
    -DMLIR_CAPI_BUILDING_LIBRARY=1
  )
endfunction()

function(add_rocmlir_tool name)
  set(exclude_from_all "")
  if(BUILD_FAT_LIBROCKCOMPILER)
    set(exclude_from_all "EXCLUDE_FROM_ALL")
    set(LLVM_BUILD_TOOLS OFF)
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_mlir_tool(${name} ${exclude_from_all} ${ARGN})
endfunction()

# Helper function for Rock-to-Triton libraries
function(add_rocmlir_triton_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_TRITON_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_triton_library)