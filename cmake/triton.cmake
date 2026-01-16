message(STATUS "Adding Triton src dependency")

#===----------------------------------------------------------------------===//
# Triton Project Configuration
# NOTE: Triton requires a pre-built LLVM/MLIR. Set MLIR_DIR before configuring.
#===----------------------------------------------------------------------===//

set(TRITON_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

# Triton's include directories are needed for tablegen to find Triton's .td files
# (e.g., triton/Dialect/Triton/IR/TritonTypes.td)
include_directories("${TRITON_PROJECT_DIR}/include")
include_directories("${TRITON_BINARY_DIR}/include") # Tablegen'd files
include_directories("${TRITON_PROJECT_DIR}/third_party")
include_directories("${TRITON_BINARY_DIR}/third_party") # Tablegen'd files

message("TRITON_PROJECT_DIR: ${TRITON_PROJECT_DIR}")

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
set(TRITON_CODEGEN_BACKENDS "amd" CACHE STRING "Enable AMD codegen backend")

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

# Set up global include directories for tablegen (needed before any subdirectories)
# These are required by mlir_tablegen to find MLIR tablegen files like mlir/IR/OpBase.td
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Triton uses its own bundled LLVM/MLIR, so we also need to include those paths
# for tablegen to work correctly when using Triton's mlir-tblgen
if(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/mlir/include")
  include_directories("${TRITON_PROJECT_DIR}/llvm-project/mlir/include")
endif()
if(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/llvm/include")
  include_directories("${TRITON_PROJECT_DIR}/llvm-project/llvm/include")
endif()
if(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/build/tools/mlir/include")
  include_directories("${TRITON_PROJECT_DIR}/llvm-project/build/tools/mlir/include")
endif()

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
# Always include NVIDIA tablegen targets
# RegisterTritonDialects.h unconditionally includes NVIDIA headers, so we need
# the tablegen files even when NVIDIA backend is not enabled
#===----------------------------------------------------------------------===//

# Check if NVIDIA backend is not in TRITON_CODEGEN_BACKENDS
string(FIND "${TRITON_CODEGEN_BACKENDS}" "nvidia" NVIDIA_BACKEND_INDEX)
if(NVIDIA_BACKEND_INDEX EQUAL -1)
  # NVIDIA backend not enabled, but we still need tablegen and some libraries for headers
  # Include just the Dialect tablegen CMakeLists.txt files
  set(NVIDIA_INCLUDE_DIR "${TRITON_PROJECT_DIR}/third_party/nvidia/include")
  set(NVIDIA_BINARY_INCLUDE_DIR "${TRITON_BINARY_DIR}/third_party/nvidia/include")
  set(NVIDIA_LIB_DIR "${TRITON_PROJECT_DIR}/third_party/nvidia/lib")
  set(NVIDIA_BINARY_LIB_DIR "${TRITON_BINARY_DIR}/third_party/nvidia/lib")
  
  # Set up include directories for NVIDIA tablegen
  include_directories(${NVIDIA_INCLUDE_DIR})
  include_directories(${NVIDIA_BINARY_INCLUDE_DIR})
  
  # Set MLIR_BINARY_DIR for tablegen (needed by the CMakeLists.txt files)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  
  # Include the Dialect CMakeLists.txt which sets up NVGPU and NVWS tablegen
  # This includes both IR and Transforms tablegen for NVWS
  add_subdirectory("${NVIDIA_INCLUDE_DIR}/Dialect" "${NVIDIA_BINARY_INCLUDE_DIR}/Dialect" EXCLUDE_FROM_ALL)
  
  # Also include TritonNVIDIAGPUToLLVM tablegen - needed by TritonNVIDIAGPUToLLVM library
  if(EXISTS "${NVIDIA_INCLUDE_DIR}/TritonNVIDIAGPUToLLVM/CMakeLists.txt")
    add_subdirectory("${NVIDIA_INCLUDE_DIR}/TritonNVIDIAGPUToLLVM" "${NVIDIA_BINARY_INCLUDE_DIR}/TritonNVIDIAGPUToLLVM" EXCLUDE_FROM_ALL)
  endif()
  
  # Also include hopper tablegen - RegisterTritonDialects.h includes hopper headers
  # Hopper is normally included via third_party/nvidia/CMakeLists.txt -> hopper/CMakeLists.txt
  # but we need it even when NVIDIA backend is not enabled
  set(HOPPER_INCLUDE_DIR "${TRITON_PROJECT_DIR}/third_party/nvidia/hopper/include")
  set(HOPPER_BINARY_INCLUDE_DIR "${TRITON_BINARY_DIR}/third_party/nvidia/hopper/include")
  if(EXISTS "${HOPPER_INCLUDE_DIR}/Transforms/CMakeLists.txt")
    add_subdirectory("${HOPPER_INCLUDE_DIR}/Transforms" "${HOPPER_BINARY_INCLUDE_DIR}/Transforms" EXCLUDE_FROM_ALL)
  endif()
  
  # Also include NVWS libraries - TritonGPUTransforms depends on NVWSIR and NVWSTransforms
  # These are needed even when NVIDIA backend is not enabled
  if(EXISTS "${NVIDIA_LIB_DIR}/Dialect/NVWS/CMakeLists.txt")
    add_subdirectory("${NVIDIA_LIB_DIR}/Dialect/NVWS" "${NVIDIA_BINARY_LIB_DIR}/Dialect/NVWS" EXCLUDE_FROM_ALL)
  endif()
  
  # Also include NVGPUIR and TritonNVIDIAGPUToLLVM libraries
  # TritonInstrumentToLLVM depends on NVGPUIR
  # Proton and other components depend on TritonNVIDIAGPUToLLVM
  if(EXISTS "${NVIDIA_LIB_DIR}/Dialect/NVGPU/IR/CMakeLists.txt")
    add_subdirectory("${NVIDIA_LIB_DIR}/Dialect/NVGPU/IR" "${NVIDIA_BINARY_LIB_DIR}/Dialect/NVGPU/IR" EXCLUDE_FROM_ALL)
  endif()
  if(EXISTS "${NVIDIA_LIB_DIR}/TritonNVIDIAGPUToLLVM/CMakeLists.txt")
    add_subdirectory("${NVIDIA_LIB_DIR}/TritonNVIDIAGPUToLLVM" "${NVIDIA_BINARY_LIB_DIR}/TritonNVIDIAGPUToLLVM" EXCLUDE_FROM_ALL)
  endif()
endif()

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
