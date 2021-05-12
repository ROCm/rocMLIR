set(LLVM_OPTIONAL_SOURCES
  BackendUtils.cpp
  rocm-runtime-wrappers.cpp
  )

if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
  message(SEND_ERROR
    "Building the mlir rocm runner requires the AMDGPU backend")
endif()

# Ensure lld is enabled.
if (NOT "lld" IN_LIST LLVM_ENABLE_PROJECTS)
  message(SEND_ERROR "lld is not enabled. Please revise LLVM_ENABLE_PROJECTS")
endif()

# lld header files.
include_directories(${CMAKE_SOURCE_DIR}/external/llvm-project/lld/include)

set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH " Path to which HIP has been installed")
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP)
if (NOT HIP_FOUND)
  message(SEND_ERROR "Build the mlir rocm runner requires a working ROCm and HIP install")
else()
  message(STATUS "ROCm HIP version: ${HIP_VERSION}")
endif()

# Set compile-time flags for ROCm path.
add_definitions(-D__ROCM_PATH__="${ROCM_PATH}")

# Locate HIP runtime library.
find_library(ROCM_RUNTIME_LIBRARY amdhip64
             PATHS "${HIP_PATH}/lib")
if (NOT ROCM_RUNTIME_LIBRARY)
  message(SEND_ERROR "Could not locate ROCm HIP runtime library")
else()
  message(STATUS "ROCm HIP runtime lib: ${ROCM_RUNTIME_LIBRARY}")
endif()

# Set HIP compile-time flags.
add_definitions(-D__HIP_PLATFORM_HCC__)

add_llvm_library(rocm-runtime-wrappers SHARED
  rocm-runtime-wrappers.cpp
)
target_include_directories(rocm-runtime-wrappers
  PRIVATE
  "${HIP_PATH}/../include"
  "${HIP_PATH}/include"
  LLVMSupport
)
target_link_libraries(rocm-runtime-wrappers
  PUBLIC
  LLVMSupport
  ${ROCM_RUNTIME_LIBRARY}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  lldCommon
  lldDriver
  lldELF
  LLVMCore
  LLVMLTO
  LLVMMC
  LLVMMCParser
  LLVMOption
  LLVMSupport
  MLIRTargetROCDLIR
  ${ROCM_RUNTIME_LIBRARY}
)

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens AllTargetsAsmParsers)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

add_llvm_library(LLVMROCmBackendUtils
  BackendUtils.cpp
  )

llvm_update_compile_flags(LLVMROCmBackendUtils)
target_include_directories(LLVMROCmBackendUtils
  PRIVATE
  "${HIP_PATH}/../include"
  "${HIP_PATH}/include"
)
target_link_libraries(LLVMROCmBackendUtils PRIVATE ${LIBS} ${targets_to_link})

