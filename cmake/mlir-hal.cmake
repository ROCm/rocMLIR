message(STATUS "Adding MHAL git-submodule src dependency")

set(MHAL_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/mlir-hal")
set(MHAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/mlir-hal")
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# Include dirs for MLIR and LLVM
list(APPEND MLIR_INCLUDE_DIRS
  ${MHAL_PROJECT_DIR}/mlir/include
  ${MHAL_BINARY_DIR}/include
)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
# A malformed _GLIBCXX_USE_CXX11_ABI can arrive via add_definitions(${LLVM_DEFINITIONS}).
# Re-assert a well-formed value on GNU-style (non-MSVC) command lines, honoring the
# GLIBCXX_USE_CXX11_ABI option selected by HandleLLVMOptions.cmake when it is set.
if(NOT MSVC)
  if(DEFINED GLIBCXX_USE_CXX11_ABI AND NOT GLIBCXX_USE_CXX11_ABI)
    set(_mhal_glibcxx_cxx11_abi 0)
  else()
    set(_mhal_glibcxx_cxx11_abi 1)
  endif()
  add_compile_options(-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=${_mhal_glibcxx_cxx11_abi})
endif()

add_subdirectory("${MHAL_PROJECT_DIR}" EXCLUDE_FROM_ALL)

# MHAL libs use upstream add_mlir_*_library, so register them under
# ROCMLIR_CONVERSION_LIBS for InitRocMLIRPasses.h consumers. Guard on TARGET
# because the MHAL build gates Conversion/ behind MHAL_ENABLE_TRANSFORMS (ON
# by default); without the guard, configuring with -DMHAL_ENABLE_TRANSFORMS=OFF
# would push a dangling target name into the property and break the link.
if(TARGET MLIRMHALToGPU)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_CONVERSION_LIBS MLIRMHALToGPU)
endif()

if(NOT WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_BINARY_DIR}/lib")
endif()

include_directories("${MHAL_PROJECT_DIR}/include")
include_directories("${MHAL_BINARY_DIR}/include")

