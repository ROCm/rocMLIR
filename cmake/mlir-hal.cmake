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

add_subdirectory("${MHAL_PROJECT_DIR}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_BINARY_DIR}/lib")

include_directories("${MHAL_PROJECT_DIR}/include")
include_directories("${MHAL_BINARY_DIR}/include")

