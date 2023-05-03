
set(MHAL_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "MHAL Source dir")
set(MHAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "MHAL Binary dir")

message(STATUS "MHAL SOURCE: ${MHAL_PROJECT_DIR}")
message(STATUS "MHAL BINARY: ${MHAL_BINARY_DIR}")

set(MHAL_MAIN_INCLUDE_DIR "${MHAL_PROJECT_DIR}/include")
option(MHAL_ENABLE_TRANSFORMS "Enable MHAL transforms" ON)

set(LLVM_ENABLE_RTTI OFF CACHE BOOL "")

include_directories("${MHAL_MAIN_INCLUDE_DIR}")
include_directories("${MHAL_BINARY_DIR}/include")

add_subdirectory(include/mlir)
add_subdirectory(lib)
