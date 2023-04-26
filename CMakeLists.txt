
set(MHAL_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "MHAL Source dir")
set(MHAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "MHAL Binary dir")

message(STATUS "MHAL SOURCE: ${MHAL_PROJECT_DIR}")
message(STATUS "MHAL BINARY: ${MHAL_BINARY_DIR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "")

include_directories("${MHAL_PROJECT_DIR}/include")
include_directories("${MHAL_BINARY_DIR}/include")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(include/mlir)
add_subdirectory(lib)
