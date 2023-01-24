message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Refer to https://reviews.llvm.org/D122088
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(LLVM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(LLVM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(ARG_PRODUCT_NAME ${CMAKE_PROJECT_NAME} CACHE STRING "")
set(ARG_VERSION_STRING ${CMAKE_PROJECT_VERSION} CACHE STRING "")

add_subdirectory(docs)
add_subdirectory(include/mlir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)

if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(RocMLIRUnitTests)
  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available LLVM_MAIN_SRC_DIR=${LLVM_MAIN_SRC_DIR}")
  endif()
  add_subdirectory(test)
endif()
add_subdirectory(tools)
add_subdirectory(utils)
