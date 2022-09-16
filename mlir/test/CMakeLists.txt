add_subdirectory(CAPI)
add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_BINDINGS_PYTHON_ENABLED
  LLVM_BUILD_EXAMPLES
  MLIR_ENABLE_ROCM_CONVERSIONS
  MLIR_ENABLE_ROCM_RUNNER

  MLIR_ROCK_DRIVER_ENABLED
  MLIR_ROCK_DRIVER_E2E_TEST_ENABLED
  MLIR_ROCK_DRIVER_MISC_E2E_TEST_ENABLED
  MLIR_ROCK_DRIVER_PR_E2E_TEST_ENABLED
  MLIR_ROCK_SQLITE_ENABLED
  )

# Passed to lit.site.cfg.py.so that the out of tree Standalone dialect test
# can find MLIR's CMake configuration
set(MLIR_CMAKE_CONFIG_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for linalg integration tests.
set(MLIR_DIALECT_LINALG_INTEGRATION_TEST_LIB_DIR ${LLVM_EXTERNAL_LIB_DIR})
set(MLIR_RUNNER_UTILS_DIR ${LLVM_EXTERNAL_BIN_DIR})

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for the rocm runner tests.
set(MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR ${MLIR_ROCK_LIB_DIR})

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "MLIR_ROCK_BIN_DIR: ${MLIR_ROCK_BIN_DIR}")
message(STATUS "MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR: ${MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR}")

# Need to explicitly give paths to llvm tools
set(LLVM_LIT_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(LLVM_EXTERNAL_LIT "${LLVM_LIT_TOOLS_DIR}/llvm-lit")
set(LLVM_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(MLIR_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(MLIR_ROCK_TOOLS_DIR ${MLIR_ROCK_BIN_DIR})

# Passed to lit.site.cfg.py.in to set the -x2 flag for the mlir E2E tests.
set(MLIR_XDLOPS "")
if(MLIR_ROCK_DRIVER_XDLOPS_TEST_ENABLED)
  set(MLIR_XDLOPS "-x2")
endif()

# Passed to lit.site.cfg.py.in to set the -rand flag for the mlir E2E tests.
set(MLIR_RANDOM_DATA "")
if(NOT MLIR_ROCK_DRIVER_RANDOM_DATA_SEED STREQUAL "none")
  set(MLIR_RANDOM_DATA "-rand ${MLIR_ROCK_DRIVER_RANDOM_DATA_SEED} ")
endif()

# Passed to lit.site.cfg.py.in to set the -pv or -pv_with_gpu flag for the mlir E2E tests.
set(MLIR_POPULATE_VALIDATION "-pv")
if(MLIR_ROCK_DRIVER_TEST_GPU_VALIDATION)
  set(MLIR_POPULATE_VALIDATION "-pv_with_gpu")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

set(MLIR_ROCK_TEST_DEPENDS
  FileCheck count not
  rock-opt
  rock-translate
  rock-lsp-server
  mlir-mixr-capi-test
  mlir-mixr-full-test
  mlir-mixr-fullc-test
  mlir-tosa-miir-test
  mlir-opt
  mlir-reduce
  mlir-translate
  mlir_runner_utils
  mlir_c_runner_utils
  mlir_async_runtime
)

list(APPEND MLIR_ROCK_TEST_DEPENDS MLIRRockUnitTests)

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND MLIR_ROCK_TEST_DEPENDS
    mlir-rocm-runner
    xmir-runner
  )
endif()

if(MLIR_BINDINGS_PYTHON_ENABLED)
  list(APPEND MLIR_ROCK_TEST_DEPENDS
    MLIRBindingsPythonExtension
    MLIRBindingsPythonTestOps
    MLIRTransformsBindingsPythonExtension
  )
endif()

# Rock dialect tools.
if(MLIR_ROCK_DRIVER_ENABLED)
  list(APPEND MLIR_ROCK_TEST_DEPENDS
    mlir-rock-driver
    rock-gen
    mlir-rock-lib-test
    mlir-translate
    opt
    llc
    )
endif()

# Enable the ROCK E2E tests if -DROCK_E2E_TEST_ENABLED=1
if(ROCK_E2E_TEST_ENABLED)
  list(APPEND MLIR_ROCK_TEST_DEPENDS e2e_tests)
  add_subdirectory(e2e)
endif()

add_custom_target(check-mlir-rock-build-only
  DEPENDS ${MLIR_ROCK_TEST_DEPENDS})

# check-mlir-rock runs MLIR-Rock regression tests including E2E tests
# when MLIR_ROCK_DRIVER_E2E_TEST_ENABLED is set
# It runs E2E tests in Resnet50 and Resnext101 when
# ROCK_E2E_TEST_ENABLED is set
add_lit_testsuite(check-mlir-rock "Running the MLIR-Rock regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_ROCK_TEST_DEPENDS}
)
set_target_properties(check-mlir-rock PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR-ROCK ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_ROCK_TEST_DEPENDS}
)

# Update the configs for E2E tests
# This needs to be done regardless of the value of ROCK_E2E_TEST_ENABLED
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/e2e/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e/lit.cfg.py
)
