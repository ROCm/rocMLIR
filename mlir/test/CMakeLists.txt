find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(CAPI)
add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_BINDINGS_PYTHON_ENABLED
  LLVM_BUILD_EXAMPLES
  MLIR_ENABLE_ROCM_CONVERSIONS
  MLIR_ENABLE_ROCM_RUNNER

  ROCMLIR_DRIVER_E2E_TEST_ENABLED
  ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED
  ROCK_E2E_TEST_ENABLED
  )

if ((ROCMLIR_DRIVER_E2E_TEST_ENABLED OR
     ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED OR
     ROCK_E2E_TEST_ENABLED) AND NOT MLIR_ENABLE_ROCM_RUNNER)
  message(FATAL_ERROR "Cannot enable E2E tests without the JIT runner being enabled")
endif()

# Passed to lit.site.cfg.py.so that the out of tree Standalone dialect test
# can find MLIR's CMake configuration
set(MLIR_CMAKE_CONFIG_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for linalg integration tests.
set(MLIR_DIALECT_LINALG_INTEGRATION_TEST_LIB_DIR ${LLVM_EXTERNAL_LIB_DIR})
set(MLIR_LIB_DIR ${LLVM_EXTERNAL_LIB_DIR})
set(MLIR_RUNNER_UTILS_DIR ${LLVM_EXTERNAL_BIN_DIR})

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for the rocm runner tests.
set(MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR ${ROCMLIR_LIB_DIR})

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "ROCMLIR_BIN_DIR: ${ROCMLIR_BIN_DIR}")
message(STATUS "MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR: ${MLIR_CONV_VALIDATION_WRAPPER_LIBRARY_DIR}")

# Need to explicitly give paths to llvm tools
set(LLVM_LIT_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(LLVM_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(MLIR_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(ROCMLIR_TOOLS_DIR ${ROCMLIR_BIN_DIR})

if(CMAKE_HOST_WIN32 AND NOT CYGWIN)
  set(LLVM_EXTERNAL_LIT "${LLVM_LIT_TOOLS_DIR}/llvm-lit.py")
else()
  set(LLVM_EXTERNAL_LIT "${LLVM_LIT_TOOLS_DIR}/llvm-lit")
endif()

message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")

# Passed to lit.site.cfg.py.in to set the -rand flag for the mlir E2E tests.
set(MLIR_RANDOM_DATA "")
if(NOT ROCMLIR_DRIVER_RANDOM_DATA_SEED STREQUAL "none")
  set(MLIR_RANDOM_DATA "-rand ${ROCMLIR_DRIVER_RANDOM_DATA_SEED} ")
endif()

# Passed to lit.site.cfg.py.in to set the -pv or -pv_with_gpu flag for the mlir E2E tests.
set(MLIR_POPULATE_VALIDATION "-pv")
if(ROCMLIR_DRIVER_TEST_GPU_VALIDATION)
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

set(ROCMLIR_TEST_DEPENDS
  FileCheck count not
  rocmlir-opt
  rocmlir-translate
  rocmlir-lsp-server
  rocmlir-driver
  rocmlir-gen
  rocmlir-lib-test

  opt
  llc

  mlir-mixr-capi-test
  mlir-mixr-full-test
  mlir-mixr-fullc-test
  mlir-tosa-miir-test
)

list(APPEND ROCMLIR_TEST_DEPENDS RocMLIRUnitTests)

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND ROCMLIR_TEST_DEPENDS
    mlir_runner_utils
    mlir_c_runner_utils
    mlir_async_runtime

    mlir_rocm_runtime
    conv-validation-wrappers
    mlir-cpu-runner
    xmir-runner
    # Not strictly a test dependency, included so we don't break it during compiles
    rocmlir-tuning-driver
  )
endif()

if(MLIR_BINDINGS_PYTHON_ENABLED)
  list(APPEND ROCMLIR_TEST_DEPENDS
    MLIRBindingsPythonExtension
    MLIRBindingsPythonTestOps
    MLIRTransformsBindingsPythonExtension
  )
endif()

# Enable the ROCK E2E tests if -DROCK_E2E_TEST_ENABLED=1
if(ROCK_E2E_TEST_ENABLED OR ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED)
  list(APPEND ROCMLIR_TEST_DEPENDS e2e_tests fusion-e2e-tests)
  add_subdirectory(e2e)
  add_subdirectory(fusion)
endif()


add_custom_target(check-rocmlir-build-only
  DEPENDS ${ROCMLIR_TEST_DEPENDS})

# check-rocmlir runs RocMLIR regression tests including E2E tests
# when ROCMLIR_DRIVER_E2E_TEST_ENABLED is set
# It runs E2E tests in Resnet50 and Resnext101 when
# ROCK_E2E_TEST_ENABLED is set
add_lit_testsuite(check-rocmlir "Running the RocMLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ROCMLIR_TEST_DEPENDS}
)
set_target_properties(check-rocmlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(ROCMLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ROCMLIR_TEST_DEPENDS}
)

# Update the configs for E2E tests
# This needs to be done regardless of the value of ROCK_E2E_TEST_ENABLED
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/e2e/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e/lit.cfg.py
)

# Update the configs for fusion E2E tests
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/fusion/e2e/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/fusion/e2e/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/fusion/e2e/lit.cfg.py
)
