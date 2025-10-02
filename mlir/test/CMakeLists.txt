find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(CAPI)
add_subdirectory(lib)
add_subdirectory(common_utils)

llvm_canonicalize_cmake_booleans(
  MLIR_BINDINGS_PYTHON_ENABLED
  LLVM_BUILD_EXAMPLES
  MLIR_ENABLE_ROCM_CONVERSIONS
  MLIR_ENABLE_ROCM_RUNNER
<<<<<<< HEAD

  ROCMLIR_DRIVER_E2E_TEST_ENABLED
  ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED
  ROCK_E2E_TEST_ENABLED
=======
  MLIR_ENABLE_SPIRV_CPU_RUNNER
  MLIR_ENABLE_VULKAN_RUNNER
  MLIR_INCLUDE_INTEGRATION_TESTS
  MLIR_RUN_AMX_TESTS
  MLIR_RUN_CUDA_TENSOR_CORE_TESTS
  MLIR_RUN_X86VECTOR_TESTS
  MLIR_RUN_ARM_SVE_TESTS
  MLIR_RUN_ARM_SME_TESTS
  MLIR_RUN_CUDA_SM80_TESTS
  MLIR_RUN_CUDA_SM80_LT_TESTS
  MLIR_RUN_CUDA_SM90_TESTS
  BUILD_SHARED_LIBS
>>>>>>> cb1c9dfd699c (Squashed 'external/llvm-project/' changes from 1e1a4d577e7e..8ffe4def7d09)
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

<<<<<<< HEAD
set(ROCMLIR_TEST_DEPENDS
  FileCheck count not
  rocmlir-opt
  rocmlir-translate
  rocmlir-lsp-server
  rocmlir-driver
  rocmlir-gen
  rocmlir-lib-test
  rocmlir-common-python-test-utils
=======
set(MLIR_TEST_DEPENDS
  mlir-capi-ir-test
  mlir-capi-irdl-test
  mlir-capi-llvm-test
  mlir-capi-pass-test
  mlir-capi-quant-test
  mlir-capi-rewrite-test
  mlir-capi-smt-test
  mlir-capi-sparse-tensor-test
  mlir-capi-transform-test
  mlir-capi-transform-interpreter-test
  mlir-capi-translation-test
  mlir-linalg-ods-yaml-gen
  mlir-lsp-server
  mlir-opt
  mlir-query
  mlir-reduce
  mlir-rewrite
  mlir-tblgen
  mlir-translate
  tblgen-lsp-server
  tblgen-to-irdl
  )
if(NOT MLIR_STANDALONE_BUILD)
  list(APPEND MLIR_TEST_DEPENDS FileCheck count not split-file yaml2obj)
endif()
# Examples/standalone/test.toy (vis-a-vis the standalone example) depends on these.
if(LLVM_INCLUDE_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS MLIRCAPIArith)
endif()
>>>>>>> cb1c9dfd699c (Squashed 'external/llvm-project/' changes from 1e1a4d577e7e..8ffe4def7d09)

  opt
  llc

  mlir-mixr-capi-test
  mlir-mixr-fullc-test
  mlir-mixr-split-k-test
  mlir-reduce-fusible-test
)

list(APPEND ROCMLIR_TEST_DEPENDS
  RocMLIRUnitTests
  rocmlir-common-python-test-utils
)

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND ROCMLIR_TEST_DEPENDS
    mlir_runner_utils
    mlir_c_runner_utils
    mlir_async_runtime

    mlir_rocm_runtime
    conv-validation-wrappers
    mlir-runner
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
