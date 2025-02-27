find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(CAPI)
add_subdirectory(lib)

if (MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

# Provide the MLIR CMake module dir so that the out of tree Standalone
# dialect and can add it to the module path.
set(MLIR_CMAKE_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if (MLIR_INCLUDE_INTEGRATION_TESTS)
  set(INTEL_SDE_EXECUTABLE "" CACHE STRING
      "If set, arch-specific integration tests are run with Intel SDE.")
  set(ARM_EMULATOR_EXECUTABLE "" CACHE STRING
      "If set, arch-specific Arm integration tests are run with an emulator.")
  set(ARM_EMULATOR_OPTIONS "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, pass these as parameters to the emulator.")
  set(ARM_EMULATOR_MLIR_RUNNER_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native mlir-runner.")
  set(ARM_EMULATOR_LLI_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native lli.")
  set(ARM_EMULATOR_UTILS_LIB_DIR "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, find Arm native utility libraries in this directory.")
  set(MLIR_GPU_COMPILATION_TEST_FORMAT "fatbin" CACHE STRING
      "The GPU compilation format used by the tests.")
  set(ARM_SME_ABI_ROUTINES_SHLIB "" CACHE STRING
      "Path to a shared library containing Arm SME ABI routines, required for Arm SME integration tests.")
  option(MLIR_RUN_AMX_TESTS "Run AMX tests.")
  option(MLIR_RUN_X86VECTOR_TESTS "Run X86Vector tests.")
  option(MLIR_RUN_CUDA_TENSOR_CORE_TESTS "Run CUDA Tensor core WMMA tests.")
  option(MLIR_RUN_CUDA_SM80_TESTS "Run CUDA A100 tests.")
  option(MLIR_RUN_CUDA_SM80_LT_TESTS "Run CUDA A100 structured sparsity tests.")
  option(MLIR_RUN_CUDA_SM90_TESTS "Run CUDA H100 tests.")
  option(MLIR_RUN_ARM_SVE_TESTS "Run Arm SVE tests.")
  option(MLIR_RUN_ARM_SME_TESTS "Run Arm SME tests.")

  # Check whether an emulator is required - if yes then make sure that it's
  # been set.
  check_emulator(MLIR_RUN_ARM_SVE_TESTS "HWCAP_SVE" ARM_EMULATOR_EXECUTABLE)
  check_emulator(MLIR_RUN_ARM_SME_TESTS "HWCAP2_SME" ARM_EMULATOR_EXECUTABLE)

  # The native target may not be enabled when cross compiling, raise an error.
  if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
    message(FATAL_ERROR "MLIR_INCLUDE_INTEGRATION_TESTS requires a native target")
  endif()

  # When the Integration tests are requested via the MLIR_INCLUDE_INTEGRATION_TESTS
  # configuration flag, we automatically include sm80 tests when build for
  # cuSparse when the configuration flag MLIR_ENABLE_CUDA_CUSPARSE is set and
  # include sm80 lt tests when the MLIR_ENABLE_CUDA_CUSPARSELT is set in
  # addition to those.
  if (MLIR_ENABLE_CUDA_CUSPARSE)
    set(MLIR_RUN_CUDA_SM80_TESTS ON)
    if (MLIR_ENABLE_CUDA_CUSPARSELT)
      set(MLIR_RUN_CUDA_SM80_LT_TESTS ON)
    endif()
  endif()

endif()

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
  rocmlir-common-python-test-utils

  opt
  llc

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(LLVM_ENABLE_PIC AND TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND MLIR_TEST_DEPENDS
    mlir-runner
    llc
    mlir_async_runtime
    mlir-capi-execution-engine-test
    mlir_c_runner_utils
    mlir_runner_utils
    mlir_float16_utils
  )
endif()

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
    mlir-cpu-runner
    xmir-runner
    # Not strictly a test dependency, included so we don't break it during compiles
    rocmlir-tuning-driver
  )
endif()

if(MLIR_ENABLE_VULKAN_RUNNER)
  list(APPEND MLIR_TEST_DEPENDS
    mlir_vulkan_runtime
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
