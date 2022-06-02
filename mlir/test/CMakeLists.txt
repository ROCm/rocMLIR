add_subdirectory(CAPI)

llvm_canonicalize_cmake_booleans(
  MLIR_BINDINGS_PYTHON_ENABLED
  LLVM_BUILD_EXAMPLES
  MLIR_CUDA_CONVERSIONS_ENABLED
  MLIR_CUDA_RUNNER_ENABLED
  MLIR_ENABLE_ROCM_CONVERSIONS
  MLIR_ENABLE_ROCM_RUNNER
  MLIR_SPIRV_CPU_RUNNER_ENABLED
  MLIR_VULKAN_RUNNER_ENABLED

  MLIR_MIOPEN_DRIVER_ENABLED
  MLIR_MIOPEN_DRIVER_E2E_TEST_ENABLED
  MLIR_MIOPEN_DRIVER_MISC_E2E_TEST_ENABLED
  MLIR_MIOPEN_DRIVER_PR_E2E_TEST_ENABLED
  MLIR_MIOPEN_DRIVER_TIMING_TEST_ENABLED
  MLIR_MIOPEN_SQLITE_ENABLED
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
set(MLIR_ROCM_WRAPPER_LIBRARY_DIR ${MLIR_MIOPEN_LIB_DIR})

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "MLIR_MIOPEN_BIN_DIR: ${MLIR_MIOPEN_BIN_DIR}")
message(STATUS "MLIR_ROCM_WRAPPER_LIBRARY_DIR: ${MLIR_ROCM_WRAPPER_LIBRARY_DIR}")

# Need to explicitly give paths to llvm tools
set(LLVM_LIT_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(LLVM_EXTERNAL_LIT "${LLVM_LIT_TOOLS_DIR}/llvm-lit")
set(LLVM_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(MLIR_TOOLS_DIR ${LLVM_EXTERNAL_BIN_DIR})
set(MLIR_MIOPEN_TOOLS_DIR ${MLIR_MIOPEN_BIN_DIR})

# Passed to lit.site.cfg.py.in to set the -x2 flag for the mlir E2E tests.
set(MLIR_XDLOPS "")
if(MLIR_MIOPEN_DRIVER_XDLOPS_TEST_ENABLED)
  set(MLIR_XDLOPS "-x2")
endif()

# Passed to lit.site.cfg.py.in to set the -rand flag for the mlir E2E tests.
set(MLIR_RANDOM_DATA "")
if(NOT MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED STREQUAL "none")
  set(MLIR_RANDOM_DATA "-rand ${MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED} ")
endif()

# Passed to lit.site.cfg.py.in to set the -pv or -pv_with_gpu flag for the mlir E2E tests.
set(MLIR_POPULATE_VALIDATION "-pv")
if(MLIR_MIOPEN_DRIVER_TEST_GPU_VALIDATION)
  set(MLIR_POPULATE_VALIDATION "-pv_with_gpu")
endif()

# Sets execution options for lit test
set(LIT_EXE_OPTION "")
# Enables time tracking of individual tests
if(MLIR_MIOPEN_DRIVER_TIMING_TEST_ENABLED)
  set(LIT_EXE_OPTION "--time-tests")
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

set(MLIR_MIOPEN_TEST_DEPENDS
  FileCheck count not
  miopen-opt
  miopen-translate
  miopen-lsp-server
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

list(APPEND MLIR_MIOPEN_TEST_DEPENDS MLIRMIOpenUnitTests)

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND MLIR_MIOPEN_TEST_DEPENDS
    mlir-rocm-runner
    xmir-runner
  )
endif()

if(MLIR_BINDINGS_PYTHON_ENABLED)
  list(APPEND MLIR_MIOPEN_TEST_DEPENDS
    MLIRBindingsPythonExtension
    MLIRBindingsPythonTestOps
    MLIRTransformsBindingsPythonExtension
  )
endif()

# MIOpen dialect tools.
if(MLIR_MIOPEN_DRIVER_ENABLED)
  list(APPEND MLIR_MIOPEN_TEST_DEPENDS
    mlir-miopen-driver
    miopen-gen
    mlir-miopen-lib-test
    mlir-translate
    opt
    llc
    )
endif()

add_custom_target(check-mlir-miopen-build-only
  DEPENDS ${MLIR_MIOPEN_TEST_DEPENDS})

# check-mlir-miopen runs MLIR-MIOpen regression tests including E2E tests
# when MLIR_MIOPEN_DRIVER_E2E_TEST_ENABLED is set
add_lit_testsuite(check-mlir-miopen "Running the MLIR-MIOpen regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_MIOPEN_TEST_DEPENDS}
  ARGS ${LIT_EXE_OPTION}
  )
set_target_properties(check-mlir-miopen PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR-MIOPEN ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_MIOPEN_TEST_DEPENDS}
)
