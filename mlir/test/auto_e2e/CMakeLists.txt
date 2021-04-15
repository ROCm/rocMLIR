llvm_canonicalize_cmake_booleans(
  MLIR_ROCM_CONVERSIONS_ENABLED
  MLIR_ROCM_RUNNER_ENABLED
  )

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for linalg integration tests.
set(MLIR_DIALECT_LINALG_INTEGRATION_TEST_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries
# for the mlir cuda / rocm / spirv / vulkan runner tests.
set(MLIR_ROCM_WRAPPER_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if(MLIR_MIOPEN_DRIVER_XDLOPV2_TEST_ENABLED)
  set(MLIR_XDLOPV2 "-x2")
endif()

if(NOT MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED STREQUAL "none")
  set(MLIR_RANDOM_DATA "-rand ${MLIR_MIOPEN_DRIVER_RANDOM_DATA_SEED} ")
endif()


configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(MLIR_E2E_TEST_DEPENDS
  FileCheck count not
  mlir-miopen-driver
  mlir-miopen-lib-test
  mlir-opt
  mlir-translate
  opt
  llc
  )

add_lit_testsuite(check-e2e "Running the MLIR end-to-end tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_E2E_TEST_DEPENDS}
  )
set_target_properties(check-e2e PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_E2E_TEST_DEPENDS}
)
