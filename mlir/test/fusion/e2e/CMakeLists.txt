# Dependencies for the E2E test only
set(FUSION_E2E_DEPENDS
  FileCheck count not
  rocmlir-opt
  rocmlir-gen
  rocmlir-driver
  mlir-cpu-runner
  xmir-runner
  mlir_rocm_runtime
  mlir_runner_utils
  mlir_c_runner_utils
  conv-validation-wrappers
  fusion-e2e-tests
)

set(E2E_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(E2E_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate-fusion-tests.py")
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LOCAL_CONFIG_FILE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg)
set(LOCAL_CONFIG_FILE_BIN ${CMAKE_CURRENT_BINARY_DIR}/lit.local.cfg)

# Add a command to copy lit.local.cfg
add_custom_command(OUTPUT ${LOCAL_CONFIG_FILE_BIN}
  COMMAND ${CMAKE_COMMAND} -E copy ${LOCAL_CONFIG_FILE_SRC} ${LOCAL_CONFIG_FILE_BIN}
  COMMENT "Copying lit.local.cfg"
)

# Add a target to trigger E2E tests generation unconditionally
add_custom_target(fusion-e2e-tests)
add_custom_command(TARGET fusion-e2e-tests
    COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating fusion E2E tests"
  )

# Add a new target to run the E2E test only
add_lit_testsuite(check-fusion-e2e "Running fusion e2e tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${FUSION_E2E_DEPENDS}
)
