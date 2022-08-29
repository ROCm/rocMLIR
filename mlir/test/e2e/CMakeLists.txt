# Dependencies for the E2E test only
set(ROCK_E2E_DEPENDS
  FileCheck count not
  miopen-opt
  miopen-gen
  mlir-miopen-driver
  mlir-rocm-runner
  e2e_tests
)

set(E2E_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(E2E_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generateE2ETest.py")
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LOCAL_CONFIG_FILE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg)
set(LOCAL_CONFIG_FILE_BIN ${CMAKE_CURRENT_BINARY_DIR}/lit.local.cfg)

# Command to generate the E2E tests and a local lit config file to control
# whether the tests are supported or not.
# The E2E tests are generated whenever any of the *.toml files are changed.
# The E2E tests generation can be triggered
#  1. conditionally if a target depends on ${LOCAL_CONFIG_FILE_BIN}.
#  2. unconditionally if a target depends on __e2e_tests_regen since it is always missing.
add_custom_command(OUTPUT ${LOCAL_CONFIG_FILE_BIN} __e2e_tests_regen
  COMMAND cp ${LOCAL_CONFIG_FILE_SRC} ${LOCAL_CONFIG_FILE_BIN}
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnet50Config.toml
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnext101Config.toml
  DEPENDS ${CONFIG_DIR}/Resnext101Config.toml ${CONFIG_DIR}/Resnet50Config.toml
  COMMENT "Generating rock E2E tests in ${E2E_DIR}"
)

# Trigger the e2e tests generation if
#   ${LOCAL_CONFIG_FILE_BIN} is missing or
#   any of *.toml files are changed
add_custom_target(e2e_tests
  DEPENDS ${LOCAL_CONFIG_FILE}
)

# Trigger the e2e tests generation unconditionally
add_custom_target(re-gen-e2e-tests
  DEPENDS __e2e_tests_regen
)

# Add a new target to run the E2E test only
add_lit_testsuite(check-rock-e2e "Running rock e2e tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ROCK_E2E_DEPENDS}
)
