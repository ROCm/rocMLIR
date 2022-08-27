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
set(LOCAL_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/lit.local.cfg)

# Command to generate the E2E tests and a local lit config file to control
# whether the tests are supported or not.
# This command will be executed when
#   The local lit config file is missing OR
#   Any of the *.toml files are changed.
add_custom_command(OUTPUT ${LOCAL_CONFIG_FILE}
  COMMAND echo "if not config.enable_rock_e2e_test or config.no_AMD_GPU:" > ${LOCAL_CONFIG_FILE}
  COMMAND echo "  config.unsupported = True" >> ${LOCAL_CONFIG_FILE}
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnet50Config.toml
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnext101Config.toml
  DEPENDS ${CONFIG_DIR}/Resnext101Config.toml ${CONFIG_DIR}/Resnet50Config.toml
  COMMENT "Generating rock E2E tests in ${E2E_DIR}"
)

# A custom target depending on the above custom command to generate E2E tests
add_custom_target(e2e_tests
  DEPENDS ${LOCAL_CONFIG_FILE}
)

# Add a new target to run the E2E test only
add_lit_testsuite(check-rock-e2e "Running rock e2e tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ROCK_E2E_DEPENDS}
  #ARGS "--show-suites"
)
