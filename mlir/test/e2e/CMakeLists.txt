# Dependencies for the E2E test only
set(ROCK_E2E_DEPENDS
  FileCheck count not
  rocmlir-opt
  rocmlir-gen
  rocmlir-driver
  mlir-cpu-runner
  mlir_rocm_runtime
  conv-validation-wrappers
  e2e_tests
)

set(E2E_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(E2E_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generateE2ETest.py")
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LOCAL_CONFIG_FILE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg)
set(LOCAL_CONFIG_FILE_BIN ${CMAKE_CURRENT_BINARY_DIR}/lit.local.cfg)
# Create a list for configs
set(CONFIGS
  Resnet50Config.toml
  Resnext101Config.toml
  PaddedGemmConfig.toml
)
# Create a list for dummy files
set(CONFIG_DUMMY_FILES ${CONFIGS})
list(TRANSFORM CONFIG_DUMMY_FILES PREPEND ${E2E_DIR}/)
list(TRANSFORM CONFIG_DUMMY_FILES APPEND .copy)

# Add a command to copy lit.local.cfg
add_custom_command(OUTPUT ${LOCAL_CONFIG_FILE_BIN}
  COMMAND ${CMAKE_COMMAND} -E copy ${LOCAL_CONFIG_FILE_SRC} ${LOCAL_CONFIG_FILE_BIN}
  COMMENT "Copying lit.local.cfg"
)

# Add a command to generate E2E tests for each config file
foreach(config ${CONFIGS})
  set(config_file ${CONFIG_DIR}/${config})
  set(config_file_copy ${E2E_DIR}/${config}.copy)
  add_custom_command(OUTPUT ${config_file_copy}
    COMMAND ${CMAKE_COMMAND} -E copy ${config_file} ${config_file_copy}
    COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${config_file}
    DEPENDS ${config_file}
    COMMENT "Generating rock E2E tests for ${config}"
  )
endforeach()

# Add a target to trigger E2E tests generation conditionally
#   1. any of the *.toml files are changed OR
#   2. ${LOCAL_CONFIG_FILE_BIN} is missing
add_custom_target(e2e_tests
  DEPENDS ${LOCAL_CONFIG_FILE_BIN} ${CONFIG_DUMMY_FILES}
)

# Add a target to trigger E2E tests generation unconditionally
add_custom_target(re-gen-e2e-tests)
foreach(config ${CONFIGS})
  add_custom_command(TARGET re-gen-e2e-tests
    COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/${config}
  )
endforeach()

# Add a new target to run the E2E test only
add_lit_testsuite(check-rock-e2e "Running rock e2e tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ROCK_E2E_DEPENDS}
)
