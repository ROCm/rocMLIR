configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Dependencies for the E2E test only
set(ROCK_E2E_DEPENDS
  FileCheck count not
  miopen-opt
  miopen-gen
  mlir-miopen-driver
  mlir-rocm-runner
  gen_e2e_tests
)

# target to generate the E2E tests
set(E2E_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(E2E_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generateE2ETest.py")
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(gen_e2e_tests
  COMMENT "Generating rock E2E tests in ${E2E_DIR}"
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnet50Config.toml
  COMMAND ${Python3_EXECUTABLE} ${E2E_GEN_SCRIPT} -d ${E2E_DIR} -i ${CONFIG_DIR}/Resnext101Config.toml
  DEPENDS ${CONFIG_DIR}/Resnext101Config.toml
)
# Add a new target to run the E2E test only
add_lit_testsuite(check-rock-e2e "Running rock e2e tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ROCK_E2E_DEPENDS}
  ARGS "--show-suites"
)
