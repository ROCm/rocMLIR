# Tests for Winograd integration
add_executable(winograd-solver-test solver_test.cpp)
target_link_libraries(winograd-solver-test
  PRIVATE
  MLIRRockWinograd
  MLIRSupport
)
