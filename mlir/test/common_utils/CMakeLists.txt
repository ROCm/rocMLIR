file(GLOB SRC_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(rocmlir-common-python-test-utils
COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${SRC_PY_FILES} ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Copying common python test utils"
)

set(ROCMLIR_COMMON_PYTHON_TESTS_UTILS ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
