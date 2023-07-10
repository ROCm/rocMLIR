set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  ${rocmlir_test_libs}
  MLIRMHAL
  MLIRMHALTransforms
  MLIRMHALPipeline
  )

if (MLIR_ENABLE_EXECUTION_ENGINE)
  list(APPEND LIBS MLIRExecutionEngine)
endif()

add_rocmlir_tool(rocmlir-driver
  PARTIAL_SOURCES_INTENDED

  rocmlir-driver.cpp

  DEPENDS
  ${LIBS}
)


target_link_libraries(rocmlir-driver PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-driver)
mlir_check_link_libraries(rocmlir-driver)
