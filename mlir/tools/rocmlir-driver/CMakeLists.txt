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
  MLIRExecutionEngine
  MLIRXModelPipeline
  )

add_llvm_executable(rocmlir-driver
  PARTIAL_SOURCES_INTENDED

  rocmlir-driver.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(rocmlir-driver)
target_link_libraries(rocmlir-driver PRIVATE ${LIBS})
mlir_check_link_libraries(rocmlir-driver)
